"""Query cache tiers (doc p.170-173). Values are compiled search queries, not results, so a cached entry always
reflects current price and stock; a small results cache sits in front of the hottest queries only.

    results cache  Redis, hot queries only, TTL (results_cache_ttl_s), purged per product on catalog_updates
    L1             in-process dict: the hot list (top 5% by frequency), loaded from S3, reloaded on list_update
    L2             Redis `qcache:v1:{hash}`: top 20% queries that have a cached query, written by the hot-list job
    warm (exact)   OpenSearch `warm_query_cache` document id = query hash (batch + fallback write-backs)
    warm (similar) k-NN over the same index, cosine >= warm_tier_min_similarity

Everything here is Tier 0 and fails open: a cache outage degrades to the fast path, never to an error.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError
from prometheus_client import Counter
from redis.asyncio import Redis

from common.normalize import query_hash
from common.schemas import ServedBy
from common.topics import WARM_QUERY_CACHE_INDEX

log = logging.getLogger(__name__)

LOOKUPS = Counter("search_cache_lookups_total", "Query cache lookups", ["tier", "result"])

L2_PREFIX = "qcache:v1:"
L2_TTL_S = 2 * 24 * 3600          # > the daily pipeline cadence; queries that leave the top 20% expire
RESULTS_PREFIX = "results:v1:"
RESULTS_INDEX_PREFIX = "results:idx:"  # product id -> results keys containing it (event-based purge)
RESULTS_MAX_IDS = 200
HOT_LIST_BUCKET, HOT_LIST_KEY = "hotlist", "hot_list.json"


def l2_key(query: str) -> str:
    return f"{L2_PREFIX}{query_hash(query)}"


def results_index_key(product_id: str) -> str:
    return f"{RESULTS_INDEX_PREFIX}{product_id}"


@dataclass
class CacheEntry:
    query: str                      # the cached query text (differs from the request on a similar hit)
    es_query: dict[str, Any]        # {"query": compiled bool query, "expansion" | "rewrite": ...}
    related_searches: list[str] = field(default_factory=list)
    source: str = "batch"           # batch | fallback | nrt
    similarity: float = 1.0

    @classmethod
    def from_doc(cls, doc: dict, similarity: float = 1.0) -> "CacheEntry":
        return cls(doc["normalized_query"], doc["es_query"], doc.get("related_searches") or [],
                   doc.get("source", "batch"), similarity)


@dataclass
class CachedResults:
    ids: list[str]
    served_by: str                  # the path that originally computed these results
    confidence: float | None
    related_searches: list[str]
    rewritten_query: str | None
    cached_query: str | None
    sims: dict[str, float] = field(default_factory=dict)  # re-rank feature for the retrieval head (Phase 8)


class QueryCache:
    def __init__(self, redis: Redis | None, os_client: AsyncOpenSearch | None, min_similarity: float,
                 results_ttl_s: int):
        self.redis = redis
        self.os_client = os_client
        self.min_similarity = min_similarity
        self.results_ttl_s = results_ttl_s
        self.hot: dict[str, dict] = {}  # L1: query -> hot-list record ({"query", "es_query"?, "related_searches"})
        self.hot_version = 0

    # -- L1: hot list --

    def load_hot_list(self, hot_list: dict) -> None:
        """{"version": int, "queries": [{"query", "es_query"?, "related_searches"?, "source"?}]}"""
        self.hot = {q["query"]: q for q in hot_list.get("queries", [])}
        self.hot_version = int(hot_list.get("version", 0))

    def is_hot(self, query: str) -> bool:
        return query in self.hot

    def related(self, query: str) -> list[str]:
        """Related searches for a hot query (log-mined, or the expansion's own), whichever path serves it."""
        rec = self.hot.get(query)
        return (rec or {}).get("related_searches") or []

    # -- exact lookups: L1 -> L2 -> warm by id --

    async def exact(self, query: str) -> tuple[CacheEntry, ServedBy] | None:
        rec = self.hot.get(query)
        if rec is not None and rec.get("es_query"):
            LOOKUPS.labels("l1", "hit").inc()
            return CacheEntry(query, rec["es_query"], rec.get("related_searches") or [],
                              rec.get("source", "batch")), ServedBy.l1
        LOOKUPS.labels("l1", "miss").inc()
        entry = await self._l2(query)
        if entry is not None:
            return entry, ServedBy.l2
        entry = await self._warm_by_id(query)
        return (entry, ServedBy.warm) if entry is not None else None

    async def _l2(self, query: str) -> CacheEntry | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(l2_key(query))
        except Exception:
            log.warning("L2 lookup failed", exc_info=True)
            return None
        LOOKUPS.labels("l2", "hit" if raw else "miss").inc()
        return CacheEntry.from_doc(json.loads(raw)) if raw else None

    async def _warm_by_id(self, query: str) -> CacheEntry | None:
        if self.os_client is None:
            return None
        try:
            doc = await self.os_client.get(index=WARM_QUERY_CACHE_INDEX, id=query_hash(query),
                                           _source_excludes=["embedding"])
        except NotFoundError:
            LOOKUPS.labels("warm_exact", "miss").inc()
            return None
        except Exception:
            log.warning("warm exact lookup failed", exc_info=True)
            return None
        LOOKUPS.labels("warm_exact", "hit").inc()
        return CacheEntry.from_doc(doc["_source"])

    # -- warm tier: similar queries --

    async def similar(self, vector: list[float]) -> CacheEntry | None:
        if self.os_client is None:
            return None
        try:
            resp = await self.os_client.search(index=WARM_QUERY_CACHE_INDEX, body={
                "size": 1, "_source": {"excludes": ["embedding"]},
                "query": {"knn": {"embedding": {"vector": vector, "k": 1}}}})
        except Exception:
            log.warning("warm similarity lookup failed", exc_info=True)
            return None
        hits = resp["hits"]["hits"]
        sim = 2 * hits[0]["_score"] - 1 if hits else 0.0  # lucene cosinesimil score is (1 + cos) / 2
        if not hits or sim < self.min_similarity:
            LOOKUPS.labels("warm_similar", "miss").inc()
            return None
        LOOKUPS.labels("warm_similar", "hit").inc()
        return CacheEntry.from_doc(hits[0]["_source"], similarity=round(sim, 3))

    # -- results cache (hot queries only) --

    def results_key(self, query: str, filters: dict, sort_by: str) -> str:
        raw = json.dumps([query, filters, sort_by], sort_keys=True)
        return f"{RESULTS_PREFIX}{self.hot_version}:{hashlib.sha1(raw.encode()).hexdigest()}"

    async def get_results(self, key: str) -> CachedResults | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(key)
        except Exception:
            log.warning("results cache lookup failed", exc_info=True)
            return None
        LOOKUPS.labels("results", "hit" if raw else "miss").inc()
        return CachedResults(**json.loads(raw)) if raw else None

    async def put_results(self, key: str, results: CachedResults) -> None:
        if self.redis is None:
            return
        results.ids = results.ids[:RESULTS_MAX_IDS]
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.set(key, json.dumps(results.__dict__), ex=self.results_ttl_s)
                for pid in results.ids:
                    pipe.sadd(results_index_key(pid), key)
                    pipe.expire(results_index_key(pid), self.results_ttl_s)
                await pipe.execute()
        except Exception:
            log.warning("results cache write failed", exc_info=True)


def purge_results_for(redis_sync, product_id: str) -> int:
    """Event-based invalidation (doc p.172): drop every cached result list containing a product whose price or
    stock changed. Sync, for the catalog_updates consumer. Returns the number of result lists purged."""
    idx = results_index_key(product_id)
    keys = redis_sync.smembers(idx)
    if keys:
        redis_sync.delete(*keys)
    redis_sync.delete(idx)
    return len(keys)
