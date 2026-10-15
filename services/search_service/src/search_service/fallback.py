"""Fallback path (doc p.157, Fig 5.14): for queries the fast path can't answer confidently, ask the LLM to
rewrite the query into system terms, compile the rewrite into a search, and retry.

Cost controls:
- only low-confidence / zero-result queries get here (~the doc's P99 slice)
- the rewrite (Tier 1, Haiku) is memoized in Redis per normalized query, including 'no usable rewrite', so a
  repeated query never pays twice
- the prompt carries ~20 embedding-preselected categories, not the whole taxonomy
- compiling the rewrite into a query is code (Tier 0)

Robustness: any gateway failure returns the fast-path results unchanged (never an error page).
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch
from prometheus_client import Counter, Histogram
from redis.asyncio import Redis

from common.db import connect
from common.embedding_service import EmbeddingService
from common.normalize import query_hash
from common.schemas import CachedQuerySource
from common.topics import WARM_QUERY_CACHE_INDEX
from search_service.filters import SearchFilters
from search_service.fusion import SEM_CEIL, SEM_FLOOR, Candidate, rrf_fuse_many
from search_service.gateway_client import GatewayClient
from search_service.retrieval import Hit, Retriever, structured_keyword_body
from search_service.taxonomy import TaxonomyIndex

log = logging.getLogger(__name__)

OUTCOMES = Counter("search_fallback_total", "Fallback path outcomes", ["outcome"])
LATENCY = Histogram("search_fallback_latency_seconds", "Fallback path latency",
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 8))

MEMO_TTL_S = 7 * 24 * 3600
MEMO_EMPTY_TTL_S = 24 * 3600
PER_KEYWORD_K = 30


@dataclass
class Rewrite:
    keywords: list[str]
    attributes: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.keywords)

    def as_text(self) -> str:
        return ", ".join(self.keywords)


@dataclass
class FallbackResult:
    ranked: list[Candidate]
    products: dict[str, dict]
    confidence: float
    rewrite: Rewrite
    memo_hit: bool


class FallbackPath:
    def __init__(self, retriever: Retriever, embeddings: EmbeddingService, gateway: GatewayClient,
                 taxonomy: "TaxonomyIndex | None", redis: Redis | None, os_client: AsyncOpenSearch | None,
                 persist: Callable[[str, dict, int], None] | None = None):
        self.retriever = retriever
        self.embeddings = embeddings
        self.gateway = gateway
        self.taxonomy = taxonomy
        self.redis = redis
        self.os_client = os_client
        self.persist = persist  # writes the persistent smart_query_cache tier; None disables it
        self._background: set[asyncio.Task] = set()

    async def run(self, normalized_query: str, query_vector: list[float],
                  filters: SearchFilters) -> FallbackResult | None:
        started = time.perf_counter()
        try:
            rewrite, memo_hit = await self._get_rewrite(normalized_query, query_vector)
            if rewrite is None or not rewrite.usable:
                OUTCOMES.labels("no_rewrite").inc()
                return None
            ranked, products, conf = await self.search_rewrite(rewrite, filters)
            if not ranked:
                OUTCOMES.labels("rewrite_no_results").inc()
                return None
            if not memo_hit:
                self._in_background(self._write_back(normalized_query, query_vector, rewrite))
            return FallbackResult(ranked, products, conf, rewrite, memo_hit)
        finally:
            LATENCY.observe(time.perf_counter() - started)

    # -- rewrite (memoized) --

    async def _get_rewrite(self, normalized: str, query_vector: list[float]) -> tuple[Rewrite | None, bool]:
        key = f"rewrite:v1:{query_hash(normalized)}"
        memo = await self._memo_get(key)
        if memo is not None:
            OUTCOMES.labels("memo_hit").inc()
            return (Rewrite(**memo) if memo else None), True
        candidates = self.taxonomy.candidates(query_vector) if self.taxonomy else []
        result = await self.gateway.rewrite(normalized, candidates)
        if result is None:  # gateway down / timed out: don't memoize, try again next time
            OUTCOMES.labels("gateway_error").inc()
            return None, False
        out = result.get("output") or {}
        rewrite = Rewrite(keywords=out.get("keywords", []), attributes=out.get("attributes", []),
                          categories=out.get("categories", []))
        valid = result.get("valid") and rewrite.usable
        await self._memo_set(key, rewrite.__dict__ if valid else {}, MEMO_TTL_S if valid else MEMO_EMPTY_TTL_S)
        return (rewrite if valid else None), False

    async def _memo_get(self, key: str) -> dict | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception:
            return None

    async def _memo_set(self, key: str, value: dict, ttl: int) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(key, json.dumps(value), ex=ttl)
        except Exception:
            log.warning("could not memoize rewrite", exc_info=True)

    # -- retry search with the rewrite (also replays cached fallback rewrites, Phase 6) --

    async def search_rewrite(self, rw: Rewrite,
                             filters: SearchFilters) -> tuple[list[Candidate], dict[str, dict], float]:
        body = structured_keyword_body(rw.keywords, rw.attributes, rw.categories, filters, size=100)
        vectors = await self.embeddings.embed(rw.keywords, "query")
        lists = await asyncio.gather(
            self.retriever.run_keyword_body(body),
            *(self.retriever.semantic(v, filters, PER_KEYWORD_K) for v in vectors))
        keyword_hits, per_keyword = lists[0], list(lists[1:])
        ids = list(dict.fromkeys([h.product_id for lst in lists for h in lst]))

        products = await self.retriever.hydrate(ids, filters)  # categories boost (in BM25), never filter
        keep = lambda hits: [h for h in hits if h.product_id in products]  # noqa: E731
        keyword_hits, per_keyword = keep(keyword_hits), [keep(lst) for lst in per_keyword]
        priors = {pid: p.get("static_score") or 0.0 for pid, p in products.items()}
        ranked = interleave_by_keyword(rrf_fuse_many([keyword_hits, *per_keyword], priors), per_keyword)
        return ranked, products, rewrite_confidence(keyword_hits, per_keyword)

    # -- write-back for Phase 6 cache tiers --

    async def _write_back(self, normalized: str, query_vector: list[float], rw: Rewrite) -> None:
        version = int(time.time())
        compiled = structured_keyword_body(rw.keywords, rw.attributes, rw.categories, SearchFilters(), size=100)
        es_query = {"rewrite": rw.__dict__, "query": compiled["query"]}
        try:
            if self.os_client is not None:
                await self.os_client.index(index=WARM_QUERY_CACHE_INDEX, id=query_hash(normalized), body={
                    "normalized_query": normalized, "embedding": query_vector, "es_query": es_query,
                    "related_searches": [], "source": CachedQuerySource.fallback.value, "version": version})
            if self.persist is not None:
                await asyncio.to_thread(self.persist, normalized, es_query, version)
        except Exception:
            log.warning("fallback write-back failed", exc_info=True)

    def _in_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)


def upsert_smart_cache(normalized: str, es_query: dict[str, Any], version: int) -> None:
    """Persistent tier. A fallback entry never replaces one produced by the offline batch pipeline."""
    from psycopg.types.json import Jsonb

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO smart_query_cache (normalized_query, es_query, source, version)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (normalized_query) DO UPDATE SET es_query = EXCLUDED.es_query,
                   version = EXCLUDED.version, updated_at = now()
               WHERE smart_query_cache.source <> 'batch'""",
            (normalized, Jsonb(es_query), CachedQuerySource.fallback.value, version))


def interleave_by_keyword(ranked: list[Candidate], per_keyword: list[list[Hit]]) -> list[Candidate]:
    """Round-robin across rewrite keywords so a multi-intent query ('party cups, chips, soda, string lights')
    shows each need near the top instead of letting the keyword with the most strong matches fill the page.
    Each candidate belongs to the keyword whose list ranks it best; candidates found only by BM25 go last;
    within a group the fused order is kept."""
    best: dict[str, tuple[int, int]] = {}  # product -> (rank, keyword index)
    for k, hits in enumerate(per_keyword):
        for rank, h in enumerate(hits):
            if h.product_id not in best or rank < best[h.product_id][0]:
                best[h.product_id] = (rank, k)
    groups: list[list[Candidate]] = [[] for _ in range(len(per_keyword) + 1)]
    for c in ranked:
        groups[best[c.product_id][1] if c.product_id in best else -1].append(c)
    out: list[Candidate] = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.extend(g[i] for g in groups[:-1] if i < len(g))
    return out + groups[-1]


def rewrite_confidence(keyword_hits: list[Hit], per_keyword: list[list[Hit]]) -> float:
    """Confidence for a rewritten search: how strongly each rewrite keyword matches catalog products
    (mean top-1 cosine across keywords, rescaled like the fast path) plus BM25 coverage."""
    tops = [lst[0].score for lst in per_keyword if lst]
    if not tops and not keyword_hits:
        return 0.0
    mean_top = sum(tops) / len(tops) if tops else 0.0
    strength = min(1.0, max(0.0, (mean_top - SEM_FLOOR) / (SEM_CEIL - SEM_FLOOR)))
    coverage = min(1.0, len(keyword_hits) / 5)
    return round(0.7 * strength + 0.3 * coverage, 3)
