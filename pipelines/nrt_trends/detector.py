"""Near-real-time trending path (doc p.169-170, Fig 5.10).

    uv run python -m pipelines.nrt_trends.detector          # NRT_THRESHOLD=5 for a demo

Consumes `query_logs`, counts searches per corrected query in a 5-minute window sliding every 10 s, and for each
query at or above the threshold:

    recently decided?  -> skip            (Redis, nrt_recheck_s)
    already cached?    -> skip            exact L2 / warm entry of any source (batch, fallback, nrt)
    fast path >= 0.6?  -> skip            Tier 0, the same gate as the batch pipeline: confident head queries
                                          never cost an LLM call
    hourly budget?     -> retry next tick nrt_max_llm_per_hour
    nrt_segment        -> Tier 1 (Haiku via the gateway, short prompt, no RAG)
    compile + count    -> skip if zero in-stock results
    write              warm tier + L2 + smart_query_cache, source=nrt; never over a batch entry, and the next
                       batch run overwrites it (doc p.170)

In this system the request path's LLM fallback already caches a hard query on its first search, so NRT mostly
catches what the request path couldn't: searches while the gateway was down or slow, or with the fallback
switched off. It does so in the background, off the request path.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from aiokafka import AIOKafkaConsumer
from opensearchpy import AsyncOpenSearch
from prometheus_client import Counter, start_http_server
from psycopg.types.json import Jsonb
from redis.asyncio import Redis

from common.config import Settings, get_settings
from common.db import connect
from common.embedding_service import EmbeddingService
from common.normalize import normalize_query, query_hash
from common.schemas import CachedQuerySource, EventType
from common.topics import PRODUCT_SEARCH_INDEX, QUERY_LOGS, WARM_QUERY_CACHE_INDEX
from pipelines.nrt_trends.window import SlidingWindow
from search_service.cache import L2_TTL_S, QueryCache, l2_key
from search_service.compile import compile_expansion, with_filters
from search_service.engine import SearchEngine
from search_service.events import EventSink
from search_service.filters import SearchFilters
from search_service.fusion import confidence
from search_service.gateway_client import GatewayClient
from search_service.negation import parse_negations
from search_service.retrieval import Retriever
from search_service.vocab import build_spell_corrector

log = logging.getLogger(__name__)

OUTCOMES = Counter("nrt_trend_outcomes_total", "What happened to each trending query", ["outcome"])
SEEN_PREFIX = "nrt:seen:"
BUDGET_PREFIX = "nrt:budget:"
TRENDING_KEY = "nrt:trending"   # sorted set query -> window count, for dashboards / a 'trending' UI


@dataclass
class Decision:
    # seen | cached | confident | budget | gateway_error | no_product_types | zero_results | cached_nrt
    outcome: str
    detail: dict | None = None

    @property
    def final(self) -> bool:
        """Transient outcomes are retried at the next tick; everything else is remembered for nrt_recheck_s."""
        return self.outcome not in {"budget", "gateway_error", "seen"}


class NrtProcessor:
    def __init__(self, engine: SearchEngine, cache: QueryCache, gateway: GatewayClient, redis: Redis,
                 os_client: AsyncOpenSearch | None, settings: Settings, persist=None):
        self.engine, self.cache, self.gateway = engine, cache, gateway
        self.redis, self.os_client, self.settings = redis, os_client, settings
        self.persist = persist  # smart_query_cache writer; None in tests

    async def process(self, query: str) -> Decision:
        seen = f"{SEEN_PREFIX}{query_hash(query)}"
        if await self.redis.exists(seen):
            return Decision("seen")
        decision = await self._decide(query)
        OUTCOMES.labels(decision.outcome).inc()
        if decision.final:
            await self.redis.set(seen, decision.outcome, ex=self.settings.nrt_recheck_s)
        return decision

    async def _decide(self, query: str) -> Decision:
        if await self.cache.exact(query) is not None:
            return Decision("cached")
        parsed = parse_negations(query)
        fast = await self.engine.fast_path(parsed, SearchFilters())
        score = confidence(fast.keyword, fast.semantic, fast.ranked).score
        if score >= self.settings.confidence_threshold:
            return Decision("confident", {"fast_confidence": score})
        if not await self._take_budget():
            return Decision("budget")
        result = await self.gateway.run("nrt_segment", {"query": query})
        if result is None:
            return Decision("gateway_error")
        out = result.get("output") or {}
        if not result.get("valid") or not out.get("product_types"):
            return Decision("no_product_types", {"output": out})
        compiled = compile_expansion(out["product_types"], out.get("attributes") or [], None)
        if not await self._count(compiled):
            return Decision("zero_results", {"output": out})
        es_query = {"query": compiled, "nrt": {"product_types": out["product_types"],
                                               "attributes": out.get("attributes") or []}}
        await self._write(query, es_query, fast.vector)
        return Decision("cached_nrt", {"fast_confidence": score, "es_query": es_query,
                                       "cost_usd": result.get("cost_usd")})

    async def _take_budget(self) -> bool:
        key = f"{BUDGET_PREFIX}{time.strftime('%Y%m%d%H', time.gmtime())}"
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.incr(key)
            pipe.expire(key, 7200)
            used, _ = await pipe.execute()
        return used <= self.settings.nrt_max_llm_per_hour

    async def _count(self, compiled: dict) -> int:
        if self.os_client is None:
            return 1
        body = {"query": with_filters(compiled, SearchFilters().product_index_clauses())}
        return (await self.os_client.count(index=PRODUCT_SEARCH_INDEX, body=body))["count"]

    async def _write(self, query: str, es_query: dict, vector: list[float]) -> None:
        version = int(time.time())
        doc = {"normalized_query": query, "es_query": es_query, "related_searches": [],
               "source": CachedQuerySource.nrt.value, "version": version}
        if self.os_client is not None:
            await self.os_client.index(index=WARM_QUERY_CACHE_INDEX, id=query_hash(query),
                                       body={**doc, "embedding": vector})
        # NX: never replace an L2 entry another source wrote meanwhile
        await self.redis.set(l2_key(query), json.dumps(doc), ex=L2_TTL_S, nx=True)
        if self.persist is not None:
            await asyncio.to_thread(self.persist, query, es_query, version)


def upsert_nrt(query: str, es_query: dict, version: int) -> None:
    """Persistent tier: insert, or refresh an earlier NRT entry; never touch a batch or fallback entry."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO smart_query_cache (normalized_query, es_query, source, version)
                       VALUES (%s, %s, 'nrt', %s)
                       ON CONFLICT (normalized_query) DO UPDATE SET es_query = EXCLUDED.es_query,
                           version = EXCLUDED.version, updated_at = now()
                       WHERE smart_query_cache.source = 'nrt'""", (query, Jsonb(es_query), version))


async def run() -> None:
    s = get_settings()
    start_http_server(s.nrt_metrics_port)
    spell = await asyncio.to_thread(build_spell_corrector)
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=10), Redis.from_url(s.redis_url)
    gateway = GatewayClient(s.genai_gateway_url, timeout_s=8.0)
    engine = SearchEngine(Retriever(os_client), EmbeddingService(redis), spell, EventSink(None, None), s)
    cache = QueryCache(redis, os_client, s.warm_tier_min_similarity, s.results_cache_ttl_s)
    processor = NrtProcessor(engine, cache, gateway, redis, os_client, s, persist=upsert_nrt)
    window = SlidingWindow(s.nrt_window_s, s.nrt_slide_s)
    consumer = AIOKafkaConsumer(QUERY_LOGS, bootstrap_servers=s.kafka_bootstrap, group_id="nrt-trends",
                                auto_offset_reset="latest")  # trends are about now; history is the batch's job
    await consumer.start()
    print(f"[nrt] consuming {QUERY_LOGS}: threshold {s.nrt_threshold} searches / {s.nrt_window_s}s, "
          f"slide {s.nrt_slide_s}s, budget {s.nrt_max_llm_per_hour} LLM calls/h", flush=True)

    async def tick() -> None:
        while True:
            await asyncio.sleep(s.nrt_slide_s)
            trending = window.trending(time.time(), s.nrt_threshold)
            if not trending:
                continue
            async with redis.pipeline(transaction=False) as pipe:
                pipe.delete(TRENDING_KEY)
                pipe.zadd(TRENDING_KEY, dict(trending))
                pipe.expire(TRENDING_KEY, s.nrt_window_s)
                await pipe.execute()
            for query, n in trending:
                try:
                    d = await processor.process(query)
                except Exception:
                    log.warning("NRT processing failed for %r", query, exc_info=True)
                    continue
                if d.outcome != "seen":
                    print(f"[nrt] {query!r} ({n} in window): {d.outcome} {json.dumps(d.detail or {})}", flush=True)

    ticker = asyncio.create_task(tick())
    try:
        async for record in consumer:
            evt = json.loads(record.value)
            if evt.get("event_type", EventType.search.value) != EventType.search.value:
                continue
            normalized = normalize_query(evt.get("raw_query") or "")
            if normalized:
                window.add(spell.correct(normalized), time.time())
    finally:
        ticker.cancel()
        await consumer.stop()
        await gateway.aclose()
        await os_client.close()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
