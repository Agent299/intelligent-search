"""Stage 4 - linking and query expansion (doc p.152-155, Fig 5.5), then caching (Fig 5.6).

1. Entity retrieval (Tier 0): embed the segmented product category ('basmati rice') and take the nearest
   catalog products; their distinct product types are the alternatives that actually exist (poha, millet ...).
2. Intelligent expansion (LLM via Batch API): `expand_head` (Sonnet) for L1 queries, `expand_tail` (Haiku) for
   the rest. The guardrail only accepts product types from the retrieved alternatives.
3. Compile (Tier 0) the result into an OpenSearch query; if it returns zero in-stock results, the item is
   escalated to the next model in its chain.
4. Cache: `smart_query_cache` (source=batch, overwrites fallback/NRT entries, doc p.170) and the warm semantic
   tier `warm_query_cache`. Related searches: the expansion's own suggestions, else session-mined pairs.
"""

import asyncio
import time

from opensearchpy import AsyncOpenSearch
from psycopg.types.json import Jsonb
from redis.asyncio import Redis

from common.config import get_settings
from common.db import connect
from common.embedding_service import EmbeddingService
from common.normalize import query_hash
from common.schemas import CachedQuerySource, SegmentedQuery
from common.topics import PRODUCT_SEARCH_INDEX, SEGMENTED_QUERIES_QUEUE, WARM_QUERY_CACHE_INDEX
from pipelines.offline.batches import BatchJob, JobResult
from pipelines.offline.enrich import _consume_all
from search_service.compile import compile_expansion, with_filters
from search_service.filters import SearchFilters
from search_service.retrieval import Retriever

STAGE = "expand"
NEIGHBOURS = 60
MAX_ALTERNATIVES = 15


def entity_text(seg: SegmentedQuery) -> str:
    r = seg.segmentation_results
    return " ".join(x for x in (r.product_quality, r.product_category) if x) or seg.canonical


async def alternatives(retriever: Retriever, embeddings: EmbeddingService, texts: list[str]) -> list[list[dict]]:
    """Distinct product types among each text's nearest catalog products, nearest first."""
    vectors = await embeddings.embed(texts, "query")
    sem = asyncio.Semaphore(16)

    async def one(vec) -> list[dict]:
        async with sem:
            hits = await retriever.semantic(vec, SearchFilters(), NEIGHBOURS)
            products = await retriever.hydrate([h.product_id for h in hits], SearchFilters())
        seen: dict[str, str] = {}
        for h in hits:
            p = products.get(h.product_id)
            if p and p.get("product_type") and p["product_type"] not in seen:
                seen[p["product_type"]] = p["name"][:80]
        return [{"product_type": t, "example": ex} for t, ex in list(seen.items())[:MAX_ALTERNATIVES]]

    return await asyncio.gather(*(one(v) for v in vectors))


async def submit(run_id: str) -> int:
    latest: dict[str, SegmentedQuery] = {}
    for p in await _consume_all(SEGMENTED_QUERIES_QUEUE, f"offline-expand-{run_id}"):
        seg = SegmentedQuery.model_validate(p)
        latest[seg.canonical] = seg
    segs = list(latest.values())
    if not segs:
        return 0
    s = get_settings()
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=30), Redis.from_url(s.redis_url)
    try:
        alts = await alternatives(Retriever(os_client), EmbeddingService(redis), [entity_text(x) for x in segs])
    finally:
        await os_client.close()
        await redis.aclose()
    items = {}
    for seg, alt in zip(segs, alts, strict=True):
        if not alt:
            continue
        items[f"exp-{query_hash(seg.canonical)}"] = {
            "task": "expand_head" if seg.tier == "l1" else "expand_tail",
            "inputs": {"query": seg.canonical, "product_category": seg.segmentation_results.product_category,
                       "attributes": seg.segmentation_results.attributes, "alternatives": alt},
            "meta": seg.model_dump(),
        }
    BatchJob(run_id, STAGE).submit(items)
    return len(items)


async def _zero_result_ids(job: BatchJob, ids: list[str]) -> list[str]:
    """Compiled queries that return nothing in stock: escalation candidates (routing: escalate_on_invalid)."""
    os_client = AsyncOpenSearch(get_settings().opensearch_url, timeout=30)
    stock = SearchFilters().product_index_clauses()
    try:
        zero = []
        for cid in ids:
            item = job.items[cid]
            out = item["output"]
            q = with_filters(compile_expansion(out["product_types"], out["must_attributes"],
                                               item["meta"]["classification_results"]["taxonomy_path"]), stock)
            if (await os_client.count(index=PRODUCT_SEARCH_INDEX, body={"query": q}))["count"] == 0:
                zero.append(cid)
        return zero
    finally:
        await os_client.close()


def collect(run_id: str) -> tuple[JobResult, str | None, list[str]]:
    job = BatchJob(run_id, STAGE)
    res = job.collect()
    zero = asyncio.run(_zero_result_ids(job, list(res.ok)))
    for cid in zero:  # treat as invalid so the cascade can try the next model
        res.ok.pop(cid)
        job.items[cid].pop("output", None)
        res.invalid[cid] = "compiled query returns zero results"
    job.save()
    esc_batch, exhausted = job.escalate(res.invalid) if res.invalid else (None, [])
    return res, esc_batch, exhausted


async def write_caches(run_id: str) -> int:
    job = BatchJob(run_id, STAGE)
    done = [it for it in job.items.values() if "output" in it]
    if not done:
        return 0
    s = get_settings()
    version = int(time.time())
    related = await asyncio.to_thread(_log_related, [it["meta"]["normalized_query"] for it in done])
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=30), Redis.from_url(s.redis_url)
    try:
        vectors = await EmbeddingService(redis).embed([it["inputs"]["query"] for it in done], "query")
        rows = []
        for it, vec in zip(done, vectors, strict=True):
            out, meta, canonical = it["output"], it["meta"], it["inputs"]["query"]
            taxonomy_path = meta["classification_results"]["taxonomy_path"]
            es_query = {
                "query": compile_expansion(out["product_types"], out["must_attributes"], taxonomy_path),
                "expansion": {"product_types": out["product_types"], "must_attributes": out["must_attributes"],
                              "taxonomy_path": taxonomy_path, "model": it["final_model"]},
            }
            # The expansion's suggestions are already paid for and on-intent; session-mined pairs are the
            # fallback (on synthetic logs they are mostly chance co-occurrence, see README decision log).
            rel = out["related_searches"] or related.get(meta["normalized_query"], [])
            await os_client.index(index=WARM_QUERY_CACHE_INDEX, id=query_hash(canonical), body={
                "normalized_query": canonical, "embedding": vec, "es_query": es_query, "related_searches": rel,
                "source": CachedQuerySource.batch.value, "version": version})
            rows.append((canonical, Jsonb(es_query), rel, meta["frequency"], version, meta["normalized_query"]))
    finally:
        await os_client.close()
        await redis.aclose()
    await asyncio.to_thread(_store, rows)
    return len(rows)


def _log_related(queries: list[str]) -> dict[str, list[str]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT normalized_query, related_searches FROM query_stats WHERE normalized_query = ANY(%s)",
                    (queries,))
        return {r["normalized_query"]: r["related_searches"] for r in cur.fetchall() if r["related_searches"]}


def _store(rows: list[tuple]) -> None:
    with connect() as conn, conn.cursor() as cur:
        # Batch results always win over fallback / NRT entries (doc p.170 'handling data overwrites').
        cur.executemany(
            """INSERT INTO smart_query_cache (normalized_query, es_query, related_searches, source, version, frequency)
               VALUES (%s, %s, %s, 'batch', %s, %s)
               ON CONFLICT (normalized_query) DO UPDATE SET es_query = EXCLUDED.es_query,
                   related_searches = EXCLUDED.related_searches, source = 'batch', version = EXCLUDED.version,
                   frequency = EXCLUDED.frequency, updated_at = now()""",
            [(c, q, rel, v, f) for c, q, rel, f, v, _ in rows])
        cur.executemany("UPDATE query_stats SET status = 'expanded', updated_at = now() WHERE normalized_query = %s",
                        [(nq,) for *_, nq in rows])
