"""Stage 3 - query segmentation and classification (doc p.150-152, Fig 5.4). Tier 1 (Haiku, Batch API),
escalating to Sonnet only when the guardrail rejects Haiku's answer (category not among the candidates, or no
product category at all).

Selects enriched queries the fast path can't answer (fast_confidence < gate) and seen at least twice; gives the
model ~20 embedding-preselected taxonomy candidates; publishes SegmentedQuery payloads to
`segmented_queries_queue`.
"""

import asyncio

from aiokafka import AIOKafkaProducer
from redis.asyncio import Redis

from common.config import get_settings
from common.db import connect
from common.embedding_service import EmbeddingService
from common.normalize import query_hash
from common.schemas import ClassificationResult, EnrichedQuery, SegmentationResult, SegmentedQuery
from common.topics import QUERY_PROCESS_QUEUE, SEGMENTED_QUERIES_QUEUE
from pipelines.offline.batches import BatchJob, JobResult
from pipelines.offline.enrich import _consume_all
from search_service.taxonomy import TaxonomyIndex

GATE = 0.6  # same threshold as the online fallback: pre-compute exactly the queries that would need the LLM
STAGE = "segment"
REFRESH_DAYS = 7  # a batch-cached (or failed) query is not paid for again within this window


def recently_processed() -> set[str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT normalized_query AS q FROM smart_query_cache
                       WHERE source = 'batch' AND updated_at > now() - make_interval(days => %s)
                       UNION SELECT coalesce(corrected_query, normalized_query) FROM query_stats
                       WHERE status = 'failed' AND updated_at > now() - make_interval(days => %s)""",
                    (REFRESH_DAYS, REFRESH_DAYS))
        return {r["q"] for r in cur.fetchall()}


def select(queries: list[EnrichedQuery], limit: int | None, skip: set[str] | None = None) -> list[EnrichedQuery]:
    skip = skip or set()
    picked = [q for q in queries
              if q.fast_confidence is not None and q.fast_confidence < GATE and q.canonical not in skip]
    picked.sort(key=lambda q: -q.frequency)  # most valuable first, so a pilot covers the head
    return picked[:limit] if limit else picked


async def submit(run_id: str, limit: int | None = None, dry_run: bool = False) -> tuple[int, int]:
    latest: dict[str, EnrichedQuery] = {}  # the topic may hold several enrichment runs: keep the latest per query
    for p in await _consume_all(QUERY_PROCESS_QUEUE, f"offline-segment-{run_id}"):
        q = EnrichedQuery.model_validate(p)
        latest[q.canonical] = q
    queries = list(latest.values())
    picked = select(queries, limit, await asyncio.to_thread(recently_processed))
    if dry_run or not picked:
        return len(queries), len(picked)
    redis = Redis.from_url(get_settings().redis_url)
    try:
        embeddings = EmbeddingService(redis)
        taxonomy = await TaxonomyIndex.build(embeddings)
        vectors = await embeddings.embed([q.canonical for q in picked], "query")
    finally:
        await redis.aclose()
    items = {
        f"seg-{query_hash(q.canonical)}": {
            "task": "segment",
            "inputs": {"query": q.canonical, "candidate_categories": taxonomy.candidates(v)},
            "meta": q.model_dump(),
        }
        for q, v in zip(picked, vectors, strict=True)
    }
    BatchJob(run_id, STAGE).submit(items)
    return len(queries), len(picked)


def collect(run_id: str) -> tuple[JobResult, str | None, list[str]]:
    """Validate results, escalate rejects to the next model. Returns (result, escalation batch, exhausted)."""
    job = BatchJob(run_id, STAGE)
    res = job.collect()
    for cid, reason in res.invalid.items():
        job.items[cid]["invalid_reason"] = reason
    job.save()
    esc_batch, exhausted = job.escalate(res.invalid) if res.invalid else (None, [])
    return res, esc_batch, exhausted


async def publish(run_id: str) -> int:
    job = BatchJob(run_id, STAGE)
    done = [it for it in job.items.values() if "output" in it]
    producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_bootstrap, linger_ms=20)
    await producer.start()
    try:
        for it in done:
            out, meta = it["output"], it["meta"]
            seg = SegmentedQuery(
                **meta,
                segmentation_results=SegmentationResult(
                    attributes=[a["value"] for a in out["attributes"]], product_category=out["product_category"],
                    product_quality=out["product_quality"], brand=out["brand"]),
                classification_results=ClassificationResult(taxonomy_path=out["taxonomy_path"]),
            )
            await producer.send(SEGMENTED_QUERIES_QUEUE, seg.model_dump_json().encode(), key=seg.canonical.encode())
        await producer.flush()
    finally:
        await producer.stop()
    failed = [it["meta"]["normalized_query"] for it in job.items.values() if "output" not in it]
    await asyncio.to_thread(_mark, [it["meta"]["normalized_query"] for it in done], failed)
    return len(done)


def _mark(segmented: list[str], failed: list[str]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany("UPDATE query_stats SET status = 'segmented', updated_at = now() WHERE normalized_query = %s",
                        [(q,) for q in segmented])
        cur.executemany("UPDATE query_stats SET status = 'failed', updated_at = now() WHERE normalized_query = %s",
                        [(q,) for q in failed])
