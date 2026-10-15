"""Stage 2 - parallel enrichment (doc p.150, Fig 5.3). Tier 0.

Consumes `log_processing_queue`; for each unique query:
- normalization + spell correction (typo variants are folded into their canonical spelling as aliases)
- context: dominant device / location / time of day (computed during aggregation)
- the offline fast-path gate: the online engine's own confidence for this query. Only queries the fast path
  can't answer well are worth an LLM call downstream (cost policy).
Queries seen only once are not enriched further (status `skipped_rare`): the online fallback + memo handles them.

Publishes EnrichedQuery payloads to `query_process_queue`.
"""

import asyncio
import json
from collections import defaultdict

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from opensearchpy import AsyncOpenSearch
from redis.asyncio import Redis

from common.config import get_settings
from common.db import connect
from common.embedding_service import EmbeddingService
from common.normalize import SpellCorrector, normalize_query
from common.schemas import EnrichedQuery, QueryContext
from common.topics import LOG_PROCESSING_QUEUE, QUERY_PROCESS_QUEUE
from search_service.engine import SearchEngine
from search_service.events import EventSink
from search_service.filters import SearchFilters
from search_service.fusion import confidence
from search_service.negation import parse_negations
from search_service.retrieval import Retriever
from search_service.vocab import build_spell_corrector

MIN_FREQUENCY = 2
CONCURRENCY = 16


async def _consume_all(topic: str, group: str) -> list[dict]:
    consumer = AIOKafkaConsumer(topic, bootstrap_servers=get_settings().kafka_bootstrap, group_id=group,
                                auto_offset_reset="earliest", enable_auto_commit=False)
    await consumer.start()
    out: list[dict] = []
    try:
        while batch := await consumer.getmany(timeout_ms=4000, max_records=10000):
            for records in batch.values():
                out.extend(json.loads(r.value) for r in records)
        await consumer.commit()
    finally:
        await consumer.stop()
    return out


def fold_aliases(payloads: list[dict], spell: SpellCorrector) -> list[EnrichedQuery]:
    """Spell-correct, then merge queries that share a corrected form: 'beef jekry' joins 'beef jerky'
    (frequencies add up; the most frequent spelling supplies context)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in payloads:
        normalized = normalize_query(p["normalized_query"])
        if normalized:
            groups[spell.correct(normalized)].append({**p, "normalized_query": normalized})
    out = []
    for corrected, members in groups.items():
        members.sort(key=lambda m: -m["frequency"])
        head = members[0]
        out.append(EnrichedQuery(
            original_query=head["original_query"], normalized_query=head["normalized_query"],
            corrected_query=corrected if corrected != head["normalized_query"] else None,
            aliases=sorted({m["normalized_query"] for m in members} - {corrected}),
            frequency=sum(m["frequency"] for m in members), tier=head.get("tier", "warm"),
            context=QueryContext(**(head.get("context") or {})),
        ))
    return out


async def gate(queries: list[EnrichedQuery], engine: SearchEngine) -> None:
    """Fill fast_confidence using the online fast path (embeddings precomputed in one batch)."""
    texts = [parse_negations(q.canonical).positive for q in queries]
    for i in range(0, len(texts), 256):
        await engine.embeddings.embed(texts[i:i + 256], "query")  # warms the Redis embedding cache
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(q: EnrichedQuery) -> None:
        async with sem:
            fast = await engine.fast_path(parse_negations(q.canonical), SearchFilters())
            q.fast_confidence = confidence(fast.keyword, fast.semantic, fast.ranked).score

    await asyncio.gather(*(one(q) for q in queries))


def _store(enriched: list[EnrichedQuery], rare: list[EnrichedQuery]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """UPDATE query_stats SET corrected_query = %s, fast_confidence = %s,
                      status = CASE WHEN status IN ('expanded', 'failed') THEN status ELSE 'enriched' END,
                      updated_at = now() WHERE normalized_query = %s""",
            [(q.canonical, q.fast_confidence, q.normalized_query) for q in enriched])
        cur.executemany(
            "UPDATE query_stats SET status = 'skipped_rare', updated_at = now() WHERE normalized_query = %s",
            [(q.normalized_query,) for q in rare])
        # The canonical spelling's own row (it may not be the most frequent variant) carries the enrichment too.
        cur.executemany(
            """UPDATE query_stats SET corrected_query = normalized_query, fast_confidence = %s, status = 'enriched',
                      updated_at = now() WHERE normalized_query = %s AND normalized_query <> %s""",
            [(q.fast_confidence, q.canonical, q.normalized_query) for q in enriched if q.corrected_query])
        # Typo variants point at their canonical query (the cache key Phase 6 resolves them to).
        cur.executemany("""UPDATE query_stats SET corrected_query = %s, status = 'alias', updated_at = now()
                           WHERE normalized_query = %s AND normalized_query <> %s""",
                        [(q.canonical, alias, q.canonical) for q in enriched + rare for alias in q.aliases])


async def run() -> tuple[int, int]:
    payloads = await _consume_all(LOG_PROCESSING_QUEUE, "offline-enrich")
    if not payloads:
        return 0, 0
    spell = await asyncio.to_thread(build_spell_corrector)
    queries = fold_aliases(payloads, spell)
    frequent = [q for q in queries if q.frequency >= MIN_FREQUENCY]
    rare = [q for q in queries if q.frequency < MIN_FREQUENCY]

    s = get_settings()
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=30), Redis.from_url(s.redis_url)
    try:
        engine = SearchEngine(Retriever(os_client), EmbeddingService(redis), spell, EventSink(None, None), s)
        await gate(frequent, engine)
    finally:
        await os_client.close()
        await redis.aclose()
    await asyncio.to_thread(_store, frequent, rare)

    producer = AIOKafkaProducer(bootstrap_servers=s.kafka_bootstrap, linger_ms=20)
    await producer.start()
    try:
        for q in frequent:
            await producer.send(QUERY_PROCESS_QUEUE, q.model_dump_json().encode(), key=q.canonical.encode())
        await producer.flush()
    finally:
        await producer.stop()
    return len(frequent), len(rare)
