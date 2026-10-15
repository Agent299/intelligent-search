"""Stage 1 - log aggregation (doc p.149, Fig 5.2). Tier 0.

Reads the data lake, groups searches by normalized query, and computes what later stages need:
frequency (and the cache tier it implies: top 5% -> L1, top 20% -> L2, rest -> warm), engagement
(clicks / carts / purchases, most-purchased products), dominant context (device, location, time of day),
and related searches mined from session co-occurrence. Results go to `query_stats` and one payload per
unique query is published to `log_processing_queue`.
"""

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import polars as pl
from aiokafka import AIOKafkaProducer

from common.config import get_settings
from common.db import connect
from common.topics import LOG_PROCESSING_QUEUE

L1_SHARE, L2_SHARE = 0.05, 0.20
RELATED_MAX = 4
RELATED_MIN_SESSIONS = 2  # a pair must co-occur in at least this many sessions
RELATED_MIN_LIFT = 3.0    # ... and far more often than chance (popular queries co-occur with everything)


@dataclass
class QueryAggregate:
    normalized_query: str
    raw_example: str
    frequency: int
    freq_rank: int
    tier: str
    sessions: int
    users: int
    clicks: int
    carts: int
    purchases: int
    top_purchased: list[str]
    related_searches: list[str]
    device: str | None
    geolocation: str | None
    time_of_day: str | None


def _time_of_day(hour: int) -> str:
    return "morning" if 5 <= hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 22 else "night"


def related_from_sessions(events: pl.DataFrame) -> dict[str, list[str]]:
    """Queries searched in the same session, ranked by lift = P(a and b) / (P(a) P(b)) (Tier 0 version of the
    doc's relatedSearches; no LLM needed when users already tell us what they search next). Raw co-occurrence
    counts were dominated by head queries ('ipad case' 'related' to a speakerphone), so pairs need lift >= 3."""
    sessions = (events.select("session_id", "normalized_query").unique()
                .group_by("session_id").agg(pl.col("normalized_query")))
    n_sessions = max(1, sessions.height)
    query_sessions: Counter = Counter()
    pair_counts: Counter = Counter()
    for queries in sessions["normalized_query"].to_list():
        query_sessions.update(queries)
        if 1 < len(queries) <= 12:
            for a in queries:
                for b in queries:
                    if a != b:
                        pair_counts[(a, b)] += 1
    related: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (a, b), n in pair_counts.items():
        lift = n * n_sessions / (query_sessions[a] * query_sessions[b])
        if n >= RELATED_MIN_SESSIONS and lift >= RELATED_MIN_LIFT:
            related[a].append((lift, b))
    return {q: [b for _, b in sorted(pairs, key=lambda x: (-x[0], x[1]))[:RELATED_MAX]] for q, pairs in related.items()}


def aggregate(events: pl.DataFrame) -> list[QueryAggregate]:
    ev = events.with_columns(
        pl.col("clicked_product_id").is_not_null().alias("clicked"),
        pl.col("added_to_cart_product_id").is_not_null().alias("carted"),
        pl.col("purchased_product_ids").list.len().alias("n_purchased"),
        pl.col("event_timestamp").dt.hour().alias("hour"),
    )
    g = (ev.group_by("normalized_query").agg(
            pl.len().alias("frequency"),
            pl.col("raw_query").first().alias("raw_example"),
            pl.col("session_id").n_unique().alias("sessions"),
            pl.col("user_id").n_unique().alias("users"),
            pl.col("clicked").sum().alias("clicks"),
            pl.col("carted").sum().alias("carts"),
            pl.col("n_purchased").sum().alias("purchases"),
            pl.col("purchased_product_ids").list.explode(keep_nulls=False, empty_as_null=False).alias("bought"),
            pl.col("device_type").mode().first().alias("device"),
            pl.col("geolocation").mode().first().alias("geolocation"),
            pl.col("hour").mode().first().alias("hour"),
         ).sort(["frequency", "normalized_query"], descending=[True, False]))
    n = g.height
    l1, l2 = max(1, int(n * L1_SHARE)), max(1, int(n * L2_SHARE))
    related = related_from_sessions(ev)
    out = []
    for rank, row in enumerate(g.iter_rows(named=True), start=1):
        out.append(QueryAggregate(
            normalized_query=row["normalized_query"], raw_example=row["raw_example"], frequency=row["frequency"],
            freq_rank=rank, tier="l1" if rank <= l1 else "l2" if rank <= l2 else "warm",
            sessions=row["sessions"], users=row["users"], clicks=row["clicks"], carts=row["carts"],
            purchases=row["purchases"], top_purchased=[p for p, _ in Counter(row["bought"]).most_common(5)],
            related_searches=related.get(row["normalized_query"], []), device=row["device"],
            geolocation=row["geolocation"], time_of_day=_time_of_day(row["hour"]) if row["hour"] is not None else None,
        ))
    return out


def store(aggs: list[QueryAggregate]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE agg (LIKE query_stats INCLUDING DEFAULTS) ON COMMIT DROP")
        with cur.copy("""COPY agg (normalized_query, frequency, freq_rank, tier, sessions, users, clicks, carts,
                         purchases, top_purchased, related_searches) FROM STDIN""") as copy:
            for a in aggs:
                copy.write_row((a.normalized_query, a.frequency, a.freq_rank, a.tier, a.sessions, a.users, a.clicks,
                                a.carts, a.purchases, a.top_purchased, a.related_searches))
        # Re-aggregation refreshes stats; pipeline progress (status, confidence, correction) is kept.
        cur.execute("""
            INSERT INTO query_stats (normalized_query, frequency, freq_rank, tier, sessions, users, clicks, carts,
                                     purchases, top_purchased, related_searches)
            SELECT normalized_query, frequency, freq_rank, tier, sessions, users, clicks, carts, purchases,
                   top_purchased, related_searches FROM agg
            ON CONFLICT (normalized_query) DO UPDATE SET
                frequency = EXCLUDED.frequency, freq_rank = EXCLUDED.freq_rank, tier = EXCLUDED.tier,
                sessions = EXCLUDED.sessions, users = EXCLUDED.users, clicks = EXCLUDED.clicks,
                carts = EXCLUDED.carts, purchases = EXCLUDED.purchases, top_purchased = EXCLUDED.top_purchased,
                related_searches = EXCLUDED.related_searches, updated_at = now()""")


async def publish(aggs: list[QueryAggregate]) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_bootstrap, linger_ms=20)
    await producer.start()
    try:
        for a in aggs:
            payload = {
                "original_query": a.raw_example, "normalized_query": a.normalized_query, "frequency": a.frequency,
                "tier": a.tier, "context": {"device": a.device, "geolocation": a.geolocation,
                                            "time_of_day": a.time_of_day},
            }
            await producer.send(LOG_PROCESSING_QUEUE, json.dumps(payload).encode(), key=a.normalized_query.encode())
        await producer.flush()
    finally:
        await producer.stop()


def run(events: pl.DataFrame) -> list[QueryAggregate]:
    aggs = aggregate(events)
    store(aggs)
    asyncio.run(publish(aggs))
    return aggs
