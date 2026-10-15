"""Hot list and L2 warming (doc p.171-172). Tier 0.

1. Hot list: the top 5% of queries by frequency (`query_stats.tier = 'l1'`), each with its related searches and
   its cached query when the offline pipeline produced one, written to S3 `hotlist/hot_list.json`; the S3
   location and version are published on `list_update`, and search instances reload their in-process L1 from it.
2. L2: every top-20% query (tiers l1 + l2) that has a `smart_query_cache` entry goes to Redis `qcache:v1:{hash}`
   with a TTL longer than the pipeline cadence, so each run refreshes it and dropped queries expire.

    uv run python -m pipelines.hot_list.build
"""

import asyncio
import json
import time

import boto3
from aiokafka import AIOKafkaProducer
from redis import Redis

from common.config import get_settings
from common.db import connect
from common.topics import LIST_UPDATE
from search_service.cache import HOT_LIST_BUCKET as BUCKET
from search_service.cache import HOT_LIST_KEY as KEY
from search_service.cache import L2_TTL_S, l2_key


def s3_client():
    s = get_settings()
    return boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                        aws_secret_access_key=s.s3_secret_key, region_name="us-east-1")


def fetch_rows() -> list[dict]:
    """Top-20% queries keyed by their search-time form (the spell-corrected query), with any cached query."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (q.query) q.query, q.tier, q.frequency, c.es_query, c.source,
                   coalesce(nullif(c.related_searches, '{}'), q.related_searches) AS related_searches
            FROM (SELECT coalesce(corrected_query, normalized_query) AS query, tier, frequency, related_searches
                  FROM query_stats WHERE tier IN ('l1', 'l2')) q
            LEFT JOIN smart_query_cache c ON c.normalized_query = q.query
            ORDER BY q.query, q.frequency DESC""")
        return cur.fetchall()


def build_hot_list(rows: list[dict], version: int) -> dict:
    hot = sorted((r for r in rows if r["tier"] == "l1"), key=lambda r: -r["frequency"])
    return {"version": version, "queries": [
        {"query": r["query"], "es_query": r["es_query"], "related_searches": r["related_searches"] or [],
         "source": r["source"]} if r["es_query"] else
        {"query": r["query"], "related_searches": r["related_searches"] or []}
        for r in hot]}


def l2_entries(rows: list[dict]) -> dict[str, str]:
    return {l2_key(r["query"]): json.dumps({
        "normalized_query": r["query"], "es_query": r["es_query"], "related_searches": r["related_searches"] or [],
        "source": r["source"]}) for r in rows if r["es_query"]}


async def publish(version: int) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(LIST_UPDATE, json.dumps({"bucket": BUCKET, "key": KEY,
                                                             "version": version}).encode())
    finally:
        await producer.stop()


def run() -> tuple[int, int, int]:
    """Returns (hot-list size, of which with a cached query, L2 entries written)."""
    version = int(time.time())
    rows = fetch_rows()
    hot_list = build_hot_list(rows, version)
    s3_client().put_object(Bucket=BUCKET, Key=KEY, Body=json.dumps(hot_list).encode(),
                           ContentType="application/json")
    entries = l2_entries(rows)
    redis = Redis.from_url(get_settings().redis_url)
    try:
        with redis.pipeline(transaction=False) as pipe:
            for key, value in entries.items():
                pipe.set(key, value, ex=L2_TTL_S)
            pipe.execute()
    finally:
        redis.close()
    asyncio.run(publish(version))
    return len(hot_list["queries"]), sum("es_query" in q for q in hot_list["queries"]), len(entries)


if __name__ == "__main__":
    n, with_query, l2 = run()
    print(f"[hot-list] {n} hot queries ({with_query} with a cached query) -> s3://{BUCKET}/{KEY}; "
          f"L2 entries: {l2}; list_update published")
