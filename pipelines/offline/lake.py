"""Data-lake access for the offline pipeline (stands in for the doc's analytical data lake read by Spark).

- read_events(): all search events from S3 `datalake/search_events/dt=*/` as one Polars frame.
- sink_live_events(): drain the live `query_logs` topic (search + click/cart/purchase rows emitted by the
  search service), fold them into one row per search, de-duplicate by event_id against the lake, and append
  a Parquet partition. Run before aggregation so live traffic is counted.
"""

import asyncio
import io
import json
from collections import defaultdict
from datetime import UTC, datetime

import boto3
import polars as pl
from aiokafka import AIOKafkaConsumer

from common.config import get_settings
from common.normalize import normalize_query
from common.topics import QUERY_LOGS

BUCKET = "datalake"
PREFIX = "search_events/"
COLUMNS = ["event_id", "event_timestamp", "user_id", "session_id", "raw_query", "normalized_query",
           "search_type", "displayed_product_ids", "clicked_product_id", "added_to_cart_product_id",
           "purchased_product_ids", "device_type", "geolocation"]


def _s3():
    s = get_settings()
    return boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                        aws_secret_access_key=s.s3_secret_key, region_name="us-east-1")


def read_events() -> pl.DataFrame:
    s3 = _s3()
    frames = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                df = pl.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()))
                frames.append(df.select([c for c in COLUMNS if c in df.columns]))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(subset="event_id", keep="first")


def fold_live_events(events: list[dict]) -> list[dict]:
    """Search-service rows (one `search` row + one row per interaction, linked by search_id) -> one lake row
    per search. Rows without a search_id (e.g. replayed synthetic rows) pass through unchanged."""
    by_search: dict[str, dict] = {}
    interactions: dict[str, list[dict]] = defaultdict(list)
    passthrough = []
    for e in events:
        sid = e.get("search_id")
        if not sid:
            passthrough.append(e)
        elif e.get("event_type", "search") == "search":
            by_search[sid] = dict(e)
        else:
            interactions[sid].append(e)
    rows = []
    for sid, row in by_search.items():
        row["purchased_product_ids"] = list(row.get("purchased_product_ids") or [])
        for it in interactions.get(sid, []):
            row["clicked_product_id"] = row.get("clicked_product_id") or it.get("clicked_product_id")
            row["added_to_cart_product_id"] = row.get("added_to_cart_product_id") or it.get("added_to_cart_product_id")
            row["purchased_product_ids"] += it.get("purchased_product_ids") or []
        rows.append(row)
    out = []
    for r in rows + passthrough:
        r = {c: r.get(c) for c in COLUMNS}
        r["normalized_query"] = r["normalized_query"] or normalize_query(r["raw_query"] or "")
        r["displayed_product_ids"] = r["displayed_product_ids"] or []
        r["purchased_product_ids"] = r["purchased_product_ids"] or []
        if isinstance(r["event_timestamp"], str):
            r["event_timestamp"] = datetime.fromisoformat(r["event_timestamp"])
        out.append(r)
    return out


async def _drain(timeout_ms: int = 5000) -> list[dict]:
    consumer = AIOKafkaConsumer(QUERY_LOGS, bootstrap_servers=get_settings().kafka_bootstrap,
                                group_id="lake-sink", auto_offset_reset="earliest", enable_auto_commit=False)
    await consumer.start()
    events = []
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=timeout_ms, max_records=5000)
            if not batch:
                break
            for records in batch.values():
                events.extend(json.loads(r.value) for r in records)
        await consumer.commit()
    finally:
        await consumer.stop()
    return events


def sink_live_events() -> int:
    events = asyncio.run(_drain())
    if not events:
        return 0
    rows = fold_live_events(events)
    existing = read_events()
    seen = set(existing["event_id"].to_list()) if existing.height else set()
    rows = [r for r in rows if r["event_id"] not in seen]
    if not rows:
        return 0
    df = pl.DataFrame(rows, schema_overrides={"event_timestamp": pl.Datetime(time_zone="UTC")})
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    buf = io.BytesIO()
    df.write_parquet(buf)
    dt = datetime.now(UTC).date().isoformat()
    _s3().put_object(Bucket=BUCKET, Key=f"{PREFIX}dt={dt}/part-live-{stamp}.parquet", Body=buf.getvalue())
    return len(rows)
