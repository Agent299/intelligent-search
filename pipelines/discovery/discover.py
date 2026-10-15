"""Discovery enrichment (doc p.160, Fig 5.8). Tier 1 for the judgment, Tier 0 for everything else.

1. Submit (Haiku, Batch API): for each primary category, the LLM picks 3-5 complements from its co-purchase list
   (the guardrail keeps only listed items). Categories whose list hasn't changed since the stored entry are not
   resubmitted, so a daily run only pays for what changed.
2. Write (code): resolve each complement category to its top in-stock products by static score and store
   `discovery_products:{category}` in Redis. Every run re-resolves all entries (stock moves; the LLM choice does
   not), and drops entries for categories that no longer have co-purchase data.
"""

import asyncio
import hashlib
import json
import time

from opensearchpy import AsyncOpenSearch
from redis.asyncio import Redis

from common.config import get_settings
from common.topics import PRODUCT_SEARCH_INDEX
from genai_gateway.core import task_spec
from pipelines.discovery import mine
from pipelines.offline.batches import BatchJob, JobResult
from search_service.discovery import PREFIX, discovery_key

STAGE = "discover"
PRODUCTS_PER_COMPLEMENT = 2


def input_hash(co_items: list[str]) -> str:
    """Changes when the co-purchase list or the prompt version changes: either is worth paying for again."""
    return hashlib.sha1(json.dumps([task_spec("discover").prompt, sorted(co_items)]).encode()).hexdigest()[:16]


async def _stored(redis: Redis, keys: list[str]) -> dict[str, dict]:
    raw = await redis.mget([discovery_key(k) for k in keys]) if keys else []
    return {k: json.loads(v) for k, v in zip(keys, raw, strict=True) if v}


async def submit(run_id: str, limit: int | None = None) -> tuple[int, int]:
    """Returns (categories with co-purchase data, submitted to the LLM). `limit` keeps the categories with the
    most co-purchase evidence (for a pilot)."""
    co = mine.load()
    redis = Redis.from_url(get_settings().redis_url)
    try:
        stored = await _stored(redis, list(co))
    finally:
        await redis.aclose()
    items = {
        f"disc-{input_hash([key])}": {"task": "discover",
                                      "inputs": {"primary_product": key, "co_purchase_data": items},
                                      "meta": {"category": key, "input_hash": input_hash(items)}}
        for key, items in co.items()
        if stored.get(key, {}).get("input_hash") != input_hash(items)
    }
    if limit:
        support = mine.support_by_key()
        items = dict(sorted(items.items(), key=lambda kv: -support.get(kv[1]["meta"]["category"], 0))[:limit])
    BatchJob(run_id, STAGE).submit(items)
    return len(co), len(items)


def collect(run_id: str) -> tuple[JobResult, str | None, list[str]]:
    job = BatchJob(run_id, STAGE)
    res = job.collect()
    for cid, reason in res.invalid.items():
        job.items[cid]["invalid_reason"] = reason
    job.save()
    esc_batch, exhausted = job.escalate(res.invalid) if res.invalid else (None, [])
    return res, esc_batch, exhausted


async def top_products(os_client: AsyncOpenSearch, category: str, n: int = PRODUCTS_PER_COMPLEMENT) -> list[str]:
    resp = await os_client.search(index=PRODUCT_SEARCH_INDEX, body={
        "size": n, "_source": False, "sort": [{"static_score": "desc"}, {"_id": "asc"}],
        "query": {"bool": {"filter": [{"term": {"category_path": category}}, {"term": {"in_stock": True}}]}}})
    return [h["_id"] for h in resp["hits"]["hits"]]


async def write(run_id: str | None) -> tuple[int, int]:
    """Returns (entries written, entries removed). run_id None: only re-resolve stored entries."""
    co = mine.load()
    s = get_settings()
    new = {}
    if run_id:
        new = {it["meta"]["category"]: it for it in BatchJob(run_id, STAGE).items.values()}
    redis, os_client = Redis.from_url(s.redis_url), AsyncOpenSearch(s.opensearch_url, timeout=30)
    try:
        stored = await _stored(redis, list(co))
        version, entries = int(time.time()), {}
        for key in co:
            if key in new:  # no output = no valid complement in the list: an empty entry, so it isn't re-paid
                chosen = new[key].get("output", {}).get("complements", [])
                h = new[key]["meta"]["input_hash"]
            elif key in stored:
                chosen, h = stored[key]["chosen"], stored[key]["input_hash"]
            else:
                continue
            complements = [{"category": c, "product_ids": await top_products(os_client, c)} for c in chosen]
            entries[discovery_key(key)] = json.dumps({
                "chosen": chosen, "complements": [c for c in complements if c["product_ids"]],
                "input_hash": h, "version": version})
        stale = [k async for k in redis.scan_iter(match=f"{PREFIX}*")
                 if k.decode() not in entries]
        async with redis.pipeline(transaction=False) as pipe:
            for k, v in entries.items():
                pipe.set(k, v)
            if stale:
                pipe.delete(*stale)
            await pipe.execute()
        return len(entries), len(stale)
    finally:
        await redis.aclose()
        await os_client.close()


def run_mine() -> tuple[int, int, int]:
    """Returns (basket sessions, pairs kept, primary categories)."""
    from pipelines.offline import lake

    sessions = mine.baskets(lake.read_events(), mine.load_categories())
    pairs = mine.mine(sessions)
    mine.store(pairs)
    return len(sessions), len(pairs), len({p.product_key for p in pairs})


def write_sync(run_id: str | None) -> tuple[int, int]:
    return asyncio.run(write(run_id))
