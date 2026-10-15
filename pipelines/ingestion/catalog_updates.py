"""Freshness path (NFR: price changes and stock-outs reach search in near real time).

Consumes {"product_id", "price"?, "in_stock"?} from the catalog_updates topic, writes the source
of truth (product_catalog), applies partial updates to both search indices, and purges every cached result
list containing the product (doc p.172 event-based invalidation; cached *queries* need no purge, they always
run against the live index).

    uv run python -m pipelines.ingestion.catalog_updates consume
    uv run python -m pipelines.ingestion.catalog_updates emit B00NE08WM6 --price 9.99 --in-stock false
"""

import argparse
import asyncio
import json

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from opensearchpy import NotFoundError
from redis import Redis

from common.config import get_settings
from common.db import connect
from common.indices import get_opensearch
from common.topics import CATALOG_UPDATES, PRODUCT_SEARCH_INDEX, SEMANTIC_SEARCH_INDEX
from search_service.cache import purge_results_for

UPDATABLE = ("price", "in_stock")


def apply_update(msg: dict, os_client=None, redis_client=None) -> bool:
    fields = {k: msg[k] for k in UPDATABLE if k in msg}
    if not fields or "product_id" not in msg:
        return False
    with connect() as conn, conn.cursor() as cur:
        sets = ", ".join(f"{k} = %({k})s" for k in fields)
        cur.execute(f"UPDATE product_catalog SET {sets}, updated_at = now() WHERE product_id = %(product_id)s",
                    {**fields, "product_id": msg["product_id"]})
        if cur.rowcount == 0:
            return False
    os_client = os_client or get_opensearch()
    for index in (PRODUCT_SEARCH_INDEX, SEMANTIC_SEARCH_INDEX):
        try:
            os_client.update(index=index, id=msg["product_id"], body={"doc": fields})
        except NotFoundError:
            pass
    if redis_client is not None:
        purge_results_for(redis_client, msg["product_id"])
    return True


async def consume() -> None:
    consumer = AIOKafkaConsumer(CATALOG_UPDATES, bootstrap_servers=get_settings().kafka_bootstrap,
                                group_id="catalog-indexer", enable_auto_commit=True,
                                auto_offset_reset="earliest")  # a new group must not skip updates sent while joining
    await consumer.start()
    os_client = get_opensearch()
    redis_client = Redis.from_url(get_settings().redis_url)
    print(f"consuming {CATALOG_UPDATES}", flush=True)
    try:
        async for record in consumer:
            msg = json.loads(record.value)
            ok = await asyncio.to_thread(apply_update, msg, os_client, redis_client)
            print(f"{'applied' if ok else 'skipped'} {msg}", flush=True)
    finally:
        await consumer.stop()
        redis_client.close()


async def emit(product_id: str, price: float | None, in_stock: bool | None) -> None:
    msg = {"product_id": product_id}
    if price is not None:
        msg["price"] = price
    if in_stock is not None:
        msg["in_stock"] = in_stock
    producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(CATALOG_UPDATES, json.dumps(msg).encode(), key=product_id.encode())
    finally:
        await producer.stop()
    print(f"emitted {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("consume")
    e = sub.add_parser("emit")
    e.add_argument("product_id")
    e.add_argument("--price", type=float)
    e.add_argument("--in-stock", choices=["true", "false"])
    args = ap.parse_args()
    if args.cmd == "consume":
        asyncio.run(consume())
    else:
        stock = None if args.in_stock is None else args.in_stock == "true"
        asyncio.run(emit(args.product_id, args.price, stock))


if __name__ == "__main__":
    main()
