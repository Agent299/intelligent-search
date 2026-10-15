"""L1 warming (doc p.171): load the hot list from S3 at startup and reload it whenever the hot-list job
publishes a new version on `list_update`. A missing list or a Kafka outage leaves L1 as it is (empty at worst),
which only costs latency: every query still has the L2 / warm / fast-path route."""

import asyncio
import json
import logging

import boto3
from aiokafka import AIOKafkaConsumer

from common.config import get_settings
from common.topics import LIST_UPDATE
from search_service.cache import HOT_LIST_BUCKET, HOT_LIST_KEY, QueryCache

log = logging.getLogger(__name__)


def fetch(bucket: str = HOT_LIST_BUCKET, key: str = HOT_LIST_KEY) -> dict | None:
    s = get_settings()
    s3 = boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                      aws_secret_access_key=s.s3_secret_key, region_name="us-east-1")
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        log.warning("hot list unavailable at s3://%s/%s", bucket, key, exc_info=True)
        return None


async def load(cache: QueryCache, bucket: str = HOT_LIST_BUCKET, key: str = HOT_LIST_KEY) -> bool:
    hot_list = await asyncio.to_thread(fetch, bucket, key)
    if hot_list is None:
        return False
    cache.load_hot_list(hot_list)
    log.info("hot list v%s loaded: %d queries", cache.hot_version, len(cache.hot))
    return True


async def follow_updates(cache: QueryCache) -> None:
    """Background task: reload on each `list_update` event. No consumer group, latest offset: every instance
    gets every event (broadcast), and the startup load already covers anything published earlier."""
    while True:
        consumer = AIOKafkaConsumer(LIST_UPDATE, bootstrap_servers=get_settings().kafka_bootstrap,
                                    auto_offset_reset="latest")
        try:
            await consumer.start()
            async for record in consumer:
                evt = json.loads(record.value)
                if int(evt.get("version", 0)) > cache.hot_version:
                    await load(cache, evt.get("bucket", HOT_LIST_BUCKET), evt.get("key", HOT_LIST_KEY))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("list_update consumer failed; retrying in 30 s", exc_info=True)
            await asyncio.sleep(30)
        finally:
            await consumer.stop()
