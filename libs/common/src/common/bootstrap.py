"""Create Kafka topics, S3 buckets and OpenSearch indices. Idempotent."""

import asyncio

import boto3
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

from common.config import get_settings
from common.indices import ensure_indices
from common.topics import ALL_TOPICS

BUCKETS = ("datalake", "hotlist")


async def ensure_topics() -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=get_settings().kafka_bootstrap)
    await admin.start()
    try:
        existing = set(await admin.list_topics())
        missing = [NewTopic(t, num_partitions=3, replication_factor=1) for t in ALL_TOPICS if t not in existing]
        if missing:
            await admin.create_topics(missing)
    finally:
        await admin.close()


def ensure_buckets() -> None:
    s = get_settings()
    s3 = boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                      aws_secret_access_key=s.s3_secret_key, region_name="us-east-1")
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for bucket in BUCKETS:
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)


if __name__ == "__main__":
    asyncio.run(ensure_topics())
    ensure_buckets()
    ensure_indices()
    print("bootstrap complete")
