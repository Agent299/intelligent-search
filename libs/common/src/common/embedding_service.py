"""Embedding service shared by the gateway (/v1/embed) and the search service: local model (tier 0, no API
cost) + Redis cache keyed by text hash, so repeated queries skip the model (doc p.153 'embedding model or cache')."""

import asyncio
import hashlib
from typing import Literal

import numpy as np
from prometheus_client import Counter
from redis.asyncio import Redis

from common.config import get_settings
from common.embeddings import embed_documents, embed_queries

Kind = Literal["query", "document"]
TTL_S = 7 * 24 * 3600
CACHE_LOOKUPS = Counter("embedding_cache_total", "Embedding cache lookups", ["result"])


def _key(kind: Kind, text: str) -> str:
    model = get_settings().embedding_model
    return f"emb:{model}:{kind}:{hashlib.sha1(text.encode()).hexdigest()}"


class EmbeddingService:
    def __init__(self, redis: Redis | None):
        self.redis = redis

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        keys = [_key(kind, t) for t in texts]
        cached: list[bytes | None] = [None] * len(texts)
        if self.redis is not None:
            try:
                cached = await self.redis.mget(keys)
            except Exception:  # cache outage must not fail embedding
                cached = [None] * len(texts)
        missing = [i for i, c in enumerate(cached) if c is None]
        CACHE_LOOKUPS.labels("hit").inc(len(texts) - len(missing))
        CACHE_LOOKUPS.labels("miss").inc(len(missing))

        vectors: list[np.ndarray | None] = [
            np.frombuffer(c, dtype=np.float32) if c is not None else None for c in cached
        ]
        if missing:
            fn = embed_queries if kind == "query" else embed_documents
            fresh = await asyncio.to_thread(fn, [texts[i] for i in missing])
            for i, vec in zip(missing, fresh, strict=True):
                vectors[i] = np.asarray(vec, dtype=np.float32)
            if self.redis is not None:
                try:
                    async with self.redis.pipeline(transaction=False) as pipe:
                        for i in missing:
                            pipe.set(keys[i], vectors[i].tobytes(), ex=TTL_S)
                        await pipe.execute()
                except Exception:
                    pass
        return [v.tolist() for v in vectors]
