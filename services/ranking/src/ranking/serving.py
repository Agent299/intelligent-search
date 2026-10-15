"""Online personalized re-rank (doc p.178, Fig 5.16). Tier 0.

One Redis round trip fetches the user's long-term profile, the session's clicked brands and this query's
(query, product) engagement; features are built with ranking.features (the same code training used) and the top
candidates are re-scored. Everything fails open: without Redis or a model the retrieval order stands.
"""

import json
import logging
import time

import numpy as np
from prometheus_client import Histogram
from redis.asyncio import Redis

from common.normalize import query_hash
from ranking import features as F
from ranking.model import LightGBMRanker, LinearRanker

log = logging.getLogger(__name__)

LATENCY = Histogram("search_rerank_seconds", "Personalized re-rank latency", ["ranker"],
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1))

QP_PREFIX = "qp:v1:"
USER_PREFIX = "ufeat:v1:"
SESSION_PREFIX = "sessb:v1:"
STATE_TTL_S = 3 * 24 * 3600   # > the daily publish cadence
SESSION_TTL_S = 30 * 60
RERANK_N = 100                # doc: top ~500 of a 1M-item catalog; our candidate lists are ~100-200
SIM_UNKNOWN = 0.65            # cached queries have no similarities: a constant leaves their relative order alone


def qp_key(query: str) -> str:
    return f"{QP_PREFIX}{query_hash(query)}"


class Reranker:
    def __init__(self, redis: Redis | None, ranker: LinearRanker | LightGBMRanker):
        self.redis = redis
        self.ranker = ranker

    @property
    def name(self) -> str:
        return self.ranker.name

    async def _context(self, query: str, user_id: str | None, session_id: str | None,
                       ids: list[str]) -> tuple[F.UserProfile | None, dict[str, float], dict[str, float]]:
        if self.redis is None:
            return None, {}, {}
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.get(f"{USER_PREFIX}{user_id}" if user_id else "ufeat:v1:-")
                pipe.hgetall(f"{SESSION_PREFIX}{session_id}" if session_id else "sessb:v1:-")
                pipe.hmget(qp_key(query), ids)
                user_raw, session_raw, qp_raw = await pipe.execute()
        except Exception:
            log.warning("re-rank context unavailable", exc_info=True)
            return None, {}, {}
        user = F.UserProfile.from_json(json.loads(user_raw)) if user_raw and user_id else None
        session = {k.decode(): float(v) for k, v in (session_raw or {}).items()}
        qp = {pid: float(v) for pid, v in zip(ids, qp_raw or [], strict=False) if v is not None}
        return user, session, qp

    async def rerank(self, query: str, user_id: str | None, session_id: str | None, ids: list[str],
                     sims: dict[str, float], products: dict[str, dict]) -> list[str]:
        """Re-order `ids` (retrieval order, all present in `products`)."""
        if len(ids) < 2:
            return ids
        started = time.perf_counter()
        user, session, qp = await self._context(query, user_id, session_id, ids)
        # k-NN only scores its own top 100: a keyword-only candidate's similarity is at most the lowest one seen
        fill = min(sims.values()) if sims else SIM_UNKNOWN
        rows = np.asarray([F.row(rank=i, sim=sims.get(pid, fill), product=products[pid], query=query, user=user,
                                 session_brands=session, qp=qp.get(pid, F.PRIOR_RATE))
                           for i, pid in enumerate(ids)], dtype=np.float64)
        try:
            scores = self.ranker.score(rows)
        except Exception:
            log.warning("ranker failed; keeping retrieval order", exc_info=True)
            return ids
        order = np.argsort(-scores, kind="stable")
        LATENCY.labels(self.name).observe(time.perf_counter() - started)
        return [ids[i] for i in order]

    async def record_click(self, session_id: str | None, brand: str | None) -> None:
        """Short-term signal: brands clicked in this session (doc: 'session brand views from Redis')."""
        if self.redis is None or not session_id or not brand:
            return
        try:
            key = f"{SESSION_PREFIX}{session_id}"
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.hincrbyfloat(key, brand, 1.0)
                pipe.expire(key, SESSION_TTL_S)
                await pipe.execute()
        except Exception:
            log.warning("could not record session click", exc_info=True)
