"""Publish ranking state for serving (doc p.177: offline static ranking). Tier 0.

Replays all logs (same code as training) and writes the final state:
- `static_score` for every product -> product_search_index (partial update; also the fast path's fusion prior and
  discovery's product choice)
- (query, product) engagement -> Redis hash `qp:v1:{query hash}` (pairs with engagement or >= 3 impressions)
- user profiles -> Redis `ufeat:v1:{user}` (top brands / categories, departments, median price)

    uv run python -m pipelines.ranking.publish
"""

import json
from collections import defaultdict

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from redis import Redis

from common.config import get_settings
from common.topics import PRODUCT_SEARCH_INDEX
from pipelines.offline import lake
from pipelines.ranking.replay import State, load_catalog, replay
from ranking import features as F
from ranking.serving import QP_PREFIX, STATE_TTL_S, USER_PREFIX, qp_key


def run() -> tuple[int, int, int]:
    """Returns (products scored, queries with engagement data, user profiles)."""
    catalog, state = load_catalog(), State()
    for _ in replay(lake.read_events(), catalog, state):
        pass
    s = get_settings()

    os_client = OpenSearch(s.opensearch_url, timeout=60)
    actions = ({"_op_type": "update", "_index": PRODUCT_SEARCH_INDEX, "_id": pid,
                "doc": {"static_score": state.static_score(pid, p)}} for pid, p in catalog.items())
    scored, _ = bulk(os_client, actions, chunk_size=1000, raise_on_error=False)

    by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for (q, pid), imp in state.qp_impressions.items():
        w = state.qp_weighted.get((q, pid), 0.0)
        if w > 0 or imp >= F.QP_MIN_IMPRESSIONS:
            by_query[q][pid] = round(F.qp_value(w, imp), 5)

    redis = Redis.from_url(s.redis_url)
    try:
        with redis.pipeline(transaction=False) as pipe:
            for q, values in by_query.items():
                key = qp_key(q)
                pipe.delete(key)
                pipe.hset(key, mapping=values)
                pipe.expire(key, STATE_TTL_S)
            for uid, profile in state.users.items():
                pipe.set(f"{USER_PREFIX}{uid}", json.dumps(profile.to_json()), ex=STATE_TTL_S)
            pipe.execute()
    finally:
        redis.close()
    return scored, len(by_query), len(state.users)


if __name__ == "__main__":
    products, queries, users = run()
    print(f"[ranking-publish] static_score for {products} products; engagement for {queries} queries "
          f"({QP_PREFIX}*); {users} user profiles ({USER_PREFIX}*)")
