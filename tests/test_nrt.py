import json

import pytest

from common.config import Settings
from common.normalize import SpellCorrector
from pipelines.nrt_trends.detector import SEEN_PREFIX, NrtProcessor
from pipelines.nrt_trends.window import SlidingWindow
from search_service.cache import QueryCache, l2_key
from search_service.engine import SearchEngine
from search_service.events import EventSink
from search_service.retrieval import Hit

# ---- window ----


def test_window_counts_slide_out_after_the_window():
    w = SlidingWindow(window_s=300, slide_s=10)
    for t in range(0, 50, 5):    # 10 searches in the first 50 s
        w.add("pumpkin spice", t)
    w.add("beef jerky", 100)
    assert w.trending(100, threshold=5) == [("pumpkin spice", 10)]
    assert w.counts(299)["pumpkin spice"] == 10
    assert w.counts(305)["pumpkin spice"] == 8      # the 0-10 s bucket (t=0, t=5) left the window
    assert "pumpkin spice" not in w.counts(360)     # all gone, zero counts dropped
    assert w.counts(360)["beef jerky"] == 1


def test_window_rejects_misaligned_sizes_and_keeps_late_events():
    with pytest.raises(ValueError):
        SlidingWindow(window_s=305, slide_s=10)
    w = SlidingWindow(300, 10)
    w.add("a", 50)
    w.add("a", 20)   # arrived late: counted, not lost
    assert w.counts(55)["a"] == 2


# ---- processor ----


class FakeRedis:
    def __init__(self):
        self.kv, self.ttl = {}, {}

    async def exists(self, key):
        return int(key in self.kv)

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        self.ttl[key] = ex
        return True

    def pipeline(self, transaction=False):
        return _Pipe(self)


class _Pipe:
    def __init__(self, r):
        self.r, self.out = r, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def incr(self, key):
        self.r.kv[key] = int(self.r.kv.get(key, 0)) + 1
        self.out.append(self.r.kv[key])

    def expire(self, key, ttl):
        self.out.append(True)

    async def execute(self):
        return self.out


class FakeRetriever:
    def __init__(self, sim):
        self.sim = sim

    async def keyword(self, query, filters, size):
        return [Hit("p1", 3.0)] if self.sim > 0.7 else []

    async def semantic(self, vector, filters, size):
        return [Hit(f"p{i}", self.sim - 0.01 * i) for i in range(3)]

    async def hydrate(self, ids, filters):
        return {pid: {"product_id": pid, "name": pid, "static_score": 0.1} for pid in ids}


class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.0, 1.0] for _ in texts]


class FakeGateway:
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def run(self, task, inputs):
        assert task == "nrt_segment"
        self.calls += 1
        return self.result


SEGMENT = {"valid": True, "output": {"product_types": ["pumpkin spice latte mix"], "attributes": ["sugar free"]},
           "cost_usd": 0.0001}


def _processor(sim=0.55, gateway_result=SEGMENT, budget=200, redis=None):
    redis = redis or FakeRedis()
    settings = Settings(nrt_max_llm_per_hour=budget)
    engine = SearchEngine(FakeRetriever(sim), FakeEmbeddings(), SpellCorrector(), EventSink(None, None), settings)
    gateway = FakeGateway(gateway_result)
    persisted = []
    p = NrtProcessor(engine, QueryCache(redis, None, 0.91, 600), gateway, redis, None, settings,
                     persist=lambda q, es, v: persisted.append((q, es)))
    return p, redis, gateway, persisted


async def test_low_confidence_trend_is_segmented_and_cached_as_nrt_once():
    p, redis, gateway, persisted = _processor()
    d = await p.process("sugar free pumpkin spice")
    assert d.outcome == "cached_nrt" and gateway.calls == 1
    entry = json.loads(redis.kv[l2_key("sugar free pumpkin spice")])
    assert entry["source"] == "nrt" and entry["es_query"]["nrt"]["product_types"] == ["pumpkin spice latte mix"]
    assert persisted[0][0] == "sugar free pumpkin spice"
    # still trending at the next tick: decided already, and now cached either way
    assert (await p.process("sugar free pumpkin spice")).outcome == "seen" and gateway.calls == 1


async def test_cached_queries_and_confident_queries_never_reach_the_llm():
    p, redis, gateway, _ = _processor(sim=0.82)
    assert (await p.process("beef jerky")).outcome == "confident"
    redis.kv[l2_key("dorm party")] = json.dumps({"normalized_query": "dorm party", "es_query": {"query": {}},
                                                 "source": "batch"})
    assert (await p.process("dorm party")).outcome == "cached"
    assert gateway.calls == 0


async def test_budget_and_gateway_errors_are_retried_next_tick_not_remembered():
    p, redis, gateway, _ = _processor(budget=0)
    assert (await p.process("q1")).outcome == "budget" and gateway.calls == 0
    assert f"{SEEN_PREFIX}" not in "".join(redis.kv)  # not marked: will be retried
    p2, redis2, _, _ = _processor(gateway_result=None)
    assert (await p2.process("q2")).outcome == "gateway_error"
    assert not any(k.startswith(SEEN_PREFIX) for k in redis2.kv)


async def test_useless_segments_are_remembered_so_they_are_not_paid_for_again():
    p, redis, gateway, persisted = _processor(gateway_result={"valid": False, "output": {"product_types": [],
                                                                                         "attributes": []}})
    assert (await p.process("asdf qwer")).outcome == "no_product_types"
    assert (await p.process("asdf qwer")).outcome == "seen" and gateway.calls == 1 and not persisted


async def test_nrt_never_replaces_an_existing_l2_entry():
    redis = FakeRedis()
    p, _, _, _ = _processor(redis=redis)
    await p._write("q", {"query": {}}, [0.0, 1.0])
    redis.kv[l2_key("q")] = json.dumps({"source": "batch"})
    await p._write("q", {"query": {"x": 1}}, [0.0, 1.0])
    assert json.loads(redis.kv[l2_key("q")])["source"] == "batch"
