import json

from common.config import Settings
from common.normalize import SpellCorrector, query_hash
from search_service.engine import SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.fallback import FallbackPath, interleave_by_keyword, rewrite_confidence
from search_service.filters import SearchFilters
from search_service.fusion import Candidate
from search_service.retrieval import Hit, structured_keyword_body


class FakeRedis:
    def __init__(self):
        self.store, self.ttls = {}, {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key], self.ttls[key] = value, ex


class FakeGateway:
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def rewrite(self, query, candidate_categories):
        self.calls += 1
        return self.result


class FakeTaxonomy:
    def candidates(self, vector, k=20):
        return ["Grocery > Snacks", "Grocery > Beverages"]


PRODUCTS = {
    "chips1": {"product_id": "chips1", "name": "Potato Chips", "product_type": "chips", "price": 3.0,
               "description": "", "static_score": 0.1},
    "chips2": {"product_id": "chips2", "name": "Tortilla Chips", "product_type": "chips", "price": 4.0,
               "description": "", "static_score": 0.1},
    "soda1": {"product_id": "soda1", "name": "Cola Soda", "product_type": "soda", "price": 5.0,
              "description": "", "static_score": 0.1},
    "noise": {"product_id": "noise", "name": "Desk Lamp", "product_type": "lamp", "price": 20.0,
              "description": "", "static_score": 0.1},
    "nutmix": {"product_id": "nutmix", "name": "Party Nut Mix", "product_type": "trail mix", "price": 6.0,
               "description": "", "static_score": 0.1},
}


class FakeRetriever:
    """Fast path finds only noise weakly; the rewrite's keywords find chips / soda strongly."""

    def __init__(self, fast_sim=0.68):  # 0.68 -> fast-path confidence ~0.31: low, but above the no-results floor
        self.fast_sim = fast_sim

    async def keyword(self, query, filters, size):
        return []

    async def semantic(self, vector, filters, size):
        if vector == "chips":
            return [Hit("chips1", 0.84), Hit("chips2", 0.82), Hit("nutmix", 0.80)]
        if vector == "soda":
            return [Hit("soda1", 0.83)]
        return [Hit("noise", self.fast_sim)]  # the fast path's weak match

    async def run_keyword_body(self, body):
        return [Hit("chips1", 5.0)]

    async def hydrate(self, ids, filters, categories=None):
        return {i: PRODUCTS[i] for i in ids if i in PRODUCTS}


class KeywordEmbeddings:
    """'Embeds' a text as itself so FakeRetriever can tell keywords apart."""

    async def embed(self, texts, kind):
        return list(texts)


GOOD = {"valid": True, "output": {"keywords": ["chips", "soda"], "attributes": ["party size"],
                                  "categories": ["Grocery > Snacks"]}}


def _fallback(gateway, redis=None):
    return FallbackPath(FakeRetriever(), KeywordEmbeddings(), gateway, FakeTaxonomy(), redis, os_client=None)


def _engine(gateway, redis=None, fast_sim=0.68):
    spell = SpellCorrector()
    retriever = FakeRetriever(fast_sim)
    fb = FallbackPath(retriever, KeywordEmbeddings(), gateway, FakeTaxonomy(), redis, os_client=None)
    return SearchEngine(retriever, KeywordEmbeddings(), spell, EventSink(None, None), Settings(), fb)


# ---- compile / fuse helpers ----


def test_categories_and_attributes_boost_but_never_filter():
    body = structured_keyword_body(["chips", "soda"], ["party size"], ["Grocery > Snacks"], SearchFilters(), 50)
    q = body["query"]["bool"]
    assert len(q["must"][0]["bool"]["should"]) == 2 and q["must"][0]["bool"]["minimum_should_match"] == 1
    assert {"prefix": {"category_path": {"value": "Grocery > Snacks", "boost": 2.0}}} in q["should"]
    assert all("prefix" not in json.dumps(f) for f in q["filter"])  # only user filters (stock) filter


def test_interleave_gives_each_keyword_a_turn():
    per_kw = [[Hit("c1", 0.9), Hit("c2", 0.8), Hit("c3", 0.7)], [Hit("s1", 0.9)]]
    ranked = [Candidate(p, 1.0 - i / 10, None, None, None) for i, p in enumerate(["c1", "c2", "c3", "s1", "bm25"])]
    assert [c.product_id for c in interleave_by_keyword(ranked, per_kw)] == ["c1", "s1", "c2", "c3", "bm25"]


def test_rewrite_confidence():
    assert rewrite_confidence([], []) == 0.0
    assert rewrite_confidence([Hit("a", 1)] * 5, [[Hit("a", 0.80)], [Hit("b", 0.80)]]) == 1.0


# ---- FallbackPath ----


async def test_rewrite_is_memoized_and_reused():
    redis, gw = FakeRedis(), FakeGateway(GOOD)
    fb = _fallback(gw, redis)
    first = await fb.run("stuff for my dorm party", "q", SearchFilters())
    second = await fb.run("stuff for my dorm party", "q", SearchFilters())
    assert gw.calls == 1 and not first.memo_hit and second.memo_hit
    assert [c.product_id for c in first.ranked][:2] == ["chips1", "soda1"]  # interleaved: chips, then soda
    assert redis.ttls[f"rewrite:v1:{query_hash('stuff for my dorm party')}"] == 7 * 24 * 3600


async def test_unusable_rewrite_is_memoized_briefly_but_gateway_errors_are_not():
    redis = FakeRedis()
    empty = FakeGateway({"valid": False, "output": {"keywords": [], "attributes": [], "categories": []}})
    assert await _fallback(empty, redis).run("xqzv blorf", "q", SearchFilters()) is None
    assert redis.store[f"rewrite:v1:{query_hash('xqzv blorf')}"] == "{}"
    assert redis.ttls[f"rewrite:v1:{query_hash('xqzv blorf')}"] == 24 * 3600
    down = FakeGateway(None)
    assert await _fallback(down, redis).run("new query", "q", SearchFilters()) is None
    assert f"rewrite:v1:{query_hash('new query')}" not in redis.store  # retry next time


# ---- engine integration ----


async def test_low_confidence_query_is_served_by_fallback():
    resp = await _engine(FakeGateway(GOOD)).search(SearchRequest(q="stuff for my dorm party", debug=True))
    assert resp.debug.served_by == "fallback" and resp.debug.rewritten_query == "chips, soda"
    assert resp.debug.fast_path_confidence < resp.debug.confidence
    assert "noise" not in [r.product_id for r in resp.results]


async def test_gateway_failure_degrades_to_fast_path_results():
    resp = await _engine(FakeGateway(None)).search(SearchRequest(q="stuff for my dorm party", debug=True))
    assert resp.debug.served_by == "fast_path"
    assert [r.product_id for r in resp.results] == ["noise"]  # weak but shown (confidence 0.15 or above)


async def test_near_zero_confidence_without_rewrite_returns_no_results():
    resp = await _engine(FakeGateway(None), fast_sim=0.57).search(SearchRequest(q="xqzv blorf", debug=True))
    assert resp.results == [] and resp.pagination.total_results == 0


async def test_negations_also_filter_fallback_results():
    resp = await _engine(FakeGateway(GOOD)).search(SearchRequest(q="party snacks without nuts", debug=True))
    assert resp.debug.served_by == "fallback"
    assert "nutmix" not in [r.product_id for r in resp.results]
