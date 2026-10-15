import json

from common.config import Settings
from common.normalize import SpellCorrector
from common.schemas import ServedBy, SortBy
from pipelines.hot_list.build import build_hot_list, l2_entries
from search_service.cache import CachedResults, CacheEntry, QueryCache, l2_key, purge_results_for
from search_service.engine import SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.filters import SearchFilters, parse_filters
from search_service.retrieval import Hit

COMPILED = {"bool": {"must": [{"match": {"name": "sauce"}}]}}


class FakeRedis:
    """Enough of redis (async get/pipeline, sync smembers/delete) for the cache."""

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    async def get(self, key):
        return self.kv.get(key)

    def pipeline(self, transaction=False):
        return _Pipe(self)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.sets.pop(k, None)


class _Pipe:
    def __init__(self, r):
        self.r = r

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def set(self, key, value, ex=None):
        self.r.kv[key] = value

    def sadd(self, key, member):
        self.r.sets.setdefault(key, set()).add(member)

    def expire(self, key, ttl):
        pass

    async def execute(self):
        pass


class FakeRetriever:
    def __init__(self, products, fast_sim=0.80):
        self.products = products
        self.fast_sim = fast_sim
        self.bodies, self.fast_calls = [], 0

    async def keyword(self, query, filters, size):
        self.fast_calls += 1
        return [Hit(pid, 10.0 - i) for i, pid in enumerate(self.products)]

    async def semantic(self, vector, filters, size):
        return [Hit(pid, self.fast_sim - 0.01 * i) for i, pid in enumerate(self.products)]

    async def run_keyword_body(self, body):
        self.bodies.append(body)
        return [Hit("p2", 5.0), Hit("p1", 4.0)]

    async def hydrate(self, ids, filters):
        return {pid: self.products[pid] for pid in ids
                if pid in self.products and (filters.include_out_of_stock or self.products[pid]["in_stock"])}


class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.0, 1.0] for _ in texts]


class FakeCache(QueryCache):
    def __init__(self, redis=None, similar_entry=None):
        super().__init__(redis, None, 0.91, 600)
        self.similar_entry, self.similar_calls = similar_entry, 0

    async def similar(self, vector):
        self.similar_calls += 1
        return self.similar_entry


def _product(pid, name, in_stock=True):
    return {"product_id": pid, "name": name, "description": "d", "brand": "B", "price": 5.0, "currency": "USD",
            "rating": 4.0, "imageUrl": None, "product_type": "sauce", "static_score": 0.5, "in_stock": in_stock}


PRODUCTS = {"p1": _product("p1", "Hot Sauce"), "p2": _product("p2", "Barbecue Sauce"),
            "p3": _product("p3", "Peanut Sauce")}


def _engine(cache, fast_sim=0.80, products=PRODUCTS):
    retriever = FakeRetriever(products, fast_sim)
    return SearchEngine(retriever, FakeEmbeddings(), SpellCorrector(), EventSink(None, None), Settings(),
                        cache=cache), retriever


def _hot(*entries):
    return {"version": 7, "queries": list(entries)}


# ---- tiers ----


async def test_l1_hit_runs_the_cached_query_with_request_filters_and_skips_the_fast_path():
    cache = FakeCache()
    cache.load_hot_list(_hot({"query": "brunch sauce", "es_query": {"query": COMPILED},
                              "related_searches": ["hot sauce"], "source": "batch"}))
    engine, retriever = _engine(cache)
    resp = await engine.search(SearchRequest(q="Brunch sauce!", filters=parse_filters('{"brand": "B"}'),
                                             debug=True))
    assert resp.debug.served_by is ServedBy.l1 and resp.debug.confidence is None
    assert [r.product_id for r in resp.results] == ["p2", "p1"] and resp.related_searches == ["hot sauce"]
    assert retriever.fast_calls == 0
    body = retriever.bodies[0]["query"]["bool"]
    assert body["must"] == [COMPILED] and {"terms": {"brand": ["B"]}} in body["filter"]


async def test_l2_hit_when_not_hot():
    redis = FakeRedis()
    redis.kv[l2_key("brunch sauce")] = json.dumps({"normalized_query": "brunch sauce", "es_query": {"query": COMPILED},
                                                   "related_searches": [], "source": "batch"})
    engine, retriever = _engine(FakeCache(redis))
    resp = await engine.search(SearchRequest(q="brunch sauce", debug=True))
    assert resp.debug.served_by is ServedBy.l2 and retriever.fast_calls == 0


async def test_confident_query_never_consults_the_similar_tier():
    cache = FakeCache(similar_entry=CacheEntry("small brunch sauce", {"query": COMPILED}, similarity=0.95))
    engine, _ = _engine(cache, fast_sim=0.82)
    resp = await engine.search(SearchRequest(q="sauce", debug=True))
    assert resp.debug.served_by is ServedBy.fast_path and cache.similar_calls == 0


async def test_low_confidence_query_uses_a_similar_cached_query_before_the_llm():
    cache = FakeCache(similar_entry=CacheEntry("brunch sauce", {"query": COMPILED}, similarity=0.93))
    engine, _ = _engine(cache, fast_sim=0.60)
    resp = await engine.search(SearchRequest(q="brunch sauces", debug=True))
    assert resp.debug.served_by is ServedBy.warm
    assert resp.debug.cached_query == "brunch sauce" and resp.debug.cache_similarity == 0.93
    assert resp.debug.fast_path_confidence is not None and resp.debug.fast_path_confidence < 0.6


async def test_negated_query_skips_the_similar_tier():
    cache = FakeCache(similar_entry=CacheEntry("snacks with nuts", {"query": COMPILED}, similarity=0.95))
    engine, _ = _engine(cache, fast_sim=0.60)
    resp = await engine.search(SearchRequest(q="sauce without peanut", debug=True))
    assert cache.similar_calls == 0 and resp.debug.served_by is ServedBy.fast_path


async def test_cached_results_still_drop_negated_products():
    cache = FakeCache()
    cache.load_hot_list(_hot({"query": "sauce without barbecue", "es_query": {"query": COMPILED}}))
    engine, _ = _engine(cache)
    resp = await engine.search(SearchRequest(q="sauce without barbecue"))
    assert [r.product_id for r in resp.results] == ["p1"]


# ---- results cache ----


async def test_hot_query_results_are_cached_and_rehydrated_fresh():
    redis = FakeRedis()
    cache = FakeCache(redis)
    cache.load_hot_list(_hot({"query": "sauce"}))  # hot, but no cached query: the fast path computes it
    products = {k: dict(v) for k, v in PRODUCTS.items()}
    engine, retriever = _engine(cache, products=products)
    first = await engine.search(SearchRequest(q="sauce", debug=True))
    for t in list(engine._background):
        await t
    assert first.debug.served_by is ServedBy.fast_path and retriever.fast_calls == 1

    products["p1"]["in_stock"] = False  # stock-out after caching: hydration drops it, no stale result shown
    second = await engine.search(SearchRequest(q="sauce", debug=True))
    assert second.debug.served_by is ServedBy.results_cache and retriever.fast_calls == 1
    assert "p1" not in [r.product_id for r in second.results] and len(second.results) == 2


def test_results_key_varies_with_filters_sort_and_hot_list_version():
    cache = QueryCache(None, None, 0.91, 600)
    a = cache.results_key("sauce", {}, SortBy.relevance.value)
    assert a != cache.results_key("sauce", {"brand": "B"}, SortBy.relevance.value)
    assert a != cache.results_key("sauce", {}, SortBy.price_asc.value)
    cache.load_hot_list(_hot())
    assert a != cache.results_key("sauce", {}, SortBy.relevance.value)


async def test_catalog_update_purges_every_result_list_containing_the_product():
    redis = FakeRedis()
    cache = QueryCache(redis, None, 0.91, 600)
    await cache.put_results("results:v1:0:a", CachedResults(["p1", "p2"], "fast_path", 0.8, [], None, None))
    await cache.put_results("results:v1:0:b", CachedResults(["p3"], "fast_path", 0.8, [], None, None))
    assert purge_results_for(redis, "p2") == 1
    assert await cache.get_results("results:v1:0:a") is None and await cache.get_results("results:v1:0:b")


async def test_cache_outage_falls_back_to_the_fast_path():
    class BrokenRedis(FakeRedis):
        async def get(self, key):
            raise ConnectionError("redis down")

    engine, _ = _engine(FakeCache(BrokenRedis()))
    resp = await engine.search(SearchRequest(q="sauce", filters=SearchFilters(), debug=True))
    assert resp.debug.served_by is ServedBy.fast_path and resp.results


# ---- hot-list job ----


def test_hot_list_takes_l1_queries_and_l2_takes_every_cached_top_20_query():
    rows = [
        {"query": "beef jerky", "tier": "l1", "frequency": 90, "es_query": None, "related_searches": None,
         "source": None},
        {"query": "brunch sauce", "tier": "l1", "frequency": 40, "es_query": {"query": COMPILED},
         "related_searches": ["hot sauce"], "source": "batch"},
        {"query": "sarong", "tier": "l2", "frequency": 9, "es_query": {"query": COMPILED}, "related_searches": [],
         "source": "fallback"},
    ]
    hot = build_hot_list(rows, version=3)
    assert hot["version"] == 3 and [q["query"] for q in hot["queries"]] == ["beef jerky", "brunch sauce"]
    assert "es_query" not in hot["queries"][0] and hot["queries"][1]["es_query"] == {"query": COMPILED}
    l2 = l2_entries(rows)
    assert set(l2) == {l2_key("brunch sauce"), l2_key("sarong")}
    assert json.loads(l2[l2_key("sarong")])["source"] == "fallback"
