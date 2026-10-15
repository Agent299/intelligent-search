import json

import polars as pl

from common.config import Settings
from common.normalize import SpellCorrector
from pipelines.discovery.discover import input_hash
from pipelines.discovery.mine import baskets, mine
from search_service.cache import QueryCache
from search_service.discovery import Discovery, discovery_key, interleave, primary_category
from search_service.engine import SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.filters import SearchFilters, parse_filters
from search_service.retrieval import Hit

LAPTOPS, MICE, SLEEVES, DRESSES, TOPS = "E > Laptops", "E > Mice", "E > Sleeves", "C > Dresses", "C > Tops"

# ---- mining ----


def test_baskets_join_purchases_and_carts_per_session_and_skip_unknown_products():
    events = pl.DataFrame([
        {"session_id": "s1", "purchased_product_ids": ["lap1"], "added_to_cart_product_id": None},
        {"session_id": "s1", "purchased_product_ids": [], "added_to_cart_product_id": "mouse1"},
        {"session_id": "s2", "purchased_product_ids": ["ghost"], "added_to_cart_product_id": None},
    ])
    assert baskets(events, {"lap1": LAPTOPS, "mouse1": MICE}) == [{LAPTOPS, MICE}]


def test_mine_keeps_lifted_pairs_and_drops_popular_coincidences():
    sessions = ([{LAPTOPS, MICE}] * 3 + [{LAPTOPS, SLEEVES}]           # sleeves: support 1, dropped
                + [{DRESSES, TOPS}] * 3 + [{DRESSES}] * 20 + [{TOPS}] * 20  # popular, low lift
                + [{MICE}] * 2)
    pairs = {(p.product_key, p.co_item): p for p in mine(sessions)}
    assert set(pairs) == {(LAPTOPS, MICE), (MICE, LAPTOPS)}
    assert pairs[(LAPTOPS, MICE)].support == 3 and pairs[(LAPTOPS, MICE)].lift >= 3


def test_input_hash_ignores_order():
    assert input_hash([MICE, SLEEVES]) == input_hash([SLEEVES, MICE]) != input_hash([MICE])


# ---- online ----


def test_primary_category_is_the_most_common_among_top_results_ties_to_rank():
    top = [{"category_path": c} for c in (MICE, LAPTOPS, LAPTOPS, MICE, SLEEVES, SLEEVES, SLEEVES)]
    assert primary_category(top) == MICE  # 2 vs 2 in the top 5: MICE ranks first; SLEEVES is mostly below
    assert primary_category([{"category_path": None}]) is None


def test_interleave_gives_each_complement_a_turn_and_skips_shown_products():
    comps = [{"category": MICE, "product_ids": ["m1", "m2"]}, {"category": SLEEVES, "product_ids": ["s1", "s2"]}]
    assert interleave(comps, exclude={"m1"}) == ["m2", "s1", "s2"]
    assert interleave(comps, exclude=set(), limit=3) == ["m1", "s1", "m2"]


class FakeRedis:
    def __init__(self, kv=None):
        self.kv = kv or {}

    async def get(self, key):
        return self.kv.get(key)


class FakeRetriever:
    def __init__(self, products):
        self.products = products
        self.hydrate_filters = []

    async def keyword(self, query, filters, size):
        return [Hit(pid, 10.0 - i) for i, pid in enumerate(["lap1", "lap2"])]

    async def semantic(self, vector, filters, size):
        return [Hit(pid, 0.85 - 0.01 * i) for i, pid in enumerate(["lap1", "lap2"])]

    async def hydrate(self, ids, filters):
        self.hydrate_filters.append(filters)
        return {pid: self.products[pid] for pid in ids
                if pid in self.products and (filters.include_out_of_stock or self.products[pid]["in_stock"])
                and (not filters.brands or self.products[pid]["brand"] in filters.brands)}


class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.0, 1.0] for _ in texts]


def _p(pid, category, brand="B", in_stock=True):
    return {"product_id": pid, "name": pid, "description": "", "brand": brand, "price": 9.0, "currency": "USD",
            "rating": 4.0, "imageUrl": None, "product_type": "x", "static_score": 0.5, "category_path": category,
            "in_stock": in_stock}


PRODUCTS = {"lap1": _p("lap1", LAPTOPS), "lap2": _p("lap2", LAPTOPS), "m1": _p("m1", MICE, brand="Logi"),
            "m2": _p("m2", MICE, in_stock=False), "s1": _p("s1", SLEEVES, brand="Case")}
ENTRY = {discovery_key(LAPTOPS): json.dumps({"chosen": [MICE, SLEEVES], "input_hash": "h", "complements": [
    {"category": MICE, "product_ids": ["m1", "m2"]}, {"category": SLEEVES, "product_ids": ["s1"]}]})}


async def test_pairs_well_with_hydrates_fresh_and_ignores_the_shoppers_brand_filter():
    retriever = FakeRetriever(PRODUCTS)
    pairs = await Discovery(FakeRedis(ENTRY), retriever).pairs_well_with(
        [PRODUCTS["lap1"], PRODUCTS["lap2"]], parse_filters('{"brand": "B"}'))
    assert [p["product_id"] for p in pairs] == ["m1", "s1"]  # m2 is out of stock; brand B doesn't apply
    assert retriever.hydrate_filters[-1].brands == []


async def test_pairs_well_with_fails_open():
    class Broken(FakeRedis):
        async def get(self, key):
            raise ConnectionError("down")

    d = Discovery(Broken(), FakeRetriever(PRODUCTS))
    assert await d.pairs_well_with([PRODUCTS["lap1"]], SearchFilters()) == []
    assert await Discovery(FakeRedis(), FakeRetriever(PRODUCTS)).pairs_well_with(
        [PRODUCTS["lap1"]], SearchFilters()) == []


async def test_search_response_carries_pairs_well_with_and_hot_related_searches():
    retriever = FakeRetriever(PRODUCTS)
    cache = QueryCache(None, None, 0.91, 600)
    cache.load_hot_list({"version": 1, "queries": [{"query": "laptop", "related_searches": ["laptop bag"]}]})
    engine = SearchEngine(retriever, FakeEmbeddings(), SpellCorrector(), EventSink(None, None), Settings(),
                          cache=cache, discovery=Discovery(FakeRedis(ENTRY), retriever))
    resp = await engine.search(SearchRequest(q="laptop"))
    assert [r.product_id for r in resp.results] == ["lap1", "lap2"]
    assert [r.product_id for r in resp.pairs_well_with] == ["m1", "s1"]
    assert resp.related_searches == ["laptop bag"]
    body = resp.model_dump(by_alias=True)
    assert "pairsWellWith" in body and "relatedSearches" in body
