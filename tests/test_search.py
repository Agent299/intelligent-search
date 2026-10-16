import json

import pytest
from fastapi.testclient import TestClient

from common.config import Settings
from common.normalize import SpellCorrector
from common.schemas import EventType, SortBy
from search_service.app import create_app
from search_service.engine import EmptyQuery, SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.filters import FilterError, SearchFilters, parse_filters
from search_service.fusion import confidence, rrf_fuse
from search_service.negation import parse_negations
from search_service.retrieval import Hit

# ---- filters ----


def test_parse_filters_maps_known_keys_and_treats_the_rest_as_attributes():
    f = parse_filters(json.dumps({"brand": "Decathlon", "size": "S", "price_max": "50", "category": "A > B"}))
    assert f.brands == ["Decathlon"] and f.price_max == 50.0 and f.category == "A > B"
    assert f.attributes == {"size": ["s"]}
    assert not f.include_out_of_stock


def test_filter_clauses_split_between_indices():
    f = parse_filters(json.dumps({"brand": ["A", "B"], "color": "Navy", "category": "X"}))
    vec = f.vector_index_clauses()
    prod = f.product_index_clauses()
    assert {"terms": {"brand": ["A", "B"]}} in vec and {"term": {"in_stock": True}} in vec
    assert not any("nested" in c or "prefix" in c for c in vec)  # semantic index lacks these fields
    assert {"prefix": {"category_path": "X"}} in prod
    nested = next(c for c in prod if "nested" in c)
    assert {"terms": {"attributes.value": ["navy"]}} in nested["nested"]["query"]["bool"]["filter"]


def test_in_stock_false_includes_out_of_stock_items():
    assert parse_filters('{"in_stock": false}').include_out_of_stock
    assert {"term": {"in_stock": True}} not in parse_filters('{"in_stock": false}').product_index_clauses()


@pytest.mark.parametrize("raw", ["{bad", "[1, 2]", '{"price_min": "cheap"}'])
def test_bad_filters_raise(raw):
    with pytest.raises(FilterError):
        parse_filters(raw)


# ---- negation ----


def test_negations_are_removed_from_retrieval_query():
    p = parse_negations("healthy snacks without nuts for kids")
    assert p.positive == "healthy snacks for kids" and p.negated == ["nuts"]
    p = parse_negations("a summer dress that's not too formal")
    assert p.positive == "a summer dress" and p.negated == ["formal"]
    assert parse_negations("basmati rice").negated == []


def test_free_phrases_stay_in_query_but_still_exclude_the_ingredient():
    p = parse_negations("gluten free pasta")
    assert p.positive == "gluten free pasta" and p.negated == ["gluten"]
    assert p.excludes("High Gluten Bread Flour") and not p.excludes("Gluten-Free Penne")


def test_excludes_matches_compound_words_but_respects_declared_absence():
    p = parse_negations("snacks without nuts")
    assert p.excludes("Raw Walnuts 2lb") and p.excludes("Peanut Butter Pretzels")
    assert not p.excludes("Nut-Free Granola Bars") and not p.excludes("Nutrition Bar Variety")


# ---- fusion & confidence ----


def test_rrf_rewards_agreement_between_retrievers():
    kw = [Hit("a", 12.0), Hit("b", 9.0)]
    sem = [Hit("b", 0.82), Hit("c", 0.70)]
    ranked = rrf_fuse(kw, sem)
    assert [c.product_id for c in ranked] == ["b", "a", "c"]
    b = ranked[0]
    assert b.keyword_rank == 2 and b.semantic_rank == 1 and b.semantic_sim == 0.82


def test_prior_moves_a_product_at_most_one_place_near_the_top():
    kw = [Hit(p, 1) for p in "abcd"]
    ranked = rrf_fuse(kw, [], priors={"c": 1.0, "d": 1.0})
    assert [c.product_id for c in ranked] == ["a", "c", "b", "d"]  # c passes b but not a
    ranked = rrf_fuse([Hit("a", 1)], [Hit("b", 0.7)], priors={"b": 1.0})
    assert ranked[0].product_id == "b"  # equal RRF, prior decides


def test_confidence_separates_strong_and_weak_matches():
    strong_sem = [Hit(str(i), 0.83) for i in range(10)]
    strong = confidence([Hit(str(i), 10) for i in range(10)], strong_sem, rrf_fuse([], strong_sem))
    weak_sem = [Hit(f"x{i}", 0.56) for i in range(10)]
    weak = confidence([], weak_sem, rrf_fuse([], weak_sem))
    assert strong.score > 0.9 and weak.score < 0.1
    assert confidence([], [], []).score == 0.0


# ---- engine ----


class FakeRetriever:
    def __init__(self, products):
        self.products = products
        self.seen_query = None

    async def keyword(self, query, filters, size):
        self.seen_query = query
        return [Hit(pid, 10.0 - i) for i, pid in enumerate(self.products) if query.split()[0] in
                self.products[pid]["name"].lower()]

    async def semantic(self, vector, filters, size):
        return [Hit(pid, 0.80 - 0.01 * i) for i, pid in enumerate(self.products)]

    async def hydrate(self, ids, filters):
        return {pid: self.products[pid] for pid in ids if pid in self.products}


class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.0, 1.0] for _ in texts]


class RecordingSink(EventSink):
    def __init__(self):
        super().__init__(None, None)
        self.searches = []

    async def search(self, **kw):
        self.searches.append(kw)


def _product(pid, name, price, ptype="snack"):
    return {"product_id": pid, "name": name, "description": "d" * 400, "brand": "B", "price": price,
            "currency": "USD", "rating": 4.5, "imageUrl": None, "product_type": ptype, "static_score": 0.5}


def _engine(products):
    spell = SpellCorrector()
    spell.add_words({"snacks": 10, "chips": 10, "walnuts": 5})
    return SearchEngine(FakeRetriever(products), FakeEmbeddings(), spell, RecordingSink(), Settings())


PRODUCTS = {
    "p1": _product("p1", "Snacks Chips", 5.0),
    "p2": _product("p2", "Snacks Walnuts Mix", 3.0),
    "p3": _product("p3", "Snacks Cookies", 9.0),
}


async def test_engine_returns_doc_shaped_response_and_logs_event():
    engine = _engine(PRODUCTS)
    resp = await engine.search(SearchRequest(q="Snakcs!", user_id="u1", page_size=2, debug=True))
    body = resp.model_dump(by_alias=True, exclude_none=True)
    assert body["correctedQuery"] == "snacks"
    assert len(body["results"]) == 2 and body["pagination"]["totalResults"] == 3
    assert body["results"][0]["description"].endswith("...")
    assert body["debug"]["servedBy"] == "fast_path"
    await __import__("asyncio").sleep(0)  # let the background event task run
    logged = engine.sink.searches[0]
    assert logged["user_id"] == "u1" and logged["displayed"] == [r["productId"] for r in body["results"]]


async def test_engine_drops_negated_products_and_strips_them_from_retrieval():
    engine = _engine(PRODUCTS)
    resp = await engine.search(SearchRequest(q="snacks without walnuts"))
    assert engine.retriever.seen_query == "snacks"
    assert "p2" not in [r.product_id for r in resp.results]


async def test_price_sort_and_empty_query():
    engine = _engine(PRODUCTS)
    resp = await engine.search(SearchRequest(q="snacks", sort_by=SortBy.price_asc))
    assert [r.price.amount for r in resp.results] == [3.0, 5.0, 9.0]
    with pytest.raises(EmptyQuery):
        await engine.search(SearchRequest(q="?!"))


# ---- HTTP ----


def test_http_search_and_events():
    engine = _engine(PRODUCTS)

    async def interaction(search_id, event_type, product_id):
        return search_id == "known" and event_type is EventType.click

    engine.sink.interaction = interaction
    with TestClient(create_app(engine)) as client:
        r = client.get("/v1/search", params={"q": "snacks", "filters": '{"price_max": 6}', "pageSize": 1})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"searchId", "results", "pagination", "relatedSearches", "facets"} and "debug" not in body
        assert client.get("/v1/search", params={"q": "snacks", "filters": "{"}).status_code == 422
        assert client.get("/v1/search", params={"q": "snacks", "sortBy": "cheapest"}).status_code == 422
        ok = {"searchId": "known", "type": "click", "productId": "p1"}
        assert client.post("/v1/events", json=ok).status_code == 202
        assert client.post("/v1/events", json={**ok, "searchId": "gone"}).status_code == 404
        assert client.post("/v1/events", json={**ok, "type": "search"}).status_code == 422


def test_search_filters_default_hides_out_of_stock():
    assert {"term": {"in_stock": True}} in SearchFilters().product_index_clauses()


# ---- facets (Phase 12) ----


def _p(pid, brand, path, price):
    return {"product_id": pid, "brand": brand, "category_path": path, "price": price}


def test_facets_count_brands_and_offer_children_of_the_shared_category_path():
    from search_service.facets import compute_facets
    products = {
        "a": _p("a", "Lays", "Food > Snacks > Chips", 3.0),
        "b": _p("b", "Lays", "Food > Snacks > Chips > Baked", 4.5),
        "c": _p("c", "Kind", "Food > Snacks > Bars", 1.5),
        "d": _p("d", None, None, 9.0),
    }
    f = compute_facets(["a", "b", "c", "d", "gone"], products)
    assert [(v.value, v.count) for v in f.brands] == [("Lays", 2), ("Kind", 1)]
    # every categorized result shares "Food > Snacks", so the options are its children, as full prefixes
    assert [(v.value, v.label, v.count) for v in f.categories] == [
        ("Food > Snacks > Chips", "Chips", 2), ("Food > Snacks > Bars", "Bars", 1)]
    assert (f.price_min, f.price_max) == (1.5, 9.0)


def test_facets_drill_down_once_a_category_is_chosen_and_are_absent_without_results():
    from search_service.facets import compute_facets
    products = {"a": _p("a", "X", "Food > Snacks > Chips > Baked", 1.0),
                "b": _p("b", "X", "Food > Snacks > Chips > Kettle", 2.0)}
    f = compute_facets(["a", "b"], products)
    assert [v.label for v in f.categories] == ["Baked", "Kettle"]
    single = compute_facets(["a"], products)  # one leaf: nothing deeper to offer
    assert single.categories == []
    assert compute_facets([], products) is None
