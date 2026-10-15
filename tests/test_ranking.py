import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from common.config import Settings
from common.normalize import SpellCorrector
from common.schemas import EventType, ServedBy, SortBy
from pipelines.ranking.replay import State, replay
from ranking import features as F
from ranking.model import LightGBMRanker, LinearRanker, ndcg_at_k
from ranking.serving import SESSION_PREFIX, USER_PREFIX, Reranker, qp_key
from search_service.engine import SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.retrieval import Hit

COL = F.FEATURES.index

# ---- features ----


def test_static_score_orders_purchase_over_cart_over_click_and_shrinks_sparse_rates():
    pop = F.popularity(4.5, 300)
    assert 0 < pop < 1 and F.popularity(None, 10) == 0.0
    assert F.static_score(pop, 3, 10) > F.static_score(pop, 2, 10) > F.static_score(pop, 1, 10)
    # one click in 2 impressions is not a 50% product: the prior keeps it close to a cold one
    assert F.static_score(pop, 1, 2) - F.static_score(pop, 0, 0) < 0.1


def test_qp_value_uses_the_prior_below_the_impression_floor_unless_engaged():
    assert F.qp_value(0, 2) == F.PRIOR_RATE
    assert F.qp_value(0, 50) < F.PRIOR_RATE < F.qp_value(3, 5)


def test_row_personal_features():
    user = F.UserProfile()
    user.add({"brand": "Acme", "category_path": "A > B", "price": 10.0}, 3.0)
    user.add({"brand": "Other", "category_path": "A > C", "price": 30.0}, 1.0)
    p = {"brand": "Acme", "category_path": "A > B", "price": 20.0, "rating": 4.0, "review_count": 10,
         "static_score": 0.4}
    r = F.row(rank=1, sim=0.8, product=p, query="acme mug", user=user, session_brands={"Acme": 2}, qp=0.1)
    assert len(r) == len(F.FEATURES)
    assert r[COL("retrieval_rank")] == 0.5 and r[COL("brand_in_query")] == 1.0
    assert r[COL("user_brand_aff")] == 0.75 and r[COL("user_cat_aff")] == 0.75 and r[COL("user_top_aff")] == 1.0
    assert r[COL("session_brand_clicks")] == 2 and r[COL("user_has_history")] == 1.0
    assert r[COL("price_band_gap")] == pytest.approx(abs(math.log(20 / 30)))  # median of [10, 30] -> 30
    anon = F.row(rank=0, sim=0.8, product=p, query="mug", user=None, session_brands={}, qp=0.1)
    assert math.isnan(anon[COL("price_band_gap")]) and anon[COL("user_brand_aff")] == 0.0


def test_profile_json_round_trip_keeps_shares_and_median():
    user = F.UserProfile()
    for price in (5.0, 7.0, 9.0):
        user.add({"brand": "A", "category_path": "X > Y", "price": price}, 1.0)
    back = F.UserProfile.from_json(json.loads(json.dumps(user.to_json())))
    assert back.median_price == 7.0 and back.brands == {"A": 3.0} and back.total == 3.0


# ---- point-in-time replay ----

CATALOG = {pid: {"product_id": pid, "name": f"{b} mug", "brand": b, "category_path": "Home > Mugs", "price": 10.0,
                 "rating": 4.0, "review_count": 50} for pid, b in (("p1", "Acme"), ("p2", "Other"), ("p3", "Acme"))}


def _event(t, shown, clicked=None, purchased=(), session="s1", user="u1", query="mug"):
    return {"event_timestamp": datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=t), "user_id": user,
            "session_id": session, "normalized_query": query, "displayed_product_ids": shown,
            "clicked_product_id": clicked, "added_to_cart_product_id": None, "purchased_product_ids": list(purchased)}


def test_replay_features_never_include_their_own_label():
    events = pl.DataFrame([_event(0, ["p1", "p2"], clicked="p1", purchased=["p1"]),
                           _event(5, ["p3", "p2"], clicked="p3")])
    state = State()
    first, second = list(replay(events, CATALOG, state))
    assert list(first.labels) == [3.0, 0.0]
    assert first.rows[0, COL("user_brand_aff")] == 0.0 and first.rows[0, COL("session_brand_clicks")] == 0.0
    # the second search sees the first one's purchase (Acme) and click, but not its own
    assert second.rows[0, COL("user_brand_aff")] == 1.0 and second.rows[0, COL("session_brand_clicks")] == 1.0
    # p2 was shown for "mug" once and ignored; p3 never shown: both below the floor, both at the prior
    assert second.rows[1, COL("qp_engagement")] == second.rows[0, COL("qp_engagement")] == F.PRIOR_RATE
    assert state.product_impressions["p2"] == 2 and state.product_weighted["p1"] == 3.0
    assert state.users["u1"].brands == {"Acme": 4.0}


def test_replay_skips_searches_without_engagement_but_still_counts_their_impressions():
    state = State()
    groups = list(replay(pl.DataFrame([_event(0, ["p1", "p2"])]), CATALOG, state))
    assert groups == [] and state.product_impressions["p1"] == 1


# ---- models ----


def test_ndcg():
    assert ndcg_at_k([0, 0, 3], [3, 2, 1]) < ndcg_at_k([0, 0, 3], [1, 2, 3]) == 1.0
    assert ndcg_at_k([0, 0], [1, 2]) is None


def test_linear_ranker_treats_missing_as_zero():
    rows = np.zeros((2, len(F.FEATURES)))
    rows[0, COL("retrieval_rank")], rows[1, COL("retrieval_rank")] = 1.0, 0.5
    rows[:, COL("price_band_gap")] = np.nan
    s = LinearRanker().score(rows)
    assert s[0] > s[1] and not np.isnan(s).any()


def test_lightgbm_ranker_checks_feature_names(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    x, y = np.random.default_rng(0).random((400, len(F.FEATURES))), np.tile([1, 0, 0, 0], 100)
    booster = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5},
                        lgb.Dataset(x, y, group=[4] * 100, feature_name=F.FEATURES), 5)
    booster.save_model(tmp_path / "ok.txt")
    assert LightGBMRanker(tmp_path / "ok.txt").score(x[:3]).shape == (3,)
    wrong = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5},
                      lgb.Dataset(x[:, :3], y, group=[4] * 100), 5)
    wrong.save_model(tmp_path / "wrong.txt")
    with pytest.raises(ValueError):
        LightGBMRanker(tmp_path / "wrong.txt")


# ---- online re-rank ----


class FakeRedis:
    def __init__(self, kv=None, hashes=None):
        self.kv, self.hashes = kv or {}, hashes or {}

    def pipeline(self, transaction=False):
        return _Pipe(self)


class _Pipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, key):
        self.ops.append(self.r.kv.get(key))

    def hgetall(self, key):
        self.ops.append({k.encode(): str(v).encode() for k, v in self.r.hashes.get(key, {}).items()})

    def hmget(self, key, fields):
        h = self.r.hashes.get(key, {})
        self.ops.append([str(h[f]).encode() if f in h else None for f in fields])

    def hincrbyfloat(self, key, field, amount):
        h = self.r.hashes.setdefault(key, {})
        h[field] = h.get(field, 0.0) + amount

    def expire(self, key, ttl):
        pass

    async def execute(self):
        return self.ops


def _products():
    return {pid: {**p, "static_score": 0.3} for pid, p in CATALOG.items()}


# A linear ranker that only listens to brand affinity, so tests isolate the personal signal.
BRAND_ONLY = LinearRanker({"retrieval_rank": 0.01, "user_brand_aff": 1.0, "session_brand_clicks": 1.0})


async def test_rerank_moves_the_users_brand_up_and_reads_context_in_one_round_trip():
    profile = F.UserProfile()
    profile.add({"brand": "Acme", "category_path": "Home > Mugs", "price": 10.0}, 3.0)
    redis = FakeRedis(kv={f"{USER_PREFIX}u1": json.dumps(profile.to_json())})
    r = Reranker(redis, BRAND_ONLY)
    assert await r.rerank("mug", "u1", None, ["p2", "p1"], {}, _products()) == ["p1", "p2"]
    assert await r.rerank("mug", None, None, ["p2", "p1"], {}, _products()) == ["p2", "p1"]  # anonymous


async def test_session_clicks_are_recorded_and_used():
    redis = FakeRedis()
    r = Reranker(redis, BRAND_ONLY)
    await r.record_click("s9", "Acme")
    assert redis.hashes[f"{SESSION_PREFIX}s9"] == {"Acme": 1.0}
    assert await r.rerank("mug", None, "s9", ["p2", "p3"], {}, _products()) == ["p3", "p2"]


async def test_rerank_fails_open():
    class Broken(FakeRedis):
        def pipeline(self, transaction=False):
            raise ConnectionError("down")

    class Exploding(LinearRanker):
        def score(self, rows):
            raise RuntimeError("bad model")

    assert await Reranker(Broken(), LinearRanker()).rerank("mug", "u1", "s1", ["p2", "p1"], {},
                                                            _products()) in (["p2", "p1"], ["p1", "p2"])
    assert await Reranker(FakeRedis(), Exploding()).rerank("mug", None, None, ["p2", "p1"], {},
                                                           _products()) == ["p2", "p1"]


def test_qp_key_matches_publish():
    assert qp_key("mug").startswith("qp:v1:")


# ---- engine ----


class FakeRetriever:
    async def keyword(self, query, filters, size):
        return [Hit(pid, 10.0 - i) for i, pid in enumerate(["p2", "p1", "p3"])]

    async def semantic(self, vector, filters, size):
        return [Hit(pid, 0.82 - 0.01 * i) for i, pid in enumerate(["p2", "p1", "p3"])]

    async def hydrate(self, ids, filters):
        return {pid: _products()[pid] for pid in ids if pid in CATALOG}


class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.0, 1.0] for _ in texts]


def _engine(reranker):
    return SearchEngine(FakeRetriever(), FakeEmbeddings(), SpellCorrector(), EventSink(None, None), Settings(),
                        reranker=reranker)


async def test_engine_personalizes_relevance_order_only():
    profile = F.UserProfile()
    profile.add({"brand": "Acme", "category_path": "Home > Mugs", "price": 10.0}, 3.0)
    engine = _engine(Reranker(FakeRedis(kv={f"{USER_PREFIX}u1": json.dumps(profile.to_json())}), BRAND_ONLY))
    resp = await engine.search(SearchRequest(q="mug", user_id="u1", debug=True))
    assert [r.product_id for r in resp.results][:2] == ["p1", "p3"] and resp.debug.ranker == "linear"
    by_price = await engine.search(SearchRequest(q="mug", user_id="u1", sort_by=SortBy.price_asc, debug=True))
    assert by_price.debug.ranker is None


def test_fallback_results_keep_their_interleaved_order():
    engine = _engine(Reranker(FakeRedis(), LinearRanker()))
    assert not engine._should_rerank(SearchRequest(q="x"), ServedBy.fallback.value)
    assert engine._should_rerank(SearchRequest(q="x"), ServedBy.l1.value)


async def test_click_events_feed_the_session_signal():
    redis = FakeRedis()
    engine = _engine(Reranker(redis, LinearRanker()))
    await engine.on_interaction({"session_id": "s1"}, EventType.click, "p3")
    await engine.on_interaction({"session_id": "s1"}, EventType.purchase, "p1")  # only clicks count
    assert redis.hashes[f"{SESSION_PREFIX}s1"] == {"Acme": 1.0}
