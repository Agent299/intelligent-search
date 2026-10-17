import json
import math
import random

import pytest
from test_gateway import OPUS, FakeCaller  # tests/ is on sys.path (rootdir-relative imports)

from eval import judge, label
from eval.golden.build import typo
from eval.metrics import ndcg_at, percentile, recall_at
from genai_gateway.core import Gateway


def test_ndcg_and_recall_against_graded_labels():
    labels = {"a": 3, "b": 2, "c": 0, "d": 1}
    assert ndcg_at(["a", "b", "d", "c"], labels) == pytest.approx(1.0)
    assert ndcg_at(["c", "d", "b", "a"], labels) < 0.6
    assert ndcg_at(["x", "y"], {"x": 0}) is None                 # no relevant product: left out
    assert ndcg_at(["unlabeled", "a"], labels) < 1.0             # unlabeled counts as 0
    assert recall_at(["a", "zz"], labels) == 0.5                 # relevant = grade >= 2 -> {a, b}
    assert recall_at(["a"], {"a": 1}) is None


def test_percentile_is_nearest_rank():
    assert percentile(list(range(1, 101)), 90) == 90
    assert percentile([5.0], 99) == 5.0 and percentile([], 50) is None


def test_typo_transposes_inside_one_long_word():
    t = typo("organic basmati rice", random.Random(0))
    assert t != "organic basmati rice" and sorted(t) == sorted("organic basmati rice")
    assert typo("tv usb", random.Random(0)) == "tv usb"  # nothing long enough to break


def test_stratified_pilot_covers_every_segment():
    qs = [{"query": f"{s}{i}", "segment": s} for s in "abc" for i in range(10)]
    picked = label.stratified(qs, 6)
    assert len(picked) == 6 and {p[0] for p in picked} == {"a", "b", "c"}


def test_pools_union_each_systems_top_results_in_order(monkeypatch):
    runs = {"baseline": {"q": {"ids": ["p1", "p2", "p3"]}}, "full": {"q": {"ids": ["p2", "p9"]}},
            "full_no_rerank": {}}
    monkeypatch.setattr(label, "load_results", lambda s: runs[s])
    monkeypatch.setattr(label, "load_golden", lambda: [{"query": "q", "segment": "x"}])
    assert label.pools(depth=2) == {"q": ["p1", "p2", "p9"]}


async def test_label_guardrail_requires_every_product_and_drops_unknown_ids():
    inp = {"query": "rice", "products": [{"id": "p1", "title": "Basmati"}, {"id": "p2", "title": "Lentils"}]}
    full = {"grades": [{"id": "p1", "grade": 3}, {"id": "p2", "grade": 1}, {"id": "p7", "grade": 3}]}
    res = await Gateway(caller=FakeCaller({OPUS: [json.dumps(full)]})).run("label", inp)
    assert res.valid and res.output == {"grades": {"p1": 3, "p2": 1}}
    partial = {"grades": [{"id": "p1", "grade": 3}]}
    res = await Gateway(caller=FakeCaller({OPUS: [json.dumps(partial)] * 3})).run("label", inp)
    assert not res.valid


def _item(query, new_is_a, score_a, score_b):
    return {"meta": {"query": query, "new_is_a": new_is_a}, "output": {"score_a": score_a, "score_b": score_b}}


def test_judge_maps_positions_back_to_systems_and_counts_flips():
    items = [_item("q1", True, 5, 3), _item("q1", False, 3, 5),   # consistent: full wins in both orders
             _item("q2", True, 4, 4), _item("q2", False, 2, 4),   # flips: tie, then full wins
             _item("q3", False, 5, 2)]                            # baseline in A wins
    r = judge.summarize(items)
    assert (r["new_wins"], r["old_wins"], r["ties"]) == (3, 1, 1)
    assert r["order_flips"] == "1/2"
    assert r["mean_score"] == {"full": pytest.approx((5 + 5 + 4 + 4 + 2) / 5), "baseline": (3 + 3 + 4 + 2 + 5) / 5}
    assert r["position_a_win_share"] == 0.5  # decided: q1a(A), q1b(B), q2b(B), q3(A)


def test_ndcg_matches_the_ranking_module():
    from ranking.model import ndcg_at_k

    labels = {"a": 3, "b": 0, "c": 2}
    assert math.isclose(ndcg_at(["b", "c", "a"], labels), ndcg_at_k([0, 2, 3], [3, 2, 1]))


def test_judge_agreement_compares_verdicts_per_judgment_and_per_query(monkeypatch):
    ref = {"c1": _item("q1", True, 5, 3), "c2": _item("q1", False, 3, 5),   # full wins both orders
           "c3": _item("q2", True, 4, 4)}                                    # tie
    cand = {"c1": _item("q1", True, 4, 3), "c2": _item("q1", False, 4, 3),  # agrees, then reverses (A bias)
            "c3": _item("q2", True, 3, 4)}                                   # decides where reference tied

    class Job:
        def __init__(self, run_id, stage):
            self.items = ref if run_id == "ref" else cand

    monkeypatch.setattr(judge, "BatchJob", Job)
    r = judge.agreement("ref", "cand")
    assert r["items"] == 3 and r["verdict_agreement"] == round(1 / 3, 3)
    assert r["agreement_where_both_decided"] == 0.5 and r["opposite_verdicts"] == 1
    # q1 margins over both orders: reference +4, candidate 1 - 1 = 0 (position bias cancels to a tie)
    assert r["queries"] == 2 and r["query_verdict_agreement"] == 0.0
