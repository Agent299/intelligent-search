"""Train the personalized re-ranker (doc p.178): LightGBM LambdaMART on point-in-time features from the logs.

    uv run python -m pipelines.ranking.train

Labels: purchase 3, cart 2, click 1, shown-but-ignored 0. Split by time (last 20% of searches = test, the 10%
before that = early-stopping validation), so the model is scored on the future, like production. The report
compares, on the same test searches: the logged order, static score alone, the linear fallback, LightGBM, and
LightGBM without the personal features (what personalization adds).
"""

import json
import time
from datetime import UTC, datetime

import numpy as np
from opensearchpy import OpenSearch

from ranking.model import META_PATH, MODEL_PATH, LinearRanker, ndcg_at_k  # isort: skip  (before torch, see module)
from opensearchpy.helpers import scan

from common.config import get_settings
from common.embeddings import embed_queries
from common.topics import SEMANTIC_SEARCH_INDEX
from pipelines.offline import lake
from pipelines.ranking.replay import Group, load_catalog, replay
from ranking.features import FEATURES, PERSONAL

TEST_SHARE, VAL_SHARE = 0.2, 0.1
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "learning_rate": 0.05,
    "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambdarank_truncation_level": 20, "verbose": -1, "seed": 7,
}


def product_vectors(ids: set[str]) -> dict[str, np.ndarray]:
    client = OpenSearch(get_settings().opensearch_url, timeout=60)
    out = {}
    for hit in scan(client, index=SEMANTIC_SEARCH_INDEX, query={"query": {"match_all": {}}},
                    _source=["embedding"], size=1000):
        if hit["_id"] in ids:
            out[hit["_id"]] = np.asarray(hit["_source"]["embedding"], dtype=np.float32)
    return out


def fill_similarity(groups: list[Group]) -> None:
    """Exact cosine for every (query, shown product), the same similarity the k-NN retriever reports."""
    queries = sorted({g.query for g in groups})
    qvec = dict(zip(queries, embed_queries(queries), strict=True))
    pvec = product_vectors({pid for g in groups for pid in g.product_ids})
    col = FEATURES.index("semantic_sim")
    for g in groups:
        q = qvec[g.query]
        g.rows[:, col] = [float(q @ pvec[pid]) if pid in pvec else np.nan for pid in g.product_ids]


def _stack(groups: list[Group]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    return (np.vstack([g.rows for g in groups]), np.concatenate([g.labels for g in groups]),
            [len(g.labels) for g in groups])


def _without_personal(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for f in PERSONAL:
        x[:, FEATURES.index(f)] = np.nan if f == "price_band_gap" else 0.0
    return x


def fit(train: list[Group], val: list[Group], personal: bool = True):
    import lightgbm as lgb  # already loaded by ranking.model, before torch

    xt, yt, gt = _stack(train)
    xv, yv, gv = _stack(val)
    if not personal:
        xt, xv = _without_personal(xt), _without_personal(xv)
    dtrain = lgb.Dataset(xt, yt, group=gt, feature_name=FEATURES)
    dval = lgb.Dataset(xv, yv, group=gv, reference=dtrain)
    return lgb.train(PARAMS, dtrain, num_boost_round=600, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def mean_ndcg(groups: list[Group], score_fn) -> float:
    vals = [ndcg_at_k(list(g.labels), list(score_fn(g.rows))) for g in groups]
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 4)


def main() -> None:
    t0 = time.perf_counter()
    groups = list(replay(lake.read_events(), load_catalog()))
    fill_similarity(groups)
    groups.sort(key=lambda g: g.ts)
    n_test, n_val = int(len(groups) * TEST_SHARE), int(len(groups) * VAL_SHARE)
    test = groups[-n_test:]
    val = groups[-(n_test + n_val):-n_test]
    train = groups[:-(n_test + n_val)]
    print(f"[ranking] {len(groups)} searches with engagement: train {len(train)}, val {len(val)}, test {len(test)}"
          f" ({sum(len(g.labels) for g in groups)} rows; {time.perf_counter() - t0:.0f}s to build)")

    model = fit(train, val)
    flat = fit(train, val, personal=False)
    col = FEATURES.index
    linear = LinearRanker()
    with_history = [g for g in test if g.rows[0, col("user_has_history")] > 0]
    report = {
        "logged order": mean_ndcg(test, lambda x: x[:, col("retrieval_rank")]),
        "static score only": mean_ndcg(test, lambda x: x[:, col("static_score")]),
        "linear fallback": mean_ndcg(test, linear.score),
        "lightgbm, no personal features": mean_ndcg(test, lambda x: flat.predict(_without_personal(x))),
        "lightgbm": mean_ndcg(test, model.predict),
    }
    history_report = {
        "logged order": mean_ndcg(with_history, lambda x: x[:, col("retrieval_rank")]),
        "lightgbm, no personal features": mean_ndcg(with_history, lambda x: flat.predict(_without_personal(x))),
        "lightgbm": mean_ndcg(with_history, model.predict),
    }
    gain = model.feature_importance("gain")
    importance = {f: round(float(v / gain.sum()), 3) for f, v in sorted(zip(FEATURES, gain, strict=True),
                                                                       key=lambda kv: -kv[1])}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_PATH))
    meta = {"trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "features": FEATURES,
            "best_iteration": model.best_iteration, "groups": {"train": len(train), "val": len(val),
                                                               "test": len(test)},
            "ndcg@10_test": report, "ndcg@10_test_users_with_history": history_report,
            "test_users_with_history": len(with_history), "importance_gain_share": importance}
    META_PATH.write_text(json.dumps(meta, indent=1))
    print("[ranking] NDCG@10 on test:")
    for k, v in report.items():
        print(f"    {k:32} {v}")
    print(f"[ranking] users with history ({len(with_history)} test searches): {history_report}")
    print(f"[ranking] gain share: {importance}")
    print(f"[ranking] saved {MODEL_PATH} (best iteration {model.best_iteration}) in "
          f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
