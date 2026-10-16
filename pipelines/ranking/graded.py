"""Train the re-ranker on graded relevance over real retrieval candidates (fixes what Phase 11 found).

    uv run python -m pipelines.ranking.graded queries [--n 400]        # sample training queries (no golden ones)
    uv run python -m pipelines.ranking.graded capture train|golden     # candidates + feature rows, as served
    uv run python -m pipelines.ranking.graded label-submit RUN [--limit N] [--model M]   # paid: Batch API
    uv run python -m pipelines.ranking.graded label-collect RUN
    uv run python -m pipelines.ranking.graded train [--install]

Why: the Phase 8 model learned from synthetic logs whose "shown" lists were random same-category products with
the query's source product near the top. Serving re-ranks the retriever's real top 100, so the model was trained
on a different distribution and lost to plain retrieval order on the golden set. Here the training lists are the
retriever's own candidates, captured through the serving engine itself (the same feature rows the live re-rank
computes, for an anonymous shopper), and the labels are 0-3 grades from the `label` task.

The golden set stays the test set: training queries never include a golden query, and `train` reports NDCG@10 on
the golden candidates for retrieval order, the current model and the new one. Unjudged products count as 0, the
pooled-evaluation convention; "condensed" NDCG (unjudged products dropped) is reported alongside, because a new
ranking that surfaces unjudged products is otherwise penalised for them.
"""

import argparse
import asyncio
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from ranking.model import META_PATH, MODEL_PATH, LightGBMRanker, load_ranker  # isort: skip  (LightGBM before torch)
from common.db import connect
from eval.golden.build import SEGMENTS
from eval.label import STAGE, collect_into, label_items, load_labels
from eval.systems import NoResultsCache, load_golden
from pipelines.offline.batches import BatchJob
from ranking.features import FEATURES
from ranking.serving import Reranker

DIR = Path("data/ranking/graded")
TRAIN_QUERIES = DIR / "train_queries.yaml"
LABELS = DIR / "labels.jsonl"
POOL_DEPTH = 30          # graded per training query, in retrieval order (the re-rank head is 100)
SEED = 23
# Over-sample the hard segments: that is where the old model lost most (Phase 11).
WEIGHTS = {"head_easy": 1.0, "head_hard": 2.0, "torso": 1.0, "torso_hard": 2.0, "tail": 1.0, "tail_hard": 2.0,
           "pipeline_failed": 1.0, "dietary_negation": 1.0}
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "learning_rate": 0.05,
    "num_leaves": 15, "min_data_in_leaf": 30, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambdarank_truncation_level": 20, "verbose": -1, "seed": 7,
}
FOLDS = 5


def candidates_path(name: str) -> Path:
    return DIR / f"candidates-{name}.jsonl"


# ---- training queries ----

def sample_queries(n: int) -> list[dict]:
    golden = load_golden()
    taken = {g["query"] for g in golden} | {g["intended"] for g in golden if g.get("intended")}
    total = sum(WEIGHTS.values())
    rng = random.Random(SEED)
    out = []
    for segment, where, _ in SEGMENTS:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT normalized_query AS q FROM query_stats WHERE {where} ORDER BY normalized_query")
            pool = [r["q"] for r in cur.fetchall() if r["q"] not in taken and 2 <= len(r["q"]) <= 60]
        for q in rng.sample(pool, min(round(n * WEIGHTS[segment] / total), len(pool))):
            out.append({"query": q, "segment": segment})
            taken.add(q)
    return out


def load_train_queries() -> list[dict]:
    return yaml.safe_load(TRAIN_QUERIES.read_text())["queries"]


# ---- capture: candidates and feature rows exactly as the live re-rank sees them ----

class _Recorder:
    """Stands in for the model: keeps the rows, returns equal scores (the stable sort keeps retrieval order)."""
    name = "capture"

    def __init__(self):
        self.rows: np.ndarray | None = None

    def score(self, rows: np.ndarray) -> np.ndarray:
        self.rows = rows
        return np.zeros(len(rows))


class CaptureReranker(Reranker):
    def __init__(self, redis):
        super().__init__(redis, _Recorder())
        self.ids: list[str] | None = None

    async def rerank(self, query, user_id, session_id, ids, sims, products):
        self.ids = list(ids)
        return await super().rerank(query, user_id, session_id, ids, sims, products)


async def capture(queries: list[dict]) -> list[dict]:
    from opensearchpy import AsyncOpenSearch
    from redis.asyncio import Redis

    from common.config import get_settings
    from common.embedding_service import EmbeddingService
    from search_service import hot_list
    from search_service.engine import SearchEngine, SearchRequest
    from search_service.events import EventSink
    from search_service.fallback import FallbackPath
    from search_service.gateway_client import GatewayClient
    from search_service.retrieval import Retriever
    from search_service.taxonomy import TaxonomyIndex
    from search_service.vocab import build_spell_corrector

    s = get_settings()
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=30), Redis.from_url(s.redis_url)
    gateway = GatewayClient(s.genai_gateway_url)
    try:
        spell = await asyncio.to_thread(build_spell_corrector)
        embeddings, retriever = EmbeddingService(redis), Retriever(os_client)
        cache = NoResultsCache(redis, os_client, s.warm_tier_min_similarity, s.results_cache_ttl_s)
        await hot_list.load(cache)
        fallback = FallbackPath(retriever, embeddings, gateway, await TaxonomyIndex.build(embeddings), redis,
                                os_client)
        recorder = CaptureReranker(redis)
        # no event sink: training traffic must not reach the logs the pipelines learn from
        engine = SearchEngine(retriever, embeddings, spell, EventSink(None, None), s, fallback, cache, None, recorder)
        out = []
        for i, g in enumerate(queries):
            recorder.ids, recorder.ranker.rows = None, None
            resp = await engine.search(SearchRequest(q=g["query"], page_size=10, debug=True))
            if recorder.ranker.rows is None:  # LLM fallback or < 2 candidates: production does not re-rank these
                continue
            out.append({"query": g["query"], "segment": g["segment"], "served_by": resp.debug.served_by.value,
                        "ids": recorder.ids, "rows": recorder.ranker.rows.tolist()})
            if (i + 1) % 50 == 0:
                print(f"[graded] captured {i + 1}/{len(queries)}", flush=True)
        return out
    finally:
        await gateway.aclose()
        await os_client.close()
        await redis.aclose()


def load_candidates(name: str) -> list[dict]:
    return [json.loads(line) for line in candidates_path(name).read_text().splitlines()]


# ---- training and evaluation ----

def dcg(grades) -> float:
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades))


def ndcg(ranked: list[str], labels: dict[str, int], k: int = 10, condensed: bool = False) -> float | None:
    """NDCG@k with the same gain and ideal as eval.metrics (ideal over every judged product for the query)."""
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    if condensed:
        ranked = [pid for pid in ranked if pid in labels]
    return dcg([labels.get(pid, 0) for pid in ranked[:k]]) / ideal


def groups(cands: list[dict], labels: dict[str, dict[str, int]]):
    """Training groups: each query's judged candidates (rows, grades)."""
    xs, ys, sizes, queries = [], [], [], []
    for c in cands:
        lab = labels.get(c["query"], {})
        idx = [i for i, pid in enumerate(c["ids"]) if pid in lab]
        grades = [lab[c["ids"][i]] for i in idx]
        if len(idx) < 2 or max(grades) == 0:
            continue
        xs.append(np.asarray(c["rows"], dtype=np.float64)[idx])
        ys.append(np.asarray(grades, dtype=np.float64))
        sizes.append(len(idx))
        queries.append(c["query"])
    return xs, ys, sizes, queries


def _fit(xs, ys, sizes, rounds: int | None = None, val=None):
    import lightgbm as lgb

    dtrain = lgb.Dataset(np.vstack(xs), np.concatenate(ys), group=sizes, feature_name=FEATURES)
    if val is None:
        return lgb.train(PARAMS, dtrain, num_boost_round=rounds or 100)
    vx, vy, vs = val
    dval = lgb.Dataset(np.vstack(vx), np.concatenate(vy), group=vs, reference=dtrain)
    return lgb.train(PARAMS, dtrain, num_boost_round=600, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def cross_validate(xs, ys, sizes) -> tuple[int, float]:
    """Grouped k-fold over training queries: picks the number of rounds and estimates in-distribution NDCG."""
    order = np.random.default_rng(SEED).permutation(len(xs))
    folds = np.array_split(order, FOLDS)
    best, scores = [], []
    for f in folds:
        hold = set(f.tolist())
        tr = [i for i in range(len(xs)) if i not in hold]
        model = _fit([xs[i] for i in tr], [ys[i] for i in tr], [sizes[i] for i in tr],
                     val=([xs[i] for i in f], [ys[i] for i in f], [sizes[i] for i in f]))
        best.append(model.best_iteration)
        for i in f:
            s = model.predict(xs[i], num_iteration=model.best_iteration)
            g = ys[i][np.argsort(-s, kind="stable")]
            ideal = dcg(sorted(ys[i], reverse=True)[:10])
            scores.append(dcg(g[:10]) / ideal)
    return max(1, int(np.median(best))), round(float(np.mean(scores)), 4)


def evaluate(model, golden_cands: list[dict], golden_labels: dict[str, dict[str, int]]) -> dict:
    """NDCG@10 on the golden set's re-ranked queries: retrieval order vs the installed model vs the new one."""
    current = load_ranker()
    rankers = {"retrieval order": None, f"current ({current.name})": current, "new (graded)": model}
    out: dict[str, dict] = {}
    per_query: dict[str, list] = {}
    for name, r in rankers.items():
        full, cond, seg = [], [], {}
        for c in golden_cands:
            lab = golden_labels.get(c["query"])
            if not lab:
                continue
            if r is None:
                ranked = c["ids"]
            else:
                scores = r.score(np.asarray(c["rows"], dtype=np.float64)) if hasattr(r, "score") \
                    else r.predict(np.asarray(c["rows"], dtype=np.float64))
                ranked = [c["ids"][i] for i in np.argsort(-scores, kind="stable")]
            v = ndcg(ranked, lab)
            if v is None:
                continue
            full.append(v)
            cond.append(ndcg(ranked, lab, condensed=True))
            seg.setdefault(c["segment"], []).append(v)
            per_query.setdefault(name, []).append(v)
        out[name] = {"ndcg@10": round(float(np.mean(full)), 4),
                     "condensed_ndcg@10": round(float(np.mean(cond)), 4), "queries": len(full),
                     "by_segment": {k: round(float(np.mean(v)), 3) for k, v in sorted(seg.items())}}
    base = np.asarray(per_query["retrieval order"])
    rng = np.random.default_rng(0)
    for name in list(rankers)[1:]:
        d = np.asarray(per_query[name]) - base
        boots = np.sort([rng.choice(d, len(d)).mean() for _ in range(5000)])
        out[name]["vs_retrieval"] = {"mean": round(float(d.mean()), 4), "ci95": [round(float(boots[125]), 4),
                                     round(float(boots[4875]), 4)], "better": int((d > 1e-3).sum()),
                                     "worse": int((d < -1e-3).sum())}
    return out


def train(install: bool) -> dict:
    cands = load_candidates("train")
    xs, ys, sizes, queries = groups(cands, load_labels(LABELS))
    rounds, cv = cross_validate(xs, ys, sizes)
    model = _fit(xs, ys, sizes, rounds=rounds)
    report = {"trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "labels": "graded (pipelines.ranking.graded)",
              "features": FEATURES, "train_queries": len(xs), "train_rows": int(sum(sizes)), "rounds": rounds,
              "cv_ndcg@10": cv,
              "golden": evaluate(model, load_candidates("golden"), load_labels())}
    gain = model.feature_importance("gain")
    report["importance_gain_share"] = {f: round(float(v / gain.sum()), 3) for f, v in
                                       sorted(zip(FEATURES, gain, strict=True), key=lambda kv: -kv[1])}
    path = DIR / "ranker-graded.txt"
    model.save_model(str(path))
    new = report["golden"]["new (graded)"]["vs_retrieval"]
    report["installed"] = install and new["mean"] > 0
    if install and not report["installed"]:
        print(f"[graded] NOT installed: the new model does not beat retrieval order on the golden set ({new})")
    if report["installed"]:
        if MODEL_PATH.exists():
            shutil.copy(MODEL_PATH, MODEL_PATH.with_name("ranker-synthetic.txt"))
            if META_PATH.exists():
                shutil.copy(META_PATH, META_PATH.with_name("ranker-synthetic.json"))
        shutil.copy(path, MODEL_PATH)
        META_PATH.write_text(json.dumps({k: v for k, v in report.items() if k != "golden"}
                                        | {"best_iteration": rounds, "golden_ndcg@10": report["golden"]}, indent=1))
        LightGBMRanker(MODEL_PATH)  # feature-name check before anything serves it
    (DIR / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("queries").add_argument("--n", type=int, default=400)
    sub.add_parser("capture").add_argument("set", choices=["train", "golden"])
    s = sub.add_parser("label-submit")
    s.add_argument("run_id")
    s.add_argument("--limit", type=int)
    s.add_argument("--model")
    sub.add_parser("label-collect").add_argument("run_id")
    sub.add_parser("train").add_argument("--install", action="store_true")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "queries":
        qs = sample_queries(args.n)
        TRAIN_QUERIES.write_text(yaml.safe_dump({"seed": SEED, "queries": qs}, sort_keys=False, allow_unicode=True))
        counts: dict[str, int] = {}
        for q in qs:
            counts[q["segment"]] = counts.get(q["segment"], 0) + 1
        print(f"[graded] {len(qs)} training queries -> {TRAIN_QUERIES} {counts}")
    elif args.cmd == "capture":
        qs = load_train_queries() if args.set == "train" else load_golden()
        out = asyncio.run(capture(qs))
        candidates_path(args.set).write_text("".join(json.dumps(c) + "\n" for c in out))
        print(f"[graded] {args.set}: {len(out)}/{len(qs)} queries re-ranked in production (rest: fallback/empty)")
    elif args.cmd == "label-submit":
        have = load_labels(LABELS)
        cands = load_candidates("train")
        if args.limit:
            cands = random.Random(SEED).sample(cands, min(args.limit, len(cands)))
        todo = {c["query"]: [p for p in c["ids"][:POOL_DEPTH] if p not in have.get(c["query"], {})] for c in cands}
        items = label_items({q: ids for q, ids in todo.items() if ids}, prefix="rk")
        model_for = {cid: args.model for cid in items} if args.model else None
        BatchJob(args.run_id, STAGE).submit(items, model_for=model_for)
        print(f"[graded] submitted {len(items)} queries, {sum(len(i['meta']['ids']) for i in items.values())} "
              f"products ({args.model or 'task default model'}, batch)")
    elif args.cmd == "label-collect":
        job = BatchJob(args.run_id, STAGE)
        if not job.done():
            print(f"[graded] not finished: {job.status()}")
            return
        n, rows, cost = collect_into(args.run_id, LABELS)
        print(f"[graded] {n} queries labeled, {rows} labels -> {LABELS}; cost ${cost:.4f}")
    else:
        r = train(args.install)
        print(f"[graded] {r['train_queries']} queries / {r['train_rows']} rows, {r['rounds']} rounds, "
              f"CV NDCG@10 {r['cv_ndcg@10']}")
        for name, v in r["golden"].items():
            print(f"    {name:22} NDCG@10 {v['ndcg@10']}  condensed {v['condensed_ndcg@10']}  "
                  f"{v.get('vs_retrieval', '')}")
        print(f"[graded] gain share: {r['importance_gain_share']}")
        if r["installed"]:
            print(f"[graded] installed -> {MODEL_PATH} (previous kept as ranker-synthetic.txt); restart search")


if __name__ == "__main__":
    main()
