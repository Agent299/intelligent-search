"""LLM-as-judge (doc p.181, completing the case study's truncated implementation): pairwise baseline vs full.

    uv run python -m eval.judge submit RUN [--limit N] [--both-orders] [--model M] [--replay EARLIER_RUN]
    uv run python -m eval.judge collect RUN
    uv run python -m eval.judge agree REFERENCE_RUN RUN      # calibrate a cheaper judge against a stronger one

Opus (cost policy tier 3) sees the query and the two systems' top 10 (title + category) and scores each list 1-5
with its reasoning first. Position bias is controlled, not assumed away: each query's systems are randomly
assigned to A/B (seeded) and mapped back; `--both-orders` judges a query in both orders to measure how often the
verdict flips. The winner is recomputed from the scores by the gateway guardrail.
"""

import argparse
import hashlib
import json
import random

from eval.label import load_results, product_text, stratified
from eval.systems import RESULTS, load_golden
from pipelines.offline.batches import BatchJob

STAGE = "judge"
TOP = 10
OLD, NEW = "baseline", "full"


def _list(ids: list[str], text: dict[str, dict]) -> list[dict]:
    return [{"title": text[pid]["name"][:120],
             "category": " > ".join((text[pid].get("category_path") or "").split(" > ")[-2:]) or None}
            for pid in ids[:TOP] if pid in text]


def submit(run_id: str, limit: int | None, both_orders: bool, seed: int = 5, model: str | None = None,
           replay: str | None = None) -> int:
    """`replay` resends an earlier run's exact inputs (same lists, same A/B orders), e.g. to another model."""
    if replay:
        earlier = BatchJob(replay, STAGE).items
        items = {cid: {k: it[k] for k in ("task", "inputs", "meta")} for cid, it in earlier.items()}
        BatchJob(run_id, STAGE).submit(items, model_for={cid: model for cid in items} if model else None)
        return len(items)
    old, new = load_results(OLD), load_results(NEW)
    golden = [g for g in load_golden() if g["query"] in old and g["query"] in new]
    queries = stratified(golden, limit) if limit else [g["query"] for g in golden]
    text = product_text(sorted({pid for q in queries for pid in old[q]["ids"][:TOP] + new[q]["ids"][:TOP]}))
    rng = random.Random(seed)
    items = {}
    for q in queries:
        orders = [True, False] if both_orders else [rng.random() < 0.5]
        for new_is_a in orders:
            a, b = (new, old) if new_is_a else (old, new)
            cid = f"jdg-{hashlib.sha1(q.encode()).hexdigest()[:14]}-{'na' if new_is_a else 'nb'}"
            items[cid] = {"task": "judge",
                          "inputs": {"query": q, "results_a": _list(a[q]["ids"], text),
                                     "results_b": _list(b[q]["ids"], text)},
                          "meta": {"query": q, "new_is_a": new_is_a}}
    BatchJob(run_id, STAGE).submit(items, model_for={cid: model for cid in items} if model else None)
    return len(items)


def summarize(items: list[dict]) -> dict:
    """items: collected judge items. Scores mapped back to systems; flips measured where both orders exist."""
    rows = []
    for it in items:
        out, new_is_a = it["output"], it["meta"]["new_is_a"]
        s_new, s_old = (out["score_a"], out["score_b"]) if new_is_a else (out["score_b"], out["score_a"])
        rows.append({"query": it["meta"]["query"], "new_is_a": new_is_a, "new": s_new, "old": s_old,
                     "winner": "new" if s_new > s_old else "old" if s_old > s_new else "tie"})
    n = len(rows)
    by_query: dict[str, list[dict]] = {}
    for r in rows:
        by_query.setdefault(r["query"], []).append(r)
    pairs = [v for v in by_query.values() if len(v) == 2]
    flips = sum(v[0]["winner"] != v[1]["winner"] for v in pairs)
    a_wins = sum((r["winner"] == "new") == r["new_is_a"] for r in rows if r["winner"] != "tie")
    decided = sum(r["winner"] != "tie" for r in rows)
    return {
        "judgments": n, "queries": len(by_query),
        "new_wins": sum(r["winner"] == "new" for r in rows), "old_wins": sum(r["winner"] == "old" for r in rows),
        "ties": sum(r["winner"] == "tie" for r in rows),
        "mean_score": {NEW: round(sum(r["new"] for r in rows) / max(1, n), 3),
                       OLD: round(sum(r["old"] for r in rows) / max(1, n), 3)},
        "position_a_win_share": round(a_wins / decided, 3) if decided else None,
        "order_flips": f"{flips}/{len(pairs)}" if pairs else None,
        "rows": rows,
    }


def agreement(reference: str, candidate: str) -> dict:
    """How far a judge run agrees with a reference run on the same items (same custom ids)."""
    ref, cand = BatchJob(reference, STAGE).items, BatchJob(candidate, STAGE).items
    both = [cid for cid in ref if "output" in ref[cid] and "output" in cand.get(cid, {})]

    def verdict(it):  # +1 new wins, -1 old wins, 0 tie, from the system's point of view
        o, new_is_a = it["output"], it["meta"]["new_is_a"]
        d = (o["score_a"] - o["score_b"]) * (1 if new_is_a else -1)
        return (d > 0) - (d < 0), d

    v = {cid: (verdict(ref[cid]), verdict(cand[cid])) for cid in both}
    decided = [cid for cid, (r, c) in v.items() if r[0] != 0 and c[0] != 0]
    scores = [(ref[c]["output"][k], cand[c]["output"][k]) for c in both for k in ("score_a", "score_b")]
    ra, ca = zip(*scores, strict=True) if scores else ((), ())
    mean_r, mean_c = sum(ra) / len(ra), sum(ca) / len(ca)
    cov = sum((x - mean_r) * (y - mean_c) for x, y in scores)
    var = (sum((x - mean_r) ** 2 for x in ra) * sum((y - mean_c) ** 2 for y in ca)) ** 0.5
    # per query: sum both orders' margins (position bias cancels), then compare signs
    per_q: dict[str, list[float]] = {}
    for cid, (r, c) in v.items():
        q = ref[cid]["meta"]["query"]
        per_q.setdefault(q, [0.0, 0.0])
        per_q[q][0] += r[1]
        per_q[q][1] += c[1]
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    return {
        "items": len(both),
        "verdict_agreement": round(sum(r[0] == c[0] for r, c in v.values()) / max(1, len(v)), 3),
        "agreement_where_both_decided": round(sum(v[c][0][0] == v[c][1][0] for c in decided) / max(1, len(decided)), 3),
        "opposite_verdicts": sum(r[0] * c[0] == -1 for r, c in v.values()),
        "score_pearson": round(cov / var, 3) if var else None,
        "score_mean_abs_diff": round(sum(abs(x - y) for x, y in scores) / max(1, len(scores)), 3),
        "query_verdict_agreement": round(sum(sign(a) == sign(b) for a, b in per_q.values()) / max(1, len(per_q)), 3),
        "queries": len(per_q),
    }


def collect(run_id: str) -> dict:
    job = BatchJob(run_id, STAGE)
    res = job.collect()
    done = [it for it in job.items.values() if "output" in it]
    report = {**summarize(done), "cost_usd": round(job.state.get("cost_usd", 0.0), 4), "invalid": len(res.invalid)}
    (RESULTS / f"judge-{run_id}.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("run_id")
    s.add_argument("--limit", type=int)
    s.add_argument("--both-orders", action="store_true")
    s.add_argument("--model")
    s.add_argument("--replay")
    sub.add_parser("collect").add_argument("run_id")
    a = sub.add_parser("agree")
    a.add_argument("reference")
    a.add_argument("run_id")
    args = ap.parse_args()
    if args.cmd == "submit":
        n = submit(args.run_id, args.limit, args.both_orders, model=args.model, replay=args.replay)
        print(f"[judge] submitted {n} judgments ({args.model or 'task default model'}, batch)")
        return
    if args.cmd == "agree":
        print(f"[judge] {args.run_id} vs {args.reference}: {json.dumps(agreement(args.reference, args.run_id))}")
        return
    job = BatchJob(args.run_id, STAGE)
    if not job.done():
        print(f"[judge] not finished: {job.status()}")
        return
    r = collect(args.run_id)
    print(f"[judge] {r['judgments']} judgments on {r['queries']} queries: full wins {r['new_wins']}, baseline wins "
          f"{r['old_wins']}, ties {r['ties']}; mean score full {r['mean_score'][NEW]} vs baseline "
          f"{r['mean_score'][OLD]}; A-position win share {r['position_a_win_share']}; order flips "
          f"{r['order_flips']}; cost ${r['cost_usd']}")


if __name__ == "__main__":
    main()
