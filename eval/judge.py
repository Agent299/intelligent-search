"""LLM-as-judge (doc p.181, completing the case study's truncated implementation): pairwise baseline vs full.

    uv run python -m eval.judge submit RUN [--limit N] [--both-orders]
    uv run python -m eval.judge collect RUN

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


def submit(run_id: str, limit: int | None, both_orders: bool, seed: int = 5) -> int:
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
    BatchJob(run_id, STAGE).submit(items)
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
    sub.add_parser("collect").add_argument("run_id")
    args = ap.parse_args()
    if args.cmd == "submit":
        print(f"[judge] submitted {submit(args.run_id, args.limit, args.both_orders)} judgments (Opus, batch)")
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
