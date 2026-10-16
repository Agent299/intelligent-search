"""Offline metrics over the golden set (doc p.181): NDCG@10, Recall@50, zero-result rate, latency by path.

    uv run python -m eval.metrics            # -> eval/results/report.json + a table on stdout

Labels are graded 0-3 (eval/golden/labels.jsonl). Unlabeled products count as 0, the usual pooled-evaluation
convention: the pool is every system's top 20, so no system is judged on products only another system showed.
Recall@50 is relative to the relevant products (grade >= 2) found by any system, so it compares systems; it is
not the share of all relevant products in the catalog. Queries with no relevant product (gibberish) are left out
of NDCG/Recall and checked through the zero-result rate instead.
"""

import json
import math
from collections import defaultdict

from eval.label import load_labels, load_results
from eval.systems import RESULTS, SYSTEMS, load_golden

RELEVANT = 2


def dcg(grades: list[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at(ids: list[str], labels: dict[str, int], k: int = 10) -> float | None:
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels.get(pid, 0) for pid in ids[:k]]) / ideal


def recall_at(ids: list[str], labels: dict[str, int], k: int = 50) -> float | None:
    relevant = {pid for pid, g in labels.items() if g >= RELEVANT}
    if not relevant:
        return None
    return len(relevant & set(ids[:k])) / len(relevant)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)]


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate(system: str, labels: dict[str, dict[str, int]], golden: list[dict]) -> dict:
    run = load_results(system)
    rows = [(g, run[g["query"]]) for g in golden if g["query"] in run]
    by_seg: dict[str, list] = defaultdict(list)
    lat_by_path: dict[str, list] = defaultdict(list)
    for g, r in rows:
        lab = labels.get(g["query"], {})
        by_seg[g["segment"]].append((ndcg_at(r["ids"], lab), recall_at(r["ids"], lab), not r["ids"]))
        lat_by_path[r["served_by"]].append(r["latency_ms"])
    labeled = [(g, r) for g, r in rows if g["query"] in labels]
    all_lat = [r["latency_ms"] for _, r in rows]
    gib = [r for g, r in rows if g["segment"] == "gibberish"]
    return {
        "queries": len(rows), "labeled_queries": len(labeled),
        "ndcg@10": _mean([ndcg_at(r["ids"], labels[g["query"]]) for g, r in labeled]),
        "recall@50": _mean([recall_at(r["ids"], labels[g["query"]]) for g, r in labeled]),
        "zero_result_rate": round(sum(not r["ids"] for _, r in rows) / max(1, len(rows)), 4),
        "gibberish_returning_results": sum(bool(r["ids"]) for r in gib),
        "latency_ms": {"p50": percentile(all_lat, 50), "p90": percentile(all_lat, 90),
                       "p99": percentile(all_lat, 99)},
        "latency_p50_ms_by_path": {k: percentile(v, 50) for k, v in sorted(lat_by_path.items())},
        "served_by": {k: len(v) for k, v in sorted(lat_by_path.items())},
        "by_segment": {seg: {"n": len(v), "ndcg@10": _mean([x[0] for x in v]),
                             "recall@50": _mean([x[1] for x in v]),
                             "zero_results": sum(x[2] for x in v)} for seg, v in sorted(by_seg.items())},
    }


def main() -> None:
    labels, golden = load_labels(), load_golden()
    report = {s: evaluate(s, labels, golden) for s in SYSTEMS if (RESULTS / f"{s}.jsonl").exists()}
    (RESULTS / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'system':16} {'labeled':>8} {'NDCG@10':>8} {'R@50':>6} {'zero%':>6} "
          f"{'p50ms':>6} {'p90ms':>6} {'p99ms':>7}")
    for s, r in report.items():
        lat = r["latency_ms"]
        print(f"{s:16} {r['labeled_queries']:>8} {r['ndcg@10'] or 0:>8.3f} {r['recall@50'] or 0:>6.3f} "
              f"{r['zero_result_rate'] * 100:>5.1f}% {lat['p50']:>6.0f} {lat['p90']:>6.0f} {lat['p99']:>7.0f}")
    segs = sorted({seg for r in report.values() for seg in r["by_segment"]})
    print("\nNDCG@10 by segment:")
    print(f"{'segment':18}" + "".join(f"{s:>16}" for s in report))
    for seg in segs:
        cells = [report[s]["by_segment"].get(seg, {}).get("ndcg@10") for s in report]
        print(f"{seg:18}" + "".join(f"{(c if c is not None else float('nan')):>16.3f}" for c in cells))


if __name__ == "__main__":
    main()
