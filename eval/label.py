"""Draft graded relevance labels for the golden set with Opus (cost policy tier 3: drafts for human review).

    uv run python -m eval.label submit RUN [--limit N]   # pool the systems' top 20 per query; Batch API
    uv run python -m eval.label collect RUN              # validate, append to eval/golden/labels.jsonl

Pooling (TREC-style): every system's top POOL_DEPTH results are labeled, so each system is judged on the same
products. Only (query, product) pairs without a label are sent, so re-running after a system change pays only for
what changed. Products are sent as p1..pN, not ASINs: shorter, and nothing for the model to mistype.
"""

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from opensearchpy import OpenSearch

from common.config import get_settings
from common.topics import PRODUCT_SEARCH_INDEX
from eval.systems import RESULTS, SYSTEMS, load_golden
from pipelines.offline.batches import BatchJob

LABELS = Path(__file__).parent / "golden" / "labels.jsonl"
POOL_DEPTH = 20
STAGE = "label"


def load_labels(path: Path = LABELS) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out.setdefault(r["query"], {})[r["product_id"]] = r["grade"]
    return out


def load_results(system: str) -> dict[str, dict]:
    path = RESULTS / f"{system}.jsonl"
    return {r["query"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def pools(systems: tuple[str, ...] = SYSTEMS, depth: int = POOL_DEPTH) -> dict[str, list[str]]:
    runs = [load_results(s) for s in systems]
    out = {}
    for g in load_golden():
        ids: list[str] = []
        for run in runs:
            for pid in run.get(g["query"], {}).get("ids", [])[:depth]:
                if pid not in ids:
                    ids.append(pid)
        out[g["query"]] = ids
    return out


def stratified(queries: list[dict], limit: int, seed: int = 3) -> list[str]:
    """A pilot that covers every segment rather than the first N rows."""
    rng = random.Random(seed)
    by_seg: dict[str, list[str]] = {}
    for q in queries:
        by_seg.setdefault(q["segment"], []).append(q["query"])
    picked: list[str] = []
    while len(picked) < limit and any(by_seg.values()):
        for seg in sorted(by_seg):
            if by_seg[seg] and len(picked) < limit:
                picked.append(by_seg[seg].pop(rng.randrange(len(by_seg[seg]))))
    return picked


def product_text(ids: list[str]) -> dict[str, dict]:
    client = OpenSearch(get_settings().opensearch_url, timeout=30)
    docs = client.mget(index=PRODUCT_SEARCH_INDEX, body={"ids": ids}, _source=["name", "category_path"])["docs"]
    return {d["_id"]: d["_source"] for d in docs if d.get("found")}


def label_items(todo: dict[str, list[str]], prefix: str = "lab") -> dict[str, dict]:
    """Batch items grading each query's products (query -> product ids). Products are sent as p1..pN."""
    text = product_text(sorted({pid for ids in todo.values() for pid in ids}))
    items = {}
    for q, ids in todo.items():
        ids = [pid for pid in ids if pid in text]
        if not ids:
            continue
        short = {f"p{i + 1}": pid for i, pid in enumerate(ids)}
        items[f"{prefix}-{hashlib.sha1(q.encode()).hexdigest()[:16]}"] = {
            "task": "label",
            "inputs": {"query": q, "products": [
                {"id": sid, "title": text[pid]["name"][:120],
                 "category": " > ".join((text[pid].get("category_path") or "").split(" > ")[-2:]) or None}
                for sid, pid in short.items()]},
            "meta": {"query": q, "ids": short},
        }
    return items


def collect_into(run_id: str, path: Path) -> tuple[int, int, float]:
    """Validate a finished label batch and append its grades to `path`. Returns (queries, labels, cost USD)."""
    job = BatchJob(run_id, STAGE)
    res = job.collect()
    for cid, reason in res.invalid.items():
        job.items[cid]["invalid_reason"] = reason
    job.save()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for cid in res.ok:
        item = job.items[cid]
        for sid, grade in item["output"]["grades"].items():
            rows.append({"query": item["meta"]["query"], "product_id": item["meta"]["ids"][sid], "grade": grade,
                         "model": item["final_model"], "labeled_at": now, "reviewed": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return len(res.ok), len(rows), job.state.get("cost_usd", 0.0)


def submit(run_id: str, limit: int | None) -> tuple[int, int]:
    """Returns (queries submitted, products to grade)."""
    have, pool = load_labels(), pools()
    queries = stratified(load_golden(), limit) if limit else [g["query"] for g in load_golden()]
    todo = {q: [pid for pid in pool[q] if pid not in have.get(q, {})] for q in queries}
    items = label_items({q: ids for q, ids in todo.items() if ids})
    BatchJob(run_id, STAGE).submit(items)
    return len(items), sum(len(it["meta"]["ids"]) for it in items.values())


def collect(run_id: str) -> tuple[int, int, float]:
    """Returns (queries labeled, labels written, cost USD so far)."""
    return collect_into(run_id, LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("run_id")
    s.add_argument("--limit", type=int)
    sub.add_parser("collect").add_argument("run_id")
    args = ap.parse_args()
    if args.cmd == "submit":
        n, products = submit(args.run_id, args.limit)
        print(f"[label] submitted {n} queries, {products} products to grade (Opus, batch)")
    else:
        job = BatchJob(args.run_id, STAGE)
        if not job.done():
            print(f"[label] not finished: {job.status()}")
            return
        n, rows, cost = collect(args.run_id)
        print(f"[label] {n} queries labeled, {rows} labels -> {LABELS}; cost ${cost:.4f}")


if __name__ == "__main__":
    main()
