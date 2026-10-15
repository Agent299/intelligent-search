"""Offline query processing pipeline driver (doc p.148-155).

    uv run python -m pipelines.offline.run aggregate              # 1. lake -> query_stats + log_processing_queue
    uv run python -m pipelines.offline.run enrich                 # 2. corrections, context, fast-path gate
    uv run python -m pipelines.offline.run segment RUN [--limit N] [--dry-run]   # 3. submit (Haiku batch)
    uv run python -m pipelines.offline.run segment-collect RUN    #    validate, escalate rejects, publish
    uv run python -m pipelines.offline.run expand RUN             # 4. retrieve alternatives, submit expansion
    uv run python -m pipelines.offline.run expand-collect RUN     #    validate, zero-result check, escalate
    uv run python -m pipelines.offline.run write RUN              # 5. smart_query_cache + warm tier
    uv run python -m pipelines.offline.run all [--limit N]        # everything, waiting on batches

Stages talk through the doc's Kafka topics; LLM stages run on the Message Batches API (50% price), so a run
takes minutes. Per-run batch state lives in data/pipeline/<RUN>/.
"""

import argparse
import asyncio
import time
from datetime import UTC, datetime

from pipelines.offline import aggregate, enrich, expand, lake, segment
from pipelines.offline.batches import BatchJob

POLL_S = 20


def cmd_aggregate() -> None:
    t = time.perf_counter()
    sunk = lake.sink_live_events()
    events = lake.read_events()
    aggs = aggregate.run(events)
    tiers = {t: sum(a.tier == t for a in aggs) for t in ("l1", "l2", "warm")}
    with_related = sum(bool(a.related_searches) for a in aggs)
    print(f"[aggregate] live events sunk: {sunk}; events: {events.height}; unique queries: {len(aggs)}; "
          f"tiers: {tiers}; with log-mined related searches: {with_related} ({time.perf_counter() - t:.0f}s)")


def cmd_enrich() -> None:
    t = time.perf_counter()
    frequent, rare = asyncio.run(enrich.run())
    print(f"[enrich] enriched + gated: {frequent}; skipped (seen once): {rare} ({time.perf_counter() - t:.0f}s)")


def cmd_segment(run_id: str, limit: int | None, dry_run: bool) -> None:
    total, picked = asyncio.run(segment.submit(run_id, limit, dry_run))
    verb = "would submit" if dry_run else "submitted"
    print(f"[segment] {total} enriched queries; {picked} below the gate ({segment.GATE}) {verb} to Haiku batch")


def _wait(run_id: str, stage: str) -> None:
    job = BatchJob(run_id, stage)
    while not job.done():
        print(f"  waiting on {stage} batches: {job.status()}", flush=True)
        time.sleep(POLL_S)


def _collect_loop(run_id: str, stage: str, collect_fn) -> None:
    """Collect; if escalations were submitted, wait for them and collect again, until nothing is pending."""
    while True:
        _wait(run_id, stage)
        res, esc_batch, exhausted = collect_fn(run_id)
        print(f"[{stage}-collect] ok={len(res.ok)} invalid={len(res.invalid)} api_errors={len(res.errored)} "
              f"escalated_batch={esc_batch} exhausted={len(exhausted)} cost=${res.cost_usd:.4f}")
        if esc_batch is None:
            return


def cmd_segment_collect(run_id: str, wait: bool) -> None:
    if wait:
        _collect_loop(run_id, "segment", segment.collect)
    else:
        res, esc, ex = segment.collect(run_id)
        print(f"[segment-collect] ok={len(res.ok)} invalid={len(res.invalid)} escalated={esc} exhausted={len(ex)}")
        if esc:
            return
    print(f"[segment] published {asyncio.run(segment.publish(run_id))} segmented queries")


def cmd_expand(run_id: str) -> None:
    print(f"[expand] submitted {asyncio.run(expand.submit(run_id))} expansions")


def cmd_expand_collect(run_id: str, wait: bool) -> None:
    if wait:
        _collect_loop(run_id, "expand", expand.collect)
    else:
        res, esc, ex = expand.collect(run_id)
        print(f"[expand-collect] ok={len(res.ok)} invalid={len(res.invalid)} escalated={esc} exhausted={len(ex)}")


def cmd_write(run_id: str) -> None:
    print(f"[write] cached {asyncio.run(expand.write_caches(run_id))} expanded queries")


def cost_report(run_id: str) -> None:
    seg, exp = BatchJob(run_id, "segment"), BatchJob(run_id, "expand")
    models: dict[str, int] = {}
    for job in (seg, exp):
        for it in job.items.values():
            m = it.get("final_model") or it.get("model")
            models[f"{it['task']}:{m}"] = models.get(f"{it['task']}:{m}", 0) + 1
    total = seg.state.get("cost_usd", 0.0) + exp.state.get("cost_usd", 0.0)
    print(f"[cost] segment ${seg.state.get('cost_usd', 0):.4f} + expand ${exp.state.get('cost_usd', 0):.4f} "
          f"= ${total:.4f}; per query ${total / max(1, len(seg.items)):.5f}; task:model counts {models}")


def cmd_all(limit: int | None) -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    print(f"run {run_id}")
    cmd_aggregate()
    cmd_enrich()
    cmd_segment(run_id, limit, dry_run=False)
    cmd_segment_collect(run_id, wait=True)
    cmd_expand(run_id)
    cmd_expand_collect(run_id, wait=True)
    cmd_write(run_id)
    cost_report(run_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("aggregate")
    sub.add_parser("enrich")
    s = sub.add_parser("segment")
    s.add_argument("run_id")
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    for name in ("segment-collect", "expand-collect"):
        c = sub.add_parser(name)
        c.add_argument("run_id")
        c.add_argument("--wait", action="store_true")
    for name in ("expand", "write", "cost"):
        sub.add_parser(name).add_argument("run_id")
    a = sub.add_parser("all")
    a.add_argument("--limit", type=int)
    args = ap.parse_args()
    {
        "aggregate": cmd_aggregate,
        "enrich": cmd_enrich,
        "segment": lambda: cmd_segment(args.run_id, args.limit, args.dry_run),
        "segment-collect": lambda: cmd_segment_collect(args.run_id, args.wait),
        "expand": lambda: cmd_expand(args.run_id),
        "expand-collect": lambda: cmd_expand_collect(args.run_id, args.wait),
        "write": lambda: cmd_write(args.run_id),
        "cost": lambda: cost_report(args.run_id),
        "all": lambda: cmd_all(args.limit),
    }[args.cmd]()


if __name__ == "__main__":
    main()
