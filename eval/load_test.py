"""Load test against the running search service (doc NFRs: P90 < 1 s, P99 < 5 s).

    uv run python -m eval.load_test [--rps 20] [--seconds 60] [--url http://localhost:8000]

Open arrival model: requests start on a fixed schedule whether or not earlier ones finished (a closed loop of N
users hides overload by slowing its own arrivals). Queries are drawn from the logs weighted by frequency, so the
cache-tier mix resembles production: head queries hit caches, the long tail reaches the fast path and fallback.
A plain asyncio + httpx client rather than Locust: it does what this test needs with no new dependency.
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from common.db import connect
from eval.metrics import percentile

OUT = Path(__file__).parent / "results" / "load_test.json"


def query_mix(n: int, seed: int = 9) -> list[str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT coalesce(corrected_query, normalized_query) AS q, frequency FROM query_stats")
        rows = cur.fetchall()
    rng = random.Random(seed)
    return rng.choices([r["q"] for r in rows], weights=[r["frequency"] for r in rows], k=n)


async def run(url: str, rps: float, seconds: float) -> dict:
    n = int(rps * seconds)
    queries = query_mix(n)
    results: list[tuple[float, str, int]] = []   # (latency s, served_by, status)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(base_url=url, timeout=15, limits=limits) as client:

        async def one(q: str) -> None:
            t0 = time.perf_counter()
            try:
                r = await client.get("/v1/search", params={"q": q, "debug": "true"})
                served = r.json().get("debug", {}).get("servedBy", "?") if r.status_code == 200 else "error"
                results.append((time.perf_counter() - t0, served, r.status_code))
            except httpx.HTTPError:
                results.append((time.perf_counter() - t0, "error", 0))

        start, tasks = time.perf_counter(), []
        for i, q in enumerate(queries):
            delay = start + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(q)))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    lat = [r[0] * 1000 for r in results]
    by_path: dict[str, list[float]] = defaultdict(list)
    for latency, served, _ in results:
        by_path[served].append(latency * 1000)
    mix = Counter(r[1] for r in results)
    report = {
        "target_rps": rps, "achieved_rps": round(len(results) / elapsed, 2), "requests": len(results),
        "errors": sum(1 for r in results if r[2] != 200),
        "latency_ms": {p: round(percentile(lat, int(p[1:])), 1) for p in ("p50", "p90", "p95", "p99")},
        "max_ms": round(max(lat), 1),
        "nfr": {"p90_under_1s": percentile(lat, 90) < 1000, "p99_under_5s": percentile(lat, 99) < 5000},
        "served_by_share": {k: round(v / len(results), 3) for k, v in mix.most_common()},
        "p90_ms_by_path": {k: round(percentile(v, 90), 1) for k, v in sorted(by_path.items())},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=20)
    ap.add_argument("--seconds", type=float, default=60)
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args.url, args.rps, args.seconds)), indent=1))
