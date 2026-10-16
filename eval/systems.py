"""Run the golden queries through search configurations, in-process (doc p.181 offline evaluation).

    uv run python -m eval.systems [--systems baseline,full,full_no_rerank]

    baseline        the Phase 3 fast path only: hybrid BM25 + k-NN, RRF, confidence (no caches, LLM or re-rank)
    full            what production serves: exact caches -> fast path -> similar warm tier -> LLM fallback,
                    then the LightGBM re-rank (anonymous shopper: personal features are zero)
    full_no_rerank  `full` without the re-rank, to isolate what ranking adds

The results cache is disabled so latency measures the real path. Writes eval/results/<system>.jsonl:
{query, segment, served_by, confidence, latency_ms, ids (top 50)}. The full system's fallback calls the gateway
(Haiku, memoized per query): start it first (`make gateway`).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml
from opensearchpy import AsyncOpenSearch
from redis.asyncio import Redis

from ranking.model import load_ranker  # isort: skip  (LightGBM before torch, see ranking.model)
from common.config import get_settings
from common.embedding_service import EmbeddingService
from ranking.serving import Reranker
from search_service import hot_list
from search_service.cache import QueryCache
from search_service.engine import SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.fallback import FallbackPath
from search_service.gateway_client import GatewayClient
from search_service.retrieval import Retriever
from search_service.taxonomy import TaxonomyIndex
from search_service.vocab import build_spell_corrector

GOLDEN = Path(__file__).parent / "golden" / "queries.yaml"
RESULTS = Path(__file__).parent / "results"
TOP_K = 50
SYSTEMS = ("baseline", "full", "full_no_rerank")


def load_golden() -> list[dict]:
    return yaml.safe_load(GOLDEN.read_text())["queries"]


class NoResultsCache(QueryCache):
    async def get_results(self, key):
        return None

    async def put_results(self, key, results):
        return None


async def run(systems: list[str]) -> dict[str, Path]:
    s = get_settings()
    os_client, redis = AsyncOpenSearch(s.opensearch_url, timeout=30), Redis.from_url(s.redis_url)
    gateway = GatewayClient(s.genai_gateway_url)
    spell = await asyncio.to_thread(build_spell_corrector)
    embeddings, retriever = EmbeddingService(redis), Retriever(os_client)
    sink = EventSink(None, None)  # evaluation traffic must not reach the logs the pipelines learn from
    cache = NoResultsCache(redis, os_client, s.warm_tier_min_similarity, s.results_cache_ttl_s)
    await hot_list.load(cache)
    fallback = FallbackPath(retriever, embeddings, gateway, await TaxonomyIndex.build(embeddings), redis, os_client)
    reranker = Reranker(redis, load_ranker())
    engines = {
        "baseline": SearchEngine(retriever, embeddings, spell, sink, s),
        "full": SearchEngine(retriever, embeddings, spell, sink, s, fallback, cache, None, reranker),
        "full_no_rerank": SearchEngine(retriever, embeddings, spell, sink, s, fallback, cache, None, None),
    }
    golden = load_golden()
    RESULTS.mkdir(exist_ok=True)
    out = {}
    try:
        for name in systems:
            engine, rows = engines[name], []
            for g in golden:
                t0 = time.perf_counter()
                resp = await engine.search(SearchRequest(q=g["query"], page_size=TOP_K, debug=True))
                ms = round((time.perf_counter() - t0) * 1000, 1)
                rows.append({"query": g["query"], "segment": g["segment"], "served_by": resp.debug.served_by.value,
                             "confidence": resp.debug.confidence, "latency_ms": ms,
                             "corrected": resp.corrected_query, "ids": [r.product_id for r in resp.results]})
            path = RESULTS / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out[name] = path
            served = {}
            for r in rows:
                served[r["served_by"]] = served.get(r["served_by"], 0) + 1
            print(f"[eval] {name}: {len(rows)} queries, served by {served}", flush=True)
    finally:
        await gateway.aclose()
        await os_client.close()
        await redis.aclose()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    args = ap.parse_args()
    asyncio.run(run(args.systems.split(",")))
