"""Search orchestration. Phase 3 implements the fast path (doc p.157):

    normalize -> spell-correct -> strip negations -> [BM25 || embed -> k-NN] -> hydrate with filters
    -> drop negated products -> RRF fusion + prior
    -> confidence -> sort/paginate -> analytic event

The fast path is Tier 0 (code, local embeddings). Queries it can't answer confidently (score < 0.6) or that
return nothing go to the fallback path (Phase 4, `fallback.py`), which asks the LLM to rewrite the query and
retries; its results are shown only if they beat the fast path's.
"""

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field

from prometheus_client import Counter, Histogram

from common.config import Settings
from common.embedding_service import EmbeddingService
from common.normalize import SpellCorrector, normalize_query
from common.schemas import (
    Pagination,
    Price,
    ProductResult,
    SearchDebug,
    SearchResponse,
    SearchType,
    ServedBy,
    SortBy,
)
from search_service.events import EventSink
from search_service.fallback import FallbackPath
from search_service.filters import SearchFilters
from search_service.fusion import SEM_FLOOR, Candidate, confidence, rrf_fuse
from search_service.negation import ParsedQuery, parse_negations
from search_service.retrieval import Hit, Retriever

CANDIDATES_PER_RETRIEVER = 100
# Below this fast-path confidence, with no usable rewrite, results are noise: show "no results" instead.
NO_RESULTS_BELOW = 0.15
# For price sorting, only results that are plausibly relevant are sorted (else a cheap unrelated item wins).
PRICE_SORT_MIN_SIM = SEM_FLOOR + 0.07

LATENCY = Histogram("search_latency_seconds", "End-to-end search latency", ["served_by"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5))
CONFIDENCE = Histogram("search_confidence", "Fast-path confidence", buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                                                            0.8, 0.9, 1.0))
OUTCOMES = Counter("search_outcomes_total", "Search outcomes", ["outcome"])  # ok | low_confidence | zero_results


class EmptyQuery(ValueError):
    pass


@dataclass
class SearchRequest:
    q: str
    user_id: str | None = None
    session_id: str | None = None
    page: int = 1
    page_size: int = 20
    filters: SearchFilters = field(default_factory=SearchFilters)
    sort_by: SortBy = SortBy.relevance
    device_type: str | None = None
    debug: bool = False


@dataclass
class FastPathResult:
    ranked: list[Candidate]
    products: dict[str, dict]
    keyword: list[Hit]
    semantic: list[Hit]
    vector: list[float]


class SearchEngine:
    def __init__(self, retriever: Retriever, embeddings: EmbeddingService, spell: SpellCorrector,
                 sink: EventSink, settings: Settings, fallback: FallbackPath | None = None):
        self.fallback = fallback
        self.retriever = retriever
        self.embeddings = embeddings
        self.spell = spell
        self.sink = sink
        self.settings = settings
        self._background: set[asyncio.Task] = set()

    async def search(self, req: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        normalized = normalize_query(req.q)
        if not normalized:
            raise EmptyQuery("query is empty after normalization")
        corrected = self.spell.correct(normalized)
        parsed = parse_negations(corrected)

        fast = await self.fast_path(parsed, req.filters)
        conf = confidence(fast.keyword, fast.semantic, fast.ranked)
        CONFIDENCE.observe(conf.score)
        outcome = ("zero_results" if not fast.ranked
                   else "low_confidence" if conf.score < self.settings.confidence_threshold else "ok")
        OUTCOMES.labels(outcome).inc()

        ranked, products, final_conf = fast.ranked, fast.products, conf.score
        served_by, search_type, rewritten = ServedBy.fast_path, SearchType.hybrid, None
        if outcome != "ok" and self.fallback is not None:
            fb = await self.fallback.run(corrected, fast.vector, req.filters)
            if fb is not None:
                fb_ranked = [c for c in fb.ranked if not parsed.excludes(_label(fb.products[c.product_id]))]
                if fb_ranked and (not ranked or fb.confidence >= conf.score):
                    ranked, products, final_conf = fb_ranked, fb.products, fb.confidence
                    served_by, search_type = ServedBy.fallback, SearchType.semantic_fallback
                    rewritten = fb.rewrite.as_text()
            if served_by is ServedBy.fast_path and conf.score < NO_RESULTS_BELOW:
                ranked = []  # nothing credible: an honest empty page beats noise
                OUTCOMES.labels("no_results").inc()

        ordered = self._sort(ranked, products, req.sort_by)
        total = len(ordered)
        start = (req.page - 1) * req.page_size
        page_ids = [c.product_id for c in ordered[start:start + req.page_size]]
        search_id = uuid.uuid4().hex

        self._in_background(self.sink.search(
            search_id=search_id, user_id=req.user_id, session_id=req.session_id, raw_query=req.q,
            filters=req.filters.as_log_map(), search_type=search_type, displayed=page_ids,
            device_type=req.device_type))

        elapsed = time.perf_counter() - started
        LATENCY.labels(served_by.value).observe(elapsed)
        return SearchResponse(
            search_id=search_id,
            corrected_query=corrected if corrected != normalized else None,
            results=[self._to_result(products[pid]) for pid in page_ids],
            related_searches=[],  # mined from session logs in Phase 5
            pagination=Pagination(current_page=req.page, total_pages=max(1, math.ceil(total / req.page_size)),
                                  total_results=total),
            debug=SearchDebug(
                served_by=served_by, confidence=final_conf, latency_ms=round(elapsed * 1000, 1),
                fast_path_confidence=conf.score if served_by is ServedBy.fallback else None,
                rewritten_query=rewritten) if req.debug else None,
        )

    async def fast_path(self, parsed: ParsedQuery, filters: SearchFilters) -> FastPathResult:
        query = parsed.positive

        vector_box: list[list[float]] = []

        async def semantic() -> list[Hit]:
            [vector] = await self.embeddings.embed([query], "query")
            vector_box.append(vector)
            return await self.retriever.semantic(vector, filters, CANDIDATES_PER_RETRIEVER)

        # Keyword search starts while the query embedding is computed.
        keyword, semantic_hits = await asyncio.gather(
            self.retriever.keyword(query, filters, CANDIDATES_PER_RETRIEVER), semantic())
        ids = list(dict.fromkeys([h.product_id for h in keyword] + [h.product_id for h in semantic_hits]))
        products = await self.retriever.hydrate(ids, filters)
        if parsed.negated:
            products = {pid: p for pid, p in products.items() if not parsed.excludes(_label(p))}
        # Vector hits that fail category/attribute filters (or mention a negated term) drop out here.
        semantic_hits = [h for h in semantic_hits if h.product_id in products]
        keyword = [h for h in keyword if h.product_id in products]
        priors = {pid: p.get("static_score") or 0.0 for pid, p in products.items()}
        ranked = rrf_fuse(keyword, semantic_hits, priors)
        return FastPathResult(ranked, products, keyword, semantic_hits, vector_box[0])

    @staticmethod
    def _sort(ranked: list[Candidate], products: dict[str, dict], sort_by: SortBy) -> list[Candidate]:
        if sort_by is SortBy.relevance:
            return ranked
        relevant = [c for c in ranked
                    if c.keyword_rank is not None or (c.semantic_sim or 0.0) >= PRICE_SORT_MIN_SIM]
        return sorted(relevant, key=lambda c: products[c.product_id]["price"],
                      reverse=sort_by is SortBy.price_desc)

    @staticmethod
    def _to_result(p: dict) -> ProductResult:
        description = p.get("description") or ""
        return ProductResult(
            product_id=p["product_id"], title=p["name"],
            description=description[:300] + ("..." if len(description) > 300 else ""),
            image_url=p.get("imageUrl"), price=Price(amount=p["price"], currency=p.get("currency") or "USD"),
            brand=p.get("brand"), rating=p.get("rating"),
        )

    def _in_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)


def _label(p: dict) -> str:
    """Text checked against negated terms: what the product is, not its description."""
    return f"{p.get('name', '')} {p.get('product_type') or ''}"
