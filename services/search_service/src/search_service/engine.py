"""Search orchestration (doc p.156-157, p.170-173):

    results cache (hot queries) -> exact cached query (L1 -> L2 -> warm) -> fast path
    -> [low confidence] similar warm-tier query -> LLM fallback

The fast path (Phase 3):

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
    CachedQuerySource,
    Pagination,
    Price,
    ProductResult,
    SearchDebug,
    SearchResponse,
    SearchType,
    ServedBy,
    SortBy,
)
from search_service.cache import CachedResults, CacheEntry, QueryCache
from search_service.compile import with_filters
from search_service.events import EventSink
from search_service.fallback import FallbackPath, Rewrite
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
class Served:
    """How a request was answered, before sorting and pagination."""
    served_by: ServedBy
    search_type: SearchType
    ranked: list[Candidate] = field(default_factory=list)
    products: dict[str, dict] = field(default_factory=dict)
    confidence: float | None = None
    fast_confidence: float | None = None
    rewritten: str | None = None
    related: list[str] = field(default_factory=list)
    cached_query: str | None = None
    cache_similarity: float | None = None


@dataclass
class FastPathResult:
    ranked: list[Candidate]
    products: dict[str, dict]
    keyword: list[Hit]
    semantic: list[Hit]
    vector: list[float]


class SearchEngine:
    def __init__(self, retriever: Retriever, embeddings: EmbeddingService, spell: SpellCorrector,
                 sink: EventSink, settings: Settings, fallback: FallbackPath | None = None,
                 cache: QueryCache | None = None):
        self.fallback = fallback
        self.cache = cache
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

        hot = self.cache is not None and self.cache.is_hot(corrected)
        results_key = (self.cache.results_key(corrected, req.filters.as_log_map(), req.sort_by.value)
                       if hot else None)
        cached = await self.cache.get_results(results_key) if results_key else None
        if cached is not None:
            ids = cached.ids
            start = (req.page - 1) * req.page_size
            page_ids = ids[start:start + req.page_size]
            products = await self.retriever.hydrate(page_ids, req.filters)  # current price; drops stock-outs
            page_ids = [pid for pid in page_ids if pid in products]
            served = Served(served_by=ServedBy.results_cache, search_type=SearchType.cached,
                            confidence=cached.confidence, related=cached.related_searches,
                            rewritten=cached.rewritten_query, cached_query=cached.cached_query)
        else:
            served = await self._serve(corrected, parsed, req.filters)
            ids = [c.product_id for c in self._sort(served.ranked, served.products, req.sort_by)]
            start = (req.page - 1) * req.page_size
            page_ids, products = ids[start:start + req.page_size], served.products
            if results_key:
                self._in_background(self.cache.put_results(results_key, CachedResults(
                    ids=ids, served_by=served.served_by.value,
                    confidence=served.confidence, related_searches=served.related,
                    rewritten_query=served.rewritten, cached_query=served.cached_query)))

        search_id = uuid.uuid4().hex
        self._in_background(self.sink.search(
            search_id=search_id, user_id=req.user_id, session_id=req.session_id, raw_query=req.q,
            filters=req.filters.as_log_map(), search_type=served.search_type, displayed=page_ids,
            device_type=req.device_type))

        elapsed = time.perf_counter() - started
        LATENCY.labels(served.served_by.value).observe(elapsed)
        return SearchResponse(
            search_id=search_id,
            corrected_query=corrected if corrected != normalized else None,
            results=[self._to_result(products[pid]) for pid in page_ids],
            related_searches=served.related,
            pagination=Pagination(current_page=req.page, total_pages=max(1, math.ceil(len(ids) / req.page_size)),
                                  total_results=len(ids)),
            debug=SearchDebug(
                served_by=served.served_by, confidence=served.confidence, latency_ms=round(elapsed * 1000, 1),
                fast_path_confidence=served.fast_confidence, rewritten_query=served.rewritten,
                cached_query=served.cached_query if served.cached_query != corrected else None,
                cache_similarity=served.cache_similarity) if req.debug else None,
        )

    async def _serve(self, corrected: str, parsed: ParsedQuery, filters: SearchFilters) -> "Served":
        """Doc lookup order, with one change (README decision 25): exact cache hits come before the fast path,
        but a *similar* warm-tier hit is only used once the fast path is unsure, in place of the LLM fallback.
        Batch-cached queries are exactly the low-confidence ones, so a confident query's nearest cached neighbour
        is a harder variant of it ('canister vacuum' -> 'small canister vacuum') and would make it worse."""
        if self.cache is not None:
            exact = await self.cache.exact(corrected)
            if exact is not None:
                entry, tier = exact
                ranked, products = await self._run_cached(entry, filters, parsed)
                if ranked:
                    return Served(served_by=tier, search_type=SearchType.cached, ranked=ranked, products=products,
                                  related=entry.related_searches, cached_query=entry.query)

        fast = await self.fast_path(parsed, filters)
        conf = confidence(fast.keyword, fast.semantic, fast.ranked)
        CONFIDENCE.observe(conf.score)
        outcome = ("zero_results" if not fast.ranked
                   else "low_confidence" if conf.score < self.settings.confidence_threshold else "ok")
        OUTCOMES.labels(outcome).inc()
        served = Served(served_by=ServedBy.fast_path, search_type=SearchType.hybrid, ranked=fast.ranked,
                        products=fast.products, confidence=conf.score)
        if outcome == "ok":
            return served

        # Similar warm-tier entry: free, so it goes before the LLM. Negated queries are excluded: 'snacks with
        # nuts' sits at 0.86 cosine from 'snacks without nuts'.
        if self.cache is not None and not parsed.negated:
            entry = await self.cache.similar(fast.vector)
            if entry is not None:
                ranked, products = await self._run_cached(entry, filters, parsed)
                if ranked:
                    return Served(served_by=ServedBy.warm, search_type=SearchType.cached, ranked=ranked,
                                  products=products, fast_confidence=conf.score, related=entry.related_searches,
                                  cached_query=entry.query, cache_similarity=entry.similarity)

        if self.fallback is not None:
            fb = await self.fallback.run(corrected, fast.vector, filters)
            if fb is not None:
                fb_ranked = [c for c in fb.ranked if not parsed.excludes(_label(fb.products[c.product_id]))]
                if fb_ranked and (not fast.ranked or fb.confidence >= conf.score):
                    return Served(served_by=ServedBy.fallback, search_type=SearchType.semantic_fallback,
                                  ranked=fb_ranked, products=fb.products, confidence=fb.confidence,
                                  fast_confidence=conf.score, rewritten=fb.rewrite.as_text())
            if conf.score < NO_RESULTS_BELOW:
                served.ranked = []  # nothing credible: an honest empty page beats noise
                OUTCOMES.labels("no_results").inc()
        return served

    async def _run_cached(self, entry: CacheEntry, filters: SearchFilters,
                          parsed: ParsedQuery) -> tuple[list[Candidate], dict[str, dict]]:
        """Execute a cached query with this request's filters (filters are never baked into the cache).
        Fallback write-backs replay their rewrite (per-keyword k-NN + interleave, as first served); batch
        expansions run their compiled bool query."""
        rewrite = entry.es_query.get("rewrite")
        if entry.source == CachedQuerySource.fallback and rewrite and self.fallback is not None:
            ranked, products, _ = await self.fallback.search_rewrite(Rewrite(**rewrite), filters)
        else:
            body = {"size": CANDIDATES_PER_RETRIEVER, "_source": False,
                    "query": with_filters(entry.es_query["query"], filters.product_index_clauses())}
            hits = await self.retriever.run_keyword_body(body)
            products = await self.retriever.hydrate([h.product_id for h in hits], filters)
            hits = [h for h in hits if h.product_id in products]
            ranked = rrf_fuse(hits, [], {pid: p.get("static_score") or 0.0 for pid, p in products.items()})
        if parsed.negated:
            ranked = [c for c in ranked if not parsed.excludes(_label(products[c.product_id]))]
        return ranked, products

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
