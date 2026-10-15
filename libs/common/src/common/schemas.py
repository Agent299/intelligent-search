"""Payload contracts from the case study. Field names follow the document so each stage
can be swapped for its production counterpart without changing producers/consumers."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---- Public search API (doc p.143-145) ----

class SortBy(StrEnum):
    relevance = "relevance"
    price_asc = "price_asc"
    price_desc = "price_desc"


class Price(CamelModel):
    amount: float
    currency: str = "INR"


class ProductResult(CamelModel):
    product_id: str
    title: str
    description: str
    image_url: str | None = None
    price: Price
    brand: str | None = None
    rating: float | None = None


class Pagination(CamelModel):
    current_page: int
    total_pages: int
    total_results: int


class ServedBy(StrEnum):
    results_cache = "results_cache"
    l1 = "l1"
    l2 = "l2"
    warm = "warm"
    fast_path = "fast_path"
    fallback = "fallback"


class SearchDebug(CamelModel):
    served_by: ServedBy
    confidence: float | None                    # None when a cached query served the request (not re-scored)
    latency_ms: float
    fast_path_confidence: float | None = None   # set when the fallback path or a similar-query hit ran
    rewritten_query: str | None = None          # the LLM rewrite's keywords, when used
    cached_query: str | None = None             # the cached query that served this one (Phase 6)
    cache_similarity: float | None = None       # cosine to that query on a warm-tier similar hit
    ranker: str | None = None                   # personalized re-rank applied: "lightgbm" | "linear" (Phase 8)


class SearchResponse(CamelModel):
    search_id: str
    corrected_query: str | None = None
    results: list[ProductResult]
    related_searches: list[str] = Field(default_factory=list)
    pairs_well_with: list[ProductResult] = Field(default_factory=list)
    pagination: Pagination
    debug: SearchDebug | None = None


# ---- Offline pipeline payloads (doc p.149-155) ----

class SearchType(StrEnum):
    hybrid = "hybrid"
    keyword_only = "keyword_only"
    cached = "cached"
    semantic_fallback = "semantic_fallback"


class EventType(StrEnum):
    search = "search"
    click = "click"
    add_to_cart = "add_to_cart"
    purchase = "purchase"


class AnalyticEvent(BaseModel):
    """Analytic logs store row (doc p.166-167). The live service emits one `search` row plus one row per
    interaction, linked by search_id; the log aggregation job folds them into one row per search (the shape
    the synthetic generator writes directly)."""
    event_id: str
    event_type: EventType = EventType.search
    search_id: str | None = None
    event_timestamp: datetime
    user_id: str | None
    session_id: str | None
    raw_query: str
    filters_applied: dict[str, str] = Field(default_factory=dict)
    search_type: SearchType
    displayed_product_ids: list[str] = Field(default_factory=list)
    clicked_product_id: str | None = None
    added_to_cart_product_id: str | None = None
    purchased_product_ids: list[str] = Field(default_factory=list)
    device_type: str | None = None
    geolocation: str | None = None


class UserProfileSnapshot(BaseModel):
    user_id: str
    past_purchases: list[str] = Field(default_factory=list)
    brand_preferences: list[str] = Field(default_factory=list)


class QueryContext(BaseModel):
    geolocation: str | None = None
    time_of_day: str | None = None
    device: str | None = None


class EnrichedQuery(BaseModel):
    """Payload on query_process_queue (doc p.150).

    The pipeline works per *unique query* (inputs are de-duplicated before any LLM call), so user_profile is
    left empty here: cached rewrites are shared by every user, and personalization happens online (Phase 8).
    Context is the query's dominant device / location / time of day across its searches."""
    original_query: str
    normalized_query: str
    corrected_query: str | None = None       # after spell correction; the canonical cache key
    aliases: list[str] = Field(default_factory=list)  # other normalized spellings folded into this query
    frequency: int = 1
    tier: str = "warm"                       # l1 | l2 | warm, from frequency rank
    fast_confidence: float | None = None     # offline fast-path gate (Tier 0)
    user_profile: UserProfileSnapshot | None = None
    context: QueryContext = Field(default_factory=QueryContext)

    @property
    def canonical(self) -> str:
        return self.corrected_query or self.normalized_query


class SegmentationResult(BaseModel):
    attributes: list[str] = Field(default_factory=list)
    product_category: str | None = None
    product_quality: str | None = None
    brand: str | None = None


class ClassificationResult(BaseModel):
    taxonomy_path: str | None = None


class SegmentedQuery(EnrichedQuery):
    """Payload on segmented_queries_queue (doc p.151-152)."""
    segmentation_results: SegmentationResult
    classification_results: ClassificationResult


class SemanticAlternative(BaseModel):
    product_id: str
    name: str


class CachedQuerySource(StrEnum):
    batch = "batch"
    nrt = "nrt"
    fallback = "fallback"


class CachedQuery(BaseModel):
    """Value in the smart_query_cache / warm tier: a compiled ES query, not a product list (doc p.155)."""
    normalized_query: str
    es_query: dict[str, Any]
    source: CachedQuerySource
    version: int  # epoch seconds; batch writes always win over nrt (doc p.170)
    related_searches: list[str] = Field(default_factory=list)
