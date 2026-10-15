"""'Pairs well with' (doc p.159-160, Fig 5.8), online half. Tier 0: one Redis GET plus one hydrate.

The offline discovery job stores, per primary category, the complement categories the LLM chose from co-purchase
data and the top in-stock products of each. At search time the primary category is the most common category
among the top results, so any query that lands in a category with co-purchase data gets complements, not only
queries the batch pipeline has seen. Products are re-hydrated with the request's stock filter, so a stock-out
never shows. Any failure returns no complements, never an error.
"""

import json
import logging
from collections import Counter

from prometheus_client import Counter as PromCounter
from redis.asyncio import Redis

from search_service.filters import SearchFilters
from search_service.retrieval import Retriever

log = logging.getLogger(__name__)

LOOKUPS = PromCounter("search_discovery_lookups_total", "Pairs-well-with lookups", ["result"])

PREFIX = "discovery_products:"
PRIMARY_FROM_TOP = 5
MAX_PAIRS = 6


def discovery_key(category_path: str) -> str:
    return f"{PREFIX}{category_path}"


def primary_category(products: list[dict]) -> str | None:
    """Most common category among the top results (ties: the higher-ranked one)."""
    cats = [p["category_path"] for p in products[:PRIMARY_FROM_TOP] if p.get("category_path")]
    if not cats:
        return None
    counts = Counter(cats)
    return max(cats, key=lambda c: (counts[c], -cats.index(c)))


def interleave(complements: list[dict], exclude: set[str], limit: int = MAX_PAIRS) -> list[str]:
    """One product per complement category in turn, so 'pairs well with' shows variety (mouse, sleeve, hub)
    before a second mouse."""
    lists = [[pid for pid in c["product_ids"] if pid not in exclude] for c in complements]
    out: list[str] = []
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x) and x[i] not in out)
    return out[:limit]


class Discovery:
    def __init__(self, redis: Redis | None, retriever: Retriever):
        self.redis = redis
        self.retriever = retriever

    async def pairs_well_with(self, top_products: list[dict], filters: SearchFilters) -> list[dict]:
        if self.redis is None:
            return []
        category = primary_category(top_products)
        if category is None:
            return []
        try:
            raw = await self.redis.get(discovery_key(category))
            LOOKUPS.labels("hit" if raw else "miss").inc()
            if not raw:
                return []
            shown = {p["product_id"] for p in top_products}
            ids = interleave(json.loads(raw)["complements"], shown)
            # Only the stock filter: the shopper's category/brand/price filters describe the primary product.
            products = await self.retriever.hydrate(ids, SearchFilters(
                include_out_of_stock=filters.include_out_of_stock))
            return [products[pid] for pid in ids if pid in products]
        except Exception:
            LOOKUPS.labels("error").inc()
            log.warning("pairs-well-with lookup failed", exc_info=True)
            return []
