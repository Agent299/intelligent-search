"""The `filters` query param (doc p.145): a URL-encoded JSON object, e.g. {"brand":"Decathlon","size":"S"}.

Known keys map to typed fields; any other key is treated as a product attribute and matched exactly on the
nested `attributes` field, so new attribute filters need no code change.
"""

import json
from typing import Any

from pydantic import BaseModel, Field


class FilterError(ValueError):
    pass


class SearchFilters(BaseModel):
    brands: list[str] = Field(default_factory=list)
    category: str | None = None          # taxonomy path prefix, e.g. "Grocery & Gourmet Food > Snacks & Sweets"
    price_min: float | None = None
    price_max: float | None = None
    include_out_of_stock: bool = False    # out-of-stock items are hidden by default
    attributes: dict[str, list[str]] = Field(default_factory=dict)

    def as_log_map(self) -> dict[str, str]:
        """Flat map for the analytic event's filters_applied column."""
        out: dict[str, str] = {}
        if self.brands:
            out["brand"] = ",".join(self.brands)
        if self.category:
            out["category"] = self.category
        if self.price_min is not None:
            out["price_min"] = str(self.price_min)
        if self.price_max is not None:
            out["price_max"] = str(self.price_max)
        out.update({k: ",".join(v) for k, v in self.attributes.items()})
        return out

    def product_index_clauses(self) -> list[dict]:
        """Filter clauses for product_search_index (all filters)."""
        clauses = self._common_clauses()
        if self.category:
            clauses.append({"prefix": {"category_path": self.category}})
        for key, values in self.attributes.items():
            clauses.append({"nested": {"path": "attributes", "query": {"bool": {"filter": [
                {"term": {"attributes.key": key}},
                {"terms": {"attributes.value": values}},
            ]}}}})
        return clauses

    def vector_index_clauses(self) -> list[dict]:
        """Filter clauses the semantic index can apply inside the k-NN search (brand, price, stock).
        Category and attribute filters are enforced when vector hits are hydrated from the product index."""
        return self._common_clauses()

    def _common_clauses(self) -> list[dict]:
        clauses: list[dict] = []
        if self.brands:
            clauses.append({"terms": {"brand": self.brands}})
        if self.price_min is not None or self.price_max is not None:
            rng: dict[str, float] = {}
            if self.price_min is not None:
                rng["gte"] = self.price_min
            if self.price_max is not None:
                rng["lte"] = self.price_max
            clauses.append({"range": {"price": rng}})
        if not self.include_out_of_stock:
            clauses.append({"term": {"in_stock": True}})
        return clauses


def _as_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(v) for v in values if v is not None and str(v) != ""]


def _as_float(key: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise FilterError(f"filter {key!r} must be a number") from e


def parse_filters(raw: str | None) -> SearchFilters:
    if not raw:
        return SearchFilters()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise FilterError("filters must be a JSON object") from e
    if not isinstance(data, dict):
        raise FilterError("filters must be a JSON object")

    f = SearchFilters()
    for key, value in data.items():
        k = key.strip().lower()
        if k == "brand":
            f.brands = _as_list(value)
        elif k == "category":
            f.category = str(value)
        elif k in ("price_min", "min_price"):
            f.price_min = _as_float(key, value)
        elif k in ("price_max", "max_price"):
            f.price_max = _as_float(key, value)
        elif k == "in_stock":
            f.include_out_of_stock = str(value).lower() in ("false", "0", "any")
        else:
            # attribute values are indexed lowercase
            f.attributes[k] = [v.lower() for v in _as_list(value)]
    return f
