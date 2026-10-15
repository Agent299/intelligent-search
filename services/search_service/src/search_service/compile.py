"""Compile an offline expansion into an OpenSearch query (Tier 0, doc p.154-155).

The doc's shape: `must` = attributes the shopper insists on ('organic', 'aged'), `should` = the base product
type plus alternatives that exist in the catalog (basmati rice, sona masoori, poha, millet), with
minimum_should_match = 1. The taxonomy path from segmentation only boosts. User filters are not baked in:
they are added when the cached query is executed (Phase 6), so one cached query serves every filter combo.
"""

from typing import Any

from search_service.retrieval import KEYWORD_FIELDS

ATTRIBUTE_FIELDS = ["tags^2", "name", "description", "product_type"]
BASE_BOOST = 3.0  # the shopper's own product type outranks alternatives (pilot: a heatsink beat 'thermal paste')


def required_attributes(product_types: list[str], must_attributes: list[str]) -> list[str]:
    """Drop must-attributes already contained in the base product type: 'conference' in 'conference speakerphone'
    adds nothing but excludes speakerphones whose text never says 'conference' (pilot: 1 result)."""
    base = set(product_types[0].lower().split()) if product_types else set()
    return [a for a in must_attributes if not set(a.lower().split()) <= base]


def compile_expansion(product_types: list[str], must_attributes: list[str],
                      taxonomy_path: str | None) -> dict[str, Any]:
    should = [{"multi_match": {"query": t, "fields": KEYWORD_FIELDS, "type": "most_fields", "operator": "and",
                               # base type first and strongest; alternatives decay with distance from intent
                               "boost": BASE_BOOST if i == 0 else 1.0 / (1 + 0.15 * i)}}
              for i, t in enumerate(product_types)]
    must = [{"multi_match": {"query": a, "fields": ATTRIBUTE_FIELDS}}
            for a in required_attributes(product_types, must_attributes)]
    boost = [{"prefix": {"category_path": {"value": taxonomy_path, "boost": 1.5}}}] if taxonomy_path else []
    return {"bool": {
        "must": must + [{"bool": {"should": should, "minimum_should_match": 1}}],
        "should": boost,
    }}


def with_filters(compiled: dict[str, Any], filter_clauses: list[dict]) -> dict[str, Any]:
    """Attach request-time filters to a cached compiled query."""
    return {"bool": {"must": [compiled], "filter": filter_clauses}}
