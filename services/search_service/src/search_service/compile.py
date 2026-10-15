"""Compile an offline expansion into an OpenSearch query (Tier 0, doc p.154-155).

The doc's shape is `must` = attributes the shopper insists on ('organic', 'aged'), `should` = the base product
type plus alternatives that exist in the catalog (basmati rice, sona masoori, poha, millet), with
minimum_should_match = 1. Here the product types are required but attributes only *boost* (README decision 26,
like decision 18 for rewrites): as hard requirements, 'elderly', 'travel' or 'cheese burst' left 38% of cached
queries with 1-4 products. The taxonomy path from segmentation also only boosts. User filters are not baked in:
they are added when the cached query is executed (Phase 6), so one cached query serves every filter combo.
"""

from typing import Any

from search_service.retrieval import KEYWORD_FIELDS

ATTRIBUTE_FIELDS = ["tags^2", "name", "description", "product_type"]
BASE_BOOST = 3.0  # the shopper's own product type outranks alternatives (pilot: a heatsink beat 'thermal paste')
ATTRIBUTE_BOOST = 2.0  # products that do say 'organic' rank first; the rest still show


def required_attributes(product_types: list[str], must_attributes: list[str]) -> list[str]:
    """Drop must-attributes already contained in the base product type: 'conference' in 'conference speakerphone'
    adds nothing but over-weights speakerphones whose text happens to repeat it."""
    base = set(product_types[0].lower().split()) if product_types else set()
    return [a for a in must_attributes if not set(a.lower().split()) <= base]


def compile_expansion(product_types: list[str], must_attributes: list[str],
                      taxonomy_path: str | None) -> dict[str, Any]:
    should = [{"multi_match": {"query": t, "fields": KEYWORD_FIELDS, "type": "most_fields", "operator": "and",
                               # base type first and strongest; alternatives decay with distance from intent
                               "boost": BASE_BOOST if i == 0 else 1.0 / (1 + 0.15 * i)}}
              for i, t in enumerate(product_types)]
    boost = [{"multi_match": {"query": a, "fields": ATTRIBUTE_FIELDS, "boost": ATTRIBUTE_BOOST}}
             for a in required_attributes(product_types, must_attributes)]
    if taxonomy_path:
        boost.append({"prefix": {"category_path": {"value": taxonomy_path, "boost": 1.5}}})
    return {"bool": {
        "must": [{"bool": {"should": should, "minimum_should_match": 1}}],
        "should": boost,
    }}


def with_filters(compiled: dict[str, Any], filter_clauses: list[dict]) -> dict[str, Any]:
    """Attach request-time filters to a cached compiled query."""
    return {"bool": {"must": [compiled], "filter": filter_clauses}}
