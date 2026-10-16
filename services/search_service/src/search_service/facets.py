"""Filter facets for the storefront (Tier 0: counted in code over the products already hydrated, no extra query).

Counted over the top FACET_N results in final order, so they describe what the shopper is looking at rather than
the whole candidate pool. Categories adapt to the results: the options are the children of the deepest taxonomy
path every result shares, so picking one (a `category` prefix filter) drills down a level.
"""

from collections import Counter

from common.schemas import FacetValue, SearchFacets

FACET_N = 100
TOP_VALUES = 8
SEP = " > "


def _common_depth(paths: list[list[str]]) -> int:
    depth = 0
    while all(len(p) > depth for p in paths) and len({p[depth] for p in paths}) == 1:
        depth += 1
    return depth


def compute_facets(ids: list[str], products: dict[str, dict]) -> SearchFacets | None:
    shown = [products[pid] for pid in ids[:FACET_N] if pid in products]
    if not shown:
        return None
    brands = Counter(p["brand"] for p in shown if p.get("brand"))

    paths = [p["category_path"].split(SEP) for p in shown if p.get("category_path")]
    categories: Counter[str] = Counter()
    if paths:
        depth = _common_depth(paths)
        categories.update(SEP.join(p[:depth + 1]) for p in paths if len(p) > depth)

    prices = [p["price"] for p in shown if p.get("price") is not None]
    return SearchFacets(
        brands=[FacetValue(value=b, label=b, count=n) for b, n in brands.most_common(TOP_VALUES)],
        categories=[FacetValue(value=c, label=c.rsplit(SEP, 1)[-1], count=n)
                    for c, n in categories.most_common(TOP_VALUES)],
        price_min=min(prices) if prices else None,
        price_max=max(prices) if prices else None,
    )
