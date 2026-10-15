"""Ranking features (doc p.176-178), one definition shared by offline training and online serving so a model
never sees a feature computed two different ways. Tier 0: plain arithmetic over counts.

Long-term signals come from the logs (built point-in-time for training, from all history for serving); the
short-term signal is the brands the shopper clicked earlier in the same session.
"""

import math
from dataclasses import dataclass, field

FEATURES = [
    "retrieval_rank",        # 1 / (1 + position in the candidate list): the retriever's own opinion
    "semantic_sim",          # cosine(query, product), bge-small
    "static_score",          # offline global score: engagement + popularity
    "popularity",            # rating weighted by review volume
    "qp_engagement",         # smoothed engagement rate of this product for this query
    "brand_in_query",        # the shopper typed this product's brand
    "user_brand_aff",        # share of the user's engagement on this brand (long-term)
    "user_cat_aff",          # ... on this leaf category
    "user_top_aff",          # ... on this top-level department
    "session_brand_clicks",  # clicks on this brand earlier in the session (short-term)
    "price_band_gap",        # |log(price / user's median engaged price)|; NaN without history
    "user_has_history",      # 1 if the user has any engagement
]
PERSONAL = {"user_brand_aff", "user_cat_aff", "user_top_aff", "session_brand_clicks", "price_band_gap",
            "user_has_history"}

WEIGHTS = {"click": 1.0, "cart": 2.0, "purchase": 3.0}  # doc: purchase > cart > click

# Engagement rate smoothing: a product shown 2 times and clicked once is not a 50% product. A Beta-style prior
# of PRIOR_N impressions at the catalog-wide rate pulls sparse rates toward the mean.
PRIOR_N = 20.0
PRIOR_RATE = 0.05
RATE_CAP = 0.5           # engagement rates at or above this map to 1.0 in the static score
QP_MIN_IMPRESSIONS = 3   # (query, product) pairs below this, with no engagement, use the prior (serving parity)


def popularity(rating: float | None, review_count: int | None) -> float:
    """Query-independent prior in [0, 1]: rating weighted by review volume."""
    if not rating:
        return 0.0
    return round((rating / 5.0) * min(1.0, math.log1p(review_count or 0) / math.log1p(5000)), 4)


def engagement_rate(weighted: float, impressions: float) -> float:
    return (weighted + PRIOR_N * PRIOR_RATE) / (impressions + PRIOR_N)


def static_score(pop: float, weighted: float, impressions: float) -> float:
    """Doc p.177's offline global relevance: purchase > cart > click per impression, plus popularity."""
    return round(0.5 * pop + 0.5 * min(1.0, engagement_rate(weighted, impressions) / RATE_CAP), 4)


def qp_value(weighted: float, impressions: float) -> float:
    if weighted <= 0 and impressions < QP_MIN_IMPRESSIONS:
        return PRIOR_RATE
    return engagement_rate(weighted, impressions)


def top_level(category_path: str | None) -> str | None:
    return category_path.split(" > ")[0] if category_path else None


@dataclass
class UserProfile:
    """Engagement-weighted counts; shares are computed on read so the profile can be updated incrementally."""
    brands: dict[str, float] = field(default_factory=dict)
    categories: dict[str, float] = field(default_factory=dict)
    tops: dict[str, float] = field(default_factory=dict)
    prices: list[float] = field(default_factory=list)
    total: float = 0.0

    def add(self, product: dict, weight: float) -> None:
        self.total += weight
        if product.get("brand"):
            self.brands[product["brand"]] = self.brands.get(product["brand"], 0.0) + weight
        if product.get("category_path"):
            c = product["category_path"]
            self.categories[c] = self.categories.get(c, 0.0) + weight
            t = top_level(c)
            self.tops[t] = self.tops.get(t, 0.0) + weight
        if product.get("price"):
            self.prices.append(float(product["price"]))

    @property
    def median_price(self) -> float | None:
        if not self.prices:
            return None
        s = sorted(self.prices)
        return s[len(s) // 2]

    def to_json(self, top_n: int = 30) -> dict:
        """Serving copy: the top entries only (a long tail of 0.01 shares adds nothing)."""
        trim = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:top_n])  # noqa: E731
        return {"brands": trim(self.brands), "categories": trim(self.categories), "tops": self.tops,
                "median_price": self.median_price, "total": self.total}

    @classmethod
    def from_json(cls, d: dict) -> "UserProfile":
        p = cls(d.get("brands", {}), d.get("categories", {}), d.get("tops", {}), [], d.get("total", 0.0))
        if d.get("median_price"):
            p.prices = [d["median_price"]]
        return p


def brand_in_query(brand: str | None, query: str) -> float:
    if not brand:
        return 0.0
    tokens = set(query.lower().split())
    words = brand.lower().split()
    return 1.0 if words and all(w in tokens for w in words) else 0.0


def row(*, rank: int, sim: float, product: dict, query: str, user: UserProfile | None,
        session_brands: dict[str, float], qp: float) -> list[float]:
    brand, cat = product.get("brand"), product.get("category_path")
    total = user.total if user else 0.0
    share = (lambda d, k: d.get(k, 0.0) / total if k and total else 0.0)  # noqa: E731
    median = user.median_price if user else None
    price = product.get("price")
    gap = abs(math.log(price / median)) if median and price else math.nan
    return [
        1.0 / (1 + rank),
        sim,
        float(product.get("static_score") or 0.0),
        popularity(product.get("rating"), product.get("review_count")),
        qp,
        brand_in_query(brand, query),
        share(user.brands, brand) if user else 0.0,
        share(user.categories, cat) if user else 0.0,
        share(user.tops, top_level(cat)) if user else 0.0,
        float(session_brands.get(brand, 0.0)) if brand else 0.0,
        gap,
        1.0 if total > 0 else 0.0,
    ]
