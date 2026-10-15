"""Co-purchase analytics (doc p.159, Fig 5.8 'co-purchase analysis'). Tier 0.

A basket is everything one session purchased or added to cart. Pairs are counted between taxonomy categories, not
products or free-text product types: 1,751 sessions bought 2+ items across 3,385 product types, far too sparse to
pair products, while 1,171 leaf categories give usable counts. Full paths are the keys because leaf names repeat
('Active' under both Women and Men).

support = sessions containing both categories; lift = P(a, b) / (P(a) P(b)) over basket sessions. Lift >= 3 drops
pairs that are just two popular categories (dresses + tops), which co-occur by chance in any large basket set.
"""

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass

import polars as pl

from common.db import connect

MIN_SUPPORT = 2
MIN_LIFT = 3.0
MAX_CO_ITEMS = 15  # per primary category, highest lift first: the LLM prompt stays small


@dataclass(frozen=True)
class Pair:
    product_key: str  # primary category path
    co_item: str      # co-bought category path
    support: int
    lift: float


def baskets(events: pl.DataFrame, category_of: dict[str, str]) -> list[set[str]]:
    """One set of categories per session that purchased or carted at least one catalog product."""
    if events.is_empty():
        return []
    rows = (events.select("session_id", "purchased_product_ids", "added_to_cart_product_id")
            .with_columns(pl.concat_list(pl.col("purchased_product_ids"),
                                         pl.concat_list(pl.col("added_to_cart_product_id"))).alias("pids"))
            .select("session_id", "pids").explode("pids").drop_nulls())
    by_session: dict[str, set[str]] = defaultdict(set)
    for sid, pid in rows.iter_rows():
        if pid in category_of:
            by_session[sid].add(category_of[pid])
    return list(by_session.values())


def mine(sessions: list[set[str]], min_support: int = MIN_SUPPORT, min_lift: float = MIN_LIFT) -> list[Pair]:
    n = len(sessions)
    single = Counter(c for s in sessions for c in s)
    pairs = Counter(p for s in sessions for p in itertools.permutations(sorted(s), 2))
    out = []
    for (a, b), k in pairs.items():
        lift = k * n / (single[a] * single[b])
        if k >= min_support and lift >= min_lift:
            out.append(Pair(a, b, k, round(lift, 2)))
    by_key: dict[str, list[Pair]] = defaultdict(list)
    for p in out:
        by_key[p.product_key].append(p)
    return [p for ps in by_key.values()
            for p in sorted(ps, key=lambda p: (-p.lift, -p.support))[:MAX_CO_ITEMS]]


def load_categories() -> dict[str, str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT product_id, category_path FROM product_catalog")
        return {r["product_id"]: r["category_path"] for r in cur.fetchall()}


def store(pairs: list[Pair]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE co_purchase")  # a full recompute each run
        cur.executemany("INSERT INTO co_purchase (product_key, co_item, support, lift) VALUES (%s, %s, %s, %s)",
                        [(p.product_key, p.co_item, p.support, p.lift) for p in pairs])


def support_by_key() -> dict[str, int]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT product_key, sum(support) AS s FROM co_purchase GROUP BY product_key")
        return {r["product_key"]: r["s"] for r in cur.fetchall()}


def load() -> dict[str, list[str]]:
    """primary category -> co-bought categories, highest lift first."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT product_key, co_item FROM co_purchase ORDER BY product_key, lift DESC, support DESC")
        out: dict[str, list[str]] = defaultdict(list)
        for r in cur.fetchall():
            out[r["product_key"]].append(r["co_item"])
        return dict(out)
