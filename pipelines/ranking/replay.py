"""Point-in-time replay of the search logs (Tier 0).

Walks every logged search in time order. For each one it emits a training group (features computed from state
*before* the search, labels from what happened in it), then folds the search's outcome into the state. Features
therefore never see their own label (a user's brand affinity can't include the purchase being predicted). The
final state is exactly what serving needs: product static scores, (query, product) engagement and user profiles,
built by the same code, so training and serving agree.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from common.db import connect
from ranking import features as F

LABELS = {"purchase": 3.0, "cart": 2.0, "click": 1.0}


@dataclass
class Group:
    query: str
    ts: object
    user_id: str | None
    product_ids: list[str]
    rows: np.ndarray        # semantic_sim column filled in later (needs embeddings)
    labels: np.ndarray


@dataclass
class State:
    product_weighted: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    product_impressions: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    qp_weighted: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    qp_impressions: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    users: dict[str, F.UserProfile] = field(default_factory=lambda: defaultdict(F.UserProfile))
    sessions: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))

    def static_score(self, pid: str, product: dict) -> float:
        return F.static_score(F.popularity(product.get("rating"), product.get("review_count")),
                              self.product_weighted[pid], self.product_impressions[pid])

    def qp(self, query: str, pid: str) -> float:
        return F.qp_value(self.qp_weighted[(query, pid)], self.qp_impressions[(query, pid)])


def load_catalog() -> dict[str, dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT product_id, brand, category_path, price, rating, review_count FROM product_catalog")
        return {r["product_id"]: r for r in cur.fetchall()}


def labels_for(e: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    if e.get("clicked_product_id"):
        out[e["clicked_product_id"]] = LABELS["click"]
    if e.get("added_to_cart_product_id"):
        out[e["added_to_cart_product_id"]] = max(out.get(e["added_to_cart_product_id"], 0), LABELS["cart"])
    for pid in e.get("purchased_product_ids") or []:
        out[pid] = LABELS["purchase"]
    return out


def replay(events: pl.DataFrame, catalog: dict[str, dict], state: State | None = None) -> Iterator[Group]:
    """Yields one group per logged search with at least one engaged product; updates `state` as it goes."""
    state = state or State()
    for e in events.sort("event_timestamp").iter_rows(named=True):
        query, uid, sid = e["normalized_query"] or "", e["user_id"], e["session_id"]
        shown = [pid for pid in (e["displayed_product_ids"] or []) if pid in catalog]
        outcome = labels_for(e)
        if shown and any(pid in outcome for pid in shown):
            user = state.users.get(uid) if uid else None
            session = state.sessions.get(sid, {}) if sid else {}
            rows = []
            for rank, pid in enumerate(shown):
                product = {**catalog[pid], "static_score": state.static_score(pid, catalog[pid])}
                rows.append(F.row(rank=rank, sim=np.nan, product=product, query=query, user=user,
                                  session_brands=session, qp=state.qp(query, pid)))
            arr = np.asarray(rows, dtype=np.float64)
            yield Group(query, e["event_timestamp"], uid, shown, arr,
                        np.asarray([outcome.get(pid, 0.0) for pid in shown]))
        # fold this search's outcome into the state, after its features were taken
        for pid in shown:
            state.product_impressions[pid] += 1
            state.qp_impressions[(query, pid)] += 1
        for pid, w in outcome.items():
            if pid not in catalog:
                continue
            state.product_weighted[pid] += w
            state.qp_weighted[(query, pid)] += w
            if uid:
                state.users[uid].add(catalog[pid], w)
        clicked = e.get("clicked_product_id")
        if sid and clicked in catalog and catalog[clicked].get("brand"):
            state.sessions[sid][catalog[clicked]["brand"]] += 1
