"""Index the catalog for keyword search (product_search_index) and semantic search
(semantic_search_index) - steps 2 and 3 of the ingestion pipeline (doc Fig 5.1).

    uv run python -m pipelines.ingestion.index [--only-ids a,b,c] [--recreate]
"""

import argparse
import math
from collections.abc import Iterator

from opensearchpy.helpers import bulk

from common.db import connect
from common.embeddings import embed_documents
from common.indices import ensure_indices, get_opensearch, semantic_document
from common.topics import PRODUCT_SEARCH_INDEX, SEMANTIC_SEARCH_INDEX

CHUNK = 512


def popularity(rating: float | None, review_count: int) -> float:
    """Query-independent prior in [0, 1]: rating weighted by review volume.
    Replaced by the offline static ranking job (Phase 8) once behavior logs exist."""
    if not rating:
        return 0.0
    return round((rating / 5.0) * min(1.0, math.log1p(review_count) / math.log1p(5000)), 4)


def search_doc(p: dict) -> dict:
    attrs = p["attributes"] or {}
    leaf = p["category_path"].split(" > ")[-1] if p["category_path"] else None
    return {
        "product_id": p["product_id"],
        "name": p["name"],
        "description": p["description"],
        "brand": p["brand"],
        "category_path": p["category_path"],
        "product_type": p["product_type"] or leaf,
        "tags": list(dict.fromkeys([*map(str, attrs.values()), *p["search_tags"]])),
        "price": p["price"],
        "currency": p["currency"],
        "in_stock": p["in_stock"],
        "rating": p["rating"],
        "review_count": p["review_count"],
        "static_score": popularity(p["rating"], p["review_count"]),
        # lowercase values so exact-match filters ({"color": "navy"}) don't depend on source casing
        "attributes": [{"key": k, "value": str(v).lower()} for k, v in attrs.items()],
        "imageUrl": p["image_url"],
        "created_at": p["created_at"].isoformat(),
    }


def semantic_text(p: dict) -> str:
    attrs = dict(p["attributes"] or {})
    if p["search_tags"]:
        attrs["tags"] = ", ".join(p["search_tags"])
    return semantic_document(p["name"], p["brand"], p["category_path"], attrs, p["description"])


def fetch(only_ids: list[str] | None) -> Iterator[list[dict]]:
    with connect() as conn, conn.cursor(name="catalog") as cur:
        if only_ids:
            cur.execute("SELECT * FROM product_catalog WHERE product_id = ANY(%s)", (only_ids,))
        else:
            cur.execute("SELECT * FROM product_catalog ORDER BY product_id")
        while rows := cur.fetchmany(CHUNK):
            yield rows


def index_products(only_ids: list[str] | None = None) -> int:
    client = get_opensearch()
    total = 0
    for rows in fetch(only_ids):
        vectors = embed_documents([semantic_text(p) for p in rows])
        actions = []
        for p, vec in zip(rows, vectors, strict=True):
            actions.append({"_index": PRODUCT_SEARCH_INDEX, "_id": p["product_id"], "_source": search_doc(p)})
            actions.append({
                "_index": SEMANTIC_SEARCH_INDEX, "_id": p["product_id"],
                "_source": {
                    "product_id": p["product_id"], "embedding": vec.tolist(), "category_id": p["category_id"],
                    "brand": p["brand"], "price": p["price"], "in_stock": p["in_stock"],
                },
            })
        bulk(client, actions, request_timeout=120)
        total += len(rows)
        print(f"indexed {total}", flush=True)
    client.indices.refresh(index=f"{PRODUCT_SEARCH_INDEX},{SEMANTIC_SEARCH_INDEX}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-ids", help="comma-separated product ids to re-index")
    ap.add_argument("--recreate", action="store_true", help="drop and recreate the indices first")
    args = ap.parse_args()
    ensure_indices(recreate=args.recreate)
    n = index_products(args.only_ids.split(",") if args.only_ids else None)
    print(f"done: {n} products indexed")


if __name__ == "__main__":
    main()
