"""Stream-sample product metadata from McAuley-Lab/Amazon-Reviews-2023.

The source files are 1-18 GB each, so we stream lines over HTTP and stop once each category has
scanned enough candidates, then take a seeded random sample. Only products with a price, a title
and a category path of depth >= 3 are kept (about half the rows have no price).

    uv run python -m pipelines.ingestion.download --scale 1.0
"""

import argparse
import json
import random
from pathlib import Path

import httpx

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"
QUOTAS = {
    "Grocery_and_Gourmet_Food": 15_000,
    "Clothing_Shoes_and_Jewelry": 15_000,
    "Electronics": 10_000,
    "Home_and_Kitchen": 10_000,
}
SCAN_FACTOR = 3  # candidates scanned per kept product, for sample diversity
OUT = Path("data/raw/products.jsonl")


def _usable(row: dict) -> bool:
    return (
        isinstance(row.get("price"), (int, float))
        and row["price"] > 0
        and bool(row.get("title"))
        and len(row.get("categories") or []) >= 3
        and bool(row.get("parent_asin"))
    )


def _slim(row: dict, source: str) -> dict:
    images = row.get("images") or []
    return {
        "product_id": row["parent_asin"],
        "source": source,
        "title": row["title"].strip(),
        "store": row.get("store"),
        "price": float(row["price"]),
        "average_rating": row.get("average_rating"),
        "rating_number": row.get("rating_number") or 0,
        "features": (row.get("features") or [])[:8],
        "description": (row.get("description") or [])[:3],
        "categories": row["categories"],
        "details": row.get("details") or {},
        "image_url": next((i.get("large") for i in images if i.get("large")), None),
    }


def sample_category(client: httpx.Client, name: str, quota: int, rng: random.Random) -> list[dict]:
    target = quota * SCAN_FACTOR
    seen: set[str] = set()
    candidates: list[dict] = []
    with client.stream("GET", f"{BASE}/meta_{name}.jsonl") as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            row = json.loads(line)
            if _usable(row) and row["parent_asin"] not in seen:
                seen.add(row["parent_asin"])
                candidates.append(_slim(row, name))
                if len(candidates) >= target:
                    break
    print(f"{name}: scanned {len(candidates)} usable candidates, keeping {min(quota, len(candidates))}")
    return rng.sample(candidates, min(quota, len(candidates)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all quotas, e.g. 0.2 for 10k products")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    products: list[dict] = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0)) as client:
        for name, quota in QUOTAS.items():
            products.extend(sample_category(client, name, int(quota * args.scale), rng))
    with OUT.open("w") as f:
        for p in products:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(products)} products to {OUT}")


if __name__ == "__main__":
    main()
