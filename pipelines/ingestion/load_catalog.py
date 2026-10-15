"""Load sampled products into product_catalog and build the taxonomy table.

Raw `details` are noisy (UPCs, dimensions, sales rank), so only a whitelist of descriptive keys
becomes product attributes. LLM enrichment later adds normalized attributes on top.

    uv run python -m pipelines.ingestion.load_catalog
"""

import hashlib
import json
import random
from pathlib import Path

from psycopg.types.json import Jsonb

from common.db import connect

SRC = Path("data/raw/products.jsonl")
TAXONOMY_DEPTH = 4

# details key -> normalized attribute key
DETAIL_KEYS = {
    "Color": "color",
    "Material": "material",
    "Fabric Type": "material",
    "Flavor": "flavor",
    "Item Form": "form",
    "Style": "style",
    "Department": "department",
    "Shape": "shape",
    "Special Feature": "feature",
    "Roast Level": "roast_level",
    "Caffeine Content": "caffeine",
    "Diet Type": "diet",
    "Specialty": "specialty",
    "Sole Material": "sole_material",
    "Closure Type": "closure",
    "Heel Type": "heel_type",
    "Age Range (Description)": "age_range",
    "Compatible Devices": "compatible_devices",
    "Connectivity Technology": "connectivity",
}


def category_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def taxonomy_path(categories: list[str]) -> str:
    return " > ".join(c.strip() for c in categories[:TAXONOMY_DEPTH])


def extract_attributes(details: dict) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for raw_key, key in DETAIL_KEYS.items():
        value = details.get(raw_key)
        if isinstance(value, str) and 0 < len(value) <= 80 and key not in attrs:
            attrs[key] = value.strip()
    return attrs


def describe(p: dict) -> str:
    parts = [" ".join(p["description"]).strip(), " ".join(f"{f}." for f in p["features"])]
    return " ".join(x for x in parts if x)[:2000]


def main() -> None:
    rng = random.Random(11)
    products = [json.loads(line) for line in SRC.open()]

    # Taxonomy: every prefix of every path, so parents exist before children.
    nodes: dict[str, tuple[str, str | None, int]] = {}
    for p in products:
        cats = [c.strip() for c in p["categories"][:TAXONOMY_DEPTH]]
        for level in range(1, len(cats) + 1):
            path = " > ".join(cats[:level])
            parent = " > ".join(cats[: level - 1]) if level > 1 else None
            nodes[path] = (category_id(path), category_id(parent) if parent else None, level)

    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO taxonomy (category_id, path, parent_id, level) VALUES (%s, %s, %s, %s)
               ON CONFLICT (category_id) DO NOTHING""",
            [(cid, path, parent, level) for path, (cid, parent, level) in
             sorted(nodes.items(), key=lambda kv: kv[1][2])],
        )
        rows = []
        for p in products:
            path = taxonomy_path(p["categories"])
            brand = (p["details"].get("Brand") or p["store"] or "").strip() or None
            rows.append((
                p["product_id"], p["title"][:500], describe(p), brand,
                Jsonb(extract_attributes(p["details"])), p["price"], "USD", p["image_url"],
                category_id(path), path,
                rng.random() > 0.05,  # synthetic stock: ~5% out of stock
                p["average_rating"], p["rating_number"],
            ))
        cur.executemany(
            """INSERT INTO product_catalog (product_id, name, description, brand, attributes, price, currency,
                   image_url, category_id, category_path, in_stock, rating, review_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (product_id) DO UPDATE SET
                   name = EXCLUDED.name, description = EXCLUDED.description, brand = EXCLUDED.brand,
                   price = EXCLUDED.price, image_url = EXCLUDED.image_url,
                   category_id = EXCLUDED.category_id, category_path = EXCLUDED.category_path,
                   rating = EXCLUDED.rating, review_count = EXCLUDED.review_count, updated_at = now()""",
            rows,
        )
    print(f"loaded {len(rows)} products, {len(nodes)} taxonomy nodes")


if __name__ == "__main__":
    main()
