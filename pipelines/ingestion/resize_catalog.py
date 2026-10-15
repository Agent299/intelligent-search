"""Shrink the loaded catalog to a stratified subset and rebuild the taxonomy to match.

Keeps the download's category mix (30/30/20/20) and prefers already-enriched products so paid LLM
work is not discarded. Removed products stay in data/raw/products.jsonl; re-run load_catalog to
restore them.

    uv run python -m pipelines.ingestion.resize_catalog --total 10000
"""

import argparse

from common.db import connect
from pipelines.ingestion.download import QUOTAS
from pipelines.ingestion.load_catalog import TAXONOMY_DEPTH, category_id

# download source -> top-level category name in the catalog
TOP_LEVEL = {
    "Grocery_and_Gourmet_Food": "Grocery & Gourmet Food",
    "Clothing_Shoes_and_Jewelry": "Clothing, Shoes & Jewelry",
    "Electronics": "Electronics",
    "Home_and_Kitchen": "Home & Kitchen",
}


def resize(total: int, seed: float = 0.42) -> dict[str, int]:
    scale = total / sum(QUOTAS.values())
    quotas = {TOP_LEVEL[src]: round(q * scale) for src, q in QUOTAS.items()}
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (seed,))
        cur.execute("CREATE TEMP TABLE quota (top TEXT PRIMARY KEY, n INT)")
        cur.executemany("INSERT INTO quota VALUES (%s, %s)", list(quotas.items()))
        cur.execute("""
            CREATE TEMP TABLE keep AS
            SELECT product_id FROM (
                SELECT p.product_id, q.n,
                       row_number() OVER (PARTITION BY split_part(p.category_path, ' > ', 1)
                                          ORDER BY (p.llm_enriched_at IS NULL), random()) AS rn
                FROM product_catalog p
                JOIN quota q ON q.top = split_part(p.category_path, ' > ', 1)
            ) ranked WHERE rn <= n""")
        cur.execute("DELETE FROM product_catalog WHERE product_id NOT IN (SELECT product_id FROM keep)")
        removed = cur.rowcount

        # Rebuild taxonomy: every prefix of every remaining path.
        cur.execute("SELECT DISTINCT category_path FROM product_catalog")
        nodes: dict[str, tuple[str, str | None, int]] = {}
        for (path,) in (tuple(r.values()) for r in cur.fetchall()):
            cats = path.split(" > ")[:TAXONOMY_DEPTH]
            for level in range(1, len(cats) + 1):
                p = " > ".join(cats[:level])
                parent = " > ".join(cats[: level - 1]) if level > 1 else None
                nodes[p] = (category_id(p), category_id(parent) if parent else None, level)
        cur.execute("DELETE FROM taxonomy")
        cur.executemany(
            "INSERT INTO taxonomy (category_id, path, parent_id, level) VALUES (%s, %s, %s, %s)",
            [(cid, p, parent, level) for p, (cid, parent, level) in sorted(nodes.items(), key=lambda kv: kv[1][2])],
        )
        cur.execute("""SELECT split_part(category_path, ' > ', 1) AS top, count(*) AS n,
                              count(llm_enriched_at) AS enriched
                       FROM product_catalog GROUP BY 1 ORDER BY 1""")
        summary = {r["top"]: (r["n"], r["enriched"]) for r in cur.fetchall()}
    print(f"removed {removed} products; taxonomy now {len(nodes)} nodes")
    for top, (n, enriched) in summary.items():
        print(f"  {top:<28} {n:>6} products ({enriched} enriched)")
    return {top: n for top, (n, _) in summary.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True)
    resize(ap.parse_args().total)


if __name__ == "__main__":
    main()
