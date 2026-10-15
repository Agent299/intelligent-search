"""OpenSearch access for the fast path: keyword (BM25) retrieval, k-NN retrieval, and hydration."""

from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch

from common.topics import PRODUCT_SEARCH_INDEX, SEMANTIC_SEARCH_INDEX
from search_service.filters import SearchFilters

# Field boosts: what a product *is* (name, type, tags) outweighs long descriptions.
KEYWORD_FIELDS = ["name^3", "product_type^4", "tags^2", "brand.text^2", "category_path.text", "description"]
SOURCE_FIELDS = ["product_id", "name", "description", "brand", "price", "currency", "rating", "review_count",
                 "imageUrl", "category_path", "product_type", "static_score", "in_stock"]


@dataclass
class Hit:
    product_id: str
    score: float


class Retriever:
    def __init__(self, client: AsyncOpenSearch):
        self.client = client

    async def keyword(self, query: str, filters: SearchFilters, size: int = 100) -> list[Hit]:
        body = {
            "size": size,
            "_source": False,
            "query": {"bool": {
                "must": [{"multi_match": {
                    "query": query,
                    "fields": KEYWORD_FIELDS,
                    "type": "most_fields",
                    # short queries: every term; longer natural-language queries: most terms
                    "minimum_should_match": "2<75%",
                }}],
                "filter": filters.product_index_clauses(),
            }},
        }
        resp = await self.client.search(index=PRODUCT_SEARCH_INDEX, body=body)
        return [Hit(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]

    async def semantic(self, vector: list[float], filters: SearchFilters, size: int = 100) -> list[Hit]:
        knn: dict = {"vector": vector, "k": size}
        clauses = filters.vector_index_clauses()
        if clauses:
            knn["filter"] = {"bool": {"filter": clauses}}  # efficient filtering inside the HNSW search
        body = {"size": size, "_source": False, "query": {"knn": {"embedding": knn}}}
        resp = await self.client.search(index=SEMANTIC_SEARCH_INDEX, body=body)
        # lucene cosinesimil scores are (1 + cos) / 2; convert back to cosine similarity
        return [Hit(h["_id"], 2 * h["_score"] - 1) for h in resp["hits"]["hits"]]

    async def hydrate(self, ids: list[str], filters: SearchFilters,
                      categories: list[str] | None = None) -> dict[str, dict]:
        """Fetch product fields for candidate ids, applying every filter (this is where category and
        attribute filters reach the vector hits). `categories` optionally restricts to taxonomy prefixes."""
        if not ids:
            return {}
        flt = [{"ids": {"values": ids}}, *filters.product_index_clauses()]
        if categories:
            flt.append(category_filter(categories))
        body = {"size": len(ids), "_source": SOURCE_FIELDS, "query": {"bool": {"filter": flt}}}
        resp = await self.client.search(index=PRODUCT_SEARCH_INDEX, body=body)
        return {h["_id"]: h["_source"] for h in resp["hits"]["hits"]}

    async def run_keyword_body(self, body: dict) -> list[Hit]:
        """Execute a compiled query body (rewrites now; cached queries in Phase 6)."""
        resp = await self.client.search(index=PRODUCT_SEARCH_INDEX, body=body)
        return [Hit(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]


def category_filter(categories: list[str]) -> dict:
    """Any-of taxonomy prefixes (rewrite categories are OR-ed, doc p.157)."""
    return {"bool": {"should": [{"prefix": {"category_path": c}} for c in categories], "minimum_should_match": 1}}


def structured_keyword_body(keywords: list[str], attributes: list[str], categories: list[str],
                            filters: SearchFilters, size: int) -> dict:
    """Compile an LLM rewrite into a BM25 query (Tier 0: code, not the LLM, writes the query).

    Each keyword must fully match (operator AND) and any one keyword is enough. Attributes and categories only
    *boost*: free-text attributes ('party size') rarely appear verbatim, and categories can only be chosen from
    the embedding-preselected candidates, which may miss part of a mixed intent (for 'dorm party' they were all
    party supplies, so a hard category filter dropped the chips and soda the keywords found). The doc's
    '(category OR ...) AND (attribute)' is therefore applied softly."""
    must = [{"bool": {"should": [
        {"multi_match": {"query": kw, "fields": KEYWORD_FIELDS, "type": "most_fields", "operator": "and"}}
        for kw in keywords], "minimum_should_match": 1}}]
    should = [{"multi_match": {"query": a, "fields": ["tags", "name", "description"], "boost": 0.5}}
              for a in attributes]
    should += [{"prefix": {"category_path": {"value": c, "boost": 2.0}}} for c in categories]
    return {"size": size, "_source": False,
            "query": {"bool": {"must": must, "should": should, "filter": filters.product_index_clauses()}}}

