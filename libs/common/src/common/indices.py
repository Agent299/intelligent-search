"""OpenSearch index definitions (doc p.163-165) and the semantic-document builder (doc p.164)."""

from typing import Any

from opensearchpy import OpenSearch

from common.config import get_settings
from common.topics import PRODUCT_SEARCH_INDEX, SEMANTIC_SEARCH_INDEX, WARM_QUERY_CACHE_INDEX


def _knn_vector(dim: int) -> dict[str, Any]:
    return {
        "type": "knn_vector",
        "dimension": dim,
        "method": {
            "name": "hnsw",
            "engine": "lucene",
            "space_type": "cosinesimil",
            "parameters": {"m": 16, "ef_construction": 128},
        },
    }


def product_search_mapping() -> dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "name": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "description": {"type": "text"},
                "brand": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "category_path": {"type": "keyword", "fields": {"text": {"type": "text"}}},
                "product_type": {"type": "text"},
                "tags": {"type": "text"},  # flattened attribute values, so 'organic' matches via BM25
                "price": {"type": "float"},
                "currency": {"type": "keyword"},
                "in_stock": {"type": "boolean"},
                "rating": {"type": "float"},
                "review_count": {"type": "integer"},
                "static_score": {"type": "float"},  # offline global relevance (doc p.177)
                "attributes": {
                    "type": "nested",
                    "properties": {"key": {"type": "keyword"}, "value": {"type": "keyword"}},
                },
                "imageUrl": {"type": "keyword", "index": False},
                "created_at": {"type": "date"},
            }
        },
    }


def semantic_search_mapping(dim: int) -> dict[str, Any]:
    return {
        "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "embedding": _knn_vector(dim),
                "category_id": {"type": "keyword"},
                "brand": {"type": "keyword"},
                "price": {"type": "float"},
                "in_stock": {"type": "boolean"},
            }
        },
    }


def warm_query_cache_mapping(dim: int) -> dict[str, Any]:
    return {
        "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "normalized_query": {"type": "keyword"},
                "embedding": _knn_vector(dim),
                "es_query": {"type": "object", "enabled": False},
                "related_searches": {"type": "keyword", "index": False},
                "source": {"type": "keyword"},
                "version": {"type": "long"},
            }
        },
    }


def semantic_document(name: str, brand: str | None, category_path: str | None,
                      attributes: dict[str, Any], description: str = "") -> str:
    """Combine catalog fields into one text for the embedding model, e.g.
    "Men's Terra Running Shoe by Olympus Athletics. Category: Footwear, Running Shoes. ..."
    """
    parts = [name + (f" by {brand}" if brand else "") + "."]
    if category_path:
        leaf = [c.strip() for c in category_path.replace("/", ">").split(">")][1:] or [category_path]
        parts.append(f"Category: {', '.join(leaf)}.")
    values = [str(v) for v in attributes.values() if v and not isinstance(v, (dict, list))]
    if values:
        parts.append("Attributes: " + ", ".join(values) + ".")
    if description:
        parts.append(description[:300])
    return " ".join(parts)


def get_opensearch() -> OpenSearch:
    return OpenSearch(get_settings().opensearch_url, timeout=10)


def ensure_indices(client: OpenSearch | None = None, recreate: bool = False) -> None:
    client = client or get_opensearch()
    dim = get_settings().embedding_dim
    for name, body in [
        (PRODUCT_SEARCH_INDEX, product_search_mapping()),
        (SEMANTIC_SEARCH_INDEX, semantic_search_mapping(dim)),
        (WARM_QUERY_CACHE_INDEX, warm_query_cache_mapping(dim)),
    ]:
        if recreate and client.indices.exists(index=name):
            client.indices.delete(index=name)
        if not client.indices.exists(index=name):
            client.indices.create(index=name, body=body)


if __name__ == "__main__":
    ensure_indices()
    print("indices ready")
