from datetime import UTC, datetime

from pipelines.ingestion.enrich_llm import ATTRIBUTE_KEYS, OUTPUT_SCHEMA, PRODUCTS_PER_REQUEST, build_requests
from pipelines.ingestion.index import popularity, search_doc
from pipelines.ingestion.load_catalog import category_id, extract_attributes, taxonomy_path


def _product(**over):
    p = {
        "product_id": "B001", "name": "Aged Basmati Rice 5lb", "description": "Long grain, aged 2 years.",
        "brand": "Royal", "category_path": "Grocery & Gourmet Food > Pantry Staples > Rice > Basmati",
        "category_id": "abc", "currency": "USD", "attributes": {"form": "grain"}, "product_type": None,
        "search_tags": [],
        "price": 12.5, "in_stock": True, "rating": 4.6, "review_count": 1200, "image_url": None,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    p.update(over)
    return p


def test_taxonomy_path_truncates_to_four_levels():
    cats = ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Slippers", "Mules"]
    assert taxonomy_path(cats) == "Clothing, Shoes & Jewelry > Women > Shoes > Slippers"
    assert category_id("a > b") == category_id("a > b") != category_id("a > c")


def test_extract_attributes_keeps_only_descriptive_short_values():
    details = {"Color": "Navy", "UPC": "047900117230", "Material": "x" * 200, "Fabric Type": "Fleece",
               "Item Form": "Ground"}
    assert extract_attributes(details) == {"color": "Navy", "material": "Fleece", "form": "Ground"}


def test_search_doc_falls_back_to_leaf_category_and_flattens_tags():
    doc = search_doc(_product(search_tags=["biryani", "grain"]))
    assert doc["product_type"] == "Basmati"
    assert doc["tags"] == ["grain", "biryani"]  # de-duplicated, attribute values first
    assert doc["attributes"] == [{"key": "form", "value": "grain"}]
    assert 0 < doc["static_score"] <= 1


def test_search_doc_prefers_llm_product_type():
    assert search_doc(_product(product_type="basmati rice"))["product_type"] == "basmati rice"


def test_popularity_rewards_rating_and_volume():
    assert popularity(None, 100) == 0.0
    assert popularity(4.5, 5000) > popularity(4.5, 10) > 0
    assert popularity(5.0, 10**6) == 1.0


def test_build_requests_chunks_products_and_constrains_keys():
    products = [_product(product_id=f"P{i}") for i in range(PRODUCTS_PER_REQUEST * 2 + 3)]
    reqs = build_requests(products)
    assert len(reqs) == 3
    params = reqs[0]["params"]
    assert params["output_config"]["format"]["schema"] is OUTPUT_SCHEMA
    key_schema = OUTPUT_SCHEMA["properties"]["products"]["items"]["properties"]["attributes"]["items"]
    assert key_schema["properties"]["key"]["enum"] == ATTRIBUTE_KEYS
    assert len({r["custom_id"] for r in reqs}) == 3
    assert all(len(r["custom_id"]) <= 64 for r in reqs)
