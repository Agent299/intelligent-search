from common.indices import semantic_document
from common.normalize import SpellCorrector, normalize_query, query_hash
from common.schemas import Pagination, Price, ProductResult, SearchResponse


def test_normalize_query_strips_case_and_punctuation():
    assert normalize_query("  Organic AGED  Basmati-Rice!!! ") == "organic aged basmati-rice"
    assert normalize_query("men's   shoes?") == "men's shoes"


def test_query_hash_is_stable_and_normalization_sensitive():
    assert query_hash("iphone") == query_hash("iphone")
    assert query_hash("iphone") != query_hash("iphone 15")


def test_spell_corrector_uses_catalog_vocabulary():
    sc = SpellCorrector()
    sc.add_words({"basmati": 500, "organic": 800, "rice": 900})
    assert sc.correct("organik basmatti rice") == "organic basmati rice"
    # short tokens and unknown-but-distant words are untouched
    assert sc.correct("tv xyzzyq") == "tv xyzzyq"


def test_semantic_document_matches_doc_format():
    doc = semantic_document(
        "Men's Terra Running Shoe", "Olympus Athletics", "Apparel/Footwear/Running Shoes",
        {"use": "trail running", "feature": "waterproof"},
    )
    assert doc.startswith("Men's Terra Running Shoe by Olympus Athletics. Category: Footwear, Running Shoes.")
    assert "waterproof" in doc


def test_search_response_serializes_to_camel_case_like_doc():
    resp = SearchResponse(
        search_id="s1",
        corrected_query="hill station outfits",
        results=[ProductResult(product_id="DEC-FLC-001S", title="Fleece", description="warm",
                               price=Price(amount=1499.0), brand="Decathlon", rating=4.7)],
        pagination=Pagination(current_page=1, total_pages=1, total_results=1),
    )
    body = resp.model_dump(by_alias=True, exclude_none=True)
    assert body["correctedQuery"] == "hill station outfits"
    assert body["results"][0]["productId"] == "DEC-FLC-001S"
    assert body["results"][0]["price"] == {"amount": 1499.0, "currency": "INR"}
    assert body["pagination"] == {"currentPage": 1, "totalPages": 1, "totalResults": 1}
