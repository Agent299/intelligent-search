"""Integration tests against the running stack (plan Phase 11). Skipped unless SEARCH_URL is set:

    make up && make search   # in another terminal
    SEARCH_URL=http://localhost:8000 uv run pytest -q -m integration

They check behaviour through the public API only, so they also work against a deployed service.
"""

import os

import httpx
import pytest

URL = os.environ.get("SEARCH_URL")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not URL, reason="set SEARCH_URL to run against a live search service")]


def _search(q: str, **params) -> dict:
    r = httpx.get(f"{URL}/v1/search", params={"q": q, "debug": "true", **params}, timeout=15)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("q", ["beef jerky", "organic aged basmati rice", "post-workout food", "laptop"])
def test_case_study_queries_return_results(q):
    body = _search(q)
    assert body["results"] and body["pagination"]["totalResults"] > 0


def test_gibberish_returns_an_honest_empty_page():
    assert _search("xqzv blorf")["results"] == []


def test_negated_ingredient_is_excluded():
    titles = " ".join(r["title"].lower() for r in _search("healthy snacks without nuts for kids")["results"][:10])
    assert "peanut" not in titles and "almond" not in titles


def test_spell_correction_is_reported():
    assert _search("beef jekry").get("correctedQuery") == "beef jerky"


def test_batch_cached_query_is_served_from_a_cache_tier():
    assert _search("conference speakerphone")["debug"]["servedBy"] in {"l1", "l2", "warm", "results_cache"}


def test_click_event_is_accepted_and_unknown_search_rejected():
    body = _search("laptop")
    ok = httpx.post(f"{URL}/v1/events", json={"searchId": body["searchId"], "type": "click",
                                               "productId": body["results"][0]["productId"]})
    assert ok.status_code == 202
    bad = httpx.post(f"{URL}/v1/events", json={"searchId": "nope", "type": "click", "productId": "x"})
    assert bad.status_code == 404


def test_bad_filters_are_a_client_error():
    assert httpx.get(f"{URL}/v1/search", params={"q": "rice", "filters": "{bad"}).status_code == 422


def test_metrics_endpoint_exposes_business_metrics():
    text = httpx.get(f"{URL}/metrics").text
    assert "search_requests_total" in text and "search_latency_seconds_bucket" in text
