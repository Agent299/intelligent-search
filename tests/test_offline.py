from datetime import UTC, datetime

import polars as pl

from common.normalize import SpellCorrector
from common.schemas import ClassificationResult, EnrichedQuery, SegmentationResult, SegmentedQuery
from pipelines.offline.aggregate import aggregate, related_from_sessions
from pipelines.offline.enrich import fold_aliases
from pipelines.offline.expand import entity_text
from pipelines.offline.lake import fold_live_events
from pipelines.offline.run import next_run
from pipelines.offline.segment import select
from search_service.compile import ATTRIBUTE_BOOST, BASE_BOOST, compile_expansion, required_attributes, with_filters


def _events(rows):
    base = {"user_id": "u1", "raw_query": None, "search_type": "hybrid", "displayed_product_ids": [],
            "clicked_product_id": None, "added_to_cart_product_id": None, "purchased_product_ids": [],
            "device_type": "web", "geolocation": "Pune", "event_timestamp": datetime(2026, 10, 1, 19, tzinfo=UTC)}
    out = []
    for i, r in enumerate(rows):
        row = {**base, "event_id": f"e{i}", **r}
        row["raw_query"] = row["raw_query"] or row["normalized_query"]
        out.append(row)
    return pl.DataFrame(out)


def test_aggregate_counts_engagement_and_assigns_tiers():
    rows = ([{"normalized_query": "beef jerky", "session_id": f"s{i}", "clicked_product_id": "p1",
              "purchased_product_ids": ["p1"]} for i in range(30)]
            + [{"normalized_query": f"rare {i}", "session_id": f"r{i}"} for i in range(39)])
    aggs = {a.normalized_query: a for a in aggregate(_events(rows))}
    top = aggs["beef jerky"]
    assert top.frequency == 30 and top.freq_rank == 1 and top.tier == "l1"
    assert top.clicks == 30 and top.purchases == 30 and top.top_purchased == ["p1"]
    assert top.device == "web" and top.time_of_day == "evening"
    assert sum(a.tier == "l1" for a in aggs.values()) == 2  # top 5% of 40 queries
    assert aggs["rare 38"].tier == "warm"


def test_related_searches_require_lift_not_just_cooccurrence():
    rows = []
    for i in range(3):  # 'chips' and 'soda' searched together in 3 sessions, and rarely otherwise
        rows += [{"normalized_query": "chips", "session_id": f"s{i}"},
                 {"normalized_query": "soda", "session_id": f"s{i}"}]
    for i in range(40):  # a popular query appears in many sessions, sometimes next to chips by chance
        rows.append({"normalized_query": "ipad case", "session_id": f"p{i}"})
        if i < 2:
            rows.append({"normalized_query": "chips", "session_id": f"p{i}"})
    rows += [{"normalized_query": f"other {i}", "session_id": f"o{i}"} for i in range(40)]
    related = related_from_sessions(_events(rows))
    assert related["chips"] == ["soda"] and related["soda"] == ["chips"]
    assert "ipad case" not in related.get("chips", [])  # co-occurs twice, but lift < 3


def test_fold_live_events_joins_interactions_to_their_search():
    events = [
        {"event_id": "a", "event_type": "search", "search_id": "S1", "raw_query": "Beef Jerky!",
         "normalized_query": None, "displayed_product_ids": ["p1", "p2"],
         "event_timestamp": "2026-10-14T10:00:00+00:00"},
        {"event_id": "b", "event_type": "click", "search_id": "S1", "clicked_product_id": "p2",
         "event_timestamp": "2026-10-14T10:00:05+00:00"},
        {"event_id": "c", "event_type": "purchase", "search_id": "S1", "purchased_product_ids": ["p2"],
         "event_timestamp": "2026-10-14T10:01:00+00:00"},
    ]
    [row] = fold_live_events(events)
    assert row["event_id"] == "a" and row["normalized_query"] == "beef jerky"
    assert row["clicked_product_id"] == "p2" and row["purchased_product_ids"] == ["p2"]


def test_fold_aliases_merges_typos_into_canonical_query():
    spell = SpellCorrector()
    spell.add_words({"beef": 50, "jerky": 50})
    payloads = [{"original_query": "beef jerky", "normalized_query": "beef jerky", "frequency": 10, "tier": "l1"},
                {"original_query": "beef jekry", "normalized_query": "beef jekry", "frequency": 3, "tier": "warm"}]
    [q] = fold_aliases(payloads, spell)
    assert q.canonical == "beef jerky" and q.frequency == 13 and q.aliases == ["beef jekry"] and q.tier == "l1"


def test_select_takes_gated_queries_most_frequent_first():
    qs = [EnrichedQuery(original_query=n, normalized_query=n, frequency=f, fast_confidence=c)
          for n, f, c in [("a", 5, 0.9), ("b", 3, 0.4), ("c", 9, 0.55), ("d", 1, None)]]
    assert [q.normalized_query for q in select(qs, None)] == ["c", "b"]
    assert [q.normalized_query for q in select(qs, 1)] == ["c"]


def test_compile_expansion_matches_doc_shape():
    q = compile_expansion(["basmati rice", "sona masoori rice", "poha"], ["organic"], "Grocery > Rice")
    [alts] = [m["bool"] for m in q["bool"]["must"]]  # product types required; attributes only boost
    assert alts["minimum_should_match"] == 1 and [s["multi_match"]["query"] for s in alts["should"]] == [
        "basmati rice", "sona masoori rice", "poha"]
    assert alts["should"][0]["multi_match"]["boost"] == BASE_BOOST > alts["should"][1]["multi_match"]["boost"]
    attr, category = q["bool"]["should"]
    assert attr["multi_match"]["query"] == "organic" and attr["multi_match"]["boost"] == ATTRIBUTE_BOOST
    assert category["prefix"]["category_path"]["value"] == "Grocery > Rice"
    wrapped = with_filters(q, [{"term": {"in_stock": True}}])
    assert wrapped["bool"]["must"] == [q] and wrapped["bool"]["filter"] == [{"term": {"in_stock": True}}]


def test_entity_text_prefers_quality_plus_category():
    seg = SegmentedQuery(original_query="x", normalized_query="organic aged basmati rice",
                         segmentation_results=SegmentationResult(product_category="rice", product_quality="basmati"),
                         classification_results=ClassificationResult())
    assert entity_text(seg) == "basmati rice"


def test_must_attributes_already_in_the_base_type_are_dropped():
    assert required_attributes(["conference speakerphone"], ["conference", "wireless"]) == ["wireless"]
    assert required_attributes(["tissue pom poms"], ["tissue"]) == []


def test_select_skips_recently_processed_queries():
    qs = [EnrichedQuery(original_query=n, normalized_query=n, frequency=5, fast_confidence=0.3) for n in "abc"]
    assert [q.normalized_query for q in select(qs, None, skip={"b"})] == ["a", "c"]


def test_next_run_daily_and_hourly():
    morning = datetime(2026, 10, 15, 1, 30, tzinfo=UTC)
    assert next_run(morning, hourly=False) == datetime(2026, 10, 15, 2, tzinfo=UTC)
    assert next_run(morning.replace(hour=2), hourly=False) == datetime(2026, 10, 16, 2, tzinfo=UTC)
    assert next_run(morning, hourly=True) == datetime(2026, 10, 15, 2, tzinfo=UTC)
