import json

import pytest
from fastapi.testclient import TestClient

from genai_gateway.app import create_app
from genai_gateway.breaker import CircuitBreaker, State
from genai_gateway.config import load_routing
from genai_gateway.core import Gateway, GatewayUnavailable, build_batch_request, next_model, parse_and_validate
from genai_gateway.llm import LLMResult, TransientLLMError
from genai_gateway.schemas import SegmentOutput, strict_schema
from genai_gateway.tasks import TASKS

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"

SEGMENT_INPUT = {
    "query": "organic aged basmati rice",
    "candidate_categories": [
        "Grocery & Gourmet Food > Pantry Staples > Rice",
        "Grocery & Gourmet Food > Pantry Staples > Pasta",
    ],
}


def _segment(path):
    return json.dumps({"product_category": "rice", "product_quality": "basmati", "brand": None,
                       "attributes": [{"key": "processing", "value": "aged"}], "taxonomy_path": path})


class FakeCaller:
    """Scripted per-model responses: a string (JSON text), an LLMResult, or an exception."""

    def __init__(self, script: dict):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls: list[dict] = []

    async def __call__(self, **kw):
        self.calls.append(kw)
        item = self.script[kw["model"]].pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResult):
            return item
        return LLMResult(text=item, stop_reason="end_turn", input_tokens=1000, output_tokens=200)


# ---- config / prompts / schema ----

def test_every_gateway_task_has_a_route_and_renders():
    routing = load_routing()
    for name, spec in TASKS.items():
        assert name in routing.tasks, name
        assert spec.system_prompt and "{{" not in spec.system_prompt
        assert spec.output_schema["additionalProperties"] is False


def test_cost_policy_tiers_in_routing():
    r = load_routing()
    assert r.tasks["rewrite"].chain == [HAIKU]  # online path stays on the cheap tier
    assert r.tasks["segment"].chain[0] == HAIKU and r.tasks["segment"].escalate_on_invalid
    assert r.tasks["judge"].chain == [OPUS]  # frontier only for evaluation
    assert r.cost_usd(HAIKU, 1_000_000, 0) == pytest.approx(0.10)
    assert r.cost_usd(SONNET, 0, 1_000_000, batch=True) == pytest.approx(5.0)
    assert r.cost_usd(HAIKU, 0, 0, cache_read=1_000_000) == pytest.approx(0.01)


def test_strict_schema_closes_nested_objects_and_requires_all_fields():
    schema = strict_schema(SegmentOutput)
    assert set(schema["required"]) == set(schema["properties"])
    attr = schema["$defs"]["Attribute"]
    assert attr["additionalProperties"] is False and attr["required"] == ["key", "value"]
    assert "enum" in attr["properties"]["key"]


# ---- circuit breaker ----

def test_breaker_opens_then_half_opens_then_closes():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, reset_after_s=10, clock=lambda: now[0])
    b.record_failure()
    assert b.allow()
    b.record_failure()
    assert b.state is State.open and not b.allow()
    now[0] = 11
    assert b.state is State.half_open
    assert b.allow() and not b.allow()  # exactly one trial call
    b.record_success()
    assert b.state is State.closed


def test_breaker_reopens_when_trial_fails():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=1, reset_after_s=10, clock=lambda: now[0])
    b.record_failure()
    now[0] = 10
    assert b.allow()
    b.record_failure()
    assert b.state is State.open


# ---- routing, escalation, guardrails ----

async def test_valid_first_model_answer_is_returned_with_cost():
    caller = FakeCaller({HAIKU: [_segment("Grocery & Gourmet Food > Pantry Staples > Rice")]})
    res = await Gateway(caller=caller).run("segment", SEGMENT_INPUT)
    assert res.valid and res.model == HAIKU and not res.escalated
    assert res.output["taxonomy_path"] == "Grocery & Gourmet Food > Pantry Staples > Rice"
    assert res.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)
    assert caller.calls[0]["effort"] == "low"


async def test_hallucinated_taxonomy_path_escalates_to_next_tier():
    caller = FakeCaller({
        HAIKU: [_segment("Groceries > Rice > Basmati")],  # not a candidate -> guardrail rejects
        SONNET: [_segment("grocery & gourmet food > pantry staples > rice")],  # case-insensitive match
    })
    res = await Gateway(caller=caller).run("segment", SEGMENT_INPUT)
    assert res.valid and res.model == SONNET and res.escalated
    assert [a.outcome for a in res.attempts] == ["invalid", "ok"]
    assert res.output["taxonomy_path"] == "Grocery & Gourmet Food > Pantry Staples > Rice"


async def test_invalid_output_without_escalation_is_returned_flagged():
    caller = FakeCaller({HAIKU: [json.dumps({"categories": ["Made Up"], "attributes": [], "keywords": []})]})
    res = await Gateway(caller=caller).run("rewrite", {"query": "x", "candidate_categories": ["A > B"]})
    assert not res.valid and res.model == HAIKU and res.output["categories"] == []


async def test_transient_errors_trip_breaker_and_later_calls_skip_the_model():
    caller = FakeCaller({HAIKU: [TransientLLMError("timeout")] * 2,
                         SONNET: [_segment(SEGMENT_INPUT["candidate_categories"][0])] * 3})
    gw = Gateway(caller=caller, failure_threshold=2)
    for _ in range(2):
        assert (await gw.run("segment", SEGMENT_INPUT)).model == SONNET
    res = await gw.run("segment", SEGMENT_INPUT)
    assert res.attempts[0].outcome == "skipped_open"
    assert gw.breaker_states()[HAIKU] == "open"
    assert sum(c["model"] == HAIKU for c in caller.calls) == 2  # no third call to the broken model


async def test_refusal_and_bad_json_move_down_the_chain():
    caller = FakeCaller({HAIKU: [LLMResult(text="", stop_reason="refusal")], SONNET: ["not json"]})
    with pytest.raises(GatewayUnavailable) as e:
        await Gateway(caller=caller).run("segment", SEGMENT_INPUT)
    assert [a.outcome for a in e.value.attempts] == ["refusal", "invalid"]


async def test_discover_keeps_only_supplied_items():
    caller = FakeCaller({HAIKU: [json.dumps({"complements": ["Vanilla Ice Cream", "caviar", "chocolate sauce"]})]})
    res = await Gateway(caller=caller).run("discover", {
        "primary_product": "brownie",
        "co_purchase_data": ["eggs", "vanilla ice cream", "chocolate sauce"]})
    assert res.output["complements"] == ["vanilla ice cream", "chocolate sauce"]


async def test_discover_accepts_an_empty_answer_but_not_only_invented_items():
    empty = await Gateway(caller=FakeCaller({HAIKU: [json.dumps({"complements": []})]})).run("discover", {
        "primary_product": "cables", "co_purchase_data": ["smartwatches"]})
    assert empty.output["complements"] == []
    invented = FakeCaller({HAIKU: [json.dumps({"complements": ["caviar"]})] * 3})
    res = await Gateway(caller=invented).run("discover", {"primary_product": "brownie",
                                                          "co_purchase_data": ["ice cream"]})
    assert not res.valid


async def test_judge_winner_is_derived_from_scores():
    caller = FakeCaller({OPUS: [json.dumps({"reasoning": "B is on-intent", "score_a": 2, "score_b": 4,
                                            "winner": "a"})]})
    res = await Gateway(caller=caller).run("judge", {"query": "q", "results_a": [{"title": "x"}],
                                                     "results_b": [{"title": "y"}]})
    assert res.output["winner"] == "b"


# ---- batch helpers ----

def test_batch_request_uses_routing_and_shared_prompt():
    req = build_batch_request("segment", "q-1", SEGMENT_INPUT)
    p = req["params"]
    assert p["model"] == HAIKU
    assert p["system"][0]["text"] == TASKS["segment"].system_prompt
    assert p["output_config"]["format"]["schema"] == TASKS["segment"].output_schema
    assert next_model("segment") == HAIKU and next_model("segment", HAIKU) == SONNET
    assert next_model("segment", SONNET) is None


def test_parse_and_validate_applies_guardrail_to_batch_text():
    ok, out, reason = parse_and_validate("segment", SEGMENT_INPUT, _segment("Nope"))
    assert not ok and out["taxonomy_path"] is None and "taxonomy_path" in reason


# ---- HTTP ----

class FakeEmbeddings:
    async def embed(self, texts, kind):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_http_endpoints():
    caller = FakeCaller({HAIKU: [json.dumps({"categories": ["A > B"], "attributes": ["party size"],
                                             "keywords": ["chips"]})]})
    app = create_app(gateway=Gateway(caller=caller), embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        r = client.post("/v1/llm/rewrite", json={"inputs": {"query": "dorm party", "candidate_categories": ["A > B"]}})
        assert r.status_code == 200 and r.json()["output"]["keywords"] == ["chips"]
        assert client.post("/v1/llm/nope", json={"inputs": {}}).status_code == 404
        assert client.post("/v1/llm/rewrite", json={"inputs": {"query": 1}}).status_code == 422
        assert client.post("/v1/embed", json={"texts": ["a", "b"]}).json()["dim"] == 3
        assert client.get("/v1/breakers").json()[HAIKU] == "closed"
        assert client.get("/metrics").status_code == 200


def test_http_returns_503_when_chain_exhausted():
    caller = FakeCaller({HAIKU: [TransientLLMError("HTTP 529")]})
    app = create_app(gateway=Gateway(caller=caller), embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        r = client.post("/v1/llm/rewrite", json={"inputs": {"query": "x", "candidate_categories": []}})
        assert r.status_code == 503 and r.json()["detail"]["attempts"][0]["outcome"] == "error"


async def test_nrt_segment_v2_returns_product_types_for_occasion_queries():
    out = {"product_types": ["Folding Chairs", "cooler", "folding chairs", "snacks", "grill", "tent"],
           "attributes": [" Portable "]}
    res = await Gateway(caller=FakeCaller({HAIKU: [json.dumps(out)]})).run(
        "nrt_segment", {"query": "tailgate party must haves"})
    assert res.output == {"product_types": ["folding chairs", "cooler", "snacks", "grill"],
                          "attributes": ["portable"]}
