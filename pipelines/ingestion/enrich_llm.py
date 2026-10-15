"""LLM attribute extraction over the catalog via the Message Batches API (50% price, async).

Step 1 of the ingestion pipeline (doc p.146, Fig 5.1). For each product the model returns a
normalized product_type, attributes restricted to a fixed key vocabulary (guardrail: the schema
enum rejects any other key), and search tags describing who/what it is for - the signals that let
'old people home footwear' find soft-soled slippers.

    uv run python -m pipelines.ingestion.enrich_llm submit --limit 200   # pilot
    uv run python -m pipelines.ingestion.enrich_llm status
    uv run python -m pipelines.ingestion.enrich_llm collect               # merge into catalog + cost report
"""

import argparse
import json
import time
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from psycopg.types.json import Jsonb

from common.db import connect
from common.llm import anthropic_client
from genai_gateway.config import load_routing
from genai_gateway.schemas import ATTRIBUTE_KEYS as GATEWAY_ATTRIBUTE_KEYS

# Tier 1 (Haiku) per the cost policy; the Sonnet pilot ran at ~$1.08 per 1k products.
DEFAULT_MODEL = load_routing().tasks["extract_attributes"].chain[0]  # routing.yaml is the source of truth
PRODUCTS_PER_REQUEST = 20
STATE = Path("data/raw/llm_batches.json")
# Batch API prices, $ per 1M tokens (50% of standard rates; cache read 0.1x, cache write 1.25x of input)
PRICES = {
    "claude-sonnet-5-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    "claude-haiku-5-5": {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write": 0.0625},
}

ATTRIBUTE_KEYS = list(GATEWAY_ATTRIBUTE_KEYS)  # shared with query segmentation so both sides speak one vocabulary

SYSTEM = f"""You enrich an e-commerce catalog so that search can understand shopper intent.

For each product, return:
- product_type: the generic noun phrase a shopper would type for this kind of item, lowercase, \
1-4 words (e.g. "basmati rice", "slippers", "fleece jacket", "usb-c cable"). No brand names.
- attributes: facts stated or clearly implied by the product data, as key/value pairs. Use only \
these keys: {", ".join(ATTRIBUTE_KEYS)}. Values are short lowercase phrases. Omit a key rather than guess.
- search_tags: 3-8 lowercase phrases describing who it suits, when it is used, and what it pairs with, \
in words a shopper might search (e.g. "soft sole", "comfortable for seniors", "post-workout snack", \
"party drinks", "hill station trip"). Only include tags supported by the product data.

Return one entry per input product, using the input id unchanged."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "product_type": {"type": "string"},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "enum": ATTRIBUTE_KEYS},
                                "value": {"type": "string"},
                            },
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "search_tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "product_type", "attributes", "search_tags"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}


def _product_payload(p: dict) -> dict:
    return {
        "id": p["product_id"],
        "title": p["name"],
        "brand": p["brand"],
        "category": p["category_path"],
        "known_attributes": p["attributes"],
        "text": p["description"][:700],
    }


def _load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"batches": []}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def build_requests(products: list[dict], model: str = DEFAULT_MODEL) -> list[Request]:
    requests = []
    for i in range(0, len(products), PRODUCTS_PER_REQUEST):
        chunk = products[i : i + PRODUCTS_PER_REQUEST]
        requests.append(Request(
            custom_id=f"p{i:06d}-{chunk[0]['product_id']}"[:64],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=8000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": json.dumps([_product_payload(p) for p in chunk])}],
            ),
        ))
    return requests


def submit(limit: int | None, model: str) -> None:
    state = _load_state()
    pending = {pid for b in state["batches"] if not b.get("collected") for pid in b["product_ids"]}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT product_id, name, brand, category_path, attributes, description FROM product_catalog
               WHERE llm_enriched_at IS NULL ORDER BY product_id""")
        products = [p for p in cur.fetchall() if p["product_id"] not in pending]
    if limit:
        products = products[:limit]
    if not products:
        print("nothing to enrich")
        return
    requests = build_requests(products, model)
    client = anthropic_client()
    batch = client.messages.batches.create(requests=requests)
    state["batches"].append({
        "id": batch.id, "model": model, "submitted_at": int(time.time()),
        "requests": len(requests), "product_ids": [p["product_id"] for p in products],
    })
    _save_state(state)
    print(f"submitted batch {batch.id}: {len(products)} products in {len(requests)} requests")


def status() -> None:
    client = anthropic_client()
    for b in _load_state()["batches"]:
        info = client.messages.batches.retrieve(b["id"])
        c = info.request_counts
        print(f"{b['id']} {info.processing_status:<12} products={len(b['product_ids'])} "
              f"ok={c.succeeded} err={c.errored} processing={c.processing} collected={b.get('collected', False)}")


def collect() -> None:
    client = anthropic_client()
    state = _load_state()
    for b in state["batches"]:
        if b.get("collected"):
            continue
        if client.messages.batches.retrieve(b["id"]).processing_status != "ended":
            print(f"{b['id']} still processing")
            continue
        allowed = set(b["product_ids"])
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        enriched, failed_requests = [], 0
        for result in client.messages.batches.results(b["id"]):
            if result.result.type != "succeeded":
                failed_requests += 1
                continue
            msg = result.result.message
            u = msg.usage
            usage["input"] += u.input_tokens
            usage["output"] += u.output_tokens
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            if msg.stop_reason != "end_turn":
                failed_requests += 1
                continue
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            for item in json.loads(text)["products"]:
                if item["id"] in allowed:  # guardrail: ignore ids the model invented
                    enriched.append(item)

        with connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """UPDATE product_catalog SET
                       product_type = %s, search_tags = %s,
                       attributes = %s || attributes,   -- catalog-sourced values win over LLM values
                       llm_enriched_at = now(), updated_at = now()
                   WHERE product_id = %s""",
                [(e["product_type"].lower()[:80],
                  list(dict.fromkeys(t.lower()[:60] for t in e["search_tags"]))[:8],
                  Jsonb({a["key"]: a["value"].lower()[:80] for a in e["attributes"]}),
                  e["id"]) for e in enriched],
            )
        price = PRICES[b.get("model", "claude-sonnet-5-5")]
        cost = sum(usage[k] * price[k] for k in usage) / 1e6
        per_product = cost / max(1, len(enriched))
        b.update(collected=True, enriched=len(enriched), failed_requests=failed_requests,
                 usage=usage, cost_usd=round(cost, 4))
        _save_state(state)
        print(f"{b['id']} ({b.get('model')}): enriched {len(enriched)}/{len(allowed)} products, "
              f"{failed_requests} failed requests")
        print(f"  tokens {usage}  cost ${cost:.4f}  (${per_product * 1000:.3f} per 1k products)")


def cancel(batch_id: str) -> None:
    """Cancel a batch. Requests already finished stay billed and are picked up by `collect`;
    unfinished products remain un-enriched and go into the next `submit`."""
    info = anthropic_client().messages.batches.cancel(batch_id)
    print(f"{batch_id}: {info.processing_status}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--limit", type=int)
    s.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES))
    sub.add_parser("status")
    sub.add_parser("collect")
    c = sub.add_parser("cancel")
    c.add_argument("batch_id")
    args = ap.parse_args()
    {
        "submit": lambda: submit(args.limit, args.model),
        "status": status,
        "collect": collect,
        "cancel": lambda: cancel(args.batch_id),
    }[args.cmd]()


if __name__ == "__main__":
    main()
