"""Task input/output models. Outputs are sent to the API as strict JSON schemas (structured outputs),
then re-validated here and checked by each task's code-side guardrail."""

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

ATTRIBUTE_KEYS = (
    "color", "material", "flavor", "form", "size", "style", "pattern", "fit",
    "audience", "age_group", "gender", "occasion", "season", "activity",
    "diet", "allergen_free", "nutrition", "origin", "processing",
    "sole_type", "closure", "heel_type", "power_source", "connectivity", "compatible_with",
    "room", "use_case", "feature",
)
AttributeKey = Literal[ATTRIBUTE_KEYS]  # type: ignore[valid-type]


class Attribute(BaseModel):
    key: AttributeKey
    value: str


# ---- segment (offline): query -> structured intent + taxonomy node (doc p.151) ----

class SegmentInput(BaseModel):
    query: str
    candidate_categories: list[str] = Field(description="taxonomy paths pre-selected by embedding similarity")


class SegmentOutput(BaseModel):
    product_category: str | None
    product_quality: str | None
    brand: str | None
    attributes: list[Attribute]
    taxonomy_path: str | None


# ---- expand (offline): pick which retrieved alternatives fit the intent (doc p.153) ----

class Alternative(BaseModel):
    product_type: str
    example: str


class ExpandInput(BaseModel):
    query: str
    product_category: str | None
    attributes: list[str]
    alternatives: list[Alternative]


class ExpandOutput(BaseModel):
    product_types: list[str] = Field(description="subset of the alternatives' product_type values, best first")
    must_attributes: list[str] = Field(description="subset of the input attributes every result must have")
    related_searches: list[str]


# ---- rewrite (online fallback): ambiguous query -> structured query (doc p.157) ----

class RewriteInput(BaseModel):
    query: str
    candidate_categories: list[str]


class RewriteOutput(BaseModel):
    categories: list[str] = Field(description="subset of candidate_categories; OR-ed together")
    attributes: list[str] = Field(description="attribute phrases that should match; AND-ed")
    keywords: list[str] = Field(description="product-type keywords to search for")


# ---- nrt_segment (near-real-time lite enrichment, doc p.170) ----

class NrtSegmentInput(BaseModel):
    query: str


class NrtSegmentOutput(BaseModel):
    product_types: list[str] = Field(description="1-4 generic product types, most likely first")
    attributes: list[str]


# ---- discover: complements from co-purchase data (doc p.160) ----

class DiscoverInput(BaseModel):
    primary_product: str
    co_purchase_data: list[str]


class DiscoverOutput(BaseModel):
    complements: list[str] = Field(description="0-5 genuine complements chosen from co_purchase_data, best first")


# ---- judge (offline evaluation, doc p.181) ----

class JudgedResult(BaseModel):
    title: str
    category: str | None = None


class JudgeInput(BaseModel):
    query: str
    results_a: list[JudgedResult]
    results_b: list[JudgedResult]


class JudgeOutput(BaseModel):
    reasoning: str
    score_a: Literal[1, 2, 3, 4, 5]
    score_b: Literal[1, 2, 3, 4, 5]
    winner: Literal["a", "b", "tie"]


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema adapted for structured outputs: every object closed and every property
    required (optional fields stay nullable), descriptions/titles dropped to save prompt tokens."""
    schema = deepcopy(model.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema
