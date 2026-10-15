"""Task registry: prompt template + input/output models + a code-side guardrail per task.

Guardrails (doc p.148: 'add in guardrails so the LLM ... only returns the categories we're passing')
are deterministic: they drop anything the model invented and report whether what is left is usable.
An unusable result escalates to the next model when the route sets `escalate_on_invalid`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from genai_gateway import schemas as s

PROMPTS = Path(__file__).with_name("prompts")
_env = Environment(loader=FileSystemLoader(PROMPTS), undefined=StrictUndefined, keep_trailing_newline=False)


class Validation(BaseModel):
    ok: bool
    output: dict
    reason: str | None = None


Validator = Callable[[BaseModel, BaseModel], Validation]


@dataclass(frozen=True)
class TaskSpec:
    prompt: str                     # template file in prompts/
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    validate: Validator
    max_tokens: int = 2048

    @property
    def system_prompt(self) -> str:
        return _render(self.prompt)

    @property
    def output_schema(self) -> dict:
        return s.strict_schema(self.output_model)

    def user_message(self, inputs: BaseModel) -> str:
        # sort_keys: identical inputs -> identical bytes, so repeated calls stay cache-friendly
        return json.dumps(inputs.model_dump(), sort_keys=True, ensure_ascii=False)


@lru_cache
def _render(template: str) -> str:
    return _env.get_template(template).render(attribute_keys=s.ATTRIBUTE_KEYS).strip()


def _subset(values: list[str], allowed: list[str]) -> list[str]:
    """Keep values that exist in `allowed` (case-insensitive), returned in allowed's spelling, de-duplicated."""
    lookup = {a.lower().strip(): a for a in allowed}
    out: list[str] = []
    for v in values:
        hit = lookup.get(v.lower().strip())
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def _validate_segment(inp: s.SegmentInput, out: s.SegmentOutput) -> Validation:
    data = out.model_dump()
    if out.taxonomy_path is not None:
        matched = _subset([out.taxonomy_path], inp.candidate_categories)
        if not matched:
            data["taxonomy_path"] = None
            return Validation(ok=False, output=data, reason="taxonomy_path not in candidate_categories")
        data["taxonomy_path"] = matched[0]
    if out.product_category is None and data["taxonomy_path"] is None:
        return Validation(ok=False, output=data, reason="no product category or taxonomy path")
    return Validation(ok=True, output=data)


def _validate_expand(inp: s.ExpandInput, out: s.ExpandOutput) -> Validation:
    data = out.model_dump()
    data["product_types"] = _subset(out.product_types, [a.product_type for a in inp.alternatives])
    data["must_attributes"] = _subset(out.must_attributes, inp.attributes)
    data["related_searches"] = [q.strip().lower() for q in out.related_searches if q.strip()][:4]
    if not data["product_types"]:
        return Validation(ok=False, output=data, reason="no valid product_types selected")
    return Validation(ok=True, output=data)


def _validate_rewrite(inp: s.RewriteInput, out: s.RewriteOutput) -> Validation:
    data = out.model_dump()
    data["categories"] = _subset(out.categories, inp.candidate_categories)
    data["attributes"] = [a.strip().lower() for a in out.attributes if a.strip()][:4]
    data["keywords"] = [k.strip().lower() for k in out.keywords if k.strip()][:6]
    if not data["categories"] and not data["keywords"]:
        return Validation(ok=False, output=data, reason="no valid categories or keywords")
    return Validation(ok=True, output=data)


def _validate_nrt_segment(inp: s.NrtSegmentInput, out: s.NrtSegmentOutput) -> Validation:
    # v2: a list, so occasion queries ("tailgate party must haves") get the products that serve them; v1's single
    # product_type came back null for both trending occasion queries in the Phase 9 demo
    types = list(dict.fromkeys(t.strip().lower() for t in out.product_types if t.strip()))[:4]
    data = {"product_types": types, "attributes": [a.strip().lower() for a in out.attributes if a.strip()]}
    ok = bool(types)
    return Validation(ok=ok, output=data, reason=None if ok else "no product types")


def _validate_discover(inp: s.DiscoverInput, out: s.DiscoverOutput) -> Validation:
    data = {"complements": _subset(out.complements, inp.co_purchase_data)[:5]}
    if out.complements and not data["complements"]:  # an empty answer is allowed (v2): nothing fits
        return Validation(ok=False, output=data, reason="no complements from co_purchase_data")
    return Validation(ok=True, output=data)


def _validate_judge(inp: s.JudgeInput, out: s.JudgeOutput) -> Validation:
    data = out.model_dump()
    # The winner is a deterministic function of the scores; don't trust the model's restatement.
    data["winner"] = "a" if out.score_a > out.score_b else "b" if out.score_b > out.score_a else "tie"
    return Validation(ok=True, output=data)


_EXPAND = TaskSpec("expand.v1.j2", s.ExpandInput, s.ExpandOutput, _validate_expand, max_tokens=2048)

TASKS: dict[str, TaskSpec] = {
    "segment": TaskSpec("segment.v1.j2", s.SegmentInput, s.SegmentOutput, _validate_segment),
    "expand_head": _EXPAND,
    "expand_tail": _EXPAND,
    "rewrite": TaskSpec("rewrite.v1.j2", s.RewriteInput, s.RewriteOutput, _validate_rewrite, max_tokens=1024),
    "nrt_segment": TaskSpec("nrt_segment.v2.j2", s.NrtSegmentInput, s.NrtSegmentOutput, _validate_nrt_segment,
                            max_tokens=1024),
    "discover": TaskSpec("discover.v2.j2", s.DiscoverInput, s.DiscoverOutput, _validate_discover),
    "judge": TaskSpec("judge.v1.j2", s.JudgeInput, s.JudgeOutput, _validate_judge, max_tokens=4096),
}
