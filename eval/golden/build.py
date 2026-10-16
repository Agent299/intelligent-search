"""Build the golden query set (doc p.181) from the logs, deterministically.

    uv run python -m eval.golden.build      -> eval/golden/queries.yaml

Segments cover what the system must handle: the case study's own examples, head queries the fast path answers,
head/torso/tail queries that needed the offline LLM pipeline, queries it failed on, dietary and negation phrasing,
typos, and gibberish (which should return nothing).
"""

import random
from pathlib import Path

import yaml

from common.db import connect

OUT = Path(__file__).parent / "queries.yaml"
SEED = 11

CASE_STUDY = [
    "post-workout food", "cheeseburst pizza", "old people home footwear", "healthy snacks without nuts for kids",
    "a summer dress that's not too formal", "organic aged basmati rice", "stuff for my dorm party",
    "hill station outfits", "brownie",
    # occasion / need queries from Phases 4 and 9
    "something warm for my grandma", "cozy movie night essentials", "tailgate party must haves",
    "stuff for a long road trip", "survival kit for finals week",
]
GIBBERISH = ["xqzv blorf", "asdkjh qwepoi", "zzkq mmrt plo"]

# (segment, SQL filter, count)
SEGMENTS = [
    ("head_easy", "tier = 'l1' AND status = 'enriched'", 25),
    ("head_hard", "tier = 'l1' AND status = 'expanded'", 15),
    ("torso", "tier = 'l2' AND status = 'enriched'", 15),
    ("torso_hard", "tier = 'l2' AND status = 'expanded'", 10),
    ("tail", "tier = 'warm' AND status = 'enriched'", 20),
    ("tail_hard", "tier = 'warm' AND status = 'expanded'", 15),
    ("pipeline_failed", "status = 'failed'", 8),
    ("dietary_negation", r"normalized_query ~ '\m(without|free|not)\M'", 10),
]
N_TYPOS = 12


def _pool(where: str) -> list[str]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT normalized_query AS q FROM query_stats WHERE {where} ORDER BY normalized_query")
        return [r["q"] for r in cur.fetchall()]


def typo(q: str, rng: random.Random) -> str:
    """One adjacent transposition inside a word longer than 4 letters (how the synthetic logs make typos)."""
    words = q.split()
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return q
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def build() -> list[dict]:
    rng = random.Random(SEED)
    taken = set(CASE_STUDY) | set(GIBBERISH)
    out = [{"query": q, "segment": "case_study"} for q in CASE_STUDY]
    for segment, where, n in SEGMENTS:
        pool = [q for q in _pool(where) if q not in taken and 2 <= len(q) <= 60]
        for q in rng.sample(pool, min(n, len(pool))):
            out.append({"query": q, "segment": segment})
            taken.add(q)
    heads = [q for q in _pool("tier = 'l1' AND status = 'enriched'") if q not in taken]
    for q in rng.sample(heads, N_TYPOS * 2):
        t = typo(q, rng)
        if t != q and len([x for x in out if x["segment"] == "typo"]) < N_TYPOS:
            out.append({"query": t, "segment": "typo", "intended": q})
    out += [{"query": q, "segment": "gibberish"} for q in GIBBERISH]
    return out


if __name__ == "__main__":
    queries = build()
    OUT.write_text(yaml.safe_dump({"seed": SEED, "queries": queries}, sort_keys=False, allow_unicode=True))
    counts = {}
    for q in queries:
        counts[q["segment"]] = counts.get(q["segment"], 0) + 1
    print(f"wrote {OUT}: {len(queries)} queries {counts}")
