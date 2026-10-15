"""Merging keyword and semantic results, and deciding whether the fast path answered the query.

BM25 scores and cosine similarities live on different scales, so results are merged with reciprocal rank
fusion (RRF), which only uses ranks. A small popularity prior breaks near-ties; full personalized re-ranking
arrives in Phase 8.
"""

from dataclasses import dataclass

from search_service.retrieval import Hit

RRF_K = 60
# Popularity prior (static_score in [0, 1]) as a tie-breaker: 0.0005 is just under the RRF gap between ranks 1 and 3,
# so near the top a maximally popular product moves up at most one place (a few places deeper in the list, where
# rank gaps shrink). Real ranking comes in Phase 8.
PRIOR_WEIGHT = 0.0005

# Cosine similarity of bge-small query/document pairs: ~0.55 is unrelated, ~0.80+ is a strong match.
SEM_FLOOR, SEM_CEIL = 0.55, 0.80


@dataclass
class Candidate:
    product_id: str
    score: float
    keyword_rank: int | None
    semantic_rank: int | None
    semantic_sim: float | None


def rrf_fuse(keyword: list[Hit], semantic: list[Hit], priors: dict[str, float] | None = None) -> list[Candidate]:
    return rrf_fuse_many([keyword, semantic], priors)


def rrf_fuse_many(lists: list[list[Hit]], priors: dict[str, float] | None = None) -> list[Candidate]:
    """RRF over one keyword list (lists[0]) and any number of semantic lists (cosine scores). Several semantic
    lists (one per rewrite keyword) interleave naturally: each list's top items score alike, so 'chips' and
    'soda' results alternate instead of one keyword taking the whole first page."""
    priors = priors or {}
    cands: dict[str, Candidate] = {}
    for i, hits in enumerate(lists):
        for rank, hit in enumerate(hits, start=1):
            c = cands.setdefault(hit.product_id, Candidate(hit.product_id, 0.0, None, None, None))
            c.score += 1 / (RRF_K + rank)
            if i == 0:
                c.keyword_rank = rank
            else:
                c.semantic_rank = rank if c.semantic_rank is None else min(c.semantic_rank, rank)
                c.semantic_sim = hit.score if c.semantic_sim is None else max(c.semantic_sim, hit.score)
    for c in cands.values():
        c.score += PRIOR_WEIGHT * priors.get(c.product_id, 0.0)
    return sorted(cands.values(), key=lambda c: c.score, reverse=True)


@dataclass
class Confidence:
    score: float
    semantic_strength: float
    agreement: float
    keyword_coverage: float


def confidence(keyword: list[Hit], semantic: list[Hit], results: list[Candidate], top_n: int = 10) -> Confidence:
    """How sure we are the fast path found what the shopper wants, in [0, 1] (doc Fig 5.14: > 0.6 to skip
    the LLM fallback). Three signals:
      - semantic strength: mean cosine of the top semantic hits, rescaled between 'unrelated' and 'strong'
      - agreement: overlap of the keyword and semantic top-N (both retrievers point at the same products)
      - keyword coverage: whether BM25 matched at all (it requires most query terms)
    """
    if not results:
        return Confidence(0.0, 0.0, 0.0, 0.0)
    sims = [h.score for h in semantic[:3]]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    strength = min(1.0, max(0.0, (mean_sim - SEM_FLOOR) / (SEM_CEIL - SEM_FLOOR)))
    kw_top = {h.product_id for h in keyword[:top_n]}
    sem_top = {h.product_id for h in semantic[:top_n]}
    agreement = len(kw_top & sem_top) / min(top_n, len(kw_top), len(sem_top)) if kw_top and sem_top else 0.0
    coverage = min(1.0, len(keyword) / 5)
    score = 0.6 * strength + 0.25 * agreement + 0.15 * coverage
    return Confidence(round(score, 3), round(strength, 3), round(agreement, 3), round(coverage, 3))
