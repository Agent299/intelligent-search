"""Re-ranking models (doc p.178, Fig 5.16): LightGBM LambdaMART when a trained model is present, else a linear
weighted scorer over the same features. Serving and tests also run without LightGBM installed.
"""

try:
    # Must load before torch. On macOS, torch ships its own libomp; if it loads first, LightGBM then brings
    # Homebrew's libomp and the second OpenMP runtime segfaults the process (verified 2026-10-15, both in
    # training and in predict, even single-threaded). Loaded first, torch reuses this runtime. Linux uses
    # libgomp for both and is unaffected. Any module that uses both must import ranking.model first.
    import lightgbm  # noqa: F401
except Exception:  # not installed (no `ml` extra) or no OpenMP runtime: the linear ranker still works
    lightgbm = None

import json
import logging
import math
from pathlib import Path

import numpy as np

from ranking.features import FEATURES

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/ranker.txt")
META_PATH = Path("models/ranker.json")

# Hand-set fallback. Retrieval order dominates; personal signals nudge within it. Evaluated against LightGBM in
# the training report, so the fallback's cost is known, not guessed.
LINEAR_WEIGHTS = {
    "retrieval_rank": 1.0, "semantic_sim": 0.5, "static_score": 0.3, "popularity": 0.1, "qp_engagement": 1.0,
    "brand_in_query": 0.3, "user_brand_aff": 0.6, "user_cat_aff": 0.1, "user_top_aff": 0.05,
    "session_brand_clicks": 0.1, "price_band_gap": -0.05, "user_has_history": 0.0,
}


class LinearRanker:
    name = "linear"

    def __init__(self, weights: dict[str, float] | None = None):
        w = weights or LINEAR_WEIGHTS
        self.w = np.array([w.get(f, 0.0) for f in FEATURES])

    def score(self, rows: np.ndarray) -> np.ndarray:
        return np.nan_to_num(rows, nan=0.0) @ self.w


class LightGBMRanker:
    name = "lightgbm"

    def __init__(self, path: Path):
        if lightgbm is None:
            raise RuntimeError("lightgbm unavailable")
        self.booster = lightgbm.Booster(model_file=str(path))
        if self.booster.feature_name() != FEATURES:
            raise ValueError(f"model features {self.booster.feature_name()} != {FEATURES}")

    def score(self, rows: np.ndarray) -> np.ndarray:
        return self.booster.predict(rows)


def load_ranker(path: Path = MODEL_PATH) -> LinearRanker | LightGBMRanker:
    if path.exists():
        try:
            ranker = LightGBMRanker(path)
            meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            log.info("LightGBM ranker loaded (trained %s)", meta.get("trained_at", "?"))
            return ranker
        except Exception:
            log.warning("could not load %s; using the linear ranker", path, exc_info=True)
    return LinearRanker()


def ndcg_at_k(labels: list[float], scores: list[float], k: int = 10) -> float | None:
    """NDCG@k with exponential gain (2^label - 1). None when the group has no relevant item."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    gains = [(2 ** labels[i] - 1) for i in order[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None
