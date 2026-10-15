"""Local sentence-transformers embeddings (stands in for the doc's hosted embedding model).

bge models expect an instruction prefix on queries but not on documents.
"""

from functools import lru_cache

import numpy as np

from common.config import get_settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model():
    """Load from the local Hugging Face cache without any network call; download only if it isn't cached.
    (Without local_files_only the library checks the Hub on load, which took ~10 s on a request path.)"""
    from sentence_transformers import SentenceTransformer

    name = get_settings().embedding_model
    try:
        return SentenceTransformer(name, local_files_only=True)
    except OSError:
        return SentenceTransformer(name)


def warm_up() -> None:
    """Load the model now (call at service startup), independent of any embedding cache."""
    _model().encode(["warm up"])


def embed_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 1000)


def embed_queries(queries: list[str]) -> np.ndarray:
    return _model().encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True)
