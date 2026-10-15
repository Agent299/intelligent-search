"""Candidate taxonomy paths for a query (Tier 0). LLM tasks get ~20 relevant paths instead of the whole
taxonomy, which keeps prompts small and lets code verify the model only picked from the list."""

import numpy as np

from common.db import connect
from common.embedding_service import EmbeddingService

MIN_LEVEL = 2  # top-level departments alone are too broad to be useful


class TaxonomyIndex:
    def __init__(self, paths: list[str], vectors: np.ndarray):
        self.paths = paths
        self.vectors = vectors  # rows L2-normalized

    @classmethod
    async def build(cls, embeddings: EmbeddingService) -> "TaxonomyIndex":
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT path FROM taxonomy WHERE level >= %s ORDER BY path", (MIN_LEVEL,))
            paths = [r["path"] for r in cur.fetchall()]
        vectors: list[list[float]] = []
        for i in range(0, len(paths), 256):  # cached in Redis after the first build
            vectors.extend(await embeddings.embed(paths[i:i + 256], "document"))
        return cls(paths, np.asarray(vectors, dtype=np.float32))

    def candidates(self, query_vector: list[float], k: int = 20) -> list[str]:
        if not self.paths:
            return []
        sims = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [self.paths[i] for i in top]
