"""Re-export: the embedding service lives in common so the search service can use it in-process."""

from common.embedding_service import EmbeddingService, Kind

__all__ = ["EmbeddingService", "Kind"]
