from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    opensearch_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://search:search@localhost:5432/search"
    kafka_bootstrap: str = "localhost:9092"
    s3_endpoint: str = "http://localhost:8333"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio12345"
    genai_gateway_url: str = "http://localhost:8001"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Search tuning (Fig 5.14: confidence > 0.6 skips the fallback path)
    confidence_threshold: float = 0.6
    # bge-small: 0.90 still merged 'usb c cable' with 'usb c charger'; >= 0.91 kept the product on sampled logs
    warm_tier_min_similarity: float = 0.91
    results_cache_ttl_s: int = 600
    fallback_enabled: bool = True   # LLM rewrite for low-confidence queries (Phase 4)


@lru_cache
def get_settings() -> Settings:
    return Settings()
