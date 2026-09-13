from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)


@dataclass
class CacheConfig:
    similarity_threshold: float = 0.80
    mismatch_penalty: float = 0.45
    max_cache_entries: int = 1000
    cache_namespace: str = "default"
    cache_store_provider: str = "memory"
    redis_url: str | None = None
    azure_ai_search_endpoint: str | None = None
    azure_ai_search_api_key: str | None = None
    azure_ai_search_index_name: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    entity_extraction_provider: str = "heuristic"
    openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_embedding_deployment: str | None = None

    @classmethod
    def from_environment(cls) -> "CacheConfig":
        return cls(
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.80")),
            mismatch_penalty=float(os.getenv("CACHE_MISMATCH_PENALTY", "0.45")),
            max_cache_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
            cache_namespace=os.getenv("CACHE_NAMESPACE", "default"),
            cache_store_provider=os.getenv("CACHE_STORE_PROVIDER", "memory"),
            redis_url=os.getenv("REDIS_URL"),
            azure_ai_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            azure_ai_search_api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
            azure_ai_search_index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "semantic-cache"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            entity_extraction_provider=os.getenv("ENTITY_EXTRACTION_PROVIDER", "heuristic"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
