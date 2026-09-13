from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Sequence

from .config import CacheConfig


def _build_cache_key(namespace: str, question: str) -> str:
    normalized = f"{namespace}:{question.strip()}".encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class BaseCacheStore(ABC):
    """Persistence abstraction for semantic-cache entries."""

    @abstractmethod
    def add(self, entry: Any) -> None:
        pass

    @abstractmethod
    def list(self) -> list[Any]:
        pass


class InMemoryCacheStore(BaseCacheStore):
    def __init__(self, config: CacheConfig):
        self.config = config
        self._entries: deque[Any] = deque(maxlen=self.config.max_cache_entries)

    def add(self, entry: Any) -> None:
        self._entries.append(entry)

    def list(self) -> list[Any]:
        return list(self._entries)


class RedisCacheStore(BaseCacheStore):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_url = config.redis_url or "redis://localhost:6379/0"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install redis-py to use Redis cache storage") from exc
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def add(self, entry: Any) -> None:
        payload = json.dumps({
            "question": entry.question,
            "answer": entry.answer,
            "vector": list(entry.vector),
            "entities": sorted(entry.entities),
        })
        key = f"{self.config.cache_namespace}:semantic-cache:{_build_cache_key(self.config.cache_namespace, entry.question)}"
        self.client.set(key, payload)

    def list(self) -> list[Any]:
        keys = self.client.keys(f"{self.config.cache_namespace}:semantic-cache:*")
        results: list[Any] = []
        for key in keys:
            payload = self.client.get(key)
            if not payload:
                continue
            item = json.loads(payload)
            results.append(type("CacheEntry", (), {
                "question": item["question"],
                "answer": item["answer"],
                "vector": item["vector"],
                "entities": set(item.get("entities", [])),
            })())
        return results


class AzureAISearchCacheStore(BaseCacheStore):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.endpoint = config.azure_ai_search_endpoint or ""
        self.api_key = config.azure_ai_search_api_key or ""
        self.index_name = config.azure_ai_search_index_name or "semantic-cache"
        self._client = None

        if not self.endpoint:
            raise ValueError("AZURE_AI_SEARCH_ENDPOINT is required when using Azure AI Search cache storage")
        if not self.api_key:
            raise ValueError("AZURE_AI_SEARCH_API_KEY is required when using Azure AI Search cache storage")

    def _build_cache_key(self, question: str) -> str:
        return _build_cache_key(self.config.cache_namespace, question)

    @property
    def client(self):
        if self._client is None:
            try:
                from azure.core.credentials import AzureKeyCredential  # type: ignore
                from azure.search.documents import SearchClient  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install azure-search-documents and azure-core to use Azure AI Search cache storage") from exc
            self._client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    def add(self, entry: Any) -> None:
        payload = {
            "id": self._build_cache_key(entry.question),
            "question": entry.question,
            "answer": entry.answer,
            "vector": list(entry.vector),
            "entities": sorted(entry.entities),
            "namespace": self.config.cache_namespace,
        }
        self.client.upload_documents(documents=[payload])

    def list(self) -> list[Any]:
        results: list[Any] = []
        filter_expression = f"namespace eq '{self.config.cache_namespace}'"
        for doc in self.client.search(
            search_text="*",
            filter=filter_expression,
            select=["question", "answer", "vector", "entities", "namespace"],
            top=1000,
        ):
            results.append(type("CacheEntry", (), {
                "question": doc.get("question"),
                "answer": doc.get("answer"),
                "vector": doc.get("vector", []),
                "entities": set(doc.get("entities", [])),
            })())
        return results


def create_cache_store(config: CacheConfig | None = None) -> BaseCacheStore:
    config = config or CacheConfig.from_environment()
    provider = (config.cache_store_provider or "memory").lower()

    if provider in {"memory", "in_memory", "default"}:
        return InMemoryCacheStore(config)
    if provider in {"redis", "redis_cache"}:
        return RedisCacheStore(config)
    if provider in {"azure_ai_search", "azure-ai-search", "azure-search", "azureai", "azure_search"}:
        return AzureAISearchCacheStore(config)

    raise ValueError(f"Unsupported cache store provider: {config.cache_store_provider}")
