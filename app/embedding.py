from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod

from .config import CacheConfig


class BaseEmbeddingProvider(ABC):
    """Shared interface for all embedding backends used by the cache."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return a numeric embedding vector for the input text."""


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic fallback embedding provider for local development/testing.

    This is intentionally simple and does not require an external embedding model.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        for token in tokens:
            idx = sum(ord(ch) for ch in token) % self.dimension
            vector[idx] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return [0.0] * self.dimension

        return [value / magnitude for value in vector]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: CacheConfig):
        self.model = config.embedding_model or "text-embedding-3-small"
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The 'openai' package is required for OpenAI embeddings. Install it with 'pip install openai'."
                ) from exc
            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def embed_text(self, text: str) -> list[float]:
        response = self._get_client().embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: CacheConfig):
        self.model = config.azure_openai_embedding_deployment or config.embedding_model
        self.api_key = config.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = config.azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = config.azure_openai_api_version
        self.client = None

        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required when EMBEDDING_PROVIDER=azure")
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required when EMBEDDING_PROVIDER=azure")
        if not self.model:
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required when EMBEDDING_PROVIDER=azure")

    def _get_client(self):
        if self.client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The 'openai' package is required for Azure OpenAI embeddings. Install it with 'pip install openai'."
                ) from exc
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        return self.client

    def embed_text(self, text: str) -> list[float]:
        response = self._get_client().embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


def create_embedding_provider(config: CacheConfig | None = None) -> BaseEmbeddingProvider:
    config = config or CacheConfig.from_environment()
    provider_name = (config.embedding_provider or "local").lower()

    if provider_name in {"local", "default", "fallback"}:
        return LocalEmbeddingProvider()
    if provider_name in {"openai", "open_ai"}:
        return OpenAIEmbeddingProvider(config)
    if provider_name in {"azure", "azure-openai", "azure_openai"}:
        return AzureOpenAIEmbeddingProvider(config)

    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
