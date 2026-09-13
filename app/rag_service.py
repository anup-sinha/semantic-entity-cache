from __future__ import annotations

from .config import CacheConfig
from .embedding import create_embedding_provider
from .semantic_cache import EntityAwareSemanticCache


class RAGService:
    """Minimal RAG wrapper that uses an entity-aware semantic cache."""

    def __init__(
        self,
        cache: EntityAwareSemanticCache | None = None,
        config: CacheConfig | None = None,
        embedding_provider=None,
    ):
        self.config = config or CacheConfig.from_environment()
        self.cache = cache or EntityAwareSemanticCache(self.config)
        self.embedding_provider = embedding_provider or create_embedding_provider(self.config)

    def answer(self, question: str, embedding: list[float] | None = None, answer_builder=None) -> str:
        if embedding is None:
            embedding = self.embedding_provider.embed_text(question)

        match = self.cache.get_best_match(question, embedding)
        if match is not None and match["effective_score"] >= self.cache.config.similarity_threshold:
            return match.get("answer", "")

        if answer_builder is None:
            raise ValueError("answer_builder is required when no cache hit is found")

        answer = answer_builder(question)
        self.cache.add(question, answer, embedding)
        return answer
