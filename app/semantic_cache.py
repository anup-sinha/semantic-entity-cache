from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from .cache_store import create_cache_store
from .config import CacheConfig
from .entity_extractor import EntityExtractor


@dataclass
class CacheEntry:
    question: str
    answer: str
    vector: list[float]
    entities: set[str]


class EntityAwareSemanticCache:
    """Cache that penalizes mismatched entities before accepting a semantic match."""

    def __init__(self, config: CacheConfig | None = None, store=None):
        self.config = config or CacheConfig()
        self.extractor = EntityExtractor()
        self.store = store or create_cache_store(self.config)
        self._entries = getattr(self.store, "_entries", [])

    def add(self, question: str, answer: str, vector: Sequence[float]) -> None:
        entry = CacheEntry(
            question=question,
            answer=answer,
            vector=list(vector),
            entities=self.extractor.extract_entities(question),
        )
        self.store.add(entry)
        self._entries = self.store.list()

    def get_best_match(
        self,
        question: str,
        query_vector: Sequence[float],
        candidate_questions: Sequence[str] | None = None,
        candidate_vectors: Sequence[Sequence[float]] | None = None,
    ) -> dict | None:
        if candidate_questions is None:
            candidates = list(self.store.list())
            best_score = None
            best_match = None

            for entry in candidates:
                raw_score = cosine_similarity(query_vector, entry.vector)
                effective_score, entity_overlap = self._effective_score(question, entry.question, raw_score)
                candidate = {
                    "question": entry.question,
                    "answer": entry.answer,
                    "raw_score": raw_score,
                    "effective_score": effective_score,
                    "entity_overlap": entity_overlap,
                }
                if best_score is None or candidate["effective_score"] > best_score:
                    best_score = candidate["effective_score"]
                    best_match = candidate

            return best_match

        if candidate_vectors is None:
            raise ValueError("candidate_vectors must be provided when candidate_questions is used")

        best_score = None
        best_match = None

        for cached_question, cached_vector in zip(candidate_questions, candidate_vectors):
            raw_score = cosine_similarity(query_vector, cached_vector)
            effective_score, entity_overlap = self._effective_score(question, cached_question, raw_score)
            candidate = {
                "question": cached_question,
                "raw_score": raw_score,
                "effective_score": effective_score,
                "entity_overlap": entity_overlap,
            }
            if best_score is None or candidate["effective_score"] > best_score:
                best_score = candidate["effective_score"]
                best_match = candidate

        return best_match

    def _effective_score(
        self,
        query_text: str,
        cached_question: str,
        raw_score: float,
    ) -> tuple[float, float]:
        query_entities = self.extractor.extract_entities(query_text)
        cached_entities = self.extractor.extract_entities(cached_question)

        if not query_entities and not cached_entities:
            return raw_score, 1.0

        if not query_entities or not cached_entities:
            return max(0.0, raw_score - (self.config.mismatch_penalty * 0.5)), 0.0

        overlap = len(query_entities & cached_entities)
        union = len(query_entities | cached_entities)
        entity_overlap = overlap / union if union else 0.0

        penalty = self.config.mismatch_penalty * (1.0 - entity_overlap)
        effective = max(0.0, raw_score - penalty)
        return effective, entity_overlap


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    mag_a = math.sqrt(sum(a * a for a in vector_a))
    mag_b = math.sqrt(sum(b * b for b in vector_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)
