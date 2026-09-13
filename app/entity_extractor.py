from __future__ import annotations

import re


class EntityExtractor:
    """Generic entity extractor for names, places, organizations, or other typed entities.

    The extractor intentionally avoids a hardcoded list of countries or states. It relies on
    generic heuristics and, when available, a spaCy NER model if configured.
    """

    QUESTION_WORDS = {
        "what",
        "why",
        "when",
        "where",
        "who",
        "how",
        "which",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "and",
        "or",
        "with",
        "from",
        "by",
        "about",
    }

    NON_ENTITY_WORDS = {
        "sky",
        "blue",
        "red",
        "green",
        "yellow",
        "color",
        "colour",
        "look",
        "looks",
        "looked",
        "cause",
        "causes",
        "caused",
        "reason",
        "because",
        "why",
        "answer",
        "question",
        "problem",
        "issue",
        "value",
        "type",
        "kind",
        "thing",
        "things",
        "data",
        "results",
        "example",
        "examples",
        "idea",
        "ideas",
        "concept",
    }

    def __init__(self, provider: str = "heuristic"):
        self.provider = (provider or "heuristic").lower()
        self._spacy_model = None
        if self.provider == "spacy":
            try:
                import spacy  # type: ignore

                self._spacy_model = spacy.blank("en")
            except Exception:
                self._spacy_model = None

    def extract_entities(self, text: str) -> set[str]:
        if self.provider == "spacy" and self._spacy_model is not None:
            entities = self._extract_with_spacy(text)
            if entities:
                return entities

        return self._extract_generic(text)

    def _extract_with_spacy(self, text: str) -> set[str]:
        if self._spacy_model is None:
            return set()

        doc = self._spacy_model(text)
        entities = set()
        for ent in doc.ents:
            value = self._normalize_text(ent.text)
            if value and value not in self.QUESTION_WORDS:
                entities.add(value)
        return entities

    def _extract_generic(self, text: str) -> set[str]:
        candidates = set()

        title_case_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z][a-z]+\b", text)
        for match in title_case_matches:
            normalized = self._normalize_text(match)
            if normalized and normalized not in self.QUESTION_WORDS:
                candidates.add(normalized)

        lowercase_matches = re.findall(
            r"\b(?:of|in|at|for|from|to|on|by)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,2})\b",
            text,
        )
        for match in lowercase_matches:
            normalized = self._normalize_text(match)
            if not normalized or normalized in self.QUESTION_WORDS:
                continue

            tokens = normalized.split()
            if len(tokens) > 2:
                continue
            if any(token in self.NON_ENTITY_WORDS for token in tokens):
                continue
            if any(token in {"look", "cause", "because"} for token in tokens):
                continue

            candidates.add(normalized)

        numeric_entity_matches = re.findall(
            r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,2}\s+\d{1,6})\b",
            text,
        )
        for match in numeric_entity_matches:
            normalized = self._normalize_text(match)
            if not normalized:
                continue

            tokens = normalized.split()
            while tokens and tokens[0] in self.QUESTION_WORDS:
                tokens = tokens[1:]

            if len(tokens) < 2:
                continue
            if any(token in self.NON_ENTITY_WORDS for token in tokens[:-1]):
                continue
            if not tokens[-1].isdigit():
                continue

            candidates.add(" ".join(tokens))

        return {candidate for candidate in candidates if candidate}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def extract_entities(text: str) -> set[str]:
    return EntityExtractor().extract_entities(text)
