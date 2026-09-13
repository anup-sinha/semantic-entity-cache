# Semantic Entity Cache for RAG

A configurable semantic cache for RAG applications that reduces false hits when two questions are textually similar but refer to different entities.

This project is designed for cases where a plain embedding similarity check is not enough. For example:

- Who won the 2022 FIFA World Cup?
- Who won the 2023 FIFA World Cup?

These questions have a high cosine similarity because the wording is nearly identical, but they are not the same request. The solution extracts entities from the incoming question and down-ranks matches when the entity set does not match, preventing incorrect cache reuse across years.

## Why this matters

In production RAG systems, semantic caching is useful only if it avoids incorrect reuse of answers across different entities, dates, products, locations, or other domain-specific values.

This repository provides a reusable pattern for:

- entity-aware cache matching
- configurable embedding providers
- configurable cache storage backends
- production-friendly extensibility for Redis or Azure AI Search

## Features

- Entity-aware semantic cache scoring
- Generic entity extraction strategy for reusable matching logic
- Configurable embedding providers for local, OpenAI, and Azure OpenAI
- Configurable cache stores for in-memory, Redis, and Azure AI Search
- Simple RAG service wrapper that can auto-generate embeddings
- Regression tests covering false-match and entity mismatch scenarios

## Architecture

```text
User Question
     |
     v
Embedding Provider --> Vector
     |
     v
Entity Extractor --> Entities
     |
     v
Semantic Cache --> Similarity + Entity Mismatch Penalty
     |
     v
Cache Store (Memory / Redis / Azure AI Search)
```

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── cache_store.py
│   ├── config.py
│   ├── embedding.py
│   ├── entity_extractor.py
│   ├── rag_service.py
│   └── semantic_cache.py
├── tests/
│   └── test_semantic_cache.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── demo.py
```

## Core logic

The semantic cache does four things:

1. Extracts named entities from the incoming question.
2. Computes cosine similarity between the embedding vectors.
3. Compares entity overlap between the query and cached question.
4. Applies a mismatch penalty when entities differ.

The effective score becomes:

`effective_score = raw_cosine_similarity - entity_mismatch_penalty`

This keeps valid cache hits fast while reducing incorrect reuse of answers across semantically similar but different queries.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Example usage

```python
from app.config import CacheConfig
from app.semantic_cache import EntityAwareSemanticCache

cache = EntityAwareSemanticCache(CacheConfig(similarity_threshold=0.80, mismatch_penalty=0.45))

question = "Who won the 2022 FIFA World Cup?"
answer = "Argentina won the 2022 FIFA World Cup."
embedding = [0.90, 0.18, 0.04]

cache.add(question, answer, embedding)

match = cache.get_best_match(
    question="Who won the 2023 FIFA World Cup?",
    query_vector=[0.88, 0.20, 0.05],
    candidate_questions=[question],
    candidate_vectors=[embedding],
)

print(match)
```

This example uses synthetic vectors in the ~0.88 similarity range to reflect the real-world case where very similar questions can still look close before entity-aware scoring is applied. The entity mismatch penalty separates `2022` from `2023`, so the cache does not incorrectly reuse the answer.

## Embedding providers

The cache does not generate embeddings itself. It expects a vector, and you can provide that vector from any embedding backend.

### Local fallback

Use the built-in local fallback when you do not want an external provider:

```python
from app.config import CacheConfig
from app.embedding import create_embedding_provider

config = CacheConfig(embedding_provider="local")
provider = create_embedding_provider(config)
vector = provider.embed_text("Who won the 2022 FIFA World Cup?")
```

### OpenAI

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-key
```

```python
from app.config import CacheConfig
from app.embedding import create_embedding_provider
from app.rag_service import RAGService

config = CacheConfig.from_environment()
provider = create_embedding_provider(config)

question = "Who won the 2022 FIFA World Cup?"
vector = provider.embed_text(question)

service = RAGService(config=config)
answer = service.answer(
    question,
    embedding=vector,
    answer_builder=lambda q: "Argentina won the 2022 FIFA World Cup.",
)

print(answer)
```

### Azure OpenAI

```env
EMBEDDING_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

```python
from app.config import CacheConfig
from app.embedding import create_embedding_provider
from app.rag_service import RAGService

config = CacheConfig.from_environment()
provider = create_embedding_provider(config)

question = "Who won the 2022 FIFA World Cup?"
vector = provider.embed_text(question)

service = RAGService(config=config)
answer = service.answer(
    question,
    embedding=vector,
    answer_builder=lambda q: "Argentina won the 2022 FIFA World Cup.",
)

print(answer)
```

## Entity extraction providers

The cache supports both a lightweight heuristic extractor and a spaCy-backed extractor for stronger entity recognition.

### Heuristic extractor (default)

This is the default and requires no extra dependency:

```env
ENTITY_EXTRACTION_PROVIDER=heuristic
```

### spaCy extractor

For more robust entity recognition in production, especially for product names, cities, and model-specific queries, enable spaCy:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Then set:

```env
ENTITY_EXTRACTION_PROVIDER=spacy
```

You can also instantiate it directly in code:

```python
from app.entity_extractor import EntityExtractor

extractor = EntityExtractor(provider="spacy")
entities = extractor.extract_entities("How much does the iPhone 15 cost?")
print(entities)
```

## Cache backends

The semantic cache can be backed by different storage implementations without changing the matching logic.

### In-memory

```env
CACHE_STORE_PROVIDER=memory
```

This is the default development setup and keeps entries in process memory.

### Redis

```env
CACHE_STORE_PROVIDER=redis
REDIS_URL=redis://localhost:6379/0
```

Use Redis for a shared, durable, production-friendly cache layer.

### Azure AI Search

```env
CACHE_STORE_PROVIDER=azure_ai_search
AZURE_AI_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<your-key>
AZURE_AI_SEARCH_INDEX_NAME=semantic-cache
```

This is useful when you want vector index semantics plus a hosted service for retrieval.

## Notes

- The entity extraction layer is designed to be reusable across entity types and domains.
- The cache logic stays the same regardless of the embedding provider or cache backend.
- For production RAG workloads, you can upgrade the extractor by using a real NER model such as spaCy or Azure AI Language.
- The project intentionally separates cache logic, embedding generation, and persistence so each layer can be replaced independently.
