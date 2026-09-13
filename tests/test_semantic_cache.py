import math

from app.cache_store import create_cache_store
from app.embedding import create_embedding_provider
from app.entity_extractor import EntityExtractor
from app.rag_service import RAGService
from app.semantic_cache import CacheConfig, EntityAwareSemanticCache


def _vector(values):
    magnitude = math.sqrt(sum(v * v for v in values))
    return [v / magnitude for v in values]


def test_same_entity_keeps_score_above_threshold():
    cache = EntityAwareSemanticCache(
        CacheConfig(
            similarity_threshold=0.75,
            mismatch_penalty=0.45,
            max_cache_entries=10,
        )
    )
    query = "What is the capital of India?"
    cached = "What is the capital of India?"

    q_vector = _vector([1.0, 0.0, 0.0])
    c_vector = _vector([1.0, 0.0, 0.0])

    match = cache.get_best_match(query, q_vector, [cached], [c_vector])
    assert match is not None
    assert match["effective_score"] >= 0.75


def test_different_entities_get_penalized():
    cache = EntityAwareSemanticCache(
        CacheConfig(
            similarity_threshold=0.75,
            mismatch_penalty=0.45,
            max_cache_entries=10,
        )
    )
    query = "What is the capital of India?"
    cached_question = "What is the capital of China?"

    q_vector = _vector([1.0, 0.0, 0.0])
    c_vector = _vector([0.98, 0.2, 0.0])

    match = cache.get_best_match(query, q_vector, [cached_question], [c_vector])
    assert match is not None
    assert match["effective_score"] < 0.75


def test_cached_question_without_entity_is_not_penalized():
    cache = EntityAwareSemanticCache(
        CacheConfig(
            similarity_threshold=0.85,
            mismatch_penalty=0.45,
            max_cache_entries=10,
        )
    )
    query = "Why is the sky blue?"
    cached_question = "What causes the sky to look blue?"

    q_vector = _vector([1.0, 0.0])
    c_vector = _vector([0.99, 0.12])

    match = cache.get_best_match(query, q_vector, [cached_question], [c_vector])
    assert match is not None
    assert match["effective_score"] >= 0.85


def test_generic_entity_extraction_detects_unlisted_names():
    extractor = EntityExtractor()
    entities = extractor.extract_entities("What is the capital of Paris?")
    assert "paris" in entities


def test_product_model_entities_are_not_collapsed():
    extractor = EntityExtractor()
    q1 = "How much does the iPhone 15 cost?"
    q2 = "How much does the iPhone 16 cost?"

    e1 = extractor.extract_entities(q1)
    e2 = extractor.extract_entities(q2)

    assert e1 != e2
    assert "iphone 15" in e1
    assert "iphone 16" in e2


def test_create_embedding_provider_for_openai(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CacheConfig.from_environment()
    provider = create_embedding_provider(config)
    assert provider.__class__.__name__ == "OpenAIEmbeddingProvider"


def test_create_embedding_provider_for_azure(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    config = CacheConfig.from_environment()
    provider = create_embedding_provider(config)
    assert provider.__class__.__name__ == "AzureOpenAIEmbeddingProvider"


def test_rag_service_generates_embedding_when_missing():
    config = CacheConfig(
        similarity_threshold=0.1,
        mismatch_penalty=0.4,
        max_cache_entries=10,
        embedding_provider="local",
    )
    cache = EntityAwareSemanticCache(config)
    service = RAGService(cache=cache, config=config)

    response = service.answer(
        "What is the capital of India?",
        answer_builder=lambda question: "New Delhi is the capital of India.",
    )

    assert response == "New Delhi is the capital of India."
    assert len(service.cache._entries) == 1


def test_create_cache_store_for_memory():
    config = CacheConfig(cache_store_provider="memory")
    store = create_cache_store(config)
    assert store.__class__.__name__ == "InMemoryCacheStore"


def test_create_cache_store_for_redis(monkeypatch):
    monkeypatch.setenv("CACHE_STORE_PROVIDER", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    config = CacheConfig.from_environment()
    store = create_cache_store(config)
    assert store.__class__.__name__ == "RedisCacheStore"


def test_create_cache_store_for_azure_ai_search(monkeypatch):
    monkeypatch.setenv("CACHE_STORE_PROVIDER", "azure_ai_search")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_INDEX_NAME", "semantic-cache")
    config = CacheConfig.from_environment()
    store = create_cache_store(config)
    assert store.__class__.__name__ == "AzureAISearchCacheStore"


def test_azure_cache_store_uses_secure_credential_and_deterministic_key(monkeypatch):
    import sys
    import types

    class FakeAzureKeyCredential:
        def __init__(self, key):
            self.key = key

    class FakeSearchClient:
        def __init__(self, endpoint, index_name, credential):
            self.endpoint = endpoint
            self.index_name = index_name
            self.credential = credential

    azure_module = types.ModuleType("azure")
    azure_search_module = types.ModuleType("azure.search")
    azure_search_documents_module = types.ModuleType("azure.search.documents")
    azure_core_module = types.ModuleType("azure.core")
    azure_core_credentials_module = types.ModuleType("azure.core.credentials")

    azure_search_documents_module.SearchClient = FakeSearchClient
    azure_core_credentials_module.AzureKeyCredential = FakeAzureKeyCredential

    azure_search_module.documents = azure_search_documents_module
    azure_core_module.credentials = azure_core_credentials_module
    azure_module.search = azure_search_module
    azure_module.core = azure_core_module

    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.search", azure_search_module)
    monkeypatch.setitem(sys.modules, "azure.search.documents", azure_search_documents_module)
    monkeypatch.setitem(sys.modules, "azure.core", azure_core_module)
    monkeypatch.setitem(sys.modules, "azure.core.credentials", azure_core_credentials_module)

    config = CacheConfig(
        cache_store_provider="azure_ai_search",
        azure_ai_search_endpoint="https://example.search.windows.net",
        azure_ai_search_api_key="test-key",
        azure_ai_search_index_name="semantic-cache",
        cache_namespace="demo",
    )
    store = create_cache_store(config)

    client = store.client
    assert isinstance(client.credential, FakeAzureKeyCredential)
    assert client.index_name == "semantic-cache"
    assert store._build_cache_key("Who won the 2022 FIFA World Cup?") == store._build_cache_key("Who won the 2022 FIFA World Cup?")
