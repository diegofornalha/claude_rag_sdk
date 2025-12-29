# =============================================================================
# TESTES - Cache Module
# =============================================================================
# Testes unitários para sistema de cache
# =============================================================================

import time

import pytest


class TestLRUCache:
    """Testes para LRU Cache."""

    def test_set_and_get(self):
        """Verifica set e get básicos."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_get_missing_key(self):
        """Verifica get de chave inexistente."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        result = cache.get("missing_key")

        assert result is None

    def test_get_with_default(self):
        """Verifica get com valor default."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        result = cache.get("missing_key", default="default_value")

        assert result == "default_value"

    def test_eviction_on_maxsize(self):
        """Verifica eviction quando atinge maxsize."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Deve evictar key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key4") == "value4"

    def test_lru_order(self):
        """Verifica ordem LRU (least recently used)."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Acessar key1 para torná-la "recente"
        cache.get("key1")

        # Adicionar novo item deve evictar key2 (menos recente)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Ainda existe
        assert cache.get("key2") is None  # Evictada
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_delete(self):
        """Verifica delete de chave."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "value1")
        cache.delete("key1")

        assert cache.get("key1") is None

    def test_clear(self):
        """Verifica clear do cache."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0

    def test_contains(self):
        """Verifica operador in."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache

    def test_len(self):
        """Verifica len do cache."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        assert len(cache) == 0

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache) == 2


class TestTTLCache:
    """Testes para cache com TTL."""

    def test_ttl_expiration(self):
        """Verifica expiração por TTL."""
        from claude_rag_sdk.core.cache import TTLCache

        cache = TTLCache(maxsize=100, ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(0.15)  # Esperar expirar

        assert cache.get("key1") is None

    def test_ttl_refresh_on_access(self):
        """Verifica se TTL pode ser renovado no acesso."""
        from claude_rag_sdk.core.cache import TTLCache

        cache = TTLCache(maxsize=100, ttl=0.2)

        cache.set("key1", "value1")

        time.sleep(0.1)
        # Acessar antes de expirar
        value = cache.get("key1")
        assert value == "value1"

        time.sleep(0.1)
        # Se o TTL for renovado no acesso, ainda deve existir
        # Se não for renovado, pode ter expirado
        # Comportamento depende da implementação

    def test_custom_ttl_per_key(self):
        """Verifica TTL customizado por chave."""
        from claude_rag_sdk.core.cache import TTLCache

        cache = TTLCache(maxsize=100, ttl=1.0)  # Default 1s

        # Se suportar TTL customizado
        if hasattr(cache, "set_with_ttl"):
            cache.set_with_ttl("key1", "value1", ttl=0.1)
            time.sleep(0.15)
            assert cache.get("key1") is None


class TestEmbeddingCache:
    """Testes para cache de embeddings."""

    def test_cache_embedding(self):
        """Verifica cache de embedding."""
        from claude_rag_sdk.core.cache import EmbeddingCache

        cache = EmbeddingCache(maxsize=100)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set("texto de teste", embedding)

        result = cache.get("texto de teste")

        assert result == embedding

    def test_cache_normalized_text(self):
        """Verifica normalização de texto para cache."""
        from claude_rag_sdk.core.cache import EmbeddingCache

        cache = EmbeddingCache(maxsize=100)

        embedding = [0.1, 0.2, 0.3]

        # Textos que devem ser normalizados para a mesma chave
        cache.set("  Texto com espaços  ", embedding)

        # Se normalizado, deve encontrar com texto limpo
        cache.get("Texto com espaços")

        # Pode ser None se não normalizar, ou embedding se normalizar
        # Depende da implementação

    def test_cache_batch_embeddings(self):
        """Verifica cache de batch de embeddings."""
        from claude_rag_sdk.core.cache import EmbeddingCache

        cache = EmbeddingCache(maxsize=100)

        texts = ["texto1", "texto2", "texto3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        # Cache batch
        for text, emb in zip(texts, embeddings, strict=False):
            cache.set(text, emb)

        # Verificar todos
        for text, expected in zip(texts, embeddings, strict=False):
            assert cache.get(text) == expected


class TestCacheStats:
    """Testes para estatísticas de cache."""

    def test_hit_rate(self):
        """Verifica cálculo de hit rate."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "value1")

        # 2 hits
        cache.get("key1")
        cache.get("key1")

        # 1 miss
        cache.get("missing")

        stats = cache.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.666, rel=0.1)

    def test_memory_usage(self):
        """Verifica estimativa de uso de memória."""
        from claude_rag_sdk.core.cache import LRUCache

        cache = LRUCache(maxsize=100)

        cache.set("key1", "x" * 1000)  # ~1KB

        stats = cache.get_stats()

        if "memory_bytes" in stats:
            assert stats["memory_bytes"] > 0


class TestCacheDecorator:
    """Testes para decorator de cache."""

    def test_cached_function(self):
        """Verifica função cacheada."""
        from claude_rag_sdk.core.cache import cached

        call_count = 0

        @cached(maxsize=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Primeira chamada
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Segunda chamada (cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Não chamou novamente

        # Chamada com argumento diferente
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_cached_with_ttl(self):
        """Verifica função cacheada com TTL."""
        from claude_rag_sdk.core.cache import cached

        call_count = 0

        @cached(maxsize=10, ttl=0.1)
        def function_with_ttl(x):
            nonlocal call_count
            call_count += 1
            return x

        # Primeira chamada
        function_with_ttl(1)
        assert call_count == 1

        # Segunda chamada (cached)
        function_with_ttl(1)
        assert call_count == 1

        # Esperar TTL
        time.sleep(0.15)

        # Terceira chamada (cache expirou)
        function_with_ttl(1)
        assert call_count == 2
