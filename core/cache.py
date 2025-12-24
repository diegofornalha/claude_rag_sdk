# =============================================================================
# CACHE - Cache LRU para RAG Agent
# =============================================================================
# Cache de embeddings e respostas para otimizar performance
# =============================================================================

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar, Generic, Callable

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Entrada do cache."""
    key: str
    value: T
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Verifica se entrada expirou."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class CacheStats:
    """Estatísticas do cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Taxa de acerto."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """
    Cache LRU thread-safe.

    Features:
    - Eviction por LRU quando cheio
    - TTL opcional por entrada
    - Estatísticas de uso
    - Thread-safe
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,  # TTL padrão em segundos
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)

    def _estimate_size(self, value: Any) -> int:
        """Estima tamanho em bytes de um valor."""
        try:
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            return 0

    def get(self, key: str) -> Optional[T]:
        """
        Obtém valor do cache.

        Args:
            key: Chave do cache

        Returns:
            Valor ou None se não encontrado/expirado
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Verificar expiração
            if entry.is_expired():
                del self._cache[key]
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes
                self._stats.misses += 1
                return None

            # Atualizar acesso (move para o final = mais recente)
            self._cache.move_to_end(key)
            entry.accessed_at = datetime.now(timezone.utc)
            entry.access_count += 1
            self._stats.hits += 1

            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Define valor no cache.

        Args:
            key: Chave do cache
            value: Valor a armazenar
            ttl: TTL em segundos (None usa default)
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            size_bytes = self._estimate_size(value)
            is_update = key in self._cache

            # Se já existe, remover primeiro para não contar no limite
            if is_update:
                old_entry = self._cache[key]
                self._stats.memory_bytes -= old_entry.size_bytes
                del self._cache[key]

            # Verificar se precisa evictar (agora não conta a entrada sendo atualizada)
            while len(self._cache) >= self.max_size:
                self._evict_oldest()

            # Criar entrada
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                ttl_seconds=ttl if ttl is not None else self.default_ttl,
                size_bytes=size_bytes,
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._stats.size = len(self._cache)
            self._stats.memory_bytes += size_bytes

    def _evict_oldest(self) -> None:
        """Remove entrada mais antiga (LRU)."""
        if not self._cache:
            return

        oldest_key = next(iter(self._cache))
        old_entry = self._cache[oldest_key]
        del self._cache[oldest_key]
        self._stats.evictions += 1
        self._stats.size -= 1
        self._stats.memory_bytes -= old_entry.size_bytes

    def delete(self, key: str) -> bool:
        """
        Remove entrada do cache.

        Args:
            key: Chave a remover

        Returns:
            True se removido, False se não existia
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes
                return True
            return False

    def clear(self) -> None:
        """Limpa todo o cache."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
            self._stats.memory_bytes = 0

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """
        Obtém do cache ou cria usando factory.

        Args:
            key: Chave do cache
            factory: Função para criar valor se não existir
            ttl: TTL em segundos

        Returns:
            Valor do cache ou criado pela factory
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    @property
    def stats(self) -> CacheStats:
        """Retorna estatísticas do cache."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
                memory_bytes=self._stats.memory_bytes,
            )

    def cleanup_expired(self) -> int:
        """
        Remove entradas expiradas.

        Returns:
            Número de entradas removidas
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes

            return len(expired_keys)


class EmbeddingCache:
    """Cache especializado para embeddings."""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self._cache = LRUCache[list[float]](max_size=max_size, default_ttl=ttl)

    def _make_key(self, text: str) -> str:
        """Cria chave de cache para texto."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def get(self, text: str) -> Optional[list[float]]:
        """Obtém embedding do cache."""
        key = self._make_key(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: list[float]) -> None:
        """Armazena embedding no cache."""
        key = self._make_key(text)
        self._cache.set(key, embedding)

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], list[float]],
    ) -> list[float]:
        """Obtém do cache ou computa embedding."""
        key = self._make_key(text)
        return self._cache.get_or_set(key, lambda: compute_fn(text))

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats


class ResponseCache:
    """Cache de respostas do RAG."""

    def __init__(self, max_size: int = 1000, ttl: int = 300):  # 5 min default
        self._cache = LRUCache[dict](max_size=max_size, default_ttl=ttl)

    def _make_key(self, query: str, top_k: int, **kwargs) -> str:
        """Cria chave de cache para query incluindo parametros extras."""
        # Incluir parametros extras na chave (ex: use_reranking)
        extra = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        content = f"{query}:{top_k}:{extra}" if extra else f"{query}:{top_k}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def get(self, query: str, top_k: int = 5, **kwargs) -> Optional[dict]:
        """Obtém resposta do cache."""
        key = self._make_key(query, top_k, **kwargs)
        return self._cache.get(key)

    def set(self, query: str, top_k: int, response: dict, **kwargs) -> None:
        """Armazena resposta no cache."""
        key = self._make_key(query, top_k, **kwargs)
        self._cache.set(key, response)

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats


# Instâncias globais
_embedding_cache: Optional[EmbeddingCache] = None
_response_cache: Optional[ResponseCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Retorna cache global de embeddings."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_response_cache() -> ResponseCache:
    """Retorna cache global de respostas."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


if __name__ == "__main__":
    print("=== Teste de Cache LRU ===\n")

    # Teste básico
    cache: LRUCache[str] = LRUCache(max_size=3)

    cache.set("a", "value_a")
    cache.set("b", "value_b")
    cache.set("c", "value_c")

    print(f"Get 'a': {cache.get('a')}")  # Move 'a' para o final
    print(f"Get 'b': {cache.get('b')}")

    cache.set("d", "value_d")  # Evicta 'c' (LRU)

    print(f"Get 'c': {cache.get('c')}")  # None - evictado
    print(f"Get 'd': {cache.get('d')}")

    print(f"\nStats: {cache.stats}")
    print(f"Hit rate: {cache.stats.hit_rate:.2%}")

    # Teste com TTL
    print("\n--- Teste com TTL ---")
    ttl_cache: LRUCache[str] = LRUCache(max_size=10, default_ttl=1)

    ttl_cache.set("x", "value_x")
    print(f"Get 'x' (fresh): {ttl_cache.get('x')}")

    time.sleep(1.5)
    print(f"Get 'x' (expired): {ttl_cache.get('x')}")

    # Teste de EmbeddingCache
    print("\n--- Teste de EmbeddingCache ---")
    emb_cache = EmbeddingCache(max_size=100, ttl=60)

    emb_cache.set("hello world", [0.1, 0.2, 0.3])
    result = emb_cache.get("hello world")
    print(f"Embedding cached: {result}")

    result = emb_cache.get("not cached")
    print(f"Not cached: {result}")

    print(f"\nEmbedding cache stats: hits={emb_cache.stats.hits}, misses={emb_cache.stats.misses}")

    # Teste de get_or_compute
    print("\n--- Teste get_or_compute ---")

    def fake_compute(text: str) -> list[float]:
        print(f"  Computing embedding for: {text}")
        return [0.5] * 384

    emb1 = emb_cache.get_or_compute("test text", fake_compute)
    print(f"First call - computed")

    emb2 = emb_cache.get_or_compute("test text", fake_compute)
    print(f"Second call - from cache")

    print(f"\nFinal stats: hits={emb_cache.stats.hits}, misses={emb_cache.stats.misses}")
