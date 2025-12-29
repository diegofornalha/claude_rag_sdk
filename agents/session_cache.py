"""Session Cache - Cache de sessões para melhor performance.

Este módulo fornece:
- Cache LRU de instâncias AgentFS
- Cache de histórico de conversas
- TTL automático para evitar dados stale
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agentfs_sdk import AgentFS, AgentFSOptions

from claude_rag_sdk.core.logger import get_logger

logger = get_logger("session_cache")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheEntry:
    """Entrada no cache com TTL."""

    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: float = 300  # 5 minutos padrão

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self):
        """Atualiza último acesso."""
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Estatísticas do cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# =============================================================================
# LRU Cache
# =============================================================================


class LRUCache:
    """Cache LRU thread-safe com TTL."""

    def __init__(self, max_size: int = 100, default_ttl: float = 300):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Obtém valor do cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Move para o final (mais recente)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None):
        """Define valor no cache."""
        async with self._lock:
            # Evict se necessário
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=ttl or self._default_ttl,
            )

    async def delete(self, key: str) -> bool:
        """Remove entrada do cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self):
        """Limpa todo o cache."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict:
        """Retorna estatísticas do cache."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "hit_rate": round(self._stats.hit_rate, 4),
        }


# =============================================================================
# Session Cache
# =============================================================================


class SessionCache:
    """Cache especializado para sessões de chat.

    Mantém em cache:
    - Histórico de conversas (evita ler do KV)
    - Metadados da sessão
    """

    def __init__(self, max_sessions: int = 50, history_ttl: float = 600):
        # Cache de histórico por sessão
        self._history_cache = LRUCache(max_size=max_sessions, default_ttl=history_ttl)
        # Cache de metadados por sessão
        self._metadata_cache = LRUCache(max_size=max_sessions, default_ttl=history_ttl)

    async def get_history(self, session_id: str) -> list | None:
        """Obtém histórico de conversa do cache."""
        return await self._history_cache.get(session_id)

    async def set_history(self, session_id: str, history: list, ttl: float | None = None):
        """Salva histórico de conversa no cache."""
        await self._history_cache.set(session_id, history, ttl)

    async def append_to_history(self, session_id: str, user_msg: str, assistant_msg: str):
        """Adiciona mensagens ao histórico em cache."""
        history = await self.get_history(session_id) or []
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        await self.set_history(session_id, history[-100:])  # Limitar a 100 mensagens

    async def get_metadata(self, session_id: str) -> dict | None:
        """Obtém metadados da sessão do cache."""
        return await self._metadata_cache.get(session_id)

    async def set_metadata(self, session_id: str, metadata: dict, ttl: float | None = None):
        """Salva metadados da sessão no cache."""
        await self._metadata_cache.set(session_id, metadata, ttl)

    async def invalidate(self, session_id: str):
        """Invalida cache de uma sessão."""
        await self._history_cache.delete(session_id)
        await self._metadata_cache.delete(session_id)

    async def clear(self):
        """Limpa todo o cache."""
        await self._history_cache.clear()
        await self._metadata_cache.clear()

    def get_stats(self) -> dict:
        """Retorna estatísticas do cache."""
        return {
            "history_cache": self._history_cache.get_stats(),
            "metadata_cache": self._metadata_cache.get_stats(),
        }


# =============================================================================
# AgentFS Pool
# =============================================================================


class AgentFSPool:
    """Pool de conexões AgentFS para reutilização.

    Mantém instâncias AgentFS abertas para evitar overhead de abrir/fechar.
    """

    def __init__(self, max_connections: int = 20, idle_timeout: float = 300):
        self._pool: dict[str, tuple[AgentFS, float]] = {}
        self._max_connections = max_connections
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, session_id: str) -> AgentFS:
        """Obtém ou cria instância AgentFS."""
        async with self._lock:
            # Limpar conexões idle
            await self._cleanup_idle()

            if session_id in self._pool:
                afs, _ = self._pool[session_id]
                self._pool[session_id] = (afs, time.time())
                self._stats.hits += 1
                return afs

            # Criar nova conexão
            self._stats.misses += 1

            # Evict se necessário
            if len(self._pool) >= self._max_connections:
                await self._evict_oldest()

            afs = await AgentFS.open(AgentFSOptions(id=session_id))
            self._pool[session_id] = (afs, time.time())
            return afs

    async def release(self, session_id: str):
        """Marca conexão como disponível (não fecha, apenas atualiza timestamp)."""
        async with self._lock:
            if session_id in self._pool:
                afs, _ = self._pool[session_id]
                self._pool[session_id] = (afs, time.time())

    async def close(self, session_id: str):
        """Fecha e remove conexão do pool."""
        async with self._lock:
            if session_id in self._pool:
                afs, _ = self._pool[session_id]
                await afs.close()
                del self._pool[session_id]

    async def _cleanup_idle(self):
        """Remove conexões idle expiradas."""
        now = time.time()
        to_remove = [
            sid for sid, (_, last_used) in self._pool.items()
            if now - last_used > self._idle_timeout
        ]
        for sid in to_remove:
            afs, _ = self._pool[sid]
            try:
                await afs.close()
            except Exception:
                pass
            del self._pool[sid]
            self._stats.evictions += 1

    async def _evict_oldest(self):
        """Remove conexão mais antiga."""
        if not self._pool:
            return

        oldest_sid = min(self._pool.keys(), key=lambda k: self._pool[k][1])
        afs, _ = self._pool[oldest_sid]
        try:
            await afs.close()
        except Exception:
            pass
        del self._pool[oldest_sid]
        self._stats.evictions += 1

    async def close_all(self):
        """Fecha todas as conexões."""
        async with self._lock:
            for sid, (afs, _) in list(self._pool.items()):
                try:
                    await afs.close()
                except Exception:
                    pass
            self._pool.clear()

    def get_stats(self) -> dict:
        """Retorna estatísticas do pool."""
        return {
            "active_connections": len(self._pool),
            "max_connections": self._max_connections,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "hit_rate": round(self._stats.hit_rate, 4),
        }


# =============================================================================
# Singleton Instances
# =============================================================================

_session_cache: SessionCache | None = None
_agentfs_pool: AgentFSPool | None = None


def get_session_cache() -> SessionCache:
    """Retorna instância singleton do SessionCache."""
    global _session_cache
    if _session_cache is None:
        _session_cache = SessionCache()
    return _session_cache


def get_agentfs_pool() -> AgentFSPool:
    """Retorna instância singleton do AgentFSPool."""
    global _agentfs_pool
    if _agentfs_pool is None:
        _agentfs_pool = AgentFSPool()
    return _agentfs_pool
