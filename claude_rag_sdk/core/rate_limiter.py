# =============================================================================
# RATE LIMITER - Controle de Taxa de Requisições
# =============================================================================
# Implementa sliding window rate limiting para proteger contra abuso
# =============================================================================

import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimitConfig:
    """Configuração de rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10  # Requisições permitidas em burst


@dataclass
class RateLimitResult:
    """Resultado da verificação de rate limit."""
    allowed: bool
    remaining: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[int] = None  # Segundos até poder tentar novamente

    def to_headers(self) -> dict[str, str]:
        """Retorna headers de rate limit."""
        headers = {
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class SlidingWindowRateLimiter:
    """
    Rate limiter com janela deslizante.

    Mais preciso que fixed window, evita bursts no limite da janela.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def _cleanup_old_requests(self, key: str, window_seconds: int) -> None:
        """Remove requisições antigas da janela."""
        cutoff = time.time() - window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def check(self, key: str) -> RateLimitResult:
        """
        Verifica se requisição é permitida.

        Args:
            key: Identificador (IP, user_id, API key, etc.)

        Returns:
            RateLimitResult com status
        """
        with self._lock:
            now = time.time()

            # Limpar requisições antigas (janela de 1 hora)
            self._cleanup_old_requests(key, 3600)

            requests = self._requests[key]

            # Verificar limite por minuto
            minute_ago = now - 60
            requests_last_minute = sum(1 for t in requests if t > minute_ago)

            if requests_last_minute >= self.config.requests_per_minute:
                # Calcular quando poderá tentar novamente
                oldest_in_minute = min((t for t in requests if t > minute_ago), default=now)
                retry_after = int(60 - (now - oldest_in_minute)) + 1

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest_in_minute + 60,
                    retry_after=retry_after,
                )

            # Verificar limite por hora
            hour_ago = now - 3600
            requests_last_hour = sum(1 for t in requests if t > hour_ago)

            if requests_last_hour >= self.config.requests_per_hour:
                oldest_in_hour = min((t for t in requests if t > hour_ago), default=now)
                retry_after = int(3600 - (now - oldest_in_hour)) + 1

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest_in_hour + 3600,
                    retry_after=retry_after,
                )

            # Permitido - registrar requisição
            requests.append(now)

            remaining = min(
                self.config.requests_per_minute - requests_last_minute - 1,
                self.config.requests_per_hour - requests_last_hour - 1,
            )

            return RateLimitResult(
                allowed=True,
                remaining=max(0, remaining),
                reset_at=now + 60,
            )

    def reset(self, key: str) -> None:
        """Reseta contadores para uma chave."""
        with self._lock:
            if key in self._requests:
                del self._requests[key]

    def get_stats(self, key: str) -> dict:
        """Retorna estatísticas para uma chave."""
        with self._lock:
            now = time.time()
            requests = self._requests.get(key, [])

            minute_ago = now - 60
            hour_ago = now - 3600

            return {
                "requests_last_minute": sum(1 for t in requests if t > minute_ago),
                "requests_last_hour": sum(1 for t in requests if t > hour_ago),
                "limit_per_minute": self.config.requests_per_minute,
                "limit_per_hour": self.config.requests_per_hour,
            }


class TokenBucketRateLimiter:
    """
    Rate limiter com token bucket.

    Permite bursts controlados enquanto mantém taxa média.
    """

    def __init__(
        self,
        rate: float = 1.0,  # Tokens por segundo
        capacity: int = 10,  # Capacidade máxima do bucket
    ):
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = threading.RLock()

    def _get_tokens(self, key: str) -> float:
        """Retorna tokens disponíveis, atualizando o bucket."""
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = (self.capacity, now)
            return self.capacity

        tokens, last_update = self._buckets[key]
        elapsed = now - last_update

        # Adicionar tokens baseado no tempo decorrido
        new_tokens = min(self.capacity, tokens + elapsed * self.rate)
        self._buckets[key] = (new_tokens, now)

        return new_tokens

    def check(self, key: str, tokens_required: int = 1) -> RateLimitResult:
        """
        Verifica se requisição é permitida.

        Args:
            key: Identificador
            tokens_required: Tokens necessários (peso da requisição)
        """
        with self._lock:
            available = self._get_tokens(key)

            if available >= tokens_required:
                # Consumir tokens
                current_tokens, last_update = self._buckets[key]
                self._buckets[key] = (current_tokens - tokens_required, last_update)

                return RateLimitResult(
                    allowed=True,
                    remaining=int(current_tokens - tokens_required),
                    reset_at=time.time() + (self.capacity - current_tokens + tokens_required) / self.rate,
                )
            else:
                # Calcular tempo até ter tokens suficientes
                tokens_needed = tokens_required - available
                retry_after = int(tokens_needed / self.rate) + 1

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=time.time() + retry_after,
                    retry_after=retry_after,
                )


# Instância global
_rate_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Retorna rate limiter global."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SlidingWindowRateLimiter()
    return _rate_limiter


def check_rate_limit(key: str) -> RateLimitResult:
    """Verifica rate limit para uma chave."""
    return get_rate_limiter().check(key)


# Aliases para compatibilidade com server.py
def get_limiter():
    """Retorna limiter para uso com decorator (compatível com slowapi)."""
    return get_rate_limiter()


# Tenta usar slowapi se disponível
SLOWAPI_AVAILABLE = False
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    SLOWAPI_AVAILABLE = True

    _slowapi_limiter = Limiter(key_func=get_remote_address)

    def get_limiter():
        return _slowapi_limiter
except ImportError:
    pass


# Rate limits padrão para endpoints
RATE_LIMITS = {
    "chat": "30/minute",
    "chat_stream": "20/minute",
    "default": "60/minute",
}


def get_client_ip(request) -> str:
    """Extrai IP do cliente da requisição."""
    # Tentar headers de proxy reverso
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback para IP direto
    return request.client.host if request.client else "unknown"


if __name__ == "__main__":
    print("=== Teste de Rate Limiter ===\n")

    # Teste sliding window
    limiter = SlidingWindowRateLimiter(RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=100,
    ))

    print("--- Sliding Window ---")
    for i in range(8):
        result = limiter.check("user123")
        status = "✓" if result.allowed else "✗"
        print(f"  Request {i+1}: {status} (remaining: {result.remaining})")
        if not result.allowed:
            print(f"    Retry after: {result.retry_after}s")

    print(f"\nStats: {limiter.get_stats('user123')}")

    # Teste token bucket
    print("\n--- Token Bucket ---")
    bucket = TokenBucketRateLimiter(rate=2.0, capacity=5)

    for i in range(8):
        result = bucket.check("api_key_abc")
        status = "✓" if result.allowed else "✗"
        print(f"  Request {i+1}: {status} (remaining: {result.remaining})")
        if not result.allowed:
            print(f"    Retry after: {result.retry_after}s")
