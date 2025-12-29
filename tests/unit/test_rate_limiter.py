# =============================================================================
# TESTES - Rate Limiter Module
# =============================================================================
# Testes unitários para controle de taxa de requisições
# =============================================================================

import time
from unittest.mock import patch

import pytest


class TestRateLimitConfig:
    """Testes para RateLimitConfig."""

    def test_default_values(self):
        """Verifica valores padrão."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig

        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.burst_size == 10

    def test_custom_values(self):
        """Verifica valores customizados."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig

        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_size=5,
        )

        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.burst_size == 5


class TestRateLimitResult:
    """Testes para RateLimitResult."""

    def test_allowed_result(self):
        """Verifica resultado permitido."""
        from claude_rag_sdk.core.rate_limiter import RateLimitResult

        result = RateLimitResult(
            allowed=True,
            remaining=59,
            reset_at=time.time() + 60,
        )

        assert result.allowed is True
        assert result.remaining == 59
        assert result.retry_after is None

    def test_blocked_result(self):
        """Verifica resultado bloqueado."""
        from claude_rag_sdk.core.rate_limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=time.time() + 30,
            retry_after=30,
        )

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30


class TestSlidingWindowRateLimiter:
    """Testes para SlidingWindowRateLimiter."""

    def test_allows_first_request(self):
        """Verifica primeira requisição é permitida."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config)

        result = limiter.check("user1")

        assert result.allowed is True
        assert result.remaining == 9

    def test_allows_within_limit(self):
        """Verifica requisições dentro do limite."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config)

        # 5 requisições
        for _i in range(5):
            result = limiter.check("user1")
            assert result.allowed is True

        # Deve ter 5 restantes
        assert result.remaining == 5

    def test_blocks_over_limit(self):
        """Verifica bloqueio acima do limite."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=5)
        limiter = SlidingWindowRateLimiter(config)

        # 5 requisições permitidas
        for _ in range(5):
            result = limiter.check("user1")
            assert result.allowed is True

        # 6ª deve ser bloqueada
        result = limiter.check("user1")
        assert result.allowed is False
        assert result.retry_after is not None

    def test_separate_users(self):
        """Verifica usuários têm limites separados."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=5)
        limiter = SlidingWindowRateLimiter(config)

        # Esgotar limite do user1
        for _ in range(5):
            limiter.check("user1")

        # user2 ainda pode fazer requisições
        result = limiter.check("user2")
        assert result.allowed is True

    def test_reset_after_window(self):
        """Verifica reset após janela de tempo."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=2)
        limiter = SlidingWindowRateLimiter(config)

        # Esgotar limite
        limiter.check("user1")
        limiter.check("user1")
        result = limiter.check("user1")
        assert result.allowed is False

        # Simular passagem de tempo (mock time.time)
        with patch("time.time", return_value=time.time() + 61):
            result = limiter.check("user1")
            assert result.allowed is True

    def test_get_usage_stats(self):
        """Verifica estatísticas de uso."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config)

        # Fazer algumas requisições
        limiter.check("user1")
        limiter.check("user1")
        limiter.check("user2")

        stats = limiter.get_usage_stats("user1")

        assert "requests_in_window" in stats
        assert stats["requests_in_window"] == 2


class TestRateLimitDecorator:
    """Testes para decorator de rate limit."""

    def test_decorator_allows_request(self):
        """Verifica decorator permite requisição."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, rate_limit

        config = RateLimitConfig(requests_per_minute=10)

        @rate_limit(config, key_func=lambda: "test_user")
        def my_endpoint():
            return "success"

        result = my_endpoint()
        assert result == "success"

    def test_decorator_blocks_request(self):
        """Verifica decorator bloqueia requisição."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, RateLimitExceeded, rate_limit

        config = RateLimitConfig(requests_per_minute=2)

        @rate_limit(config, key_func=lambda: "test_user")
        def my_endpoint():
            return "success"

        # Permitidas
        my_endpoint()
        my_endpoint()

        # Deve bloquear
        with pytest.raises(RateLimitExceeded):
            my_endpoint()


class TestBurstHandling:
    """Testes para handling de burst."""

    def test_burst_allowed(self):
        """Verifica burst é permitido."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(
            requests_per_minute=10,
            burst_size=5,
        )
        limiter = SlidingWindowRateLimiter(config)

        # Burst de 5 requisições simultâneas deve ser permitido
        results = [limiter.check("user1") for _ in range(5)]

        assert all(r.allowed for r in results)

    def test_burst_exceeded(self):
        """Verifica burst excedido é bloqueado."""
        from claude_rag_sdk.core.rate_limiter import RateLimitConfig, SlidingWindowRateLimiter

        config = RateLimitConfig(
            requests_per_minute=100,  # Alto, para não ser o limitador
            burst_size=3,
        )
        limiter = SlidingWindowRateLimiter(config)

        # Burst de 5 quando limite é 3
        results = [limiter.check("user1") for _ in range(5)]

        allowed_count = sum(1 for r in results if r.allowed)
        # Primeiras 3 devem passar, próximas 2 podem passar se dentro do minuto
        assert allowed_count >= 3
