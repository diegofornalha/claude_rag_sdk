# =============================================================================
# TESTES - Circuit Breaker Module
# =============================================================================
# Testes unitários para resiliência contra falhas em cascata
# =============================================================================

import pytest
import time
from unittest.mock import patch, MagicMock


class TestCircuitState:
    """Testes para CircuitState enum."""

    def test_states_exist(self):
        """Verifica todos os estados existem."""
        from claude_rag_sdk.core.circuit_breaker import CircuitState

        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Testes para CircuitBreaker."""

    def test_initial_state_closed(self):
        """Verifica estado inicial é CLOSED."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        assert cb.state.value == "closed"

    def test_success_keeps_closed(self):
        """Verifica sucesso mantém circuito fechado."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_success()
        cb.record_success()

        assert cb.state.value == "closed"

    def test_failures_open_circuit(self):
        """Verifica falhas consecutivas abrem circuito."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_blocks_calls(self):
        """Verifica circuito aberto bloqueia chamadas."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=2)

        # Abrir circuito
        cb.record_failure()
        cb.record_failure()

        # Verificar se bloqueia
        assert cb.can_execute() is False

    def test_half_open_after_timeout(self):
        """Verifica transição para HALF_OPEN após timeout."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Abrir circuito
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simular passagem de tempo
        time.sleep(1.1)

        # Deve transicionar para half-open ao verificar
        can_exec = cb.can_execute()
        assert can_exec is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """Verifica sucesso em HALF_OPEN fecha circuito."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Abrir e esperar
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.2)

        # Transicionar para half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Sucesso fecha
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Verifica falha em HALF_OPEN reabre circuito."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Abrir e esperar
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.2)

        # Transicionar para half-open
        cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Falha reabre
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Verifica sucesso reseta contador de falhas."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)

        # 2 falhas
        cb.record_failure()
        cb.record_failure()

        # 1 sucesso
        cb.record_success()

        # Mais 2 falhas não devem abrir (reset)
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.CLOSED

    def test_get_stats(self):
        """Verifica estatísticas do circuit breaker."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        stats = cb.get_stats()

        assert stats["state"] == "closed"
        assert stats["success_count"] >= 2
        assert stats["failure_count"] >= 1


class TestCircuitBreakerDecorator:
    """Testes para decorator do circuit breaker."""

    def test_decorator_allows_success(self):
        """Verifica decorator permite função com sucesso."""
        from claude_rag_sdk.core.circuit_breaker import circuit_breaker

        @circuit_breaker(failure_threshold=3)
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_decorator_records_failure(self):
        """Verifica decorator registra falha."""
        from claude_rag_sdk.core.circuit_breaker import circuit_breaker, CircuitOpenError

        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=10)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        # Primeiras 2 chamadas falham normalmente
        with pytest.raises(ValueError):
            failing_function()
        with pytest.raises(ValueError):
            failing_function()

        # Terceira deve ser bloqueada pelo circuit breaker
        with pytest.raises(CircuitOpenError):
            failing_function()

        # A função só foi chamada 2 vezes (não 3)
        assert call_count == 2

    def test_decorator_with_fallback(self):
        """Verifica decorator com fallback."""
        from claude_rag_sdk.core.circuit_breaker import circuit_breaker

        @circuit_breaker(failure_threshold=2, fallback=lambda: "fallback_value")
        def failing_function():
            raise ValueError("Error")

        # Falhar até abrir circuito
        try:
            failing_function()
        except ValueError:
            pass
        try:
            failing_function()
        except ValueError:
            pass

        # Próxima chamada deve retornar fallback
        result = failing_function()
        assert result == "fallback_value"


class TestCircuitBreakerAsync:
    """Testes para circuit breaker assíncrono."""

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Verifica circuit breaker com funções async."""
        from claude_rag_sdk.core.circuit_breaker import async_circuit_breaker

        @async_circuit_breaker(failure_threshold=3)
        async def async_function():
            return "async_success"

        result = await async_function()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_circuit_opens(self):
        """Verifica circuit breaker async abre com falhas."""
        from claude_rag_sdk.core.circuit_breaker import async_circuit_breaker, CircuitOpenError

        @async_circuit_breaker(failure_threshold=2, recovery_timeout=10)
        async def async_failing():
            raise ValueError("Async error")

        # Falhar até abrir
        with pytest.raises(ValueError):
            await async_failing()
        with pytest.raises(ValueError):
            await async_failing()

        # Deve bloquear
        with pytest.raises(CircuitOpenError):
            await async_failing()


class TestMultipleCircuits:
    """Testes para múltiplos circuitos independentes."""

    def test_named_circuits_independent(self):
        """Verifica circuitos nomeados são independentes."""
        from claude_rag_sdk.core.circuit_breaker import CircuitBreaker

        cb1 = CircuitBreaker(name="service1", failure_threshold=2)
        cb2 = CircuitBreaker(name="service2", failure_threshold=2)

        # Abrir cb1
        cb1.record_failure()
        cb1.record_failure()

        # cb2 deve continuar fechado
        assert cb1.state.value == "open"
        assert cb2.state.value == "closed"
        assert cb2.can_execute() is True
