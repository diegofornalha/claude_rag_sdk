# =============================================================================
# CIRCUIT BREAKER - Resiliência para Chamadas Externas
# =============================================================================
# Implementa padrão Circuit Breaker para proteger contra falhas em cascata
# =============================================================================

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, TypeVar, Generic
from functools import wraps

T = TypeVar('T')


class CircuitState(str, Enum):
    """Estados do circuit breaker."""
    CLOSED = "closed"       # Normal, permitindo chamadas
    OPEN = "open"           # Bloqueando chamadas
    HALF_OPEN = "half_open"  # Testando se pode reabrir


@dataclass
class CircuitStats:
    """Estatísticas do circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Erro quando circuit está aberto."""
    pass


class CircuitBreaker:
    """
    Circuit Breaker para proteger chamadas a serviços externos.

    Estados:
    - CLOSED: Normal, todas as chamadas passam
    - OPEN: Bloqueado, chamadas são rejeitadas imediatamente
    - HALF_OPEN: Testando, permite algumas chamadas para verificar recuperação
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,      # Falhas para abrir circuito
        success_threshold: int = 3,      # Sucessos para fechar em half-open
        timeout: float = 30.0,           # Segundos antes de tentar half-open
        half_open_max_calls: int = 3,    # Chamadas permitidas em half-open
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._opened_at: Optional[float] = None
        self._half_open_calls: int = 0

    @property
    def state(self) -> CircuitState:
        """Retorna estado atual, verificando timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def stats(self) -> CircuitStats:
        """Retorna estatísticas."""
        with self._lock:
            return CircuitStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes,
            )

    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar resetar para half-open."""
        if self._opened_at is None:
            return False
        return time.time() - self._opened_at >= self.timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transiciona para novo estado."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._stats.consecutive_failures = 0

    def _can_execute(self) -> bool:
        """Verifica se pode executar chamada."""
        current_state = self.state  # Isso pode atualizar o estado

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        elif current_state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls

        return False

    def _record_success(self) -> None:
        """Registra sucesso."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Registra falha."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _record_rejection(self) -> None:
        """Registra rejeição."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.rejected_calls += 1

    def call(self, func: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """
        Executa função com proteção do circuit breaker.

        Args:
            func: Função a executar
            fallback: Função de fallback se circuit aberto

        Returns:
            Resultado da função ou fallback

        Raises:
            CircuitBreakerError: Se circuit aberto e sem fallback
        """
        with self._lock:
            if not self._can_execute():
                self._record_rejection()
                if fallback:
                    return fallback()
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is {self._state.value}. "
                    f"Consecutive failures: {self._stats.consecutive_failures}"
                )

            # Incrementar ANTES de liberar o lock para evitar race condition
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                # Verificar novamente se excedemos o limite após incremento
                if self._half_open_calls > self.half_open_max_calls:
                    self._half_open_calls -= 1
                    self._record_rejection()
                    if fallback:
                        return fallback()
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is {self._state.value}. "
                        f"Max half-open calls ({self.half_open_max_calls}) exceeded."
                    )

        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            if fallback:
                return fallback()
            raise

    def reset(self) -> None:
        """Reseta o circuit breaker manualmente."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitStats()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
    fallback: Optional[Callable] = None,
):
    """
    Decorator para adicionar circuit breaker a uma função.

    Args:
        name: Nome do circuit
        failure_threshold: Falhas para abrir
        timeout: Timeout em segundos
        fallback: Função de fallback
    """
    cb = CircuitBreaker(name=name, failure_threshold=failure_threshold, timeout=timeout)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(
                lambda: func(*args, **kwargs),
                fallback=fallback
            )

        wrapper.circuit_breaker = cb
        return wrapper

    return decorator


# Registry global de circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_or_create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
) -> CircuitBreaker:
    """
    Obtém ou cria circuit breaker pelo nome.

    Args:
        name: Nome único do circuit
        failure_threshold: Falhas para abrir
        timeout: Timeout em segundos

    Returns:
        CircuitBreaker
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout,
        )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Retorna todos os circuit breakers registrados."""
    return _circuit_breakers.copy()


if __name__ == "__main__":
    import random

    print("=== Teste de Circuit Breaker ===\n")

    cb = CircuitBreaker(
        name="test-service",
        failure_threshold=3,
        success_threshold=2,
        timeout=5.0,
    )

    def unstable_service():
        """Serviço que falha aleatoriamente."""
        if random.random() < 0.7:  # 70% chance de falha
            raise Exception("Service unavailable")
        return "Success!"

    def fallback_response():
        return "Fallback response"

    print("Simulando chamadas ao serviço instável...")
    for i in range(15):
        try:
            result = cb.call(unstable_service, fallback=fallback_response)
            print(f"  Call {i+1}: {result} | State: {cb.state.value}")
        except CircuitBreakerError as e:
            print(f"  Call {i+1}: BLOCKED | State: {cb.state.value}")

        time.sleep(0.5)

    print(f"\n--- Estatísticas ---")
    stats = cb.stats
    print(f"  Total: {stats.total_calls}")
    print(f"  Sucesso: {stats.successful_calls}")
    print(f"  Falha: {stats.failed_calls}")
    print(f"  Rejeitadas: {stats.rejected_calls}")

    # Teste do decorator
    print("\n--- Teste do Decorator ---")

    @circuit_breaker(name="decorated-service", failure_threshold=2, timeout=3.0)
    def decorated_function():
        if random.random() < 0.5:
            raise Exception("Error")
        return "OK"

    for i in range(10):
        try:
            result = decorated_function()
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: {type(e).__name__}")
        time.sleep(0.3)
