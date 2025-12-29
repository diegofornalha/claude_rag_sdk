"""Metrics module - Rastreamento de tokens, custo e performance.

Este módulo fornece:
- Contagem de tokens (input/output)
- Estimativa de custo por modelo
- Latência por request
- Agregação por sessão
- Persistência no AgentFS
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from claude_rag_sdk.core.logger import get_logger

logger = get_logger("metrics")


# =============================================================================
# Pricing (USD per 1M tokens) - Atualizado Dezembro 2024
# =============================================================================

PRICING = {
    "haiku": {"input": 0.25, "output": 1.25},      # Claude 3.5 Haiku
    "sonnet": {"input": 3.00, "output": 15.00},    # Claude 3.5 Sonnet
    "opus": {"input": 15.00, "output": 75.00},     # Claude 3 Opus
}

# Fallback para modelos desconhecidos
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RequestMetrics:
    """Métricas de um único request."""

    request_id: str
    session_id: str
    model: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    latency_ms: float = 0

    # Status
    success: bool = True
    error: str | None = None

    # Tool calls
    tool_calls: int = 0

    def finish(self, input_tokens: int = 0, output_tokens: int = 0, error: str | None = None):
        """Finaliza a métrica com tokens e status."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if error:
            self.success = False
            self.error = error

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Calcula custo em USD."""
        pricing = PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "success": self.success,
            "error": self.error,
            "tool_calls": self.tool_calls,
        }


@dataclass
class SessionMetrics:
    """Métricas agregadas de uma sessão."""

    session_id: str
    model: str = "opus"

    # Totais
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0

    # Timing
    total_latency_ms: float = 0
    first_request: datetime | None = None
    last_request: datetime | None = None

    # Erros
    error_count: int = 0

    def add_request(self, metrics: RequestMetrics):
        """Adiciona métricas de um request."""
        self.total_requests += 1
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        self.total_latency_ms += metrics.latency_ms
        self.total_tool_calls += metrics.tool_calls

        if not metrics.success:
            self.error_count += 1

        if self.first_request is None:
            self.first_request = metrics.timestamp
        self.last_request = metrics.timestamp

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        pricing = PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.error_count) / self.total_requests

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 4),
            "first_request": self.first_request.isoformat() if self.first_request else None,
            "last_request": self.last_request.isoformat() if self.last_request else None,
        }


# =============================================================================
# Metrics Manager
# =============================================================================


class MetricsManager:
    """Gerenciador de métricas com persistência opcional no AgentFS."""

    def __init__(self):
        self._sessions: dict[str, SessionMetrics] = {}
        self._requests: list[RequestMetrics] = []
        self._max_requests = 1000  # Limite de requests em memória

    def start_request(self, request_id: str, session_id: str, model: str = "opus") -> RequestMetrics:
        """Inicia rastreamento de um novo request."""
        metrics = RequestMetrics(
            request_id=request_id,
            session_id=session_id,
            model=model,
        )
        return metrics

    def finish_request(
        self,
        metrics: RequestMetrics,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
        error: str | None = None,
    ):
        """Finaliza e registra métricas de um request."""
        metrics.finish(input_tokens, output_tokens, error)
        metrics.tool_calls = tool_calls

        # Adicionar à lista de requests
        self._requests.append(metrics)
        if len(self._requests) > self._max_requests:
            self._requests = self._requests[-self._max_requests:]

        # Atualizar métricas da sessão
        if metrics.session_id not in self._sessions:
            self._sessions[metrics.session_id] = SessionMetrics(
                session_id=metrics.session_id,
                model=metrics.model,
            )
        self._sessions[metrics.session_id].add_request(metrics)

        logger.info(
            "Request metrics recorded",
            session_id=metrics.session_id,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            latency_ms=round(metrics.latency_ms, 2),
            cost_usd=round(metrics.cost_usd, 6),
        )

    def get_session_metrics(self, session_id: str) -> SessionMetrics | None:
        """Retorna métricas agregadas de uma sessão."""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> list[SessionMetrics]:
        """Retorna métricas de todas as sessões."""
        return list(self._sessions.values())

    def get_recent_requests(self, limit: int = 100) -> list[RequestMetrics]:
        """Retorna requests mais recentes."""
        return self._requests[-limit:]

    def get_global_stats(self) -> dict:
        """Retorna estatísticas globais."""
        total_requests = sum(s.total_requests for s in self._sessions.values())
        total_input = sum(s.total_input_tokens for s in self._sessions.values())
        total_output = sum(s.total_output_tokens for s in self._sessions.values())
        total_cost = sum(s.total_cost_usd for s in self._sessions.values())
        total_errors = sum(s.error_count for s in self._sessions.values())

        return {
            "total_sessions": len(self._sessions),
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "total_errors": total_errors,
            "success_rate": round((total_requests - total_errors) / max(total_requests, 1), 4),
        }

    async def persist_to_agentfs(self, afs, session_id: str):
        """Persiste métricas da sessão no AgentFS."""
        if session_id in self._sessions:
            metrics = self._sessions[session_id]
            await afs.kv.set("session:metrics", metrics.to_dict())
            logger.debug(f"Metrics persisted for session {session_id}")

    async def load_from_agentfs(self, afs, session_id: str) -> SessionMetrics | None:
        """Carrega métricas de uma sessão do AgentFS."""
        try:
            data = await afs.kv.get("session:metrics")
            if data:
                metrics = SessionMetrics(
                    session_id=data["session_id"],
                    model=data.get("model", "opus"),
                    total_requests=data.get("total_requests", 0),
                    total_input_tokens=data.get("total_input_tokens", 0),
                    total_output_tokens=data.get("total_output_tokens", 0),
                    total_tool_calls=data.get("total_tool_calls", 0),
                    total_latency_ms=data.get("total_latency_ms", 0),
                    error_count=data.get("error_count", 0),
                )
                self._sessions[session_id] = metrics
                return metrics
        except Exception as e:
            logger.warning(f"Failed to load metrics from AgentFS: {e}")
        return None


# =============================================================================
# Singleton Instance
# =============================================================================

_metrics_manager: MetricsManager | None = None


def get_metrics_manager() -> MetricsManager:
    """Retorna instância singleton do MetricsManager."""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


# =============================================================================
# Helper Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimativa simples de tokens (aproximadamente 4 chars por token)."""
    return len(text) // 4


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula custo para um modelo específico."""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
