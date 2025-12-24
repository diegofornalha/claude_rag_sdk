# =============================================================================
# METRICS - Coleta e Exposição de Métricas para RAG Agent
# =============================================================================
# Métricas de latência, custo, erros e uso do sistema
# =============================================================================

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Optional
import json


@dataclass
class MetricPoint:
    """Ponto de métrica com timestamp."""
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    """Coletor de métricas thread-safe."""

    def __init__(self, max_history: int = 1000):
        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._max_history = max_history

        # Métricas específicas do RAG
        self._query_latencies: list[float] = []
        self._llm_latencies: list[float] = []
        self._token_counts: dict[str, int] = {"input": 0, "output": 0}
        self._costs: dict[str, float] = defaultdict(float)
        self._errors: dict[str, int] = defaultdict(int)

        # Timestamp de início
        self._start_time = datetime.now(timezone.utc)

    # --- Counters ---

    def increment(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None:
        """Incrementa um contador."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value

    def get_counter(self, name: str, labels: Optional[dict] = None) -> int:
        """Retorna valor de um contador."""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0)

    # --- Gauges ---

    def set_gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Define valor de um gauge."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

    def get_gauge(self, name: str, labels: Optional[dict] = None) -> float:
        """Retorna valor de um gauge."""
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key, 0.0)

    # --- Histograms ---

    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Adiciona observação a um histograma."""
        with self._lock:
            key = self._make_key(name, labels)
            hist = self._histograms[key]
            hist.append(value)
            # Manter apenas últimas N observações
            if len(hist) > self._max_history:
                self._histograms[key] = hist[-self._max_history:]

    def get_histogram_stats(self, name: str, labels: Optional[dict] = None) -> dict:
        """Retorna estatísticas de um histograma."""
        with self._lock:
            key = self._make_key(name, labels)
            values = self._histograms.get(key, [])
            if not values:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

            sorted_values = sorted(values)
            n = len(sorted_values)
            return {
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(values) / n,
                "p50": sorted_values[int(n * 0.5)],
                "p95": sorted_values[int(n * 0.95)] if n > 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1],
            }

    # --- Métricas RAG específicas ---

    def record_query(self, latency_ms: float, results_count: int) -> None:
        """Registra uma query de busca."""
        self.increment("rag_queries_total")
        self.observe("rag_query_latency_ms", latency_ms)
        self.increment("rag_results_total", results_count)
        with self._lock:
            self._query_latencies.append(latency_ms)
            if len(self._query_latencies) > self._max_history:
                self._query_latencies = self._query_latencies[-self._max_history:]

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float = 0.0,
    ) -> None:
        """Registra uma chamada ao LLM."""
        self.increment("llm_calls_total", labels={"model": model})
        self.observe("llm_latency_ms", latency_ms, labels={"model": model})
        self.increment("llm_tokens_input", input_tokens, labels={"model": model})
        self.increment("llm_tokens_output", output_tokens, labels={"model": model})

        with self._lock:
            self._token_counts["input"] += input_tokens
            self._token_counts["output"] += output_tokens
            self._costs[model] += cost_usd
            self._llm_latencies.append(latency_ms)
            if len(self._llm_latencies) > self._max_history:
                self._llm_latencies = self._llm_latencies[-self._max_history:]

    def record_error(self, error_type: str) -> None:
        """Registra um erro."""
        self.increment("errors_total", labels={"type": error_type})
        with self._lock:
            self._errors[error_type] += 1

    def record_rbac_decision(self, allowed: bool) -> None:
        """Registra decisão RBAC."""
        if allowed:
            self.increment("rbac_allowed_total")
        else:
            self.increment("rbac_denied_total")

    # --- Export ---

    def get_all_metrics(self) -> dict:
        """Retorna todas as métricas em formato JSON."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": round(uptime, 2),

                # Counters
                "counters": dict(self._counters),

                # Gauges
                "gauges": dict(self._gauges),

                # Histogramas resumidos
                "histograms": {
                    key: self._get_histogram_stats_internal(key)
                    for key in self._histograms.keys()
                },

                # Métricas RAG específicas
                "rag": {
                    "queries_total": self._counters.get("rag_queries_total", 0),
                    "query_latency": self._calculate_stats(self._query_latencies),
                    "llm_latency": self._calculate_stats(self._llm_latencies),
                    "tokens": dict(self._token_counts),
                    "costs_by_model": dict(self._costs),
                    "total_cost_usd": sum(self._costs.values()),
                    "errors_by_type": dict(self._errors),
                },

                # RBAC
                "rbac": {
                    "allowed_total": self._counters.get("rbac_allowed_total", 0),
                    "denied_total": self._counters.get("rbac_denied_total", 0),
                },
            }

    def get_prometheus_format(self) -> str:
        """Retorna métricas em formato Prometheus."""
        lines = []

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                label_str = self._format_labels(labels)
                lines.append(f"{name}{label_str} {value}")

            # Gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                label_str = self._format_labels(labels)
                lines.append(f"{name}{label_str} {value}")

            # Histogram summaries
            for key, values in self._histograms.items():
                name, labels = self._parse_key(key)
                label_str = self._format_labels(labels)
                if values:
                    stats = self._calculate_stats(values)
                    lines.append(f"{name}_count{label_str} {stats['count']}")
                    lines.append(f"{name}_sum{label_str} {sum(values):.2f}")
                    lines.append(f"{name}_avg{label_str} {stats['avg']:.2f}")
                    lines.append(f"{name}_p95{label_str} {stats['p95']:.2f}")

        return "\n".join(lines)

    # --- Helpers ---

    def _make_key(self, name: str, labels: Optional[dict]) -> str:
        """Cria chave única para métrica com labels."""
        if not labels:
            return name
        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"

    def _parse_key(self, key: str) -> tuple[str, dict]:
        """Parse chave em nome e labels."""
        if "{" not in key:
            return key, {}
        name = key[:key.index("{")]
        label_str = key[key.index("{") + 1:-1]
        labels = {}
        if label_str:
            for part in label_str.split(","):
                k, v = part.split("=")
                labels[k] = v
        return name, labels

    def _format_labels(self, labels: dict) -> str:
        """Formata labels para Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"

    def _get_histogram_stats_internal(self, key: str) -> dict:
        """Calcula stats de histograma (sem lock)."""
        values = self._histograms.get(key, [])
        return self._calculate_stats(values)

    def _calculate_stats(self, values: list[float]) -> dict:
        """Calcula estatísticas de uma lista de valores."""
        if not values:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        n = len(sorted_values)
        return {
            "count": n,
            "min": round(sorted_values[0], 2),
            "max": round(sorted_values[-1], 2),
            "avg": round(sum(values) / n, 2),
            "p50": round(sorted_values[int(n * 0.5)], 2),
            "p95": round(sorted_values[min(int(n * 0.95), n - 1)], 2),
            "p99": round(sorted_values[min(int(n * 0.99), n - 1)], 2),
        }


# Instância global
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Retorna instância global do coletor de métricas."""
    return _metrics


# Context manager para medir latência
class Timer:
    """Context manager para medir latência."""

    def __init__(self, callback=None):
        self.callback = callback
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.callback:
            self.callback(self.elapsed_ms)


if __name__ == "__main__":
    # Teste do coletor de métricas
    import random

    metrics = get_metrics()

    print("=== Simulando métricas ===\n")

    # Simular queries
    for i in range(50):
        latency = random.uniform(50, 500)
        results = random.randint(1, 10)
        metrics.record_query(latency, results)

    # Simular chamadas LLM
    for i in range(20):
        metrics.record_llm_call(
            model="claude-haiku-4-5",
            input_tokens=random.randint(100, 1000),
            output_tokens=random.randint(50, 500),
            latency_ms=random.uniform(500, 3000),
            cost_usd=random.uniform(0.0001, 0.001),
        )

    # Simular erros
    for _ in range(5):
        metrics.record_error("ValidationError")
    for _ in range(2):
        metrics.record_error("TimeoutError")

    # Simular RBAC
    for _ in range(30):
        metrics.record_rbac_decision(True)
    for _ in range(5):
        metrics.record_rbac_decision(False)

    # Mostrar métricas JSON
    print("=== JSON Format ===")
    print(json.dumps(metrics.get_all_metrics(), indent=2))

    print("\n=== Prometheus Format ===")
    print(metrics.get_prometheus_format())
