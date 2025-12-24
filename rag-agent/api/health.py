# =============================================================================
# HEALTH CHECK - Endpoint de Saúde e SLOs
# =============================================================================
# Verificação de saúde do sistema e monitoramento de SLOs
# =============================================================================

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import apsw
import sqlite_vec


class HealthStatus(str, Enum):
    """Status de saúde."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Saúde de um componente."""
    name: str
    status: HealthStatus
    latency_ms: float
    message: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class SLO:
    """Definição de SLO."""
    name: str
    target: float           # Target (ex: 0.99 para 99%)
    current: float          # Valor atual
    unit: str               # Unidade (ex: "ms", "%", "requests/s")
    window: str             # Janela de tempo (ex: "1h", "24h", "30d")
    compliant: bool         # Se está em compliance


@dataclass
class HealthReport:
    """Relatório completo de saúde."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: list[ComponentHealth]
    slos: list[SLO]
    version: str = "1.0.0"


class HealthChecker:
    """Verificador de saúde do sistema."""

    def __init__(self, db_path: str, start_time: Optional[datetime] = None):
        self.db_path = db_path
        self.start_time = start_time or datetime.now(timezone.utc)

        # SLO targets
        self.slo_targets = {
            "availability": 0.999,           # 99.9% uptime
            "latency_p95_ms": 3000,          # p95 < 3s
            "latency_p99_ms": 5000,          # p99 < 5s
            "error_rate": 0.01,              # < 1% erros
            "throughput_rps": 10,            # > 10 req/s capacity
        }

    def check_database(self) -> ComponentHealth:
        """Verifica saúde do banco de dados."""
        start = time.perf_counter()
        try:
            conn = apsw.Connection(self.db_path)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)

            cursor = conn.cursor()

            # Verificar tabelas
            tables = []
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ):
                tables.append(row[0])

            # Verificar contagem
            doc_count = 0
            vec_count = 0
            for row in cursor.execute("SELECT COUNT(*) FROM documentos"):
                doc_count = row[0]
            for row in cursor.execute("SELECT COUNT(*) FROM vec_documentos"):
                vec_count = row[0]

            conn.close()

            latency = (time.perf_counter() - start) * 1000

            if doc_count == 0:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="No documents in database",
                    details={"doc_count": doc_count, "vec_count": vec_count},
                )

            if doc_count != vec_count:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="Document/embedding count mismatch",
                    details={"doc_count": doc_count, "vec_count": vec_count},
                )

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="OK",
                details={"doc_count": doc_count, "vec_count": vec_count, "tables": tables},
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )

    def check_embedding_model(self) -> ComponentHealth:
        """Verifica saúde do modelo de embeddings."""
        start = time.perf_counter()
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding("BAAI/bge-small-en-v1.5")
            embeddings = list(model.embed(["health check test"]))

            latency = (time.perf_counter() - start) * 1000

            if len(embeddings[0]) != 384:
                return ComponentHealth(
                    name="embedding_model",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message=f"Unexpected embedding dimension: {len(embeddings[0])}",
                )

            # Importar config para pegar modelo atual
            from core.config import get_config
            config = get_config()

            return ComponentHealth(
                name="embedding_model",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="OK",
                details={
                    "model": config.embedding_model.value,
                    "short_name": config.embedding_model.short_name,
                    "dimensions": config.embedding_dimensions,
                },
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="embedding_model",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )

    def check_vector_search(self) -> ComponentHealth:
        """Verifica saúde da busca vetorial."""
        start = time.perf_counter()
        try:
            from fastembed import TextEmbedding

            model = TextEmbedding("BAAI/bge-small-en-v1.5")
            embeddings = list(model.embed(["test query"]))
            query_vec = sqlite_vec.serialize_float32(embeddings[0].tolist())

            conn = apsw.Connection(self.db_path)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)
            cursor = conn.cursor()

            results = []
            for row in cursor.execute("""
                SELECT v.doc_id, v.distance
                FROM vec_documentos v
                WHERE v.embedding MATCH ? AND k = 1
            """, (query_vec,)):
                results.append(row)

            conn.close()
            latency = (time.perf_counter() - start) * 1000

            if not results:
                return ComponentHealth(
                    name="vector_search",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="No results from vector search",
                )

            return ComponentHealth(
                name="vector_search",
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                message="OK",
                details={"search_latency_ms": latency},
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="vector_search",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )

    def get_slos(self, metrics: Optional[dict] = None) -> list[SLO]:
        """
        Calcula status dos SLOs.

        Args:
            metrics: Métricas do sistema (de api/metrics.py)

        Returns:
            Lista de SLOs com status atual
        """
        slos = []

        # SLO: Availability (baseado em uptime)
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        # Assumir 100% se rodando
        availability = 1.0 if uptime > 0 else 0.0
        slos.append(SLO(
            name="availability",
            target=self.slo_targets["availability"],
            current=availability,
            unit="%",
            window="since_start",
            compliant=availability >= self.slo_targets["availability"],
        ))

        if metrics:
            # SLO: Latency P95
            query_latency = metrics.get("rag", {}).get("query_latency", {})
            p95 = query_latency.get("p95", 0)
            slos.append(SLO(
                name="latency_p95",
                target=self.slo_targets["latency_p95_ms"],
                current=p95,
                unit="ms",
                window="all_time",
                compliant=p95 <= self.slo_targets["latency_p95_ms"],
            ))

            # SLO: Error Rate
            total = metrics.get("rag", {}).get("queries_total", 0)
            errors = sum(metrics.get("rag", {}).get("errors_by_type", {}).values())
            error_rate = errors / max(total, 1)
            slos.append(SLO(
                name="error_rate",
                target=self.slo_targets["error_rate"],
                current=error_rate,
                unit="%",
                window="all_time",
                compliant=error_rate <= self.slo_targets["error_rate"],
            ))

        return slos

    def check_health(self, include_details: bool = True) -> HealthReport:
        """
        Executa verificação completa de saúde.

        Args:
            include_details: Se deve incluir verificações detalhadas

        Returns:
            HealthReport com status de todos os componentes
        """
        components = []

        # Verificar componentes
        components.append(self.check_database())

        if include_details:
            components.append(self.check_embedding_model())
            components.append(self.check_vector_search())

        # Determinar status geral
        unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in components)
        degraded = any(c.status == HealthStatus.DEGRADED for c in components)

        if unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Calcular uptime
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Obter SLOs
        slos = self.get_slos()

        return HealthReport(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=uptime,
            components=components,
            slos=slos,
        )

    def to_dict(self, report: HealthReport) -> dict:
        """Converte HealthReport para dict."""
        return {
            "status": report.status.value,
            "timestamp": report.timestamp.isoformat(),
            "uptime_seconds": round(report.uptime_seconds, 2),
            "version": report.version,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": round(c.latency_ms, 2),
                    "message": c.message,
                    "details": c.details,
                }
                for c in report.components
            ],
            "slos": [
                {
                    "name": s.name,
                    "target": s.target,
                    "current": round(s.current, 4),
                    "unit": s.unit,
                    "window": s.window,
                    "compliant": s.compliant,
                }
                for s in report.slos
            ],
        }


# Liveness e Readiness probes (Kubernetes-style)
def liveness_check(db_path: str) -> tuple[bool, str]:
    """
    Liveness check - verifica se o processo está vivo.

    Returns:
        (is_alive, message)
    """
    return True, "OK"


def readiness_check(db_path: str) -> tuple[bool, str]:
    """
    Readiness check - verifica se pode receber tráfego.

    Returns:
        (is_ready, message)
    """
    try:
        conn = apsw.Connection(db_path)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        cursor = conn.cursor()

        # Verificar se tem dados
        for row in cursor.execute("SELECT COUNT(*) FROM documentos"):
            if row[0] == 0:
                conn.close()
                return False, "No documents loaded"

        conn.close()
        return True, "Ready"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    import json

    db_path = str(Path(__file__).parent.parent.parent / "teste" / "documentos.db")

    print("=== Health Check ===\n")

    checker = HealthChecker(db_path)

    # Full health check
    report = checker.check_health(include_details=True)
    report_dict = checker.to_dict(report)

    print(json.dumps(report_dict, indent=2))

    # Kubernetes probes
    print("\n--- Kubernetes Probes ---")
    alive, msg = liveness_check(db_path)
    print(f"Liveness: {alive} - {msg}")

    ready, msg = readiness_check(db_path)
    print(f"Readiness: {ready} - {msg}")
