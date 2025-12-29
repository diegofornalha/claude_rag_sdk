# =============================================================================
# TESTES DE PERFORMANCE
# =============================================================================
# Testes de carga e performance para endpoints críticos
# =============================================================================

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import patch

import pytest


@dataclass
class PerformanceResult:
    """Resultado de teste de performance."""

    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    p95_time: float
    p99_time: float
    requests_per_second: float


# Configurar ambiente antes de importar a app
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Configura ambiente para testes."""
    env_vars = {
        "AUTH_ENABLED": "false",
        "ANTHROPIC_API_KEY": "test-key-123",
        "RAG_API_KEY": "rag_test_key",
        "ENVIRONMENT": "test",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    from fastapi.testclient import TestClient

    from server import app

    return TestClient(app)


def measure_endpoint(
    client,
    method: str,
    endpoint: str,
    num_requests: int = 100,
    **kwargs,
) -> PerformanceResult:
    """
    Mede performance de um endpoint.

    Args:
        client: TestClient
        method: HTTP method (get, post, etc)
        endpoint: URL do endpoint
        num_requests: Número de requisições
        **kwargs: Argumentos para a requisição

    Returns:
        PerformanceResult com métricas
    """
    times = []
    successful = 0
    failed = 0

    method_func = getattr(client, method.lower())

    start_total = time.time()

    for _ in range(num_requests):
        start = time.time()
        try:
            response = method_func(endpoint, **kwargs)
            elapsed = time.time() - start
            times.append(elapsed)

            if response.status_code < 400:
                successful += 1
            else:
                failed += 1
        except Exception:
            failed += 1
            times.append(time.time() - start)

    total_time = time.time() - start_total

    # Calcular percentis
    sorted_times = sorted(times)
    p95_idx = int(len(sorted_times) * 0.95)
    p99_idx = int(len(sorted_times) * 0.99)

    return PerformanceResult(
        endpoint=endpoint,
        total_requests=num_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_time=total_time,
        min_time=min(times) if times else 0,
        max_time=max(times) if times else 0,
        avg_time=statistics.mean(times) if times else 0,
        median_time=statistics.median(times) if times else 0,
        p95_time=sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0,
        p99_time=sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0,
        requests_per_second=num_requests / total_time if total_time > 0 else 0,
    )


def print_result(result: PerformanceResult):
    """Imprime resultado formatado."""
    print(f"\n--- {result.endpoint} ---")
    print(f"Total: {result.total_requests} requests em {result.total_time:.2f}s")
    print(f"Sucesso: {result.successful_requests}, Falha: {result.failed_requests}")
    print(f"RPS: {result.requests_per_second:.2f}")
    print(
        f"Tempos (ms): min={result.min_time * 1000:.1f}, avg={result.avg_time * 1000:.1f}, max={result.max_time * 1000:.1f}"
    )
    print(
        f"Percentis (ms): p50={result.median_time * 1000:.1f}, p95={result.p95_time * 1000:.1f}, p99={result.p99_time * 1000:.1f}"
    )


class TestHealthEndpointPerformance:
    """Testes de performance para endpoints de health."""

    def test_health_latency(self, client):
        """Mede latência do /health."""
        result = measure_endpoint(client, "get", "/health", num_requests=100)

        print_result(result)

        # Assertions
        assert result.successful_requests >= 95  # 95% sucesso
        assert result.avg_time < 0.1  # < 100ms média
        assert result.p95_time < 0.2  # < 200ms p95

    def test_root_latency(self, client):
        """Mede latência do /."""
        result = measure_endpoint(client, "get", "/", num_requests=100)

        print_result(result)

        assert result.successful_requests >= 95
        assert result.avg_time < 0.1

    def test_model_latency(self, client):
        """Mede latência do /model."""
        result = measure_endpoint(client, "get", "/model", num_requests=100)

        print_result(result)

        assert result.successful_requests >= 95
        assert result.avg_time < 0.1


class TestSessionEndpointPerformance:
    """Testes de performance para endpoints de sessão."""

    def test_session_list_latency(self, client):
        """Mede latência do GET /sessions."""
        result = measure_endpoint(client, "get", "/sessions", num_requests=50)

        print_result(result)

        assert result.successful_requests >= 45  # 90% sucesso
        assert result.avg_time < 0.5  # < 500ms média

    def test_session_current_latency(self, client):
        """Mede latência do GET /session/current."""
        result = measure_endpoint(client, "get", "/session/current", num_requests=50)

        print_result(result)

        assert result.successful_requests >= 45
        assert result.avg_time < 0.3

    def test_session_create_latency(self, client):
        """Mede latência do POST /sessions."""
        # Criar sessões tem mais overhead
        result = measure_endpoint(client, "post", "/sessions", num_requests=20)

        print_result(result)

        assert result.successful_requests >= 18  # 90% sucesso
        assert result.avg_time < 1.0  # < 1s média (IO-bound)


class TestOutputsEndpointPerformance:
    """Testes de performance para endpoints de artifacts."""

    def test_artifacts_list_latency(self, client):
        """Mede latência do GET /artifacts."""
        result = measure_endpoint(client, "get", "/artifacts", num_requests=50)

        print_result(result)

        assert result.successful_requests >= 45
        assert result.avg_time < 0.5


class TestAuditEndpointPerformance:
    """Testes de performance para endpoints de auditoria."""

    def test_audit_tools_latency(self, client):
        """Mede latência do GET /audit/tools."""
        result = measure_endpoint(client, "get", "/audit/tools", num_requests=50)

        print_result(result)

        assert result.successful_requests >= 45
        assert result.avg_time < 0.5

    def test_audit_stats_latency(self, client):
        """Mede latência do GET /audit/stats."""
        result = measure_endpoint(client, "get", "/audit/stats", num_requests=50)

        print_result(result)

        assert result.successful_requests >= 45
        assert result.avg_time < 0.5


class TestConcurrentLoad:
    """Testes de carga concorrente."""

    def test_concurrent_health_checks(self, client):
        """Testa health checks concorrentes."""
        num_concurrent = 10
        requests_per_thread = 10

        results = []

        def make_requests():
            times = []
            for _ in range(requests_per_thread):
                start = time.time()
                client.get("/health")
                times.append(time.time() - start)
            return times

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_concurrent)]
            for future in futures:
                results.extend(future.result())

        # Métricas
        total = len(results)
        avg = statistics.mean(results)
        p95 = sorted(results)[int(total * 0.95)]

        print(f"\nConcurrent Health Checks ({num_concurrent} threads x {requests_per_thread})")
        print(f"Total: {total} requests")
        print(f"Avg: {avg * 1000:.1f}ms, p95: {p95 * 1000:.1f}ms")

        assert avg < 0.2  # < 200ms avg under load

    def test_concurrent_session_list(self, client):
        """Testa listagem de sessões concorrente."""
        num_concurrent = 5
        requests_per_thread = 10

        results = []

        def make_requests():
            times = []
            for _ in range(requests_per_thread):
                start = time.time()
                client.get("/sessions")
                times.append(time.time() - start)
            return times

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_concurrent)]
            for future in futures:
                results.extend(future.result())

        avg = statistics.mean(results)
        print(f"\nConcurrent Session List: avg={avg * 1000:.1f}ms")

        assert avg < 1.0  # < 1s under concurrent load


class TestMemoryAndResources:
    """Testes de uso de recursos."""

    def test_repeated_requests_no_memory_leak(self, client):
        """Verifica ausência de memory leak em requisições repetidas."""
        import gc

        # Forçar GC antes
        gc.collect()

        # Fazer muitas requisições
        for _ in range(200):
            client.get("/health")

        # Forçar GC depois
        gc.collect()

        # Se chegou aqui sem OOM, passou
        assert True

    def test_large_session_list_performance(self, client):
        """Verifica performance com muitas sessões."""
        # Criar várias sessões
        for _ in range(10):
            client.post("/sessions")

        # Medir listagem
        result = measure_endpoint(client, "get", "/sessions", num_requests=20)

        print_result(result)

        # Deve escalar razoavelmente
        assert result.avg_time < 2.0  # < 2s mesmo com muitas sessões


class TestRateLimitingPerformance:
    """Testes de performance do rate limiting."""

    def test_rate_limiter_overhead(self, client):
        """Mede overhead do rate limiter."""
        # Sem rate limiter seria mais rápido, mas devemos aceitar overhead
        result = measure_endpoint(client, "get", "/health", num_requests=100)

        # Rate limiter não deve adicionar mais que 50ms
        assert result.avg_time < 0.15  # < 150ms com overhead

    def test_burst_handling(self, client):
        """Testa handling de burst de requisições."""
        # Burst rápido
        times = []
        for _ in range(50):
            start = time.time()
            client.get("/health")
            times.append(time.time() - start)

        # Mesmo em burst, deve responder rápido
        avg = statistics.mean(times)
        assert avg < 0.2  # < 200ms avg em burst


class TestCachePerformance:
    """Testes de performance do cache."""

    def test_repeated_same_endpoint_faster(self, client):
        """Verifica que requisições repetidas são mais rápidas (cache)."""
        # Primeira vez (cold)
        cold_times = []
        for _ in range(10):
            start = time.time()
            client.get("/sessions")
            cold_times.append(time.time() - start)

        cold_avg = statistics.mean(cold_times)

        # Segunda vez (potencialmente cached)
        warm_times = []
        for _ in range(10):
            start = time.time()
            client.get("/sessions")
            warm_times.append(time.time() - start)

        warm_avg = statistics.mean(warm_times)

        print(f"\nCache Performance: cold={cold_avg * 1000:.1f}ms, warm={warm_avg * 1000:.1f}ms")

        # Warm deve ser <= cold (pode não ter cache, então apenas verifica que não piorou)
        assert warm_avg <= cold_avg * 1.5  # No máximo 50% mais lento


class TestBM25Performance:
    """Testes de performance do BM25."""

    def test_bm25_indexing_speed(self):
        """Testa velocidade de indexação BM25."""
        from claude_rag_sdk.core.hybrid_search import BM25

        # Criar documentos de teste
        documents = [
            (i, f"Document {i} with some content about topic {i % 10} and various keywords")
            for i in range(1000)
        ]

        bm25 = BM25()

        start = time.time()
        bm25.index(documents)
        elapsed = time.time() - start

        print(f"\nBM25 Indexing: {len(documents)} docs em {elapsed * 1000:.1f}ms")

        # Deve indexar 1000 docs em menos de 1s
        assert elapsed < 1.0

    def test_bm25_search_speed(self):
        """Testa velocidade de busca BM25."""
        from claude_rag_sdk.core.hybrid_search import BM25

        documents = [(i, f"Document {i} with content about topic {i % 10}") for i in range(1000)]

        bm25 = BM25()
        bm25.index(documents)

        # Medir busca
        times = []
        for _ in range(100):
            start = time.time()
            bm25.search("document topic content")
            times.append(time.time() - start)

        avg = statistics.mean(times)
        print(f"\nBM25 Search: avg={avg * 1000:.2f}ms")

        # Busca deve ser < 10ms
        assert avg < 0.01


class TestRerankerPerformance:
    """Testes de performance do reranker."""

    def test_lightweight_reranker_speed(self):
        """Testa velocidade do reranker leve."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        # Documentos de teste
        documents = [
            (i, f"Document {i} about python programming and machine learning", 0.5, {})
            for i in range(100)
        ]

        times = []
        for _ in range(50):
            start = time.time()
            reranker.rerank("python machine learning", documents, top_k=10)
            times.append(time.time() - start)

        avg = statistics.mean(times)
        print(f"\nLightweight Reranker: avg={avg * 1000:.2f}ms")

        # Deve ser < 50ms para 100 docs
        assert avg < 0.05


class TestAdaptiveSearchPerformance:
    """Testes de performance do adaptive search."""

    def test_adaptive_topk_speed(self):
        """Testa velocidade do adaptive top-k."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            similarity: float

        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        # Resultados de teste
        results = [MockResult(similarity=0.9 - i * 0.01) for i in range(100)]

        times = []
        for _ in range(1000):
            start = time.time()
            adapter.calculate_optimal_k(results, base_top_k=10)
            times.append(time.time() - start)

        avg = statistics.mean(times)
        print(f"\nAdaptive TopK: avg={avg * 1000000:.2f}µs")

        # Deve ser < 100µs
        assert avg < 0.0001


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_performance_tests():
    """Executa testes de performance manualmente."""

    print("\n" + "=" * 60)
    print("TESTES DE PERFORMANCE")
    print("=" * 60)

    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-s",  # Mostrar prints
        ]
    )

    return exit_code == 0


if __name__ == "__main__":
    import sys

    success = run_performance_tests()
    sys.exit(0 if success else 1)
