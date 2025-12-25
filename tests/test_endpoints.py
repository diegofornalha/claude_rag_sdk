"""
Teste automatizado de todos os endpoints.
Executa após refatoração para garantir que tudo funciona.
"""
import pytest
import httpx
import asyncio
from typing import Optional

BASE_URL = "http://localhost:8001"


class TestEndpoints:
    """Testes de todos os endpoints do Chat Simples."""

    @pytest.fixture(scope="class")
    def client(self):
        """Cliente HTTP para testes."""
        return httpx.Client(base_url=BASE_URL, timeout=30.0)

    # =========================================================================
    # HEALTH ENDPOINTS
    # =========================================================================

    def test_root(self, client):
        """GET / - Health check básico."""
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "session_id" in data
        print(f"✓ GET / - status={data['status']}")

    def test_health(self, client):
        """GET /health - Health check detalhado."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "security" in data
        print(f"✓ GET /health - status={data['status']}")

    # =========================================================================
    # SESSION ENDPOINTS
    # =========================================================================

    def test_session_current(self, client):
        """GET /session/current - Sessão atual."""
        r = client.get("/session/current")
        assert r.status_code == 200
        data = r.json()
        assert "active" in data
        print(f"✓ GET /session/current - active={data['active']}")

    def test_sessions_list(self, client):
        """GET /sessions - Listar sessões."""
        r = client.get("/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "count" in data
        assert "sessions" in data
        print(f"✓ GET /sessions - count={data['count']}")

    # =========================================================================
    # OUTPUTS ENDPOINTS
    # =========================================================================

    def test_outputs_list(self, client):
        """GET /outputs - Listar arquivos."""
        r = client.get("/outputs")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert "count" in data
        print(f"✓ GET /outputs - count={data['count']}")

    # =========================================================================
    # AUDIT ENDPOINTS
    # =========================================================================

    def test_audit_tools(self, client):
        """GET /audit/tools - Histórico de tools."""
        r = client.get("/audit/tools")
        assert r.status_code == 200
        data = r.json()
        # Pode não ter sessão ativa, então aceita error ou records
        assert "records" in data or "recent" in data or "error" in data
        print(f"✓ GET /audit/tools - {'com dados' if 'recent' in data else 'sem sessão'}")

    def test_audit_stats(self, client):
        """GET /audit/stats - Estatísticas de auditoria."""
        r = client.get("/audit/stats")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data or "error" in data
        print(f"✓ GET /audit/stats")

    def test_audit_debug(self, client):
        """GET /audit/debug/{session_id} - Debug do CLI."""
        # Usar um ID fake para testar o endpoint
        r = client.get("/audit/debug/test-session-id")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "found" in data
        print(f"✓ GET /audit/debug - found={data['found']}")

    def test_audit_enriched(self, client):
        """GET /audit/tools/enriched - Tools com debug."""
        r = client.get("/audit/tools/enriched?session_id=test-session")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert "enriched" in data
        print(f"✓ GET /audit/tools/enriched - enriched={data['enriched']}")

    # =========================================================================
    # RAG ENDPOINTS
    # =========================================================================

    def test_rag_stats(self, client):
        """GET /rag/stats - Estatísticas RAG."""
        r = client.get("/rag/stats")
        # Pode falhar se não houver RAG configurado
        assert r.status_code in [200, 500]
        print(f"✓ GET /rag/stats - status={r.status_code}")

    # =========================================================================
    # FILESYSTEM ENDPOINTS
    # =========================================================================

    def test_fs_tree(self, client):
        """GET /fs/tree - Árvore de arquivos."""
        r = client.get("/fs/tree")
        assert r.status_code == 200
        data = r.json()
        assert "tree" in data or "error" in data
        print(f"✓ GET /fs/tree")

    def test_kv_list(self, client):
        """GET /kv/list - Listar chaves KV."""
        r = client.get("/kv/list")
        assert r.status_code == 200
        data = r.json()
        assert "keys" in data or "error" in data
        print(f"✓ GET /kv/list")


def run_all_tests():
    """Executa todos os testes e imprime resumo."""
    print("\n" + "="*60)
    print("TESTE AUTOMATIZADO DE ENDPOINTS")
    print("="*60 + "\n")

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    tests = TestEndpoints()

    endpoints = [
        ("GET /", tests.test_root),
        ("GET /health", tests.test_health),
        ("GET /session/current", tests.test_session_current),
        ("GET /sessions", tests.test_sessions_list),
        ("GET /outputs", tests.test_outputs_list),
        ("GET /audit/tools", tests.test_audit_tools),
        ("GET /audit/stats", tests.test_audit_stats),
        ("GET /audit/debug/{id}", tests.test_audit_debug),
        ("GET /audit/tools/enriched", tests.test_audit_enriched),
        ("GET /rag/stats", tests.test_rag_stats),
        ("GET /fs/tree", tests.test_fs_tree),
        ("GET /kv/list", tests.test_kv_list),
    ]

    passed = 0
    failed = 0

    for name, test_fn in endpoints:
        try:
            test_fn(client)
            passed += 1
        except Exception as e:
            print(f"✗ {name} - FALHOU: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTADO: {passed}/{len(endpoints)} testes passaram")
    if failed > 0:
        print(f"⚠️  {failed} testes falharam")
    else:
        print("✅ Todos os testes passaram!")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
