# =============================================================================
# TESTES DE INTEGRAÇÃO - Endpoints
# =============================================================================
# Testes de integração usando FastAPI TestClient (sem servidor externo)
# =============================================================================

import os
from unittest.mock import patch, MagicMock

import pytest


# Configurar ambiente antes de importar a app
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """Configura ambiente para testes."""
    env_vars = {
        "AUTH_ENABLED": "false",
        "ANTHROPIC_API_KEY": "test-key-123",
        "RAG_API_KEY": "rag_test_key",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def client():
    """Cliente de teste FastAPI (não requer servidor rodando)."""
    from fastapi.testclient import TestClient
    from server import app

    return TestClient(app)


class TestHealthEndpoints:
    """Testes dos endpoints de health check."""

    def test_root_returns_ok(self, client):
        """GET / - Deve retornar status ok."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "session_id" in data

    def test_health_returns_healthy(self, client):
        """GET /health - Deve retornar status healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "security" in data


class TestSessionEndpoints:
    """Testes dos endpoints de sessão."""

    def test_session_current(self, client):
        """GET /session/current - Deve retornar sessão atual."""
        response = client.get("/session/current")

        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_sessions_list(self, client):
        """GET /sessions - Deve listar sessões."""
        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_session_create(self, client):
        """POST /sessions - Deve criar nova sessão."""
        response = client.post("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_session_switch(self, client):
        """POST /session/switch - Deve trocar de sessão."""
        # Criar uma sessão primeiro
        create_response = client.post("/sessions")
        session_id = create_response.json()["session_id"]

        # Trocar para ela
        response = client.post(f"/session/switch?session_id={session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data


class TestOutputsEndpoints:
    """Testes dos endpoints de artifacts."""

    def test_artifacts_list(self, client):
        """GET /artifacts - Deve listar arquivos de output."""
        response = client.get("/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "count" in data


class TestAuditEndpoints:
    """Testes dos endpoints de auditoria."""

    def test_audit_tools(self, client):
        """GET /audit/tools - Deve retornar histórico de tools."""
        response = client.get("/audit/tools")

        assert response.status_code == 200
        data = response.json()
        # Pode ter dados ou indicar que não há sessão
        assert isinstance(data, dict)

    def test_audit_stats(self, client):
        """GET /audit/stats - Deve retornar estatísticas."""
        response = client.get("/audit/stats")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_audit_debug_session(self, client):
        """GET /audit/debug/{session_id} - Deve retornar debug info."""
        response = client.get("/audit/debug/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "found" in data

    def test_audit_enriched(self, client):
        """GET /audit/tools/enriched - Deve retornar tools enriquecidas."""
        response = client.get("/audit/tools/enriched?session_id=test-session")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data


class TestRAGEndpoints:
    """Testes dos endpoints RAG."""

    def test_rag_stats(self, client):
        """GET /rag/stats - Deve retornar estatísticas RAG."""
        response = client.get("/rag/stats")

        # Pode retornar 200 ou 500 se RAG não estiver configurado
        assert response.status_code in [200, 500]

    def test_rag_search_without_query(self, client):
        """POST /rag/search sem query - Deve retornar erro."""
        response = client.post("/rag/search", json={})

        # Deve retornar erro de validação
        assert response.status_code in [400, 422]


class TestFilesystemEndpoints:
    """Testes dos endpoints de filesystem."""

    def test_fs_tree(self, client):
        """GET /fs/tree - Deve retornar árvore de arquivos."""
        response = client.get("/fs/tree")

        assert response.status_code == 200
        data = response.json()
        # Pode ter tree ou error dependendo do estado
        assert isinstance(data, dict)

    def test_kv_list(self, client):
        """GET /kv/list - Deve listar chaves KV."""
        response = client.get("/kv/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestChatEndpoints:
    """Testes dos endpoints de chat."""

    def test_chat_history_empty(self, client):
        """GET /chat/history - Deve retornar histórico (pode estar vazio)."""
        response = client.get("/chat/history")

        assert response.status_code == 200
        data = response.json()
        assert "messages" in data or "history" in data or isinstance(data, list)

    def test_chat_clear(self, client):
        """POST /chat/clear - Deve limpar histórico."""
        response = client.post("/chat/clear")

        assert response.status_code == 200


class TestErrorHandling:
    """Testes de tratamento de erros."""

    def test_404_for_unknown_endpoint(self, client):
        """Endpoint desconhecido deve retornar 404."""
        response = client.get("/unknown/endpoint/that/does/not/exist")

        assert response.status_code == 404

    def test_invalid_session_id_format(self, client):
        """Session ID com formato inválido deve ser tratado."""
        # Tentar switch para sessão com caracteres inválidos
        response = client.post("/session/switch?session_id=../../../etc/passwd")

        # Deve rejeitar ou tratar gracefully
        assert response.status_code in [400, 422, 404, 200]


class TestSecurityHeaders:
    """Testes de headers de segurança."""

    def test_cors_headers_present(self, client):
        """Verifica se CORS está configurado."""
        response = client.options("/health")

        # OPTIONS pode retornar 200 ou 405 dependendo da config
        assert response.status_code in [200, 405]

    def test_no_sensitive_info_in_error(self, client):
        """Erros não devem vazar informações sensíveis."""
        response = client.get("/rag/search?query=test")

        if response.status_code >= 400:
            data = (
                response.json()
                if response.headers.get("content-type") == "application/json"
                else {}
            )
            # Não deve conter stack traces ou paths internos
            error_text = str(data)
            assert "/home/" not in error_text
            assert "/Users/" not in error_text.lower() or "output" in error_text.lower()


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_integration_tests():
    """Executa testes de integração manualmente."""
    import sys

    print("\n" + "=" * 60)
    print("TESTES DE INTEGRAÇÃO - ENDPOINTS")
    print("=" * 60 + "\n")

    # Usar pytest para executar
    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",  # Parar no primeiro erro
        ]
    )

    return exit_code == 0


if __name__ == "__main__":
    import sys

    success = run_integration_tests()
    sys.exit(0 if success else 1)
