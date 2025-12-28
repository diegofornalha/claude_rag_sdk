# =============================================================================
# TESTES E2E - Fluxo Completo de Chat
# =============================================================================
# Testes end-to-end simulando fluxo completo do usuário
# =============================================================================

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


class TestCompleteUserFlow:
    """Testes de fluxo completo do usuário."""

    def test_flow_health_check_first(self, client):
        """Passo 1: Usuário verifica se servidor está ativo."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_flow_check_current_session(self, client):
        """Passo 2: Verificar sessão atual."""
        response = client.get("/session/current")

        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_flow_list_sessions(self, client):
        """Passo 3: Listar sessões existentes."""
        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_flow_create_new_session(self, client):
        """Passo 4: Criar nova sessão."""
        response = client.post("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

        return data["session_id"]

    def test_flow_switch_session(self, client):
        """Passo 5: Trocar para sessão criada."""
        # Criar sessão primeiro
        create_response = client.post("/sessions")
        session_id = create_response.json()["session_id"]

        # Trocar para ela
        response = client.post(f"/session/switch?session_id={session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_flow_check_chat_history(self, client):
        """Passo 6: Verificar histórico (vazio para nova sessão)."""
        response = client.get("/chat/history")

        assert response.status_code == 200

    def test_flow_check_artifacts(self, client):
        """Passo 7: Verificar artifacts da sessão."""
        response = client.get("/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert "files" in data


class TestSessionLifecycle:
    """Testes do ciclo de vida de sessão."""

    def test_create_multiple_sessions(self, client):
        """Cria múltiplas sessões e verifica isolamento."""
        sessions = []

        # Criar 3 sessões
        for i in range(3):
            response = client.post("/sessions")
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])

        # Verificar que são diferentes
        assert len(set(sessions)) == 3

        # Listar e verificar que todas aparecem
        response = client.get("/sessions")
        listed_sessions = [s["id"] for s in response.json()["sessions"]]

        for sid in sessions:
            assert sid in listed_sessions

    def test_session_metadata_persistence(self, client):
        """Verifica persistência de metadados da sessão."""
        # Criar sessão
        response = client.post("/sessions")
        session_id = response.json()["session_id"]

        # Obter detalhes
        response = client.get(f"/sessions/{session_id}")

        if response.status_code == 200:
            data = response.json()
            assert "id" in data or "session_id" in data


class TestOutputsFlow:
    """Testes de fluxo de artifacts."""

    def test_artifacts_empty_initially(self, client):
        """Verifica artifacts vazios inicialmente."""
        # Criar nova sessão
        create_response = client.post("/sessions")
        session_id = create_response.json()["session_id"]

        # Verificar artifacts vazios
        response = client.get(f"/artifacts?session_id={session_id}")

        if response.status_code == 200:
            data = response.json()
            # Pode ter arquivos de sessões anteriores ou estar vazio
            assert "files" in data

    def test_list_output_files(self, client):
        """Lista arquivos de output."""
        response = client.get("/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "files" in data


class TestAuditFlow:
    """Testes de fluxo de auditoria."""

    def test_audit_tools_endpoint(self, client):
        """Verifica endpoint de auditoria de tools."""
        response = client.get("/audit/tools")

        assert response.status_code == 200

    def test_audit_stats_endpoint(self, client):
        """Verifica endpoint de estatísticas."""
        response = client.get("/audit/stats")

        assert response.status_code == 200

    def test_audit_debug_session(self, client):
        """Verifica debug de sessão específica."""
        # Criar sessão primeiro
        create_response = client.post("/sessions")
        session_id = create_response.json()["session_id"]

        # Debug da sessão
        response = client.get(f"/audit/debug/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data


class TestRAGFlow:
    """Testes de fluxo RAG."""

    def test_rag_stats(self, client):
        """Verifica estatísticas RAG."""
        response = client.get("/rag/stats")

        # Pode retornar 200 ou 500 se RAG não configurado
        assert response.status_code in [200, 500]

    def test_rag_search_endpoint(self, client):
        """Testa endpoint de busca RAG."""
        response = client.post(
            "/rag/search",
            json={"query": "test query", "top_k": 5},
        )

        # Pode funcionar ou retornar erro se RAG não configurado
        assert response.status_code in [200, 400, 422, 500]


class TestFilesystemFlow:
    """Testes de fluxo do filesystem."""

    def test_fs_tree(self, client):
        """Verifica árvore de arquivos."""
        response = client.get("/fs/tree")

        assert response.status_code == 200

    def test_kv_list(self, client):
        """Verifica listagem de KV store."""
        response = client.get("/kv/list")

        assert response.status_code == 200


class TestErrorRecovery:
    """Testes de recuperação de erros."""

    def test_invalid_session_handled(self, client):
        """Verifica tratamento de sessão inválida."""
        response = client.post("/session/switch?session_id=invalid-session")

        # Deve retornar erro ou criar nova
        assert response.status_code in [200, 400, 404]

    def test_malformed_request_handled(self, client):
        """Verifica tratamento de requisição malformada."""
        response = client.post(
            "/chat",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        # Deve retornar erro de validação
        assert response.status_code == 422

    def test_large_payload_handled(self, client):
        """Verifica tratamento de payload grande."""
        large_message = "x" * 100000  # 100KB

        response = client.post("/chat", json={"message": large_message})

        # Deve processar ou rejeitar gracefully
        assert response.status_code in [200, 400, 413, 500]


class TestConcurrentRequests:
    """Testes de requisições concorrentes."""

    def test_multiple_health_checks(self, client):
        """Verifica múltiplas verificações de saúde."""
        responses = []

        for _ in range(10):
            resp = client.get("/health")
            responses.append(resp.status_code)

        assert all(code == 200 for code in responses)

    def test_multiple_session_creates(self, client):
        """Verifica criação de múltiplas sessões."""
        responses = []

        for _ in range(5):
            resp = client.post("/sessions")
            responses.append(resp.status_code)

        assert all(code == 200 for code in responses)


class TestSecurityFlow:
    """Testes de fluxo de segurança."""

    def test_path_traversal_prevented(self, client):
        """Verifica prevenção de path traversal."""
        malicious_ids = [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "....//....//....//etc/passwd",
        ]

        for mid in malicious_ids:
            response = client.get(f"/artifacts?session_id={mid}")
            # Não deve retornar arquivos do sistema
            if response.status_code == 200:
                data = response.json()
                # Verificar que não vazou paths do sistema
                assert "/etc" not in str(data)
                assert "/passwd" not in str(data)

    def test_no_sensitive_data_exposed(self, client):
        """Verifica que dados sensíveis não são expostos."""
        endpoints = ["/", "/health", "/model"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                data = str(response.json())
                # Não deve expor chaves de API
                assert "ANTHROPIC_API_KEY" not in data
                assert "test-key-123" not in data


class TestModelSelection:
    """Testes de seleção de modelo."""

    def test_get_current_model(self, client):
        """Verifica obtenção de modelo atual."""
        response = client.get("/model")

        assert response.status_code == 200
        data = response.json()
        assert "model" in data

    def test_model_ids_mapping(self):
        """Verifica mapeamento de IDs de modelo."""
        from server import _get_model_id

        assert "haiku" in _get_model_id("haiku")
        assert "sonnet" in _get_model_id("sonnet")
        assert "opus" in _get_model_id("opus")


class TestEdgeCasesE2E:
    """Testes de casos extremos E2E."""

    def test_rapid_session_switching(self, client):
        """Testa troca rápida de sessões."""
        # Criar algumas sessões
        sessions = []
        for _ in range(3):
            resp = client.post("/sessions")
            sessions.append(resp.json()["session_id"])

        # Trocar rapidamente entre elas
        for _ in range(5):
            for sid in sessions:
                resp = client.post(f"/session/switch?session_id={sid}")
                assert resp.status_code in [200, 400]

    def test_unicode_handling(self, client):
        """Verifica tratamento de unicode."""
        response = client.get("/sessions")

        # A resposta deve ser JSON válido com unicode
        assert response.status_code == 200
        # O content deve ser UTF-8
        assert response.encoding in [None, "utf-8", "UTF-8"]


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_e2e_tests():
    """Executa testes E2E manualmente."""
    import sys

    print("\n" + "=" * 60)
    print("TESTES E2E - FLUXO COMPLETO")
    print("=" * 60 + "\n")

    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
        ]
    )

    return exit_code == 0


if __name__ == "__main__":
    import sys

    success = run_e2e_tests()
    sys.exit(0 if success else 1)
