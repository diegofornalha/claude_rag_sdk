# =============================================================================
# CONFTEST - Pytest Fixtures Globais
# =============================================================================
# Fixtures mockadas para testes unitários sem dependências externas
# =============================================================================

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adicionar root ao path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# FIXTURES DE AMBIENTE
# =============================================================================


@pytest.fixture(autouse=True)
def setup_test_env():
    """Configura variáveis de ambiente para testes."""
    env_vars = {
        "AUTH_ENABLED": "false",
        "ANTHROPIC_API_KEY": "test-key-123",
        "RAG_API_KEY": "rag_test_key_for_testing",
        "EMBEDDING_MODEL": "bge-small",
        "DEFAULT_TOP_K": "5",
        "ADAPTIVE_TOPK_ENABLED": "false",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Retorna path temporário para banco de dados."""
    return tmp_path / "test_rag.db"


# =============================================================================
# FIXTURES DE MOCK - AGENTFS
# =============================================================================


@pytest.fixture
def mock_agentfs():
    """Mock completo do AgentFS."""
    mock = MagicMock()

    # Filesystem mock
    mock.filesystem.read.return_value = b"test content"
    mock.filesystem.write.return_value = True
    mock.filesystem.exists.return_value = True
    mock.filesystem.list.return_value = ["file1.txt", "file2.txt"]

    # KV Store mock
    mock.kv.get.return_value = {"key": "value"}
    mock.kv.set.return_value = True
    mock.kv.delete.return_value = True

    # Toolcalls mock
    mock.toolcalls.list.return_value = []
    mock.toolcalls.add.return_value = {"id": "tc_123"}

    return mock


@pytest.fixture
def mock_agentfs_manager(mock_agentfs):
    """Mock do AgentFSManager."""
    with patch("claude_rag_sdk.core.agentfs_manager.AgentFSManager") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_client.return_value = mock_agentfs
        mock_instance.is_available = True
        mock_cls.return_value = mock_instance
        yield mock_instance


# =============================================================================
# FIXTURES DE MOCK - CLAUDE CLIENT
# =============================================================================


@pytest.fixture
def mock_claude_response():
    """Resposta mock do Claude."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Esta é uma resposta de teste do Claude."}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@pytest.fixture
def mock_claude_client(mock_claude_response):
    """Mock do cliente Claude."""
    mock = MagicMock()

    # Mock de mensagem síncrona
    mock.messages.create.return_value = MagicMock(
        id="msg_123",
        content=[MagicMock(type="text", text="Resposta de teste")],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=MagicMock(input_tokens=100, output_tokens=50),
    )

    # Mock de streaming
    async def mock_stream():
        yield MagicMock(type="content_block_start", content_block=MagicMock(text=""))
        yield MagicMock(type="content_block_delta", delta=MagicMock(text="Resposta "))
        yield MagicMock(type="content_block_delta", delta=MagicMock(text="de teste"))
        yield MagicMock(type="message_stop")

    mock.messages.stream = MagicMock(return_value=mock_stream())

    return mock


@pytest.fixture
def mock_anthropic(mock_claude_client):
    """Patch do módulo anthropic."""
    with patch("anthropic.Anthropic", return_value=mock_claude_client):
        yield mock_claude_client


# =============================================================================
# FIXTURES DE MOCK - EMBEDDINGS
# =============================================================================


@pytest.fixture
def mock_embedding_model():
    """Mock do modelo de embedding."""
    mock = MagicMock()

    # Retorna embedding de dimensão 384 (bge-small)
    import numpy as np

    def mock_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [np.random.rand(384).tolist() for _ in texts]

    mock.embed.side_effect = mock_embed
    mock.passage_embed.side_effect = mock_embed
    mock.query_embed.side_effect = mock_embed

    return mock


@pytest.fixture
def mock_fastembed(mock_embedding_model):
    """Patch do fastembed."""
    with patch("fastembed.TextEmbedding", return_value=mock_embedding_model):
        yield mock_embedding_model


# =============================================================================
# FIXTURES DE MOCK - DATABASE
# =============================================================================


@pytest.fixture
def mock_db_connection(temp_db_path):
    """Mock de conexão com banco de dados."""
    import apsw

    # Criar banco em memória para testes
    conn = apsw.Connection(":memory:")

    # Criar schema básico
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            nome TEXT NOT NULL,
            tipo TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    """
    )

    return conn


# =============================================================================
# FIXTURES DE MOCK - FASTAPI
# =============================================================================


@pytest.fixture
def test_client():
    """Cliente de teste FastAPI."""
    from fastapi.testclient import TestClient

    # Importar após patches
    from server import app

    return TestClient(app)


@pytest.fixture
def async_test_client():
    """Cliente de teste assíncrono."""
    from httpx import ASGITransport, AsyncClient

    from server import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# =============================================================================
# FIXTURES DE DADOS DE TESTE
# =============================================================================


@pytest.fixture
def sample_document():
    """Documento de exemplo para testes."""
    return {
        "nome": "Política de IA",
        "tipo": "policy",
        "content": "Este documento descreve a política de uso de IA na empresa. "
        "A IA deve ser usada de forma ética e responsável. "
        "Todos os modelos devem ser auditados regularmente.",
        "source": "test_document.txt",
    }


@pytest.fixture
def sample_documents():
    """Lista de documentos de exemplo."""
    return [
        {
            "nome": "Política de IA",
            "tipo": "policy",
            "content": "Política de uso de IA na empresa.",
            "source": "policy.txt",
        },
        {
            "nome": "Manual do Funcionário",
            "tipo": "manual",
            "content": "Manual com diretrizes para funcionários.",
            "source": "manual.txt",
        },
        {
            "nome": "Guia de Segurança",
            "tipo": "guide",
            "content": "Guia de práticas de segurança da informação.",
            "source": "security.txt",
        },
    ]


@pytest.fixture
def sample_query():
    """Query de exemplo para testes."""
    return "Quais são as políticas de IA da empresa?"


@pytest.fixture
def sample_chat_message():
    """Mensagem de chat de exemplo."""
    return {
        "role": "user",
        "content": "Olá, me explique sobre as políticas de IA.",
    }


@pytest.fixture
def sample_session_id():
    """ID de sessão de exemplo."""
    return "test-session-123"


# =============================================================================
# FIXTURES UTILITÁRIAS
# =============================================================================


@pytest.fixture
def capture_logs(caplog):
    """Captura logs durante testes."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def freeze_time():
    """Congela o tempo para testes determinísticos."""
    from datetime import datetime, timezone
    from unittest.mock import patch

    frozen_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = frozen_time
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield frozen_time
