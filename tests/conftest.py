# =============================================================================
# CONFTEST - Fixtures compartilhadas para todos os testes
# =============================================================================
# Centraliza mocks, fixtures e configurações comuns
# =============================================================================

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# FIXTURES DE AMBIENTE
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configura ambiente de testes globalmente."""
    env_vars = {
        "AUTH_ENABLED": "false",
        "ANTHROPIC_API_KEY": "test-key-123",
        "RAG_API_KEY": "rag_test_key",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",  # Reduzir logs em testes
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def clean_env():
    """Limpa variáveis de ambiente para testes isolados."""
    with patch.dict(os.environ, {}, clear=True):
        yield


# =============================================================================
# FIXTURES DO FASTAPI
# =============================================================================


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    from fastapi.testclient import TestClient
    from server import app

    return TestClient(app)


@pytest.fixture
def async_client():
    """Cliente assíncrono para testes async."""
    from httpx import ASGITransport, AsyncClient
    from server import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# =============================================================================
# FIXTURES DO CLAUDE CLIENT
# =============================================================================


@pytest.fixture
def mock_claude_client():
    """Mock do cliente Claude."""
    mock = AsyncMock()
    mock.query = AsyncMock()
    mock.receive_response = AsyncMock()
    return mock


@pytest.fixture
def mock_claude_response():
    """Mock de resposta do Claude."""
    from dataclasses import dataclass

    @dataclass
    class MockTextBlock:
        text: str = "Resposta mock do Claude"

    @dataclass
    class MockMessage:
        content: list = None

        def __post_init__(self):
            self.content = [MockTextBlock()]

    return MockMessage()


# =============================================================================
# FIXTURES DO AGENTFS
# =============================================================================


@pytest.fixture
def mock_agentfs():
    """Mock completo do AgentFS."""
    mock = MagicMock()

    # KV Store
    mock.kv = AsyncMock()
    mock.kv.get = AsyncMock(return_value=None)
    mock.kv.set = AsyncMock()
    mock.kv.delete = AsyncMock()
    mock.kv.list = AsyncMock(return_value=[])

    # File System
    mock.fs = AsyncMock()
    mock.fs.write_file = AsyncMock()
    mock.fs.read_file = AsyncMock(return_value="")
    mock.fs.list_files = AsyncMock(return_value=[])

    # Tools
    mock.tools = AsyncMock()
    mock.tools.start = AsyncMock(return_value="call-123")
    mock.tools.success = AsyncMock()
    mock.tools.error = AsyncMock()
    mock.tools.finish = AsyncMock()

    # Lifecycle
    mock.close = AsyncMock()

    return mock


@pytest.fixture
def mock_agentfs_with_data():
    """Mock do AgentFS com dados pré-populados."""
    mock = MagicMock()
    _storage = {}

    async def mock_get(key):
        return _storage.get(key)

    async def mock_set(key, value):
        _storage[key] = value

    async def mock_delete(key):
        _storage.pop(key, None)

    async def mock_list(prefix=""):
        return [{"key": k} for k in _storage if k.startswith(prefix)]

    mock.kv = AsyncMock()
    mock.kv.get = mock_get
    mock.kv.set = mock_set
    mock.kv.delete = mock_delete
    mock.kv.list = mock_list
    mock._storage = _storage

    mock.close = AsyncMock()

    return mock


# =============================================================================
# FIXTURES DO RAG
# =============================================================================


@pytest.fixture
def mock_rag():
    """Mock do ClaudeRAG."""
    from dataclasses import dataclass

    @dataclass
    class MockSearchResult:
        content: str
        source: str
        similarity: float

    mock = MagicMock()
    mock.search = AsyncMock(
        return_value=[
            MockSearchResult(
                content="Conteúdo relevante do regulamento",
                source="regulamento.pdf",
                similarity=0.85,
            ),
            MockSearchResult(
                content="Outro trecho do documento",
                source="regulamento.pdf",
                similarity=0.75,
            ),
        ]
    )

    return mock


@pytest.fixture
def mock_rag_empty():
    """Mock do RAG sem resultados."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=[])
    return mock


# =============================================================================
# FIXTURES DO QUIZ
# =============================================================================


@pytest.fixture
def sample_alternativa():
    """Alternativa de exemplo para testes."""
    from quiz.models.schemas_supabase import Alternativa

    return Alternativa(
        texto="Texto da alternativa",
        correta=False,
        explicacao="Explicação do motivo",
    )


@pytest.fixture
def sample_pergunta_quiz():
    """PerguntaQuiz de exemplo para testes."""
    from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

    return PerguntaQuiz(
        numero=1,
        texto="Qual é a idade mínima para participar do programa Renda Extra?",
        alternativas={
            "A": Alternativa(
                texto="16 anos",
                correta=False,
                explicacao="Incorreto. A idade mínima é 18 anos.",
            ),
            "B": Alternativa(
                texto="18 anos",
                correta=True,
                explicacao="Correto! A idade mínima é 18 anos conforme regulamento.",
            ),
            "C": Alternativa(
                texto="21 anos",
                correta=False,
                explicacao="Incorreto. A idade mínima é 18 anos.",
            ),
            "D": Alternativa(
                texto="25 anos",
                correta=False,
                explicacao="Incorreto. A idade mínima é 18 anos.",
            ),
        },
        dificuldade="facil",
        topico="Elegibilidade",
        regulamento_ref="Item 2.1 - Renda Extra",
    )


@pytest.fixture
def sample_quiz_questions():
    """Lista de perguntas de quiz para testes."""
    from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

    questions = []
    dificuldades = ["facil", "media", "media", "dificil", "facil"]
    topicos = ["Elegibilidade", "Pagamentos", "Penalidades", "LGPD", "Geral"]

    for i in range(5):
        questions.append(
            PerguntaQuiz(
                numero=i + 1,
                texto=f"Pergunta {i + 1} sobre {topicos[i]}?",
                alternativas={
                    "A": Alternativa(
                        texto="Opção A",
                        correta=(i % 4 == 0),
                        explicacao="Explicação A",
                    ),
                    "B": Alternativa(
                        texto="Opção B",
                        correta=(i % 4 == 1),
                        explicacao="Explicação B",
                    ),
                    "C": Alternativa(
                        texto="Opção C",
                        correta=(i % 4 == 2),
                        explicacao="Explicação C",
                    ),
                    "D": Alternativa(
                        texto="Opção D",
                        correta=(i % 4 == 3),
                        explicacao="Explicação D",
                    ),
                },
                dificuldade=dificuldades[i],
                topico=topicos[i],
                regulamento_ref=f"Item {i + 1}.1 - Renda Extra",
            )
        )

    return questions


@pytest.fixture
def sample_quiz_state(sample_pergunta_quiz):
    """QuizState de exemplo para testes."""
    from quiz.models.state import QuizState

    state = QuizState(quiz_id="test-123")
    state.add_question(1, sample_pergunta_quiz)
    state.context = "Contexto RAG de teste"

    return state


# =============================================================================
# FIXTURES DE LOGGING
# =============================================================================


@pytest.fixture
def capture_logs(caplog):
    """Captura logs para verificação em testes."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


# =============================================================================
# FIXTURES DE TEMPO
# =============================================================================


@pytest.fixture
def mock_datetime():
    """Mock do datetime para testes com tempo."""
    fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


# =============================================================================
# FIXTURES DE CACHE
# =============================================================================


@pytest.fixture
def clean_quiz_cache():
    """Limpa cache de quiz entre testes."""
    from quiz.engine.quiz_engine import QuizEngine

    # Limpar cache antes do teste
    QuizEngine._memory_cache.clear()
    yield
    # Limpar após o teste
    QuizEngine._memory_cache.clear()


# =============================================================================
# HELPERS
# =============================================================================


@pytest.fixture
def make_mock_response():
    """Factory para criar respostas mock do Claude."""

    def _make_response(text: str):
        from dataclasses import dataclass

        @dataclass
        class MockTextBlock:
            text: str

        @dataclass
        class MockResponse:
            answer: str
            content: list

        return MockResponse(answer=text, content=[MockTextBlock(text=text)])

    return _make_response


@pytest.fixture
def json_quiz_response():
    """JSON de resposta de pergunta do Claude."""
    return """
{
    "texto": "Qual é o prazo máximo para pagamento?",
    "alternativas": {
        "A": {"texto": "15 dias", "correta": false, "explicacao": "Incorreto."},
        "B": {"texto": "30 dias", "correta": true, "explicacao": "Correto!"},
        "C": {"texto": "45 dias", "correta": false, "explicacao": "Incorreto."},
        "D": {"texto": "60 dias", "correta": false, "explicacao": "Incorreto."}
    },
    "dificuldade": "media",
    "topico": "Pagamentos",
    "regulamento_ref": "Item 5.1 - Renda Extra"
}
"""
