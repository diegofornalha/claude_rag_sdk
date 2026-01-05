# =============================================================================
# TESTES - Quiz Engine Module
# =============================================================================
# Testes unitarios para motor de geracao de quiz
# =============================================================================

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQuizEngineInit:
    """Testes para inicializacao do QuizEngine."""

    def test_init_with_agentfs(self, mock_agentfs, mock_rag):
        """Verifica inicializacao com AgentFS."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        assert engine.store is not None
        assert engine.rag == mock_rag
        assert engine.llm_factory is not None
        assert engine.dedup is not None

    def test_init_without_agentfs(self, mock_rag):
        """Verifica inicializacao sem AgentFS."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(None, mock_rag)

        assert engine.store is None
        assert engine.rag == mock_rag

    def test_init_with_custom_factory(self, mock_agentfs, mock_rag):
        """Verifica inicializacao com factory customizada."""
        from quiz.engine.quiz_engine import QuizEngine
        from quiz.llm.factory import LLMClientFactory

        custom_factory = LLMClientFactory()
        engine = QuizEngine(mock_agentfs, mock_rag, llm_factory=custom_factory)

        assert engine.llm_factory == custom_factory


class TestQuizEngineMemoryCache:
    """Testes para cache em memoria."""

    def test_get_state_not_found(self, mock_agentfs, mock_rag, clean_quiz_cache):
        """Verifica retorno None quando estado nao existe."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        result = engine._get_state("nonexistent")

        assert result is None

    def test_set_and_get_state(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica salvamento e recuperacao de estado."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        engine._set_state(sample_quiz_state)
        result = engine._get_state("test-123")

        assert result is not None
        assert result.quiz_id == "test-123"

    def test_cache_is_shared_between_instances(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica que cache e compartilhado globalmente."""
        from quiz.engine.quiz_engine import QuizEngine

        engine1 = QuizEngine(mock_agentfs, mock_rag)
        engine2 = QuizEngine(mock_agentfs, mock_rag)

        engine1._set_state(sample_quiz_state)
        result = engine2._get_state("test-123")

        assert result is not None


class TestQuizEngineExtractJson:
    """Testes para extracao de JSON."""

    def test_extract_plain_json(self, mock_agentfs, mock_rag):
        """Verifica extracao de JSON puro."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        text = '{"texto": "Pergunta?", "dificuldade": "facil"}'
        result = engine._extract_json(text)

        assert result["texto"] == "Pergunta?"
        assert result["dificuldade"] == "facil"

    def test_extract_json_from_markdown(self, mock_agentfs, mock_rag):
        """Verifica extracao de JSON de bloco markdown."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        text = """
Aqui esta a pergunta:

```json
{"texto": "Pergunta?", "dificuldade": "media"}
```
"""
        result = engine._extract_json(text)

        assert result["texto"] == "Pergunta?"
        assert result["dificuldade"] == "media"

    def test_extract_json_with_surrounding_text(self, mock_agentfs, mock_rag):
        """Verifica extracao de JSON com texto ao redor."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        text = 'Texto antes {"texto": "Pergunta?", "topico": "Geral"} texto depois'
        result = engine._extract_json(text)

        assert result["texto"] == "Pergunta?"
        assert result["topico"] == "Geral"

    def test_extract_json_invalid(self, mock_agentfs, mock_rag):
        """Verifica erro com JSON invalido."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        with pytest.raises(json.JSONDecodeError):
            engine._extract_json("texto sem json valido")


class TestQuizEngineCreateQuestion:
    """Testes para criacao de perguntas."""

    def test_create_question_from_data_supabase_format(self, mock_agentfs, mock_rag):
        """Verifica criacao a partir de dados formato Supabase."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        q_data = {
            "texto": "Qual a idade minima para participar?",
            "alternativas": {
                "A": {"texto": "16 anos", "correta": False, "explicacao": "Errado"},
                "B": {"texto": "18 anos", "correta": True, "explicacao": "Correto!"},
                "C": {"texto": "21 anos", "correta": False, "explicacao": "Errado"},
                "D": {"texto": "25 anos", "correta": False, "explicacao": "Errado"},
            },
            "dificuldade": "facil",
            "topico": "Elegibilidade",
            "regulamento_ref": "Item 2.1",
        }

        question = engine._create_question_from_data(1, q_data, "easy")

        assert question.numero == 1
        assert "idade" in question.texto.lower()
        assert question.alternativa_correta == "B"
        assert question.dificuldade == "facil"

    def test_create_question_difficulty_normalization(self, mock_agentfs, mock_rag):
        """Verifica normalizacao de dificuldade."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        base_alts = {
            "A": {"texto": "A", "correta": True, "explicacao": "X"},
            "B": {"texto": "B", "correta": False, "explicacao": "X"},
            "C": {"texto": "C", "correta": False, "explicacao": "X"},
            "D": {"texto": "D", "correta": False, "explicacao": "X"},
        }

        # Testa EN -> PT-BR
        q_data = {
            "texto": "Pergunta teste sobre o programa",
            "alternativas": base_alts,
            "dificuldade": "easy",
            "topico": "Geral",
            "regulamento_ref": "Ref",
        }

        question = engine._create_question_from_data(1, q_data, "easy")
        assert question.dificuldade == "facil"

        # Testa medium -> media
        q_data["dificuldade"] = "medium"
        question = engine._create_question_from_data(2, q_data, "medium")
        assert question.dificuldade == "media"

        # Testa hard -> dificil
        q_data["dificuldade"] = "hard"
        question = engine._create_question_from_data(3, q_data, "hard")
        assert question.dificuldade == "dificil"


class TestQuizEngineCreateFallback:
    """Testes para criacao de pergunta fallback."""

    def test_create_fallback_question(self, mock_agentfs, mock_rag):
        """Verifica criacao de fallback."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        question = engine._create_fallback_question(5, "medium")

        assert question.numero == 5
        assert question.dificuldade == "media"
        assert len(question.alternativas) == 4
        assert question.alternativa_correta == "A"

    def test_create_fallback_difficulty_mapping(self, mock_agentfs, mock_rag):
        """Verifica mapeamento de dificuldade no fallback."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        # Easy -> facil
        q = engine._create_fallback_question(1, "easy")
        assert q.dificuldade == "facil"

        # Hard -> dificil
        q = engine._create_fallback_question(2, "hard")
        assert q.dificuldade == "dificil"


class TestQuizEngineStartQuiz:
    """Testes para inicio de quiz."""

    @pytest.mark.asyncio
    async def test_start_quiz_success(self, mock_agentfs, mock_rag, clean_quiz_cache, json_quiz_response):
        """Verifica inicio de quiz com sucesso."""
        from quiz.engine.quiz_engine import QuizEngine

        # Mock do LLM factory
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = json_quiz_response
        mock_agent.query = AsyncMock(return_value=mock_response)

        mock_factory = MagicMock()
        mock_factory.create_first_question_agent = MagicMock(return_value=mock_agent)
        mock_factory.create_remaining_questions_agent = MagicMock(return_value=mock_agent)

        engine = QuizEngine(mock_agentfs, mock_rag, llm_factory=mock_factory)

        quiz_id, first_question = await engine.start_quiz()

        assert quiz_id is not None
        assert len(quiz_id) == 8
        assert first_question is not None
        assert first_question.numero == 1

    @pytest.mark.asyncio
    async def test_start_quiz_creates_state(self, mock_agentfs, mock_rag, clean_quiz_cache, json_quiz_response):
        """Verifica que estado e criado no cache."""
        from quiz.engine.quiz_engine import QuizEngine

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = json_quiz_response
        mock_agent.query = AsyncMock(return_value=mock_response)

        mock_factory = MagicMock()
        mock_factory.create_first_question_agent = MagicMock(return_value=mock_agent)
        mock_factory.create_remaining_questions_agent = MagicMock(return_value=mock_agent)

        engine = QuizEngine(mock_agentfs, mock_rag, llm_factory=mock_factory)

        quiz_id, _ = await engine.start_quiz()

        state = engine._get_state(quiz_id)
        assert state is not None
        assert state.quiz_id == quiz_id
        assert 1 in state.questions


class TestQuizEngineGetQuestion:
    """Testes para buscar pergunta."""

    @pytest.mark.asyncio
    async def test_get_question_ready(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica busca de pergunta pronta."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)
        engine._set_state(sample_quiz_state)

        question = await engine.get_question("test-123", 1, timeout=1.0)

        assert question is not None
        assert question.numero == 1

    @pytest.mark.asyncio
    async def test_get_question_timeout(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica timeout quando pergunta nao esta pronta."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)
        engine._set_state(sample_quiz_state)

        # Pergunta 2 nao existe
        question = await engine.get_question("test-123", 2, timeout=0.5)

        assert question is None

    @pytest.mark.asyncio
    async def test_get_question_nonexistent_quiz(self, mock_agentfs, mock_rag, clean_quiz_cache):
        """Verifica timeout quando quiz nao existe."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        question = await engine.get_question("nonexistent", 1, timeout=0.5)

        assert question is None


class TestQuizEngineGetStatus:
    """Testes para status do quiz."""

    @pytest.mark.asyncio
    async def test_get_status_found(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica status de quiz existente."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)
        engine._set_state(sample_quiz_state)

        status = await engine.get_status("test-123")

        assert status["found"] is True
        assert status["quiz_id"] == "test-123"
        assert "generated_count" in status

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, mock_agentfs, mock_rag, clean_quiz_cache):
        """Verifica status de quiz inexistente."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        status = await engine.get_status("nonexistent")

        assert status["found"] is False
        assert "erro" in status.get("error", "erro").lower() or "não encontrado" in status.get("error", "").lower()


class TestQuizEngineIsComplete:
    """Testes para verificacao de completude."""

    @pytest.mark.asyncio
    async def test_is_complete_true(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica True quando completo."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)
        sample_quiz_state.mark_complete()
        engine._set_state(sample_quiz_state)

        result = await engine.is_complete("test-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_complete_false(self, mock_agentfs, mock_rag, clean_quiz_cache, sample_quiz_state):
        """Verifica False quando incompleto."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)
        engine._set_state(sample_quiz_state)

        result = await engine.is_complete("test-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_complete_nonexistent(self, mock_agentfs, mock_rag, clean_quiz_cache):
        """Verifica False para quiz inexistente."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        result = await engine.is_complete("nonexistent")

        assert result is False


class TestQuizEngineRagContext:
    """Testes para busca de contexto RAG."""

    @pytest.mark.asyncio
    async def test_get_rag_context_success(self, mock_agentfs, mock_rag):
        """Verifica busca de contexto com sucesso."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag)

        context = await engine._get_rag_context()

        assert context != ""
        assert "Trecho" in context

    @pytest.mark.asyncio
    async def test_get_rag_context_empty(self, mock_agentfs, mock_rag_empty):
        """Verifica retorno vazio quando RAG nao retorna nada."""
        from quiz.engine.quiz_engine import QuizEngine

        engine = QuizEngine(mock_agentfs, mock_rag_empty)

        context = await engine._get_rag_context()

        assert context == ""
