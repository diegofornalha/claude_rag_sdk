# =============================================================================
# TESTES - Quiz Store Module
# =============================================================================
# Testes unitarios para persistencia de quiz no AgentFS
# =============================================================================

from unittest.mock import AsyncMock

import pytest


class TestQuizStoreKeys:
    """Testes para geracao de chaves."""

    def test_state_key_format(self, mock_agentfs):
        """Verifica formato da chave de estado."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs)

        key = store._state_key("abc-123")

        assert key == "quiz:abc-123:state"

    def test_question_key_format(self, mock_agentfs):
        """Verifica formato da chave de pergunta."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs)

        key = store._question_key("abc-123", 5)

        assert key == "quiz:abc-123:questions:5"


class TestQuizStoreSaveState:
    """Testes para salvar estado."""

    @pytest.mark.asyncio
    async def test_save_state_basic(self, mock_agentfs, sample_quiz_state):
        """Verifica salvamento basico de estado."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs)

        await store.save_state(sample_quiz_state)

        mock_agentfs.kv.set.assert_called_once()
        call_args = mock_agentfs.kv.set.call_args
        assert "quiz:test-123:state" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_state_serializes_questions(self, mock_agentfs, sample_quiz_state):
        """Verifica que perguntas sao serializadas."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs)

        await store.save_state(sample_quiz_state)

        call_args = mock_agentfs.kv.set.call_args
        data = call_args[0][1]

        assert "questions" in data
        # Chaves devem ser strings no JSON
        assert "1" in data["questions"] or 1 in data["questions"]


class TestQuizStoreLoadState:
    """Testes para carregar estado."""

    @pytest.mark.asyncio
    async def test_load_state_not_found(self, mock_agentfs):
        """Verifica retorno None quando nao encontrado."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        result = await store.load_state("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_state_found(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica carregamento de estado existente."""
        from quiz.storage.quiz_store import QuizStore

        # Salvar primeiro
        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        # Carregar
        loaded = await store.load_state("test-123")

        assert loaded is not None
        assert loaded.quiz_id == "test-123"


class TestQuizStoreUpdateQuestion:
    """Testes para atualizar pergunta."""

    @pytest.mark.asyncio
    async def test_update_question_not_found(self, mock_agentfs, sample_pergunta_quiz):
        """Verifica erro quando quiz nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        with pytest.raises(ValueError) as exc_info:
            await store.update_question("nonexistent", 1, sample_pergunta_quiz)

        assert "não encontrado" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_question_success(
        self, mock_agentfs_with_data, sample_quiz_state, sample_pergunta_quiz
    ):
        """Verifica atualizacao de pergunta com sucesso."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        # Criar nova pergunta
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        new_question = PerguntaQuiz(
            numero=2,
            texto="Nova pergunta sobre o programa Renda Extra",
            alternativas={
                "A": Alternativa(texto="A", correta=True, explicacao="Certo"),
                "B": Alternativa(texto="B", correta=False, explicacao="Errado"),
                "C": Alternativa(texto="C", correta=False, explicacao="Errado"),
                "D": Alternativa(texto="D", correta=False, explicacao="Errado"),
            },
            dificuldade="media",
            topico="Geral",
            regulamento_ref="Ref",
        )

        await store.update_question("test-123", 2, new_question)

        # Verificar
        loaded = await store.load_state("test-123")
        assert 2 in loaded.questions


class TestQuizStoreAddTopic:
    """Testes para adicionar topico."""

    @pytest.mark.asyncio
    async def test_add_topic_not_found(self, mock_agentfs):
        """Verifica erro quando quiz nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        with pytest.raises(ValueError):
            await store.add_topic("nonexistent", "Topico")

    @pytest.mark.asyncio
    async def test_add_topic_success(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica adicao de topico."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        await store.add_topic("test-123", "Novo Topico")

        loaded = await store.load_state("test-123")
        assert "Novo Topico" in loaded.previous_topics


class TestQuizStoreMarkComplete:
    """Testes para marcar como completo."""

    @pytest.mark.asyncio
    async def test_mark_complete_not_found(self, mock_agentfs):
        """Verifica erro quando quiz nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        with pytest.raises(ValueError):
            await store.mark_complete("nonexistent")

    @pytest.mark.asyncio
    async def test_mark_complete_success(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica marcacao como completo."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        await store.mark_complete("test-123")

        loaded = await store.load_state("test-123")
        assert loaded.complete is True


class TestQuizStoreSetError:
    """Testes para definir erro."""

    @pytest.mark.asyncio
    async def test_set_error_not_found(self, mock_agentfs):
        """Verifica erro quando quiz nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        with pytest.raises(ValueError):
            await store.set_error("nonexistent", "Erro de teste")

    @pytest.mark.asyncio
    async def test_set_error_success(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica definicao de erro."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        await store.set_error("test-123", "Erro de teste")

        loaded = await store.load_state("test-123")
        assert loaded.error == "Erro de teste"


class TestQuizStoreListQuizzes:
    """Testes para listar quizzes."""

    @pytest.mark.asyncio
    async def test_list_quizzes_empty(self, mock_agentfs):
        """Verifica lista vazia."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.list = AsyncMock(return_value=[])
        store = QuizStore(mock_agentfs)

        result = await store.list_quizzes()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_quizzes_with_data(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica lista com dados."""
        from quiz.models.state import QuizState
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)

        # Salvar multiplos quizzes
        await store.save_state(sample_quiz_state)

        state2 = QuizState(quiz_id="test-456")
        await store.save_state(state2)

        result = await store.list_quizzes()

        assert len(result) == 2
        assert "test-123" in result
        assert "test-456" in result


class TestQuizStoreDeleteQuiz:
    """Testes para deletar quiz."""

    @pytest.mark.asyncio
    async def test_delete_quiz(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica delecao de quiz."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        await store.delete_quiz("test-123")

        loaded = await store.load_state("test-123")
        assert loaded is None


class TestQuizStoreGetQuestion:
    """Testes para buscar pergunta."""

    @pytest.mark.asyncio
    async def test_get_question_not_found(self, mock_agentfs):
        """Verifica retorno None quando nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        result = await store.get_question("nonexistent", 1)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_question_success(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica busca de pergunta existente."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        result = await store.get_question("test-123", 1)

        assert result is not None
        assert result.numero == 1


class TestQuizStoreIsQuestionReady:
    """Testes para verificar pergunta pronta."""

    @pytest.mark.asyncio
    async def test_is_question_ready_false(self, mock_agentfs):
        """Verifica False quando quiz nao existe."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        result = await store.is_question_ready("nonexistent", 1)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_question_ready_true(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica True quando pergunta existe."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        result = await store.is_question_ready("test-123", 1)

        assert result is True


class TestQuizStoreGetStatus:
    """Testes para status do quiz."""

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, mock_agentfs):
        """Verifica status quando nao encontrado."""
        from quiz.storage.quiz_store import QuizStore

        mock_agentfs.kv.get = AsyncMock(return_value=None)
        store = QuizStore(mock_agentfs)

        result = await store.get_status("nonexistent")

        assert result["found"] is False
        assert "erro" in result.get("error", "erro").lower() or "não encontrado" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_status_found(self, mock_agentfs_with_data, sample_quiz_state):
        """Verifica status quando encontrado."""
        from quiz.storage.quiz_store import QuizStore

        store = QuizStore(mock_agentfs_with_data)
        await store.save_state(sample_quiz_state)

        result = await store.get_status("test-123")

        assert result["found"] is True
        assert result["quiz_id"] == "test-123"
        assert "generated_count" in result
        assert "complete" in result
