"""Quiz Store - Abstração sobre AgentFS para persistência de quiz."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentfs_sdk import AgentFS

from ..models.schemas import QuizQuestion
from ..models.state import QuizState

logger = logging.getLogger(__name__)


class QuizStore:
    """Abstração sobre AgentFS para persistência de quiz.

    Gerencia o armazenamento de estados de quiz no KV store do AgentFS,
    permitindo persistência entre requests e recuperação de sessões.

    Estrutura de chaves:
        - quiz:{quiz_id}:state -> Estado completo do quiz (QuizState)
        - quiz:{quiz_id}:questions:{index} -> Pergunta individual (backup)

    Example:
        >>> store = QuizStore(agentfs)
        >>> state = QuizState(quiz_id="abc123")
        >>> await store.save_state(state)
        >>> loaded = await store.load_state("abc123")
    """

    KEY_PREFIX = "quiz"

    def __init__(self, agentfs: AgentFS):
        """Inicializa store com instância do AgentFS.

        Args:
            agentfs: Instância configurada do AgentFS
        """
        self.agentfs = agentfs

    def _state_key(self, quiz_id: str) -> str:
        """Gera chave para estado do quiz."""
        return f"{self.KEY_PREFIX}:{quiz_id}:state"

    def _question_key(self, quiz_id: str, index: int) -> str:
        """Gera chave para pergunta individual."""
        return f"{self.KEY_PREFIX}:{quiz_id}:questions:{index}"

    async def save_state(self, state: QuizState) -> None:
        """Persiste estado completo no KV store.

        Args:
            state: Estado do quiz a persistir
        """
        key = self._state_key(state.quiz_id)
        data = state.to_dict()

        # Serializar questions para JSON-compatible (int keys -> str keys)
        data["questions"] = {
            str(k): v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in state.questions.items()
        }

        await self.agentfs.kv.set(key, data)
        logger.debug(f"Quiz state salvo: {state.quiz_id}")

    async def load_state(self, quiz_id: str) -> QuizState | None:
        """Carrega estado do KV store.

        Args:
            quiz_id: ID do quiz

        Returns:
            QuizState se encontrado, None caso contrário
        """
        key = self._state_key(quiz_id)
        data = await self.agentfs.kv.get(key)

        if not data:
            logger.debug(f"Quiz não encontrado: {quiz_id}")
            return None

        return QuizState.from_dict(data)

    async def update_question(
        self, quiz_id: str, index: int, question: QuizQuestion
    ) -> None:
        """Atualiza pergunta individual no estado.

        Carrega estado, adiciona pergunta e salva de volta.

        Args:
            quiz_id: ID do quiz
            index: Índice da pergunta (1-10)
            question: Pergunta a adicionar
        """
        state = await self.load_state(quiz_id)
        if state is None:
            logger.error(f"Quiz não encontrado para update: {quiz_id}")
            raise ValueError(f"Quiz {quiz_id} não encontrado")

        state.add_question(index, question)
        await self.save_state(state)

        # Backup individual (para recuperação)
        backup_key = self._question_key(quiz_id, index)
        await self.agentfs.kv.set(backup_key, question.model_dump())

        logger.debug(f"Pergunta {index} atualizada: {quiz_id}")

    async def add_topic(self, quiz_id: str, topic: str) -> None:
        """Adiciona tópico à lista de usados.

        Args:
            quiz_id: ID do quiz
            topic: Tópico a adicionar
        """
        state = await self.load_state(quiz_id)
        if state is None:
            raise ValueError(f"Quiz {quiz_id} não encontrado")

        state.add_topic(topic)
        await self.save_state(state)

    async def mark_complete(self, quiz_id: str) -> None:
        """Marca quiz como completo.

        Args:
            quiz_id: ID do quiz
        """
        state = await self.load_state(quiz_id)
        if state is None:
            raise ValueError(f"Quiz {quiz_id} não encontrado")

        state.mark_complete()
        await self.save_state(state)
        logger.info(f"Quiz marcado como completo: {quiz_id}")

    async def set_error(self, quiz_id: str, error: str) -> None:
        """Define mensagem de erro no estado.

        Args:
            quiz_id: ID do quiz
            error: Mensagem de erro
        """
        state = await self.load_state(quiz_id)
        if state is None:
            raise ValueError(f"Quiz {quiz_id} não encontrado")

        state.set_error(error)
        await self.save_state(state)
        logger.error(f"Erro no quiz {quiz_id}: {error}")

    async def list_quizzes(self) -> list[str]:
        """Lista todos os quiz IDs armazenados.

        Returns:
            Lista de quiz IDs
        """
        prefix = f"{self.KEY_PREFIX}:"
        entries = await self.agentfs.kv.list(prefix=prefix)

        # Extrair quiz IDs únicos das chaves
        quiz_ids = set()
        for entry in entries:
            key = entry.get("key", "") if isinstance(entry, dict) else str(entry)
            parts = key.split(":")
            if len(parts) >= 2:
                quiz_ids.add(parts[1])

        return list(quiz_ids)

    async def delete_quiz(self, quiz_id: str) -> None:
        """Remove quiz do store.

        Remove estado principal e backups de perguntas.

        Args:
            quiz_id: ID do quiz
        """
        # Deletar estado principal
        state_key = self._state_key(quiz_id)
        await self.agentfs.kv.delete(state_key)

        # Deletar backups de perguntas (1-10)
        for i in range(1, 11):
            question_key = self._question_key(quiz_id, i)
            try:
                await self.agentfs.kv.delete(question_key)
            except Exception:
                pass  # Ignorar se não existir

        logger.info(f"Quiz deletado: {quiz_id}")

    async def get_question(self, quiz_id: str, index: int) -> QuizQuestion | None:
        """Busca pergunta específica do estado.

        Args:
            quiz_id: ID do quiz
            index: Índice da pergunta (1-10)

        Returns:
            QuizQuestion se pronta, None caso contrário
        """
        state = await self.load_state(quiz_id)
        if state is None:
            return None

        return state.questions.get(index)

    async def is_question_ready(self, quiz_id: str, index: int) -> bool:
        """Verifica se uma pergunta específica está pronta.

        Args:
            quiz_id: ID do quiz
            index: Índice da pergunta (1-10)

        Returns:
            True se pronta, False caso contrário
        """
        state = await self.load_state(quiz_id)
        if state is None:
            return False

        return state.is_ready(index)

    async def get_status(self, quiz_id: str) -> dict[str, Any]:
        """Retorna status resumido do quiz.

        Args:
            quiz_id: ID do quiz

        Returns:
            Dict com status do quiz
        """
        state = await self.load_state(quiz_id)
        if state is None:
            return {
                "quiz_id": quiz_id,
                "found": False,
                "error": "Quiz não encontrado",
            }

        return {
            "quiz_id": quiz_id,
            "found": True,
            "generated_count": state.generated_count,
            "total_questions": 10,
            "complete": state.complete,
            "error": state.error,
            "max_score": state.max_score,
            "questions_ready": list(state.questions.keys()),
            "previous_topics": state.previous_topics,
        }
