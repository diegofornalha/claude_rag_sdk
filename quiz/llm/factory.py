"""LLM Client Factory - Abstração para criação de AgentEngine."""

from claude_rag_sdk.agent import AgentEngine
from claude_rag_sdk.options import AgentModel, ClaudeRAGOptions

from ..prompts import QUIZ_SYSTEM_PROMPT


class LLMClientFactory:
    """Factory para criar AgentEngine com diferentes configurações.

    Centraliza a criação de agentes LLM para o quiz, permitindo:
    - Configuração consistente de system prompt
    - Seleção de modelo (HAIKU para velocidade, OPUS para qualidade)
    - IDs únicos por quiz para tracking

    Example:
        >>> factory = LLMClientFactory()
        >>> agent = factory.create_first_question_agent("quiz-abc123")
        >>> response = await agent.query("Gere a primeira pergunta...")
    """

    DEFAULT_MODEL = AgentModel.HAIKU  # Rápido e econômico
    QUALITY_MODEL = AgentModel.OPUS  # Melhor qualidade

    @staticmethod
    def create_agent(
        agent_id: str,
        model: AgentModel = AgentModel.HAIKU,
        system_prompt: str | None = None,
    ) -> AgentEngine:
        """Cria um AgentEngine genérico para quiz.

        Args:
            agent_id: ID único do agente (usado para tracking)
            model: Modelo Claude a usar (HAIKU, SONNET, OPUS)
            system_prompt: Prompt de sistema customizado

        Returns:
            AgentEngine configurado
        """
        options = ClaudeRAGOptions(
            id=agent_id,
            agent_model=model,
            system_prompt=system_prompt or QUIZ_SYSTEM_PROMPT,
        )
        return AgentEngine(options=options)

    @classmethod
    def create_first_question_agent(cls, quiz_id: str) -> AgentEngine:
        """Cria agente otimizado para primeira pergunta.

        Usa HAIKU para resposta rápida na P1 (primeira impressão do usuário).

        Args:
            quiz_id: ID do quiz

        Returns:
            AgentEngine configurado para P1
        """
        return cls.create_agent(
            agent_id=f"quiz-{quiz_id}-p1",
            model=AgentModel.HAIKU,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )

    @classmethod
    def create_remaining_questions_agent(cls, quiz_id: str) -> AgentEngine:
        """Cria agente para perguntas P2-P10.

        Usa HAIKU para manter velocidade na geração em background.

        Args:
            quiz_id: ID do quiz

        Returns:
            AgentEngine configurado para P2-P10
        """
        return cls.create_agent(
            agent_id=f"quiz-{quiz_id}-remaining",
            model=AgentModel.HAIKU,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )

    @classmethod
    def create_quality_agent(cls, quiz_id: str) -> AgentEngine:
        """Cria agente de alta qualidade (OPUS) para casos especiais.

        Use quando qualidade é mais importante que velocidade.

        Args:
            quiz_id: ID do quiz

        Returns:
            AgentEngine com OPUS
        """
        return cls.create_agent(
            agent_id=f"quiz-{quiz_id}-quality",
            model=AgentModel.OPUS,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )

    @classmethod
    def create_batch_agent(cls, quiz_id: str) -> AgentEngine:
        """Cria agente para geração em lote (todas as perguntas de uma vez).

        Usa OPUS para melhor qualidade em geração complexa.

        Args:
            quiz_id: ID do quiz

        Returns:
            AgentEngine configurado para batch
        """
        return cls.create_agent(
            agent_id=f"quiz-{quiz_id}-batch",
            model=AgentModel.OPUS,
            system_prompt=QUIZ_SYSTEM_PROMPT,
        )
