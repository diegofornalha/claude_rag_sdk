"""Topic Deduplication Engine - Motor de deduplicacao de topicos."""

import logging

from ..prompts import TOPIC_KEYWORDS

logger = logging.getLogger(__name__)


class TopicDeduplicationEngine:
    """Motor de deduplicação de tópicos para evitar perguntas repetitivas.

    Usa um mapeamento de palavras-chave para tópicos canônicos,
    permitindo identificar quando duas perguntas diferentes abordam
    o mesmo assunto.

    Example:
        >>> engine = TopicDeduplicationEngine()
        >>> topic = engine.extract_topic("Qual é o prazo de pagamento das recompensas?")
        >>> print(topic)  # "prazo de pagamento das recompensas"
        >>> engine.is_duplicate("Quando os valores são pagos?", ["prazo de pagamento das recompensas"])
        True
    """

    # Usar keywords do templates.py
    TOPIC_KEYWORDS = TOPIC_KEYWORDS

    def __init__(self):
        """Inicializa engine com keywords padrão."""
        self._keywords = dict(self.TOPIC_KEYWORDS)

    def extract_topic(self, question_text: str) -> str:
        """Extrai o tópico principal de uma pergunta.

        Busca palavras-chave na pergunta para mapear para um tópico canônico.
        A ordem das keywords importa - mais específicas primeiro.

        Args:
            question_text: Texto da pergunta

        Returns:
            Tópico canônico identificado ou primeiros 60 chars como fallback
        """
        q_lower = question_text.lower()

        for keyword, topic in self._keywords.items():
            if keyword in q_lower:
                return topic

        # Fallback: usar primeiros 60 chars
        return question_text[:60].strip()

    def is_duplicate(self, question_text: str, used_topics: list[str]) -> bool:
        """Verifica se a pergunta é sobre um tópico já usado.

        Args:
            question_text: Texto da nova pergunta
            used_topics: Lista de tópicos já utilizados

        Returns:
            True se o tópico já foi usado, False caso contrário
        """
        topic = self.extract_topic(question_text)
        is_dup = topic in used_topics

        if is_dup:
            logger.debug(f"Tópico duplicado detectado: '{topic}'")

        return is_dup

    def add_keyword(self, keyword: str, topic: str) -> None:
        """Adiciona nova keyword ao mapeamento.

        Args:
            keyword: Palavra-chave a detectar (lowercase)
            topic: Tópico canônico associado
        """
        self._keywords[keyword.lower()] = topic

    def get_topic_for_question(self, question_text: str) -> str:
        """Alias para extract_topic - compatibilidade."""
        return self.extract_topic(question_text)

    def validate_and_get_topic(
        self, question_text: str, used_topics: list[str]
    ) -> tuple[bool, str]:
        """Valida pergunta e retorna tópico.

        Args:
            question_text: Texto da pergunta
            used_topics: Tópicos já usados

        Returns:
            Tuple de (is_valid, topic) onde is_valid é False se duplicado
        """
        topic = self.extract_topic(question_text)
        is_valid = topic not in used_topics
        return is_valid, topic
