"""Quiz State - Gerenciamento de estado do quiz."""

from dataclasses import dataclass, field
from typing import Any

from .schemas import QuizQuestion


@dataclass
class QuizState:
    """Estado completo de um quiz em andamento.

    Attributes:
        quiz_id: ID unico do quiz
        questions: Dict de perguntas geradas (index -> QuizQuestion)
        generated_count: Numero de perguntas geradas ate o momento
        complete: Se a geracao foi concluida
        error: Mensagem de erro (se houver)
        max_score: Pontuacao maxima possivel
        context: Contexto RAG usado para geracao
        previous_topics: Lista de topicos ja usados (para deduplicacao)
    """

    quiz_id: str
    questions: dict[int, QuizQuestion] = field(default_factory=dict)
    generated_count: int = 0
    complete: bool = False
    error: str | None = None
    max_score: int = 0
    context: str = ""
    previous_topics: list[str] = field(default_factory=list)

    def is_ready(self, index: int) -> bool:
        """Verifica se uma pergunta especifica esta pronta."""
        return index in self.questions

    def add_question(self, index: int, question: QuizQuestion) -> None:
        """Adiciona uma pergunta ao estado."""
        self.questions[index] = question
        self.generated_count = max(self.generated_count, index)
        self.max_score = sum(q.points for q in self.questions.values())

    def mark_complete(self) -> None:
        """Marca o quiz como completo."""
        self.complete = True
        self.max_score = sum(q.points for q in self.questions.values())

    def set_error(self, error: str) -> None:
        """Define mensagem de erro."""
        self.error = error

    def add_topic(self, topic: str) -> None:
        """Adiciona topico a lista de usados."""
        if topic not in self.previous_topics:
            self.previous_topics.append(topic)

    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionario (para persistencia)."""
        return {
            "quiz_id": self.quiz_id,
            "questions": {
                k: v.model_dump() if hasattr(v, "model_dump") else v
                for k, v in self.questions.items()
            },
            "generated_count": self.generated_count,
            "complete": self.complete,
            "error": self.error,
            "max_score": self.max_score,
            "context": self.context,
            "previous_topics": self.previous_topics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuizState":
        """Cria instancia a partir de dicionario."""
        questions = {}
        for k, v in data.get("questions", {}).items():
            if isinstance(v, dict):
                questions[int(k)] = QuizQuestion(**v)
            else:
                questions[int(k)] = v

        return cls(
            quiz_id=data["quiz_id"],
            questions=questions,
            generated_count=data.get("generated_count", 0),
            complete=data.get("complete", False),
            error=data.get("error"),
            max_score=data.get("max_score", 0),
            context=data.get("context", ""),
            previous_topics=data.get("previous_topics", []),
        )
