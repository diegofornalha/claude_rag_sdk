"""Quiz Schemas - Modelos Pydantic para request/response."""

from pydantic import BaseModel, Field

from .enums import QuizDifficulty, QuizRank


class QuizOption(BaseModel):
    """Alternativa de multipla escolha."""

    label: str = Field(..., description="Letra da alternativa (A, B, C, D)")
    text: str = Field(..., description="Texto da alternativa")


class QuizQuestion(BaseModel):
    """Questao do quiz com metadata educacional."""

    id: int = Field(..., description="ID da questao (1-N)")
    question: str = Field(..., description="Enunciado da questao")
    options: list[QuizOption] = Field(..., description="4 alternativas")
    correct_index: int = Field(..., ge=0, le=3, description="Indice da resposta correta (0-3)")
    difficulty: QuizDifficulty = Field(..., description="Nivel de dificuldade")
    points: int = Field(..., description="Pontos atribuidos (1=facil, 2=medio, 3=dificil)")
    explanation: str = Field(..., description="Explicacao detalhada da resposta correta")
    wrong_feedback: dict[int, str] = Field(
        ...,
        description="Feedback especifico para cada alternativa incorreta (index -> feedback)",
    )
    learning_tip: str = Field(..., description="Dica de memorizacao ou conceito-chave")
    source_reference: str = Field(
        default="", description="Referencia ao trecho do documento (pagina/secao)"
    )


class GenerateQuizRequest(BaseModel):
    """Request para geracao de quiz."""

    num_questions: int = Field(default=10, ge=5, le=20, description="Numero de questoes (5-20)")
    focus_topics: list[str] = Field(
        default=[],
        description="Topicos especificos para focar (vazio = todos os topicos do documento)",
    )
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        description="Distribuicao de dificuldade (deve somar 1.0)",
    )


class GenerateQuizResponse(BaseModel):
    """Response com quiz gerado."""

    quiz_id: str = Field(..., description="ID unico do quiz gerado")
    title: str = Field(..., description="Titulo do quiz")
    description: str = Field(..., description="Descricao do conteudo")
    total_questions: int = Field(..., description="Total de questoes")
    max_score: int = Field(..., description="Pontuacao maxima possivel")
    questions: list[QuizQuestion] = Field(..., description="Lista de questoes")
    difficulty_breakdown: dict[str, int] = Field(
        ..., description="Contagem por dificuldade (easy/medium/hard)"
    )


class QuizAnswerRequest(BaseModel):
    """Request para avaliar uma resposta."""

    quiz_id: str = Field(..., description="ID do quiz")
    question_id: int = Field(..., description="ID da questao")
    selected_index: int = Field(..., ge=0, le=3, description="Indice selecionado (0-3)")


class QuizAnswerResponse(BaseModel):
    """Response da avaliacao de resposta."""

    is_correct: bool = Field(..., description="Se a resposta esta correta")
    points_earned: int = Field(..., description="Pontos ganhos (0 se errado)")
    correct_index: int = Field(..., description="Indice da resposta correta")
    feedback: str = Field(..., description="Feedback educativo detalhado")
    explanation: str = Field(..., description="Explicacao da resposta correta")
    learning_tip: str = Field(..., description="Dica de aprendizado")


class QuizResultsRequest(BaseModel):
    """Request para calcular resultado final."""

    quiz_id: str = Field(..., description="ID do quiz")
    answers: list[int] = Field(..., description="Lista de indices selecionados para cada questao")


class QuizResultsResponse(BaseModel):
    """Response com resultado final e ranking."""

    total_questions: int = Field(..., description="Total de questoes")
    correct_answers: int = Field(..., description="Respostas corretas")
    score: int = Field(..., description="Pontuacao obtida")
    max_score: int = Field(..., description="Pontuacao maxima")
    percentage: float = Field(..., description="Percentual de aproveitamento")
    rank: QuizRank = Field(..., description="Ranking alcancado")
    rank_title: str = Field(..., description="Titulo do ranking")
    rank_message: str = Field(..., description="Mensagem personalizada de feedback")
    breakdown: dict[str, dict[str, int]] = Field(
        ..., description="Analise por dificuldade (corretas/total)"
    )


class StartQuizResponse(BaseModel):
    """Response ao iniciar quiz com lazy generation."""

    quiz_id: str = Field(..., description="ID unico do quiz")
    total_questions: int = Field(default=10, description="Total de questoes")
    first_question: QuizQuestion = Field(..., description="Primeira pergunta")


class QuestionStatusResponse(BaseModel):
    """Status de uma pergunta especifica."""

    quiz_id: str
    index: int
    ready: bool = Field(..., description="Se a pergunta esta pronta")
    question: QuizQuestion | None = Field(None, description="Pergunta se pronta")
