"""Quiz Router - Endpoints FastAPI refatorados."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

if TYPE_CHECKING:
    pass

import app_state

from .engine.quiz_engine import QuizEngine
from .engine.scoring_engine import QuizScoringEngine
from .models.enums import QuizDifficulty
from .models.schemas import (
    GenerateQuizRequest,
    GenerateQuizResponse,
    QuizAnswerRequest,
    QuizAnswerResponse,
    QuizQuestion,
    QuizResultsRequest,
    QuizResultsResponse,
    StartQuizResponse,
)
from .models.state import QuizState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz"])

# =============================================================================
# IN-MEMORY CACHE (fallback quando AgentFS não disponível)
# =============================================================================

# Cache global de estados de quiz (quiz_id -> QuizState)
_quiz_cache: dict[str, QuizState] = {}

# Referência ao engine singleton para geração em background
_engine_instance: QuizEngine | None = None


def get_cached_state(quiz_id: str) -> QuizState | None:
    """Busca estado do cache em memória."""
    return _quiz_cache.get(quiz_id)


def set_cached_state(state: QuizState) -> None:
    """Salva estado no cache em memória."""
    _quiz_cache[state.quiz_id] = state


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


async def get_quiz_engine() -> QuizEngine:
    """Dependency para obter QuizEngine configurado."""
    global _engine_instance

    agentfs = await app_state.get_agentfs()
    rag = await app_state.get_rag()

    # Criar engine com callbacks para cache
    engine = QuizEngine(agentfs=agentfs, rag=rag)

    # Armazenar referência global para background tasks
    _engine_instance = engine

    return engine


async def get_scoring_engine() -> QuizScoringEngine:
    """Dependency para obter ScoringEngine."""
    return QuizScoringEngine()


def verify_api_key() -> str | None:
    """Placeholder para autenticação (usar implementação existente)."""
    # TODO: Integrar com sistema de autenticação existente
    return None


# =============================================================================
# LAZY GENERATION ENDPOINTS (Principal)
# =============================================================================


@router.post("/start", response_model=StartQuizResponse)
async def start_quiz(
    engine: QuizEngine = Depends(get_quiz_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Inicia um quiz com lazy generation.

    - Valida que existem documentos no RAG antes de iniciar
    - Gera a primeira pergunta dinamicamente baseada no documento
    - Inicia geração das perguntas 2-10 em background
    - Frontend pode buscar perguntas via /question/{quiz_id}/{index}

    Esta arquitetura permite UX fluida enquanto as demais
    perguntas são geradas em paralelo.
    """
    # Validar RAG
    search_results = await engine.rag.search(
        "programa Renda Extra Ton regras benefícios",
        top_k=5,
    )

    if not search_results:
        logger.error("Quiz não pode iniciar: RAG vazio")
        raise HTTPException(
            status_code=400,
            detail="Nenhum documento encontrado no RAG. Faça a ingestão do regulamento primeiro.",
        )

    # Iniciar quiz
    quiz_id, first_question = await engine.start_quiz()

    logger.info(f"[Quiz {quiz_id}] Iniciado com lazy generation")

    return StartQuizResponse(
        quiz_id=quiz_id,
        total_questions=10,
        first_question=first_question,
    )


@router.get("/question/{quiz_id}/{index}", response_model=QuizQuestion)
async def get_question(
    quiz_id: str,
    index: int,
    engine: QuizEngine = Depends(get_quiz_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Busca uma pergunta específica do quiz.

    - Se a pergunta já foi gerada, retorna imediatamente
    - Se ainda está sendo gerada, aguarda com polling (max 30s)
    - Se houver erro ou timeout, retorna HTTP 408/404

    Args:
        quiz_id: ID do quiz retornado por /start
        index: Número da pergunta (1-10)
    """
    if index < 1 or index > 10:
        raise HTTPException(status_code=400, detail="Index deve ser entre 1 e 10")

    # Buscar pergunta com timeout
    question = await engine.get_question(quiz_id, index, timeout=30.0)

    if question is None:
        # Verificar se quiz existe
        status = await engine.get_status(quiz_id)
        if not status.get("found"):
            raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

        if status.get("error"):
            raise HTTPException(status_code=500, detail=f"Erro na geração: {status['error']}")

        raise HTTPException(
            status_code=408,
            detail=f"Timeout aguardando pergunta {index}. "
            f"Geradas até agora: {status.get('generated_count', 0)}",
        )

    return question


@router.get("/status/{quiz_id}")
async def get_quiz_status(
    quiz_id: str,
    engine: QuizEngine = Depends(get_quiz_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Retorna status do quiz (para debug/monitoramento).

    Útil para verificar quantas perguntas já foram geradas.
    """
    status = await engine.get_status(quiz_id)

    if not status.get("found"):
        raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

    return status


@router.get("/all/{quiz_id}")
async def get_all_questions(
    quiz_id: str,
    engine: QuizEngine = Depends(get_quiz_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Retorna todas as perguntas do quiz (quando geração completa).

    Útil para o frontend obter max_score e calcular resultado final.
    """
    status = await engine.get_status(quiz_id)

    if not status.get("found"):
        raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

    if not status.get("complete"):
        raise HTTPException(
            status_code=202,
            detail=f"Quiz ainda em geração. Perguntas prontas: {status.get('generated_count', 0)}/10",
        )

    # Buscar todas as perguntas
    questions = []
    for i in range(1, 11):
        question = await engine.store.get_question(quiz_id, i)
        if question:
            questions.append(question)

    return {
        "quiz_id": quiz_id,
        "total_questions": len(questions),
        "max_score": status.get("max_score", 0),
        "questions": questions,
    }


# =============================================================================
# ANSWER & RESULTS ENDPOINTS
# =============================================================================


@router.post("/answer", response_model=QuizAnswerResponse)
async def evaluate_answer(
    request: QuizAnswerRequest,
    engine: QuizEngine = Depends(get_quiz_engine),
    scoring: QuizScoringEngine = Depends(get_scoring_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Avalia uma resposta individual.

    - Retorna se está correta
    - Fornece feedback educativo específico
    - Explica a resposta correta
    - Oferece dica de aprendizado
    """
    question = await engine.store.get_question(request.quiz_id, request.question_id)

    if question is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pergunta {request.question_id} não encontrada no quiz {request.quiz_id}",
        )

    result = scoring.evaluate_answer(question, request.selected_index)

    return QuizAnswerResponse(
        is_correct=result["is_correct"],
        points_earned=result["points_earned"],
        correct_index=result["correct_index"],
        feedback=result["feedback"],
        explanation=result["explanation"],
        learning_tip=result["learning_tip"],
    )


@router.post("/results", response_model=QuizResultsResponse)
async def calculate_results(
    request: QuizResultsRequest,
    engine: QuizEngine = Depends(get_quiz_engine),
    scoring: QuizScoringEngine = Depends(get_scoring_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Calcula resultado final e ranking.

    - Analisa desempenho por dificuldade
    - Calcula percentual de aproveitamento
    - Atribui ranking na trilha de carreira
    - Fornece feedback personalizado
    """
    status = await engine.get_status(request.quiz_id)

    if not status.get("found"):
        raise HTTPException(status_code=404, detail=f"Quiz {request.quiz_id} não encontrado")

    if not status.get("complete"):
        raise HTTPException(
            status_code=400,
            detail=f"Quiz ainda em geração. Aguarde conclusão: {status.get('generated_count', 0)}/10",
        )

    # Validar número de respostas
    if len(request.answers) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Esperado 10 respostas, recebido {len(request.answers)}",
        )

    # Buscar todas as perguntas
    questions = []
    for i in range(1, 11):
        question = await engine.store.get_question(request.quiz_id, i)
        if question:
            questions.append(question)

    if len(questions) != 10:
        raise HTTPException(
            status_code=500,
            detail=f"Erro: quiz tem apenas {len(questions)} perguntas",
        )

    # Calcular resultado
    result = scoring.calculate_score(questions, request.answers)

    return QuizResultsResponse(
        total_questions=result["total_questions"],
        correct_answers=result["correct_answers"],
        score=result["score"],
        max_score=result["max_score"],
        percentage=result["percentage"],
        rank=result["rank"],
        rank_title=result["rank_title"],
        rank_message=result["rank_message"],
        breakdown=result["breakdown"],
    )


# =============================================================================
# BATCH GENERATION (Legacy - Mantido para compatibilidade)
# =============================================================================


@router.post("/generate", response_model=GenerateQuizResponse)
async def generate_quiz(
    request: GenerateQuizRequest,
    engine: QuizEngine = Depends(get_quiz_engine),
    _api_key: str | None = Depends(verify_api_key),
):
    """Gera um quiz dinâmico usando RAG + Claude (modo batch).

    NOTA: Prefira usar /start para lazy generation com melhor UX.

    - Busca contexto relevante no documento ingerido
    - Gera todas as questões de uma vez
    - Cada questão tem feedback educativo detalhado
    - Pontuação ponderada por dificuldade
    """

    logger.info("Gerando quiz (batch)", num_questions=request.num_questions)

    # Iniciar quiz e aguardar todas as perguntas
    quiz_id, first_question = await engine.start_quiz()

    # Aguardar conclusão (max 120s)
    for _ in range(120):
        if await engine.is_complete(quiz_id):
            break
        await asyncio.sleep(1)

    # Buscar todas as perguntas
    questions = []
    for i in range(1, request.num_questions + 1):
        q = await engine.store.get_question(quiz_id, i)
        if q:
            questions.append(q)

    if not questions:
        raise HTTPException(status_code=500, detail="Erro ao gerar quiz")

    difficulty_breakdown = {
        "easy": sum(1 for q in questions if q.difficulty == QuizDifficulty.EASY),
        "medium": sum(1 for q in questions if q.difficulty == QuizDifficulty.MEDIUM),
        "hard": sum(1 for q in questions if q.difficulty == QuizDifficulty.HARD),
    }

    max_score = sum(q.points for q in questions)

    return GenerateQuizResponse(
        quiz_id=quiz_id,
        title="Quiz: Renda Extra Ton",
        description="Avalie seu conhecimento sobre o programa",
        total_questions=len(questions),
        max_score=max_score,
        questions=questions,
        difficulty_breakdown=difficulty_breakdown,
    )
