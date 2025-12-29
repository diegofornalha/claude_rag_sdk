"""Evaluation endpoints - Avaliação de qualidade do RAG."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from claude_rag_sdk.core.auth import verify_api_key

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# =============================================================================
# Models
# =============================================================================


class EvaluationRequest(BaseModel):
    """Request para avaliação."""

    question_ids: list[int] | None = None  # None = todas
    max_questions: int | None = None


class QuestionEvalRequest(BaseModel):
    """Request para avaliar uma pergunta específica."""

    question: str
    expected_answer: str | None = None
    expected_sources: str | None = None
    evidence_keywords: list[str] | None = None


# =============================================================================
# State
# =============================================================================

# Armazena último relatório
_last_report: dict | None = None
_evaluation_running: bool = False


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/config")
async def get_evaluation_config():
    """Retorna configuração de avaliação (qa_pairs, etc)."""
    from agents.evaluator import get_evaluator

    evaluator = get_evaluator()

    return {
        "total_questions": len(evaluator.qa_pairs),
        "questions": [
            {
                "id": q.get("id"),
                "question": q.get("question", "")[:100] + "...",
                "expected_sources": q.get("expected_sources", ""),
            }
            for q in evaluator.qa_pairs
        ],
        "config_loaded": bool(evaluator.config),
    }


@router.post("/run")
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
):
    """Executa avaliação completa (pode demorar).

    Returns:
        Relatório de avaliação com métricas
    """
    global _evaluation_running, _last_report

    if _evaluation_running:
        raise HTTPException(
            status_code=409,
            detail="Avaliação já em andamento. Aguarde ou consulte /evaluate/status"
        )

    from agents.evaluator import get_evaluator

    evaluator = get_evaluator()

    _evaluation_running = True

    try:
        report = await evaluator.run_evaluation(
            question_ids=request.question_ids,
            max_questions=request.max_questions,
        )

        _last_report = report.to_dict()

        return _last_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        _evaluation_running = False


@router.post("/question")
async def evaluate_single_question(
    request: QuestionEvalRequest,
    _: None = Depends(verify_api_key),
):
    """Avalia uma pergunta específica (ad-hoc)."""
    from agents.evaluator import get_evaluator

    evaluator = get_evaluator()

    qa_pair = {
        "id": 0,
        "question": request.question,
        "expected_answer": request.expected_answer or "",
        "expected_sources": request.expected_sources or "",
        "evidence_keywords": request.evidence_keywords or [],
    }

    try:
        metrics = await evaluator.evaluate_question(qa_pair)
        return metrics.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status")
async def evaluation_status():
    """Retorna status da avaliação."""
    return {
        "running": _evaluation_running,
        "has_report": _last_report is not None,
        "last_report_summary": {
            "pass_rate": _last_report["summary"]["pass_rate"],
            "total_questions": _last_report["summary"]["total_questions"],
            "timestamp": _last_report["timestamp"],
        } if _last_report else None,
    }


@router.get("/report")
async def get_last_report():
    """Retorna último relatório de avaliação."""
    if not _last_report:
        raise HTTPException(
            status_code=404,
            detail="Nenhum relatório disponível. Execute /evaluate/run primeiro."
        )

    return _last_report


@router.get("/metrics")
async def get_evaluation_metrics():
    """Retorna métricas resumidas da última avaliação."""
    if not _last_report:
        return {
            "available": False,
            "message": "Execute /evaluate/run para gerar métricas",
        }

    return {
        "available": True,
        "timestamp": _last_report["timestamp"],
        "summary": _last_report["summary"],
        "scores": _last_report["scores"],
        "latency": _last_report["latency"],
        "recommendations": _last_report["recommendations"],
    }


@router.get("/questions/{question_id}")
async def get_question_result(question_id: int):
    """Retorna resultado de uma pergunta específica."""
    if not _last_report:
        raise HTTPException(
            status_code=404,
            detail="Nenhum relatório disponível"
        )

    for result in _last_report.get("results", []):
        if result.get("question_id") == question_id:
            return result

    raise HTTPException(
        status_code=404,
        detail=f"Pergunta {question_id} não encontrada no relatório"
    )
