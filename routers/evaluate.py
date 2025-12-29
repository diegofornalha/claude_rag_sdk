"""Evaluation endpoints - Avaliação de qualidade do RAG."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
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
# Database (SQLite para histórico)
# =============================================================================

DB_PATH = Path(__file__).parent.parent / "data" / "evaluations.db"


def _init_db():
    """Inicializa banco de dados de avaliações."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            pass_rate REAL NOT NULL,
            passed INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            avg_latency_ms REAL NOT NULL,
            scores_json TEXT NOT NULL,
            recommendations_json TEXT NOT NULL,
            results_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _save_evaluation(report: dict) -> str:
    """Salva avaliação no banco e retorna ID."""
    _init_db()
    eval_id = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO evaluations
        (id, timestamp, pass_rate, passed, failed, total_questions, avg_latency_ms,
         scores_json, recommendations_json, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            eval_id,
            report["timestamp"],
            report["summary"]["pass_rate"],
            report["summary"]["passed"],
            report["summary"]["failed"],
            report["summary"]["total_questions"],
            report["latency"]["avg_ms"],
            json.dumps(report["scores"]),
            json.dumps(report["recommendations"]),
            json.dumps(report["results"]),
        ),
    )
    conn.commit()
    conn.close()
    return eval_id


def _get_evaluations_list() -> list[dict]:
    """Lista todas as avaliações (resumo)."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT id, timestamp, pass_rate, passed, failed, total_questions, avg_latency_ms
        FROM evaluations
        ORDER BY timestamp DESC
        LIMIT 50
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_evaluation_by_id(eval_id: str) -> dict | None:
    """Retorna avaliação completa por ID."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM evaluations WHERE id = ?",
        (eval_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    data["scores"] = json.loads(data["scores_json"])
    data["recommendations"] = json.loads(data["recommendations_json"])
    data["results"] = json.loads(data["results_json"])
    del data["scores_json"]
    del data["recommendations_json"]
    del data["results_json"]
    return data


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
                "question": q.get("question", ""),
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

        # Salvar no banco de dados
        eval_id = _save_evaluation(_last_report)
        _last_report["id"] = eval_id

        return _last_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        _evaluation_running = False


@router.post("/run/stream")
async def run_evaluation_stream(
    request: EvaluationRequest,
):
    """Executa avaliação com streaming SSE.

    Envia eventos conforme cada questão é avaliada:
    - type: 'start' - Início da avaliação
    - type: 'progress' - Resultado de cada questão
    - type: 'complete' - Relatório final
    """
    global _evaluation_running, _last_report

    if _evaluation_running:
        raise HTTPException(
            status_code=409,
            detail="Avaliação já em andamento. Aguarde ou consulte /evaluate/status"
        )

    from agents.evaluator import get_evaluator, EvaluationReport

    evaluator = get_evaluator()

    async def generate_events():
        global _evaluation_running, _last_report
        _evaluation_running = True

        try:
            # Filtrar questões
            qa_to_evaluate = evaluator.qa_pairs

            if request.question_ids:
                qa_to_evaluate = [q for q in qa_to_evaluate if q.get("id") in request.question_ids]

            if request.max_questions:
                qa_to_evaluate = qa_to_evaluate[:request.max_questions]

            total = len(qa_to_evaluate)

            # Evento de início
            yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

            # Criar report
            report = EvaluationReport()

            # Avaliar cada questão
            for i, qa_pair in enumerate(qa_to_evaluate):
                question_id = qa_pair.get("id", i + 1)
                question_text = qa_pair.get("question", "")[:50]

                # Evento de início da questão
                yield f"data: {json.dumps({'type': 'evaluating', 'question': question_id, 'current': i + 1, 'total': total, 'text': question_text})}\n\n"

                # Avaliar
                metrics = await evaluator.evaluate_question(qa_pair)
                report.add_result(metrics)

                # Evento de resultado
                result_data = {
                    'type': 'progress',
                    'question': question_id,
                    'current': i + 1,
                    'total': total,
                    'passed': metrics.passed,
                    'score': round(metrics.overall_score * 100, 1),
                    'latency_ms': round(metrics.latency_ms),
                    'citations': metrics.citations_count,
                }
                yield f"data: {json.dumps(result_data)}\n\n"

            # Gerar recomendações e finalizar
            report.generate_recommendations()
            _last_report = report.to_dict()

            # Salvar no banco
            eval_id = _save_evaluation(_last_report)
            _last_report["id"] = eval_id

            # Evento de conclusão
            complete_data = {
                'type': 'complete',
                'id': eval_id,
                'pass_rate': round(report.pass_rate * 100, 1),
                'passed': report.passed_count,
                'failed': report.failed_count,
                'total': report.total_questions,
                'avg_latency_ms': round(report.avg_latency_ms),
                'recommendations': report.recommendations,
            }
            yield f"data: {json.dumps(complete_data)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        finally:
            _evaluation_running = False

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/history")
async def get_evaluation_history():
    """Lista histórico de avaliações."""
    evaluations = _get_evaluations_list()
    return {
        "total": len(evaluations),
        "evaluations": evaluations,
    }


@router.get("/history/{eval_id}")
async def get_evaluation_details(eval_id: str):
    """Retorna detalhes de uma avaliação específica."""
    evaluation = _get_evaluation_by_id(eval_id)
    if not evaluation:
        raise HTTPException(
            status_code=404,
            detail=f"Avaliação {eval_id} não encontrada"
        )
    return evaluation


@router.delete("/history/{eval_id}")
async def delete_evaluation(eval_id: str):
    """Deleta uma avaliação do histórico."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("DELETE FROM evaluations WHERE id = ?", (eval_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Avaliação {eval_id} não encontrada"
        )

    return {"deleted": True, "id": eval_id}


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
