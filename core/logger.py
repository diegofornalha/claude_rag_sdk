# =============================================================================
# LOGGER - Logging JSON Estruturado para RAG Agent
# =============================================================================
# Logs em formato JSON com campos padronizados para observabilidade
# =============================================================================

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from contextvars import ContextVar

# Context var para rastrear conversation_id entre chamadas
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


class JSONFormatter(logging.Formatter):
    """Formatter que gera logs em JSON estruturado."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "conversation_id": conversation_id_var.get() or None,
            "request_id": request_id_var.get() or None,
            "session_id": session_id_var.get() or None,
        }

        # Adicionar campos extras se existirem
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Adicionar exception info se houver
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Adicionar source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class RAGLogger:
    """Logger estruturado para o RAG Agent."""

    def __init__(self, name: str = "rag-agent"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Evitar handlers duplicados
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def _log(
        self,
        level: int,
        message: str,
        **extra_fields: Any,
    ) -> None:
        """Log interno com campos extras."""
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)

    def debug(self, message: str, **extra: Any) -> None:
        self._log(logging.DEBUG, message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        self._log(logging.INFO, message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._log(logging.WARNING, message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self._log(logging.ERROR, message, **extra)

    def critical(self, message: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, message, **extra)

    # Métodos específicos para RAG

    def log_query(
        self,
        query: str,
        top_k: int,
        results_count: int,
        latency_ms: float,
        **extra: Any,
    ) -> None:
        """Log de busca semântica."""
        self.info(
            "semantic_search",
            event_type="query",
            query=query[:200],  # Truncar queries longas
            top_k=top_k,
            results_count=results_count,
            latency_ms=round(latency_ms, 2),
            **extra,
        )

    def log_retrieval(
        self,
        doc_ids: list[int],
        similarities: list[float],
        latency_ms: float,
        **extra: Any,
    ) -> None:
        """Log de documentos recuperados."""
        self.info(
            "document_retrieval",
            event_type="retrieval",
            doc_ids=doc_ids,
            similarities=[round(s, 3) for s in similarities],
            latency_ms=round(latency_ms, 2),
            **extra,
        )

    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log de chamada ao LLM."""
        self.info(
            "llm_call",
            event_type="llm",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6) if cost_usd else None,
            **extra,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        **extra: Any,
    ) -> None:
        """Log de erro estruturado."""
        self.error(
            "error_occurred",
            event_type="error",
            error_type=error_type,
            error_message=str(error_message)[:500],
            **extra,
        )

    def log_rbac(
        self,
        user_id: str,
        action: str,
        resource: str,
        allowed: bool,
        **extra: Any,
    ) -> None:
        """Log de decisão RBAC."""
        level = logging.INFO if allowed else logging.WARNING
        self._log(
            level,
            "rbac_decision",
            event_type="rbac",
            user_id=user_id,
            action=action,
            resource=resource,
            allowed=allowed,
            **extra,
        )


# Funções de contexto
def set_conversation_id(conversation_id: Optional[str] = None) -> str:
    """Define conversation_id para o contexto atual."""
    cid = conversation_id or str(uuid.uuid4())
    conversation_id_var.set(cid)
    return cid


def get_conversation_id() -> str:
    """Retorna conversation_id do contexto atual."""
    return conversation_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Define request_id para o contexto atual."""
    rid = request_id or str(uuid.uuid4())
    request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    """Retorna request_id do contexto atual."""
    return request_id_var.get()


def set_session_id(session_id: Optional[str] = None) -> str:
    """Define session_id para o contexto atual."""
    sid = session_id or "default"
    session_id_var.set(sid)
    return sid


def get_session_id() -> str:
    """Retorna session_id do contexto atual."""
    return session_id_var.get() or "default"


# Instância global
logger = RAGLogger()


if __name__ == "__main__":
    # Teste do logger
    set_conversation_id("test-conv-123")
    set_request_id("test-req-456")

    logger.info("Sistema iniciado", version="1.0.0")
    logger.log_query("politica de IA", top_k=5, results_count=3, latency_ms=150.5)
    logger.log_retrieval([1, 2, 3], [0.95, 0.87, 0.72], latency_ms=50.2)
    logger.log_llm_call("claude-haiku-4-5", 500, 200, latency_ms=1200.0, cost_usd=0.0005)
    logger.log_rbac("user@email.com", "read", "doc:123", allowed=True)
    logger.log_error("ValidationError", "Campo obrigatório ausente")
