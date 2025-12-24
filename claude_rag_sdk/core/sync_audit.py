"""
=============================================================================
SYNC AUDIT - Auditoria Sincrona para MCP Tools
=============================================================================
Registra tool calls de forma sincrona usando threading para nao bloquear.
Funciona com o AgentFS compartilhado entre processos.
=============================================================================
"""

import os
import time
import json
import functools
import threading
import queue
from typing import Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Importar logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.logger import logger


@dataclass
class ToolCallRecord:
    """Registro de uma tool call."""
    tool_name: str
    started_at: int
    completed_at: int
    duration_ms: int
    parameters: dict
    result: Optional[Any] = None
    error: Optional[str] = None
    session_id: Optional[str] = None


class SyncAuditQueue:
    """
    Fila de auditoria sincrona que persiste em arquivo.
    Usa threading para nao bloquear as tools.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._session_id: Optional[str] = None
        self._audit_file: Optional[Path] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._records: list[ToolCallRecord] = []

        # Tentar obter session_id do ambiente
        self._init_from_env()

    def _init_from_env(self):
        """Inicializa a partir de variaveis de ambiente ou arquivo compartilhado."""
        # Primeiro tentar variavel de ambiente
        session_id = os.environ.get("AGENTFS_SESSION_ID")

        # Se nao houver, tentar arquivo compartilhado
        if not session_id:
            session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
            if session_file.exists():
                try:
                    session_id = session_file.read_text().strip()
                except Exception:
                    pass

        if session_id:
            self.set_session(session_id)

    def set_session(self, session_id: str):
        """Define a sessao atual e inicia o worker."""
        self._session_id = session_id

        # Criar diretorio de auditoria
        audit_dir = Path.home() / ".claude" / ".agentfs" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        # Arquivo de auditoria por sessao
        self._audit_file = audit_dir / f"{session_id}.jsonl"

        # Carregar registros existentes
        self._load_existing_records()

        # Iniciar worker thread se nao estiver rodando
        if not self._running:
            self._start_worker()

        logger.info("SyncAudit inicializado", session_id=session_id, audit_file=str(self._audit_file))

    def _load_existing_records(self):
        """Carrega registros existentes do arquivo."""
        self._records = []
        if self._audit_file and self._audit_file.exists():
            try:
                with open(self._audit_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._records.append(ToolCallRecord(**data))
            except Exception as e:
                logger.warning(f"Erro ao carregar audit file: {e}")

    def _start_worker(self):
        """Inicia thread worker para processar fila."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        """Loop do worker que processa a fila."""
        while self._running:
            try:
                # Aguardar item com timeout para permitir shutdown gracioso
                record = self._queue.get(timeout=1.0)
                self._persist_record(record)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no audit worker: {e}")

    def _persist_record(self, record: ToolCallRecord):
        """Persiste record no arquivo JSONL."""
        if not self._audit_file:
            return

        try:
            with open(self._audit_file, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
            self._records.append(record)
        except Exception as e:
            logger.error(f"Erro ao persistir audit record: {e}")

    def record(self,
               tool_name: str,
               started_at: int,
               completed_at: int,
               parameters: dict,
               result: Optional[Any] = None,
               error: Optional[str] = None):
        """
        Enfileira um registro de tool call.
        Non-blocking - retorna imediatamente.
        """
        duration_ms = (completed_at - started_at) * 1000

        record = ToolCallRecord(
            tool_name=tool_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            parameters=parameters,
            result=result,
            error=error,
            session_id=self._session_id
        )

        self._queue.put(record)

    def get_records(self, limit: int = 100) -> list[dict]:
        """Retorna os ultimos N registros."""
        records = self._records[-limit:] if limit else self._records
        return [asdict(r) for r in records]

    def get_stats(self) -> dict:
        """Retorna estatisticas de auditoria."""
        if not self._records:
            return {
                "total_calls": 0,
                "by_tool": {},
                "errors": 0,
                "avg_duration_ms": 0
            }

        by_tool = {}
        total_duration = 0
        errors = 0

        for r in self._records:
            by_tool[r.tool_name] = by_tool.get(r.tool_name, 0) + 1
            total_duration += r.duration_ms
            if r.error:
                errors += 1

        return {
            "total_calls": len(self._records),
            "by_tool": by_tool,
            "errors": errors,
            "avg_duration_ms": round(total_duration / len(self._records), 2),
            "session_id": self._session_id
        }

    def shutdown(self):
        """Para o worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)


# Instancia global
_audit_queue: Optional[SyncAuditQueue] = None


def get_audit_queue() -> SyncAuditQueue:
    """Retorna ou cria a fila de auditoria."""
    global _audit_queue
    if _audit_queue is None:
        _audit_queue = SyncAuditQueue()
    return _audit_queue


def audit_sync_tool(tool_name: str):
    """
    Decorator para auditar tool calls sincronas e assincronas.
    Non-blocking - usa threading para persistir.

    Usage:
        @audit_sync_tool("search_documents")
        def search_documents(query: str):
            # ... codigo da tool ...
            return results

        @audit_sync_tool("create_file")
        async def create_file(path: str, content: str):
            # ... codigo async da tool ...
            return results
    """
    def decorator(func: Callable):
        def _serialize_result(result):
            """Serializa result de forma segura"""
            try:
                if isinstance(result, (dict, list)):
                    result_str = json.dumps(result)
                    if len(result_str) > 1000:
                        return {"_truncated": True, "_size": len(result_str)}
                    return result
                elif isinstance(result, (str, int, float, bool)):
                    return result
                else:
                    return str(result)[:500]
            except:
                return "[not serializable]"

        def _get_parameters(*args, **kwargs):
            """Captura parametros de forma segura"""
            return {
                "args": [str(arg)[:200] for arg in args] if args else [],
                "kwargs": {k: str(v)[:200] for k, v in kwargs.items()} if kwargs else {}
            }

        # Wrapper async para funcoes assincronas
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            audit = get_audit_queue()
            start_time = int(time.time())
            parameters = _get_parameters(*args, **kwargs)

            try:
                result = await func(*args, **kwargs)
                end_time = int(time.time())

                audit.record(
                    tool_name=tool_name,
                    started_at=start_time,
                    completed_at=end_time,
                    parameters=parameters,
                    result=_serialize_result(result)
                )

                return result

            except Exception as e:
                end_time = int(time.time())
                audit.record(
                    tool_name=tool_name,
                    started_at=start_time,
                    completed_at=end_time,
                    parameters=parameters,
                    error=str(e)
                )
                raise

        # Wrapper sync para funcoes sincronas
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            audit = get_audit_queue()
            start_time = int(time.time())
            parameters = _get_parameters(*args, **kwargs)

            try:
                result = func(*args, **kwargs)
                end_time = int(time.time())

                audit.record(
                    tool_name=tool_name,
                    started_at=start_time,
                    completed_at=end_time,
                    parameters=parameters,
                    result=_serialize_result(result)
                )

                return result

            except Exception as e:
                end_time = int(time.time())
                audit.record(
                    tool_name=tool_name,
                    started_at=start_time,
                    completed_at=end_time,
                    parameters=parameters,
                    error=str(e)
                )
                raise

        # Detecta se funcao eh async e retorna o wrapper apropriado
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
