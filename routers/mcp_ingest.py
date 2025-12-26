"""
MCP Ingest Router - Endpoints para ingestão de documentos via MCP servers.

Este router carrega dinamicamente apenas os adapters que estão habilitados.
Se nenhum adapter estiver habilitado, os endpoints retornam erro informativo.

Endpoints:
    GET  /mcp/adapters          - Lista adapters disponíveis
    GET  /mcp/adapters/{name}   - Info de um adapter específico
    POST /mcp/adapters/{name}/enable   - Habilita adapter
    POST /mcp/adapters/{name}/disable  - Desabilita adapter
    POST /mcp/ingest/{adapter}  - Ingestão via adapter específico
    POST /mcp/ingest/{adapter}/bulk - Ingestão em massa
    GET  /mcp/status            - Status geral do sistema MCP
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from claude_rag_sdk.core.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Ingest"])

# RAG Knowledge base path
RAG_DB_PATH = Path.cwd() / "data" / "rag_knowledge.db"


# === Pydantic Models ===


class AdapterInfo(BaseModel):
    """Informações de um adapter MCP."""

    name: str
    description: str
    version: str
    enabled: bool
    status: str
    tools: List[str] = []


class IngestRequest(BaseModel):
    """Request para ingestão de documentos."""

    queries: List[str] = Field(default=[], description="Lista de queries para buscar documentação")
    include_examples: bool = Field(default=True, description="Incluir exemplos de código")
    include_best_practices: bool = Field(default=True, description="Incluir best practices")


class IngestResult(BaseModel):
    """Resultado de uma operação de ingestão."""

    success: bool
    documents_ingested: int = 0
    errors: List[str] = []
    duration_seconds: float = 0.0


class MCPStatus(BaseModel):
    """Status geral do sistema MCP."""

    available: bool
    adapters_registered: int
    adapters_enabled: int
    adapters_connected: int
    enabled_list: List[str] = []


# === Helper Functions ===


def _get_registry():
    """Obtém o registry MCP de forma lazy."""
    try:
        # Garante que adapters foram registrados
        from claude_rag_sdk import mcp_adapters  # noqa: F401
        from claude_rag_sdk.mcp import get_mcp_registry

        return get_mcp_registry()
    except ImportError as e:
        logger.error(f"MCP module not available: {e}")
        return None


def _get_config():
    """Obtém configuração MCP."""
    try:
        from claude_rag_sdk.mcp import get_mcp_config

        return get_mcp_config()
    except ImportError:
        return None


async def _get_rag_engine():
    """Obtém engine de ingestão do RAG."""
    from claude_rag_sdk.ingest import IngestEngine

    return IngestEngine(
        db_path=str(RAG_DB_PATH),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=1500,
        chunk_overlap=150,
    )


# === Endpoints ===


@router.get("/status")
async def get_mcp_status() -> MCPStatus:
    """
    Retorna status geral do sistema MCP.

    Não requer autenticação - útil para health checks.
    """
    registry = _get_registry()

    if registry is None:
        return MCPStatus(
            available=False,
            adapters_registered=0,
            adapters_enabled=0,
            adapters_connected=0,
        )

    return MCPStatus(
        available=True,
        adapters_registered=len(registry.list_registered()),
        adapters_enabled=len(registry.list_enabled()),
        adapters_connected=len(registry.list_active()),
        enabled_list=registry.list_enabled(),
    )


@router.get("/adapters")
async def list_adapters() -> List[AdapterInfo]:
    """
    Lista todos os adapters MCP registrados.

    Retorna informações incluindo se está habilitado ou não.
    """
    registry = _get_registry()

    if registry is None:
        return []

    adapters = []
    for info in registry.get_all_adapters_info():
        adapters.append(
            AdapterInfo(
                name=info.name,
                description=info.description,
                version=info.version,
                enabled=info.enabled,
                status=info.status.value,
                tools=info.tools,
            )
        )

    return adapters


@router.get("/adapters/{name}")
async def get_adapter_info(name: str) -> AdapterInfo:
    """Retorna informações de um adapter específico."""
    registry = _get_registry()

    if registry is None:
        raise HTTPException(status_code=503, detail="MCP system not available")

    info = registry.get_adapter_info(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

    return AdapterInfo(
        name=info.name,
        description=info.description,
        version=info.version,
        enabled=info.enabled,
        status=info.status.value,
        tools=info.tools,
    )


@router.post("/adapters/{name}/enable")
async def enable_adapter(name: str, api_key: str = Depends(verify_api_key)) -> dict:
    """
    Habilita um adapter MCP.

    Requer API key. O adapter será carregado na próxima chamada.
    """
    config = _get_config()

    if config is None:
        raise HTTPException(status_code=503, detail="MCP system not available")

    if name not in config.adapters:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found in configuration")

    config.enable_adapter(name)
    logger.info(f"Adapter '{name}' enabled")

    return {
        "success": True,
        "message": f"Adapter '{name}' enabled",
        "adapter": name,
    }


@router.post("/adapters/{name}/disable")
async def disable_adapter(name: str, api_key: str = Depends(verify_api_key)) -> dict:
    """
    Desabilita um adapter MCP.

    Se o adapter estiver conectado, será desconectado.
    """
    config = _get_config()
    registry = _get_registry()

    if config is None or registry is None:
        raise HTTPException(status_code=503, detail="MCP system not available")

    # Desconecta se estiver ativo
    if registry.is_active(name):
        await registry.disconnect_adapter(name)

    config.disable_adapter(name)
    logger.info(f"Adapter '{name}' disabled")

    return {
        "success": True,
        "message": f"Adapter '{name}' disabled",
        "adapter": name,
    }


@router.post("/ingest/{adapter}")
async def ingest_from_adapter(
    adapter: str,
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Inicia ingestão de documentos de um adapter MCP.

    A ingestão roda em background para não bloquear a requisição.

    Args:
        adapter: Nome do adapter (ex: 'angular-cli')
        request: Configuração da ingestão

    Returns:
        Status da tarefa iniciada
    """
    registry = _get_registry()

    if registry is None:
        raise HTTPException(status_code=503, detail="MCP system not available")

    if not registry.is_registered(adapter):
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter}' not registered")

    if not registry.is_enabled(adapter):
        raise HTTPException(
            status_code=400,
            detail=f"Adapter '{adapter}' is disabled. Enable it first via POST /mcp/adapters/{adapter}/enable",
        )

    # Agenda tarefa em background
    task_id = f"mcp-ingest-{adapter}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    background_tasks.add_task(
        _run_ingest_task,
        adapter,
        request.queries,
        request.include_examples,
        request.include_best_practices,
    )

    return {
        "success": True,
        "message": f"Ingest task started for adapter '{adapter}'",
        "task_id": task_id,
        "adapter": adapter,
        "queries_count": len(request.queries) if request.queries else "default",
    }


@router.post("/ingest/{adapter}/sync")
async def ingest_from_adapter_sync(
    adapter: str, request: IngestRequest, api_key: str = Depends(verify_api_key)
) -> IngestResult:
    """
    Executa ingestão de documentos de forma síncrona.

    ATENÇÃO: Pode demorar bastante dependendo da quantidade de queries.
    Prefira usar o endpoint assíncrono /ingest/{adapter} para produção.
    """
    registry = _get_registry()

    if registry is None:
        raise HTTPException(status_code=503, detail="MCP system not available")

    if not registry.is_enabled(adapter):
        raise HTTPException(status_code=400, detail=f"Adapter '{adapter}' is disabled")

    return await _run_ingest_task(
        adapter,
        request.queries,
        request.include_examples,
        request.include_best_practices,
    )


async def _run_ingest_task(
    adapter_name: str,
    queries: Optional[List[str]],
    include_examples: bool,
    include_best_practices: bool,
) -> IngestResult:
    """
    Executa a tarefa de ingestão.

    Esta função é chamada tanto pelo endpoint síncrono quanto pelo assíncrono.
    """
    start_time = datetime.now()
    errors: List[str] = []
    documents_ingested = 0

    try:
        from claude_rag_sdk.mcp import get_adapter

        # Obtém adapter (conecta se necessário)
        adapter = await get_adapter(adapter_name)

        # Obtém documentos do MCP
        documents = await adapter.get_documents_for_ingest(
            queries=queries if queries else None,
            include_examples=include_examples,
            include_best_practices=include_best_practices,
        )

        if not documents:
            logger.warning(f"No documents returned from adapter '{adapter_name}'")
            return IngestResult(
                success=True,
                documents_ingested=0,
                errors=["No documents returned from MCP"],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        # Obtém engine de ingestão
        engine = await _get_rag_engine()

        # CORREÇÃO: Deletar documentos do adapter ANTES de reingerir (evita duplicação)
        if adapter_name == "angular-cli":
            try:
                from claude_rag_sdk import ClaudeRAG

                # Abrir RAG para deletar documentos Angular antigos
                rag_options = engine.options
                async with ClaudeRAG.open(rag_options) as rag:
                    all_docs = await rag.list_documents(limit=1000)
                    angular_docs = [
                        d
                        for d in all_docs
                        if "angular" in (d.get("source", "") or "").lower()
                        or "angular-cli-mcp" in (d.get("source", "") or "").lower()
                    ]

                    deleted_count = 0
                    for doc in angular_docs:
                        doc_id = doc.get("id")
                        if doc_id and await rag.ingest.delete_document(doc_id):
                            deleted_count += 1

                    logger.info(f"Deleted {deleted_count} old Angular documents before reingest")
            except Exception as e:
                logger.warning(f"Error deleting old Angular docs: {e}")

        # Ingere cada documento
        for doc in documents:
            try:
                # Formata conteúdo com header
                content = _format_document_for_ingest(doc)

                result = await engine.add_text(
                    content=content,
                    source=doc.source,
                    doc_type=doc.doc_type,
                    metadata=doc.metadata,
                )

                if result.success:
                    documents_ingested += 1
                else:
                    errors.append(f"Failed to ingest '{doc.title}': {result.error}")

                # Rate limiting entre documentos
                await asyncio.sleep(0.1)

            except Exception as e:
                errors.append(f"Error ingesting '{doc.title}': {str(e)}")
                logger.error(f"Ingest error for {doc.title}: {e}")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"MCP ingest completed: {documents_ingested}/{len(documents)} docs, "
            f"{len(errors)} errors, {duration:.2f}s"
        )

        return IngestResult(
            success=len(errors) == 0,
            documents_ingested=documents_ingested,
            errors=errors,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(f"Ingest task failed: {e}")
        return IngestResult(
            success=False,
            errors=[str(e)],
            duration_seconds=(datetime.now() - start_time).total_seconds(),
        )


def _format_document_for_ingest(doc) -> str:
    """Formata documento MCP para ingestão no RAG."""
    lines = []

    # Header
    lines.append(f"# {doc.title}")
    lines.append("")

    # Metadados
    if doc.url:
        lines.append(f"_Fonte: {doc.url}_")
    if doc.metadata.get("mcp_server"):
        lines.append(f"_MCP Server: {doc.metadata['mcp_server']}_")
    if doc.metadata.get("fetched_at"):
        lines.append(f"_Obtido em: {doc.metadata['fetched_at']}_")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Conteúdo
    lines.append(doc.content)

    return "\n".join(lines)
