"""API v1 - Versioned routers.

Este módulo agrupa todos os routers da API v1.
Permite versionamento futuro sem quebrar compatibilidade.

Uso:
    from routers.v1 import v1_router
    app.include_router(v1_router, prefix="/v1")
"""

from fastapi import APIRouter

# Import routers do módulo pai
from ..artifacts import router as artifacts_router
from ..audit import router as audit_router
from ..chat import router as chat_router
from ..fs import router as fs_router
from ..rag import router as rag_router
from ..sessions import router as sessions_router

# Import opcional do MCP
try:
    from ..mcp_ingest import router as mcp_router

    _mcp_available = True
except ImportError:
    mcp_router = None
    _mcp_available = False

# Router principal v1 que agrupa todos os sub-routers
v1_router = APIRouter()

# Montar todos os routers
v1_router.include_router(chat_router)
v1_router.include_router(sessions_router)
v1_router.include_router(rag_router)
v1_router.include_router(artifacts_router)
v1_router.include_router(audit_router)
v1_router.include_router(fs_router)

if _mcp_available and mcp_router:
    v1_router.include_router(mcp_router)

__all__ = ["v1_router"]


def is_mcp_available() -> bool:
    """Verifica se o módulo MCP está disponível na v1."""
    return _mcp_available
