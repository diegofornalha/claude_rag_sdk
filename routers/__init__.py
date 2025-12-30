"""Routers module for Chat Simples."""

from .artifacts import router as artifacts_router
from .audit import router as audit_router
from .chat import router as chat_router
from .evaluate import router as evaluate_router
from .fs import router as fs_router
from .rag import router as rag_router
from .sessions import router as sessions_router

# Quiz router refatorado (usa novo módulo quiz/)
from quiz.router import router as quiz_router

# MCP router é opcional - pode ser removido sem afetar o sistema
try:
    from .mcp_ingest import router as mcp_router

    _mcp_available = True
except ImportError:
    mcp_router = None
    _mcp_available = False

__all__ = [
    "audit_router",
    "chat_router",
    "evaluate_router",
    "quiz_router",
    "rag_router",
    "sessions_router",
    "artifacts_router",
    "fs_router",
    "mcp_router",
]


def is_mcp_available() -> bool:
    """Verifica se o módulo MCP está disponível."""
    return _mcp_available
