"""Routers module for Chat Simples."""
from .audit import router as audit_router
from .chat import router as chat_router
from .rag import router as rag_router
from .sessions import router as sessions_router
from .outputs import router as outputs_router
from .fs import router as fs_router

__all__ = [
    "audit_router",
    "chat_router",
    "rag_router",
    "sessions_router",
    "outputs_router",
    "fs_router",
]
