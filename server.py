"""
Chat Simples Server - Powered by Claude RAG SDK

Production-ready FastAPI server with:
- Claude RAG SDK integration
- Session management via AgentFS
- Rate limiting, CORS, authentication
- Prompt injection protection
- Streaming responses
- Audit trail
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app_state
from claude_rag_sdk.core.auth import is_auth_enabled
from claude_rag_sdk.core.rate_limiter import SLOWAPI_AVAILABLE
from routers import (
    audit_router,
    chat_router,
    fs_router,
    outputs_router,
    rag_router,
    sessions_router,
)

# =============================================================================
# FASTAPI APP
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    print("[INFO] Starting Chat Simples...")
    yield
    # Cleanup watcher before app_state
    try:
        from utils.file_watcher import get_watcher

        watcher = get_watcher()
        if watcher.is_active():
            watcher.stop()
    except Exception as e:
        print(f"[WARN] Error stopping watcher: {e}")
    await app_state.cleanup()


app = FastAPI(
    title="Chat Simples",
    description="Chat backend powered by Claude RAG SDK",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4200",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:8001",
        "null",  # Allow file:// protocol for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter
if SLOWAPI_AVAILABLE:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded

    from claude_rag_sdk.core.rate_limiter import get_limiter

    app.state.limiter = get_limiter()
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(sessions_router)
app.include_router(outputs_router)
app.include_router(audit_router)
app.include_router(fs_router)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    """Health check."""
    env = os.getenv("ENVIRONMENT", "development")
    response = {
        "status": "ok",
        "session_active": app_state.client is not None,
        "session_id": app_state.current_session_id,
        "message": "Chat Simples v3 - Claude RAG SDK",
        "auth_enabled": is_auth_enabled(),
    }

    # Em desenvolvimento, expor dev key para facilitar testes
    if env == "development" and is_auth_enabled():
        from claude_rag_sdk.core.auth import VALID_API_KEYS

        if VALID_API_KEYS:
            response["dev_key"] = list(VALID_API_KEYS)[0]

    return response


@app.get("/model")
async def get_current_model():
    """Return current active model."""
    return {
        "model": app_state.current_model,
        "model_id": _get_model_id(app_state.current_model),
        "session_id": app_state.current_session_id,
        "client_active": app_state.client is not None,
    }


def _get_model_id(model_name: str) -> str:
    """Get full model ID from short name."""
    MODEL_IDS = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-5-20251101",
    }
    return MODEL_IDS.get(model_name, f"claude-{model_name}-latest")


@app.get("/health")
async def health_check():
    """Detailed health check."""
    env = os.getenv("ENVIRONMENT", "development")

    rag_stats = None
    if app_state.agentfs:
        rag_stats = {"agentfs": "active"}

    return {
        "status": "healthy",
        "environment": env,
        "session_active": app_state.client is not None,
        "session_id": app_state.current_session_id,
        "rag_stats": rag_stats,
        "security": {
            "auth_enabled": is_auth_enabled(),
            "rate_limiter": "slowapi" if SLOWAPI_AVAILABLE else "simple",
            "prompt_guard": "active",
        },
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
