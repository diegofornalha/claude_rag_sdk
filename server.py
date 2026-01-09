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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app_state
from claude_rag_sdk.core.auth import is_auth_enabled
from claude_rag_sdk.core.exceptions import RAGException
from claude_rag_sdk.core.logger import get_logger
from claude_rag_sdk.core.rate_limiter import SLOWAPI_AVAILABLE

logger = get_logger("server")
from routers import (
    artifacts_router,
    audit_router,
    chat_router,
    evaluate_router,
    fs_router,
    is_mcp_available,
    mcp_router,
    quiz_router,
    rag_router,
    sessions_router,
)
from routers.v1 import v1_router

# from routers.neo4j_mcp import router as neo4j_mcp_router  # Bridge desnecessário - SDK funciona agora!

# =============================================================================
# FASTAPI APP
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    logger.info("Starting Chat Simples...")
    yield
    # Cleanup watcher before app_state
    try:
        from utils.file_watcher import get_watcher

        watcher = get_watcher()
        if watcher.is_active():
            watcher.stop()
    except Exception as e:
        logger.warning("Error stopping watcher", error=str(e))
    await app_state.cleanup()


app = FastAPI(
    title="Claude RAG SDK API",
    description="""
## Chat Simples - Backend API

API REST para chat com **RAG (Retrieval-Augmented Generation)** usando Claude.

### Funcionalidades

- 🤖 **Chat**: Conversação com streaming SSE
- 🔍 **RAG Search**: Busca semântica em documentos
- 📁 **Sessions**: Gerenciamento de sessões de chat
- 📊 **Audit**: Logs de tool calls e debug
- 📄 **Artifacts**: Artefatos gerados pelo Claude

### Autenticação

Use o header `X-API-Key` com sua chave de API:

```
X-API-Key: sua-chave-aqui
```

### Streaming

O endpoint `/chat/stream` retorna Server-Sent Events (SSE):

```
data: {"text": "Olá"}
data: {"text": "! Como posso ajudar?"}
data: [DONE]
```
""",
    version="3.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Chat", "description": "Endpoints de conversação com Claude"},
        {"name": "RAG", "description": "Busca semântica e ingestão de documentos"},
        {"name": "Sessions", "description": "Gerenciamento de sessões de chat"},
        {"name": "Artifacts", "description": "Artefatos gerados pelo Claude"},
        {"name": "Audit", "description": "Logs de tool calls e debug"},
        {"name": "MCP", "description": "Model Context Protocol adapters"},
        {"name": "Health", "description": "Endpoints de status e health check"},
    ],
    contact={
        "name": "Claude RAG SDK",
        "url": "https://github.com/your-org/claude-rag-sdk",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4200",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "null",  # Allow file:// protocol for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """Handler para exceções customizadas do RAG.

    Converte RAGException em respostas HTTP apropriadas com JSON estruturado.
    """
    logger.error(
        "RAG exception occurred",
        error_type=exc.code,
        error_message=exc.message,
        http_status=exc.http_status,
        path=str(request.url.path),
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler para exceções genéricas não tratadas.

    Garante que erros não capturados retornem JSON estruturado.
    """
    logger.error(
        "Unhandled exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=str(request.url.path),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Erro interno do servidor",
            "details": {"type": type(exc).__name__},
        },
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

# API v1 - versionada (recomendada para novos clientes)
app.include_router(v1_router, prefix="/v1", tags=["v1"])

# Routers sem prefixo - compatibilidade legada (deprecated)
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(quiz_router)
app.include_router(sessions_router)
app.include_router(artifacts_router)
app.include_router(audit_router)
app.include_router(fs_router)
app.include_router(evaluate_router)
# app.include_router(neo4j_mcp_router)  # Não necessário - SDK funciona!

# MCP router é opcional - só inclui se disponível
if is_mcp_available() and mcp_router:
    app.include_router(mcp_router)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Health check básico.

    Retorna status do servidor e informações da sessão atual.
    """
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


@app.get("/model", tags=["Health"])
async def get_current_model():
    """Retorna o modelo Claude ativo.

    Mostra qual modelo está sendo usado na sessão atual.
    """
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


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check detalhado.

    Retorna status completo incluindo RAG stats, autenticação e ambiente.
    """
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
        "mcp": {
            "available": is_mcp_available(),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
