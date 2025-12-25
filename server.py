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

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import json
import os
import asyncio
import time
import uuid

# Claude RAG SDK (for RAG endpoints only)
from claude_rag_sdk import AgentModel, EmbeddingModel

# Core modules from SDK
from claude_rag_sdk.core.security import get_allowed_origins
from claude_rag_sdk.core.rate_limiter import get_limiter, RATE_LIMITS, SLOWAPI_AVAILABLE
from claude_rag_sdk.core.prompt_guard import PromptGuard
from claude_rag_sdk.core.auth import verify_api_key, is_auth_enabled

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTFS_DIR = Path.cwd() / ".agentfs"

# Global client and AgentFS instances
client: Optional["ClaudeSDKClient"] = None
agentfs: Optional["AgentFS"] = None
current_session_id: Optional[str] = None
prompt_guard = PromptGuard(strict_mode=False)

# Sessions directory (Claude Code uses cwd-based path)
# Format: -Users-2a--claude-hello-agent-chat-simples-backend-outputs
SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs"


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def extract_session_id_from_jsonl() -> Optional[str]:
    """Extrai session_id do arquivo JSONL mais recente."""
    if not SESSIONS_DIR.exists():
        return None

    jsonl_files = sorted(
        SESSIONS_DIR.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    if not jsonl_files:
        return None

    latest_jsonl = jsonl_files[0]
    try:
        with open(latest_jsonl, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return data.get("sessionId", latest_jsonl.stem)
    except:
        return latest_jsonl.stem

    return None


async def get_client() -> "ClaudeSDKClient":
    """Get ClaudeSDKClient instance (manages sessions automatically)."""
    global client, agentfs, current_session_id

    # Verificar se sessão atual ainda existe, se não, resetar
    if client is not None and current_session_id:
        session_db = AGENTFS_DIR / f"{current_session_id}.db"
        if not session_db.exists():
            print(f"⚠️ Sessão {current_session_id} foi deletada, resetando...")
            if client is not None:
                await client.__aexit__(None, None, None)
                client = None
            if agentfs is not None:
                await agentfs.close()
                agentfs = None
            current_session_id = None

    if client is None:
        # Build options using AgentEngine helper
        from claude_rag_sdk.agent import AgentEngine
        from claude_rag_sdk import ClaudeRAGOptions
        from claude_agent_sdk import ClaudeSDKClient

        # System prompt para instruir o agente sobre onde salvar arquivos
        # O session_id será inserido dinamicamente quando disponível
        outputs_base = str(Path.cwd() / "outputs")
        system_prompt = f"""Você é um assistente RAG especializado em responder perguntas usando uma base de conhecimento.

## Regras para criação de arquivos:
- SEMPRE salve arquivos em: {outputs_base}/[SESSION_ID]/
- Substitua [SESSION_ID] pelo ID da sessão atual
- Use nomes descritivos e extensões apropriadas (ex: relatorio.txt, dados.json)
- NUNCA use /tmp/ ou outros diretórios
- Confirme ao usuário o caminho completo do arquivo criado

## Importante:
- Responda com base nos documentos da base de conhecimento
- Forneça citações com fonte e trecho quando aplicável"""

        temp_options = ClaudeRAGOptions(
            id="temp",
            agent_model=AgentModel.HAIKU,
            system_prompt=system_prompt
        )
        engine = AgentEngine(options=temp_options, mcp_server_path=None)
        client_options = engine._get_agent_options()

        # Create ClaudeSDKClient (manages session automatically)
        client = ClaudeSDKClient(options=client_options)
        await client.__aenter__()

        # Wait for SDK to write JSONL
        await asyncio.sleep(0.2)

        # Extract session_id from JSONL
        current_session_id = extract_session_id_from_jsonl()
        if not current_session_id:
            raise RuntimeError("Failed to extract session_id from ClaudeSDKClient")

        # Initialize AgentFS with extracted session_id
        from agentfs_sdk import AgentFS, AgentFSOptions
        agentfs = await AgentFS.open(AgentFSOptions(id=current_session_id))

        # Store session info in KV
        await agentfs.kv.set("session:info", {
            "id": current_session_id,
            "created_at": time.time(),
        })

        # Write session start log
        await agentfs.fs.write_file(
            f"/logs/session_start.txt",
            f"Session {current_session_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Write current session file
        session_file = AGENTFS_DIR / "current_session"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(current_session_id)

        print(f"🚀 Session created: {current_session_id}")

    return client


async def get_agentfs() -> "AgentFS":
    """Get AgentFS instance."""
    global agentfs
    if agentfs is None:
        await get_client()  # Will initialize both
    return agentfs


async def reset_session():
    """Reset session (creates new ClaudeSDKClient + AgentFS)."""
    global client, agentfs, current_session_id

    old_session = current_session_id

    if client is not None:
        await client.__aexit__(None, None, None)
        client = None

    if agentfs is not None:
        await agentfs.close()
        agentfs = None

    current_session_id = None

    # Create new session
    await get_client()
    print(f"🔄 Session reset: {old_session} -> {current_session_id}")


def get_current_session_id() -> Optional[str]:
    """Get current session ID."""
    global current_session_id

    if current_session_id:
        return current_session_id

    # Try to read from file
    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        try:
            return session_file.read_text().strip()
        except:
            pass

    return None


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    print("🚀 Starting Chat Simples...")
    yield
    # Cleanup
    global client, agentfs
    if client is not None:
        await client.__aexit__(None, None, None)
        print("👋 ClaudeSDKClient closed!")
    if agentfs is not None:
        await agentfs.close()
        print("👋 AgentFS closed!")


app = FastAPI(
    title="Chat Simples",
    description="Chat backend powered by Claude RAG SDK",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter
limiter = get_limiter()
if SLOWAPI_AVAILABLE:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Permite continuar conversa de sessão específica


class ChatResponse(BaseModel):
    response: str


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    global client, agentfs
    env = os.getenv("ENVIRONMENT", "development")
    response = {
        "status": "ok",
        "session_active": client is not None,
        "session_id": current_session_id,
        "message": "Chat Simples v3 - Claude RAG SDK",
        "auth_enabled": is_auth_enabled()
    }
    if env != "production" and is_auth_enabled():
        from claude_rag_sdk.core.auth import VALID_API_KEYS
        if VALID_API_KEYS:
            response["dev_key"] = list(VALID_API_KEYS)[0]
    return response


@app.get("/health")
async def health_check():
    """Detailed health check."""
    global client, agentfs
    env = os.getenv("ENVIRONMENT", "development")

    # Get session stats if available
    rag_stats = None
    if agentfs:
        try:
            # Basic stats from AgentFS
            rag_stats = {"agentfs": "active"}
        except:
            pass

    return {
        "status": "healthy",
        "environment": env,
        "session_active": client is not None,
        "session_id": current_session_id,
        "rag_stats": rag_stats,
        "security": {
            "auth_enabled": is_auth_enabled(),
            "rate_limiter": "slowapi" if SLOWAPI_AVAILABLE else "simple",
            "prompt_guard": "active"
        }
    }


# =============================================================================
# CHAT ENDPOINTS
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat with RAG-powered AI."""
    # Validate prompt
    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    try:
        from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock

        c = await get_client()

        # Se session_id foi passado, usar AgentFS específico da sessão
        session_specific_afs = None
        target_session_id = chat_request.session_id or current_session_id

        if chat_request.session_id and chat_request.session_id != current_session_id:
            # Abrir AgentFS da sessão específica
            session_specific_afs = await AgentFS.open(AgentFSOptions(id=chat_request.session_id))
            afs = session_specific_afs
            print(f"📂 Usando sessão específica: {chat_request.session_id}")
        else:
            afs = await get_agentfs()

        # Track the query
        call_id = await afs.tools.start("chat", {"message": chat_request.message[:100]})

        # Incluir session_id como contexto para o LLM saber onde salvar arquivos
        outputs_path = str(Path.cwd() / "outputs" / target_session_id)
        context_message = f"""[CONTEXTO DO SISTEMA - Session ID: {target_session_id}]
Ao criar arquivos, use EXATAMENTE este caminho: {outputs_path}/
Exemplo: {outputs_path}/meu_arquivo.txt

[MENSAGEM DO USUÁRIO]
{chat_request.message}"""

        # Query with ClaudeSDKClient
        await c.query(context_message)

        # Collect response and track tool calls
        response_text = ""
        tool_calls = {}  # tool_use_id -> call_id for tracking

        async for message in c.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        # Registrar início da tool call
                        tool_call_id = await afs.tools.start(
                            block.name,
                            {"input": str(block.input)[:500]}  # Limitar tamanho
                        )
                        tool_calls[block.id] = tool_call_id
                        print(f"🔧 Tool call: {block.name} (id: {block.id})")
                    elif isinstance(block, ToolResultBlock):
                        # Registrar resultado da tool call
                        if block.tool_use_id in tool_calls:
                            await afs.tools.success(
                                tool_calls[block.tool_use_id],
                                {"result": str(block.content)[:500]}
                            )
                            print(f"✅ Tool result: {block.tool_use_id}")

        # Complete pending tool calls (ToolResultBlock é processado internamente pelo SDK)
        for tool_use_id, tool_call_id in tool_calls.items():
            await afs.tools.success(tool_call_id, {"status": "completed_by_sdk"})
            print(f"✅ Tool completed: {tool_use_id}")

        # Complete tracking
        await afs.tools.success(call_id, {"response_length": len(response_text)})

        # Store in conversation history (KV) - incluir todas as mensagens para consistência
        history = await afs.kv.get("conversation:history") or []

        # Adicionar mensagem do usuário
        history.append({
            "role": "user",
            "content": chat_request.message,
        })

        # Adicionar tool calls como mensagens separadas (para consistência com .jsonl)
        for tool_use_id, tool_call_id in tool_calls.items():
            history.append({
                "role": "assistant",
                "content": f"[Tool Call: {tool_use_id}]",
                "type": "tool_use"
            })
            history.append({
                "role": "tool",
                "content": f"[Tool Result: {tool_use_id}]",
                "type": "tool_result"
            })

        # Adicionar resposta final do assistente
        history.append({
            "role": "assistant",
            "content": response_text,
        })

        await afs.kv.set("conversation:history", history[-100:])  # Keep last 100

        # Save response to filesystem for persistence
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        await afs.fs.write_file(
            f"/outputs/chat_{timestamp}.json",
            json.dumps({
                "timestamp": timestamp,
                "question": chat_request.message,
                "answer": response_text,
            }, indent=2, ensure_ascii=False)
        )

        # Organize outputs: move files from various locations to session folder
        import shutil
        outputs_root = Path.cwd() / "outputs"
        session_outputs = outputs_root / target_session_id

        # Create session folder if it doesn't exist
        session_outputs.mkdir(parents=True, exist_ok=True)

        # Move any files from backend/outputs root to session folder
        if outputs_root.exists():
            for item in outputs_root.iterdir():
                if item.is_file():  # Only move files, not directories
                    target = session_outputs / item.name
                    shutil.move(str(item), str(target))

        # Move files from default folders (default, default_session) to session folder
        for default_folder in ["default", "default_session"]:
            default_path = outputs_root / default_folder
            if default_path.exists() and default_path.is_dir():
                for item in default_path.iterdir():
                    if item.is_file():
                        target = session_outputs / item.name
                        if not target.exists():  # Avoid overwriting
                            shutil.move(str(item), str(target))
                            print(f"📦 Moved {item.name} from {default_folder}/ to {target_session_id}/")

        # Also check and move files from /tmp/outputs
        tmp_outputs = Path("/tmp/outputs")
        if tmp_outputs.exists():
            for item in tmp_outputs.iterdir():
                if item.is_file():
                    target = session_outputs / item.name
                    shutil.move(str(item), str(target))

        # Fechar AgentFS específico da sessão se foi aberto
        if session_specific_afs:
            await session_specific_afs.close()
            print(f"📂 Sessão específica fechada: {chat_request.session_id}")

        return ChatResponse(response=response_text)

    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit(RATE_LIMITS.get("chat_stream", "20/minute"))
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat with streaming response."""
    # Validate prompt
    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    try:
        r = await get_rag()

        async def generate():
            try:
                # For now, get full response and stream it
                # TODO: Implement true streaming with AgentEngine
                response = await r.query(chat_request.message)

                # Stream the response in chunks
                text = response.answer
                chunk_size = 50
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01)

                # Send citations at the end
                if response.citations:
                    yield f"data: {json.dumps({'citations': response.citations})}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RAG ENDPOINTS
# =============================================================================

@app.post("/rag/search")
async def rag_search(
    request: Request,
    query: str,
    top_k: int = 5,
    api_key: str = Depends(verify_api_key)
):
    """Search documents using RAG."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        # Create temporary RAG for search
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=current_session_id or "temp"))
        results = await temp_rag.search(query, top_k=top_k)
        await temp_rag.close()
        return {
            "query": query,
            "results": [res.to_dict() for res in results],
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ingest")
async def rag_ingest(
    request: Request,
    content: str,
    source: str,
    api_key: str = Depends(verify_api_key)
):
    """Add document to RAG."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=current_session_id or "temp"))
        result = await temp_rag.add_text(content, source)
        await temp_rag.close()
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
async def rag_stats():
    """Get RAG statistics."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=current_session_id or "temp"))
        stats = await temp_rag.stats()
        await temp_rag.close()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================

@app.get("/session/current")
async def get_current_session():
    """Get current session info."""
    global agentfs

    session_id = get_current_session_id()

    if not session_id:
        return {
            "active": False,
            "session_id": None,
            "message": "No active session"
        }

    # Verificar se a sessão existe (DB ou pasta de outputs)
    session_db = AGENTFS_DIR / f"{session_id}.db"
    session_outputs = Path.cwd() / "outputs" / session_id

    # Considera sessão ativa se tiver DB OU pasta de outputs
    if not session_db.exists() and not session_outputs.exists():
        # Sessão foi completamente deletada, retornar como inativa
        return {
            "active": False,
            "session_id": None,
            "message": "No active session"
        }

    # Get session info from KV
    session_info = None
    output_files = []
    if agentfs:
        try:
            session_info = await agentfs.kv.get("session:info")
            # List outputs from AgentFS filesystem
            output_files = await agentfs.fs.readdir("/outputs")
        except:
            pass

    return {
        "active": True,
        "session_id": session_id,
        "info": session_info,
        "has_outputs": len(output_files) > 0,
        "output_count": len(output_files),
        "outputs": output_files[:10],  # Last 10 files
    }


@app.post("/reset")
@limiter.limit("5/minute")
async def reset_endpoint(request: Request, api_key: str = Depends(verify_api_key)):
    """Start new session."""
    old_session = current_session_id
    await reset_session()

    return {
        "status": "ok",
        "message": "New session started!",
        "old_session_id": old_session,
        "new_session_id": current_session_id
    }


@app.get("/sessions")
async def list_sessions():
    """List all sessions (AgentFS + old JSONL sessions)."""
    from agentfs_sdk import AgentFS, AgentFSOptions
    sessions = []

    # 1. List AgentFS sessions (new system)
    if AGENTFS_DIR.exists():
        # Get all session DBs (UUID.db or chat-*_rag.db)
        db_files = list(AGENTFS_DIR.glob("*.db"))
        # Filter out WAL and SHM files, and only keep main DB files
        db_files = [f for f in db_files if not f.name.endswith(('-wal', '-shm', '.db-wal', '.db-shm'))]
        for db_file in sorted(db_files, key=lambda f: f.stat().st_mtime, reverse=True):
            session_id = db_file.stem

            # Remove _rag suffix from session_id for display
            if session_id.endswith("_rag"):
                session_id = session_id[:-4]  # Remove "_rag"

            # Try to get message count - preferir .jsonl como fonte (dados completos)
            message_count = 0
            output_count = 0

            # Primeiro: verificar se existe .jsonl correspondente (fonte completa)
            jsonl_file = SESSIONS_DIR / f"{session_id}.jsonl"
            if jsonl_file.exists():
                try:
                    with open(jsonl_file, 'r') as f:
                        message_count = len(f.readlines())
                except:
                    pass

            try:
                # Open AgentFS for this session to get data
                session_agentfs = await AgentFS.open(AgentFSOptions(id=session_id))

                # Se não tem .jsonl, usar KV store como fallback
                if message_count == 0:
                    history = await session_agentfs.kv.get("conversation:history")
                    if history:
                        message_count = len(history)

                # Get output count from filesystem
                try:
                    outputs = await session_agentfs.fs.readdir("/outputs")
                    output_count = len(outputs) if outputs else 0
                except:
                    output_count = 0

                await session_agentfs.close()
            except Exception as e:
                print(f"[WARN] Could not read session {session_id}: {e}")

            sessions.append({
                "session_id": session_id,
                "file": f"chat-simples/{session_id}",  # Path amigável para frontend
                "file_name": session_id,  # Nome da sessão
                "db_file": str(db_file),
                "db_size": db_file.stat().st_size,
                "updated_at": db_file.stat().st_mtime * 1000,
                "is_current": session_id == current_session_id,
                "message_count": message_count,
                "has_outputs": output_count > 0,
                "output_count": output_count,
                "model": "claude-3-5-haiku",  # Modelo padrão usado
            })

    # 2. List JSONL sessions (Claude Code standard location)
    # Use SESSIONS_DIR which points to correct location
    if SESSIONS_DIR.exists():
        for jsonl_file in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
            session_id = jsonl_file.stem

            # Count messages in JSONL
            message_count = 0
            try:
                with open(jsonl_file, 'r') as f:
                    message_count = len(f.readlines())
            except:
                pass

            sessions.append({
                "session_id": session_id,
                "file": f"backend/{jsonl_file.name}",
                "file_name": jsonl_file.name,
                "db_file": str(jsonl_file),
                "db_size": jsonl_file.stat().st_size,
                "updated_at": jsonl_file.stat().st_mtime * 1000,
                "is_current": False,
                "message_count": message_count,
                "has_outputs": False,  # Legacy sessions don't track this
                "output_count": 0,
                "model": "claude-3-5-haiku (legacy)",
            })

    # Sort all sessions by update time
    sessions.sort(key=lambda s: s['updated_at'], reverse=True)

    return {"count": len(sessions), "sessions": sessions}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (AgentFS or legacy JSONL)."""
    # Don't delete current session
    if session_id == current_session_id:
        return {"success": False, "error": "Cannot delete active session"}

    deleted = []

    # Delete AgentFS database files (includes fs, kv, tools data)
    for pattern in [f"{session_id}.db", f"{session_id}.db-wal", f"{session_id}.db-shm", f"{session_id}_rag.db"]:
        f = AGENTFS_DIR / pattern
        if f.exists():
            f.unlink()
            deleted.append(str(f))

    # Delete audit file if exists
    audit_file = AGENTFS_DIR / "audit" / f"{session_id}.jsonl"
    if audit_file.exists():
        audit_file.unlink()
        deleted.append(str(audit_file))

    # Delete legacy JSONL session if exists (check multiple locations)
    old_sessions_dirs = [
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend",
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs",
    ]
    for old_dir in old_sessions_dirs:
        jsonl_file = old_dir / f"{session_id}.jsonl"
        if jsonl_file.exists():
            jsonl_file.unlink()
            deleted.append(str(jsonl_file))

    # Delete outputs folder for this session
    import shutil
    outputs_dir = Path.cwd() / "outputs" / session_id
    if outputs_dir.exists() and outputs_dir.is_dir():
        shutil.rmtree(outputs_dir)
        deleted.append(str(outputs_dir))

    return {"success": True, "deleted": deleted}


# =============================================================================
# OUTPUT ENDPOINTS (using AgentFS filesystem)
# =============================================================================

@app.get("/outputs")
async def list_outputs(directory: str = "outputs", session_id: str = None):
    """List output files from physical filesystem."""
    try:
        # If session_id is provided, append it to the directory path
        if session_id:
            outputs_dir = Path.cwd() / directory / session_id
        else:
            outputs_dir = Path.cwd() / directory

        if not outputs_dir.exists():
            return {"files": [], "directory": str(outputs_dir), "count": 0, "session_id": session_id}

        files = []
        for file in outputs_dir.iterdir():
            if file.is_file():
                stat = file.stat()
                files.append({
                    "name": file.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime * 1000
                })

        files.sort(key=lambda f: f["modified"], reverse=True)
        return {
            "files": files,
            "directory": str(outputs_dir),
            "count": len(files),
            "session_id": session_id,
        }
    except Exception as e:
        return {"files": [], "error": str(e), "session_id": session_id}


@app.get("/outputs/file/{filename:path}")
async def get_output_file(filename: str):
    """Get output file content from AgentFS filesystem."""
    afs = await get_agentfs()

    try:
        # Ensure path starts with /
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        content = await afs.fs.read_file(filepath)
        return {
            "filename": filename,
            "content": content,
            "session_id": current_session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")


@app.post("/outputs/write")
async def write_output_file(filename: str, content: str, directory: str = "/outputs"):
    """Write file to AgentFS filesystem."""
    afs = await get_agentfs()

    try:
        filepath = f"{directory}/{filename}"
        await afs.fs.write_file(filepath, content)
        return {
            "success": True,
            "filepath": filepath,
            "session_id": current_session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/outputs/{filename:path}")
async def delete_output(filename: str):
    """Delete file from AgentFS filesystem."""
    try:
        afs = await get_agentfs()
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        await afs.fs.unlink(filepath)
        return {"success": True, "deleted": filepath}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get session details with messages."""
    messages_data = await get_session_messages(session_id)
    return messages_data


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get messages from a session (supports both AgentFS and legacy JSONL)."""
    import json

    # Try JSONL first (legacy sessions - check multiple locations)
    old_sessions_dirs = [
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend",
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs",
    ]

    jsonl_file = None
    for old_dir in old_sessions_dirs:
        candidate = old_dir / f"{session_id}.jsonl"
        if candidate.exists():
            jsonl_file = candidate
            break

    if jsonl_file:
        messages = []
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('type') in ['user', 'assistant']:
                            msg = entry.get('message', {})
                            messages.append({
                                'role': msg.get('role'),
                                'content': msg.get('content'),
                                'timestamp': entry.get('timestamp'),
                            })
                    except:
                        continue
            return {"messages": messages, "count": len(messages), "type": "jsonl"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Try AgentFS KV (new system)
    global agentfs
    if agentfs and current_session_id == session_id:
        try:
            history = await agentfs.kv.get("conversation:history") or []
            return {"messages": history, "count": len(history), "type": "agentfs"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/fs/tree")
async def get_filesystem_tree(path: str = "/"):
    """Get filesystem tree from AgentFS."""
    afs = await get_agentfs()

    try:
        # Recursively list directories
        async def list_tree(dir_path: str, depth: int = 0) -> list:
            if depth > 3:  # Limit depth
                return []
            items = []
            try:
                entries = await afs.fs.readdir(dir_path)
                for entry in entries:
                    full_path = f"{dir_path}/{entry}".replace("//", "/")
                    item = {"name": entry, "path": full_path}
                    # Try to list as directory
                    try:
                        children = await list_tree(full_path, depth + 1)
                        if children:
                            item["children"] = children
                            item["type"] = "directory"
                        else:
                            item["type"] = "file"
                    except:
                        item["type"] = "file"
                    items.append(item)
            except:
                pass
            return items

        tree = await list_tree(path)
        return {
            "path": path,
            "tree": tree,
            "session_id": current_session_id,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@app.get("/audit/tools")
async def get_audit_tools(limit: int = 100, session_id: Optional[str] = None):
    """Get tool call history."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session", "records": []}

    # Para audit, o DB é OBRIGATÓRIO (agentfs.tools precisa dele)
    session_db = AGENTFS_DIR / f"{sid}.db"
    if not session_db.exists():
        return {"error": "No active session", "session_id": None, "records": [], "count": 0}

    # Abrir AgentFS específico para a sessão solicitada
    try:
        session_afs = await AgentFS.open(AgentFSOptions(id=sid))
        stats = await session_afs.tools.get_stats()

        # get_recent requer timestamp 'since' - buscar últimas 24h
        since_timestamp = int(time.time()) - 86400  # 24 horas atrás
        recent = await session_afs.tools.get_recent(since=since_timestamp, limit=limit)
        await session_afs.close()

        # Converter ToolCall objects para dicts serializáveis
        recent_dicts = []
        for call in recent[:limit]:
            recent_dicts.append({
                "id": call.id,
                "name": call.name,
                "started_at": call.started_at,
                "completed_at": call.completed_at,
                "duration_ms": call.duration_ms,
                "status": call.status,
                "parameters": call.parameters,
                "result": call.result,
                "error": call.error,
            })

        return {
            "session_id": sid,
            "stats": [{"name": s.name, "calls": s.total_calls, "avg_ms": s.avg_duration_ms} for s in stats],
            "recent": recent_dicts,
            "count": len(recent_dicts),
        }
    except Exception as e:
        print(f"[AUDIT] Error getting tools for {sid}: {e}")
        return {"session_id": sid, "records": [], "count": 0, "error": str(e)}


@app.get("/audit/stats")
async def get_audit_stats(session_id: Optional[str] = None):
    """Get audit statistics."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session"}

    # Para audit/stats, o DB é OBRIGATÓRIO (agentfs.tools precisa dele)
    session_db = AGENTFS_DIR / f"{sid}.db"
    if not session_db.exists():
        return {"error": "No active session", "session_id": None}

    # Abrir AgentFS específico para a sessão solicitada
    try:
        session_afs = await AgentFS.open(AgentFSOptions(id=sid))
        stats = await session_afs.tools.get_stats()
        await session_afs.close()

        return {
            "session_id": sid,
            "total_calls": sum(s.total_calls for s in stats),
            "by_tool": {s.name: s.total_calls for s in stats},
            "avg_duration_ms": sum(s.avg_duration_ms for s in stats) / len(stats) if stats else 0,
        }
    except Exception as e:
        print(f"[AUDIT] Error getting stats for {sid}: {e}")
        return {"session_id": sid, "total_calls": 0, "by_tool": {}, "error": str(e)}


# =============================================================================
# AGENTFS KV ENDPOINTS (for debugging)
# =============================================================================

@app.get("/kv/list")
async def list_kv(prefix: str = ""):
    """List KV store keys."""
    afs = await get_agentfs()

    try:
        keys = await afs.kv.list(prefix=prefix if prefix else None)
        return {"keys": keys, "count": len(keys)}
    except Exception as e:
        return {"keys": [], "error": str(e)}


@app.get("/kv/{key}")
async def get_kv(key: str):
    """Get KV value."""
    afs = await get_agentfs()

    try:
        value = await afs.kv.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"key": key, "value": value}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
