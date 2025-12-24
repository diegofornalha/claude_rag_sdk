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

# Claude RAG SDK
from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions, AgentModel, EmbeddingModel

# Core modules from SDK
from claude_rag_sdk.core.security import get_allowed_origins
from claude_rag_sdk.core.rate_limiter import get_limiter, RATE_LIMITS, SLOWAPI_AVAILABLE
from claude_rag_sdk.core.prompt_guard import PromptGuard
from claude_rag_sdk.core.auth import verify_api_key, is_auth_enabled

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTFS_DIR = Path.home() / ".claude" / ".agentfs"

# Global RAG instance
rag: Optional[ClaudeRAG] = None
current_session_id: Optional[str] = None
prompt_guard = PromptGuard(strict_mode=False)


# =============================================================================
# RAG SESSION MANAGEMENT
# =============================================================================

async def get_rag() -> ClaudeRAG:
    """Get or create RAG instance."""
    global rag, current_session_id

    if rag is None:
        # Generate session ID
        current_session_id = f"chat-{uuid.uuid4().hex[:8]}"

        # Create RAG instance
        options = ClaudeRAGOptions(
            id=current_session_id,
            agent_model=AgentModel.HAIKU,
            embedding_model=EmbeddingModel.BGE_SMALL,
            enable_reranking=True,
            enable_prompt_guard=True,
            enable_adaptive_topk=True,
        )

        rag = await ClaudeRAG.open(options)

        # Store session info in KV
        await rag.kv.set("session:info", {
            "id": current_session_id,
            "created_at": time.time(),
        })

        # Create output directories in AgentFS filesystem
        await rag.fs.mkdir("/outputs")
        await rag.fs.mkdir("/reports")
        await rag.fs.mkdir("/logs")

        # Write session start log
        await rag.fs.write_file(
            f"/logs/session_start.txt",
            f"Session {current_session_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Write current session file (for external tools)
        session_file = AGENTFS_DIR / "current_session"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(current_session_id)

        print(f"🚀 RAG Session created: {current_session_id}")

    return rag


async def reset_rag() -> ClaudeRAG:
    """Reset RAG instance (new session)."""
    global rag, current_session_id

    old_session = current_session_id

    if rag is not None:
        await rag.close()
        rag = None
        current_session_id = None

    new_rag = await get_rag()
    print(f"🔄 Session reset: {old_session} -> {current_session_id}")

    return new_rag


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
    print("🚀 Starting Chat Simples with Claude RAG SDK...")
    yield
    # Cleanup
    global rag
    if rag is not None:
        await rag.close()
        print("👋 RAG session closed!")


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


class ChatResponse(BaseModel):
    response: str


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check."""
    global rag
    env = os.getenv("ENVIRONMENT", "development")
    response = {
        "status": "ok",
        "session_active": rag is not None,
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
    global rag
    env = os.getenv("ENVIRONMENT", "development")

    # Get RAG stats if available
    rag_stats = None
    if rag:
        try:
            rag_stats = await rag.stats()
        except:
            pass

    return {
        "status": "healthy",
        "environment": env,
        "session_active": rag is not None,
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
        r = await get_rag()

        # Track the query
        call_id = await r.tools.start("chat", {"message": chat_request.message[:100]})

        # Query with RAG
        response = await r.query(chat_request.message)

        # Complete tracking
        await r.tools.success(call_id, {
            "citations": len(response.citations),
            "confidence": response.confidence,
        })

        # Store in conversation history (KV)
        history = await r.kv.get("conversation:history") or []
        history.append({
            "role": "user",
            "content": chat_request.message,
        })
        history.append({
            "role": "assistant",
            "content": response.answer,
            "citations": response.citations,
        })
        await r.kv.set("conversation:history", history[-50:])  # Keep last 50

        # Save response to filesystem for persistence
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        await r.fs.write_file(
            f"/outputs/chat_{timestamp}.json",
            json.dumps({
                "timestamp": timestamp,
                "question": chat_request.message,
                "answer": response.answer,
                "citations": response.citations,
                "confidence": response.confidence,
            }, indent=2, ensure_ascii=False)
        )

        return ChatResponse(response=response.answer)

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
        r = await get_rag()
        results = await r.search(query, top_k=top_k)
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
        r = await get_rag()
        result = await r.add_text(content, source)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
async def rag_stats():
    """Get RAG statistics."""
    try:
        r = await get_rag()
        return await r.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================

@app.get("/session/current")
async def get_current_session():
    """Get current session info."""
    global rag

    session_id = get_current_session_id()

    if not session_id:
        return {
            "active": False,
            "session_id": None,
            "message": "No active session"
        }

    # Get session info from KV
    session_info = None
    output_files = []
    if rag:
        try:
            session_info = await rag.kv.get("session:info")
            # List outputs from AgentFS filesystem
            output_files = await rag.fs.readdir("/outputs")
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
async def reset_session(request: Request, api_key: str = Depends(verify_api_key)):
    """Start new session."""
    old_session = current_session_id
    await reset_rag()

    return {
        "status": "ok",
        "message": "New session started!",
        "old_session_id": old_session,
        "new_session_id": current_session_id
    }


@app.get("/sessions")
async def list_sessions():
    """List all sessions."""
    sessions = []

    # Check AgentFS directory for session databases
    if AGENTFS_DIR.exists():
        for db_file in sorted(AGENTFS_DIR.glob("chat-*.db"), key=lambda f: f.stat().st_mtime, reverse=True):
            session_id = db_file.stem

            sessions.append({
                "session_id": session_id,
                "db_file": str(db_file),
                "db_size": db_file.stat().st_size,
                "updated_at": db_file.stat().st_mtime * 1000,
                "is_current": session_id == current_session_id,
            })

    return {"count": len(sessions), "sessions": sessions}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (deletes AgentFS database which includes fs, kv, tools)."""
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

    return {"success": True, "deleted": deleted}


# =============================================================================
# OUTPUT ENDPOINTS (using AgentFS filesystem)
# =============================================================================

@app.get("/outputs")
async def list_outputs(directory: str = "/outputs"):
    """List output files from AgentFS filesystem."""
    global rag
    if not rag:
        return {"files": [], "error": "No active session"}

    try:
        files = await rag.fs.readdir(directory)
        return {
            "files": files,
            "directory": directory,
            "count": len(files),
            "session_id": current_session_id,
        }
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.get("/outputs/file/{filename:path}")
async def get_output_file(filename: str):
    """Get output file content from AgentFS filesystem."""
    global rag
    if not rag:
        raise HTTPException(status_code=404, detail="No session")

    try:
        # Ensure path starts with /
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        content = await rag.fs.read_file(filepath)
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
    global rag
    if not rag:
        raise HTTPException(status_code=404, detail="No session")

    try:
        filepath = f"{directory}/{filename}"
        await rag.fs.write_file(filepath, content)
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
    global rag
    if not rag:
        return {"success": False, "error": "No session"}

    try:
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        await rag.fs.unlink(filepath)
        return {"success": True, "deleted": filepath}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/fs/tree")
async def get_filesystem_tree(path: str = "/"):
    """Get filesystem tree from AgentFS."""
    global rag
    if not rag:
        return {"error": "No active session"}

    try:
        # Recursively list directories
        async def list_tree(dir_path: str, depth: int = 0) -> list:
            if depth > 3:  # Limit depth
                return []
            items = []
            try:
                entries = await rag.fs.readdir(dir_path)
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
    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session", "records": []}

    # Get from AgentFS tools
    if rag:
        try:
            stats = await rag.tools.get_stats()
            recent = await rag.tools.get_recent(limit=limit)
            return {
                "session_id": sid,
                "stats": [{"name": s.name, "calls": s.total_calls, "avg_ms": s.avg_duration_ms} for s in stats],
                "recent": recent[:limit],
                "count": len(recent),
            }
        except:
            pass

    return {"session_id": sid, "records": [], "count": 0}


@app.get("/audit/stats")
async def get_audit_stats(session_id: Optional[str] = None):
    """Get audit statistics."""
    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session"}

    if rag:
        try:
            stats = await rag.tools.get_stats()
            return {
                "session_id": sid,
                "total_calls": sum(s.total_calls for s in stats),
                "by_tool": {s.name: s.total_calls for s in stats},
                "avg_duration_ms": sum(s.avg_duration_ms for s in stats) / len(stats) if stats else 0,
            }
        except:
            pass

    return {"session_id": sid, "total_calls": 0, "by_tool": {}}


# =============================================================================
# AGENTFS KV ENDPOINTS (for debugging)
# =============================================================================

@app.get("/kv/list")
async def list_kv(prefix: str = ""):
    """List KV store keys."""
    if not rag:
        return {"keys": [], "error": "No session"}

    try:
        keys = await rag.kv.list(prefix=prefix if prefix else None)
        return {"keys": keys, "count": len(keys)}
    except Exception as e:
        return {"keys": [], "error": str(e)}


@app.get("/kv/{key}")
async def get_kv(key: str):
    """Get KV value."""
    if not rag:
        raise HTTPException(status_code=404, detail="No session")

    try:
        value = await rag.kv.get(key)
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
