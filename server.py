from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import json
import os
import sys
import importlib.util
import asyncio
import shutil

from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock, ProcessError

# Importa config do RAG Agent
rag_config_path = Path(__file__).parent / "rag-agent" / "config.py"
spec = importlib.util.spec_from_file_location("rag_config", rag_config_path)
rag_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_config)
RAG_AGENT_OPTIONS = rag_config.RAG_AGENT_OPTIONS

# Importa modulos de seguranca
sys.path.insert(0, str(Path(__file__).parent))
from core.security import get_allowed_origins, get_allowed_methods, get_allowed_headers, SECURITY_CONFIG
from core.rate_limiter import get_limiter, RATE_LIMITS, get_client_ip, SLOWAPI_AVAILABLE
from core.prompt_guard import validate_prompt
from core.auth import verify_api_key, is_auth_enabled

# Importa funções de logger do RAG agent para session tracking
rag_logger_path = Path(__file__).parent / "rag-agent" / "core" / "logger.py"
spec_logger = importlib.util.spec_from_file_location("rag_logger", rag_logger_path)
rag_logger = importlib.util.module_from_spec(spec_logger)
spec_logger.loader.exec_module(rag_logger)
set_session_id = rag_logger.set_session_id
get_session_id = rag_logger.get_session_id

SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend-rag-agent"
RAG_OUTPUTS_DIR = Path(__file__).parent / "rag-agent" / "outputs"

client: ClaudeSDKClient | None = None


def extract_session_id_from_jsonl() -> str:
    """Extrai session_id do arquivo JSONL mais recente."""
    if not SESSIONS_DIR.exists():
        return "default"

    # Pegar JSONL mais recente (por mtime)
    jsonl_files = sorted(
        SESSIONS_DIR.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    if not jsonl_files:
        return "default"

    latest_jsonl = jsonl_files[0]

    # Ler primeira linha para extrair sessionId
    try:
        with open(latest_jsonl, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                session_id = data.get("sessionId", latest_jsonl.stem)
                return session_id
    except Exception as e:
        print(f"[WARN] Não foi possível extrair sessionId: {e}")
        return latest_jsonl.stem  # Fallback: usar nome do arquivo

    return "default"


async def get_client() -> ClaudeSDKClient:
    """Retorna o cliente, criando se necessário."""
    global client
    if client is None:
        # Criar cliente único (evita mismatch de session_id)
        client = ClaudeSDKClient(options=RAG_AGENT_OPTIONS)
        try:
            await client.__aenter__()
            print("🔗 Nova sessão criada!")

            # Aguardar SDK escrever primeira linha do JSONL
            await asyncio.sleep(0.2)

            # Extrair session_id do cliente ativo
            session_id = extract_session_id_from_jsonl()
            set_session_id(session_id)
            print(f"📁 Session ID: {session_id}")

            # Criar pasta da sessão para outputs
            session_output_dir = RAG_OUTPUTS_DIR / session_id
            session_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 Pasta da sessão criada: {session_output_dir}")

            # Inicializar AgentFS para a sessão
            from core.agentfs_manager import init_agentfs
            await init_agentfs(session_id)
            print(f"🗄️  AgentFS inicializado: ~/.claude/.agentfs/{session_id}.db")

            # Definir variável de ambiente para MCP server usar auditoria
            os.environ["AGENTFS_SESSION_ID"] = session_id

            # Escrever session_id em arquivo compartilhado para subprocessos
            session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
            session_file.write_text(session_id)

        except Exception as e:
            # Cleanup em caso de erro durante inicialização
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            client = None
            raise e

    return client

async def reset_client():
    """Reseta o cliente (nova sessão)."""
    global client
    if client is not None:
        await client.__aexit__(None, None, None)
        client = None
    return await get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida do app."""
    print("🚀 Iniciando Chat Simples...")
    yield
    # Cleanup ao desligar
    global client
    if client is not None:
        await client.__aexit__(None, None, None)
        print("👋 Sessão encerrada!")

    # Fechar AgentFS
    from core.agentfs_manager import close_agentfs
    await close_agentfs()
    print("🗄️  AgentFS fechado")

app = FastAPI(
    title="Chat Simples",
    description="Backend com sessão persistente - Claude Agent SDK",
    version="2.0.0",
    lifespan=lifespan
)

# CORS - permitir localhost em dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8001", "http://127.0.0.1:3000", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter (Debito #2)
limiter = get_limiter()
if SLOWAPI_AVAILABLE:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    """Health check."""
    global client
    env = os.getenv("ENVIRONMENT", "development")
    response = {
        "status": "ok",
        "session_active": client is not None,
        "message": "Chat Simples v2 - Sessão Persistente",
        "auth_enabled": is_auth_enabled()
    }
    # Em dev, expor a API key
    if env != "production" and is_auth_enabled():
        from core.auth import VALID_API_KEYS
        if VALID_API_KEYS:
            response["dev_key"] = list(VALID_API_KEYS)[0]
    return response

@app.get("/health")
async def health_check():
    """Health check detalhado com status de segurança."""
    global client
    env = os.getenv("ENVIRONMENT", "development")
    return {
        "status": "healthy",
        "environment": env,
        "session_active": client is not None,
        "security": {
            "auth_enabled": is_auth_enabled(),
            "cors_origins": len(get_allowed_origins()),
            "rate_limiter": "slowapi" if SLOWAPI_AVAILABLE else "simple",
            "prompt_guard": "active"
        }
    }


def _get_current_session_id() -> str:
    """Obtém session_id do arquivo compartilhado com fallback."""
    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        try:
            session_id = session_file.read_text().strip()
            if session_id:
                return session_id
        except:
            pass
    return get_session_id()  # Fallback para logger


@app.get("/session/current")
async def get_current_session():
    """Retorna informações da sessão atual."""
    global client

    # Usar a mesma lógica de _get_current_session_id() com fallback
    session_id = _get_current_session_id()

    # Se não houver sessão válida, retornar inativo
    if not session_id or session_id == "default":
        return {
            "active": False,
            "session_id": None,
            "message": "Nenhuma sessão ativa"
        }
    session_file = SESSIONS_DIR / f"{session_id}.jsonl"

    # Contar mensagens
    message_count = 0
    if session_file.exists():
        message_count = len(session_file.read_text().strip().split('\n'))

    # Verificar outputs no rag-agent
    session_output_dir = RAG_OUTPUTS_DIR / session_id
    has_outputs = session_output_dir.exists()
    output_count = len(list(session_output_dir.iterdir())) if has_outputs else 0

    return {
        "active": True,
        "session_id": session_id,
        "message_count": message_count,
        "has_outputs": has_outputs,
        "output_count": output_count,
        "output_dir": str(session_output_dir) if has_outputs else None
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS["chat"])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat com sessão persistente."""
    # Validacao anti-injection (Debito #3)
    validation = validate_prompt(chat_request.message)
    if not validation.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Mensagem bloqueada: {validation.message}"
        )

    try:
        c = await get_client()

        # Envia mensagem
        await c.query(chat_request.message)

        # Coleta resposta
        response_text = ""
        async for message in c.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        return ChatResponse(response=response_text)

    except ProcessError as e:
        raise HTTPException(status_code=503, detail=f"Erro ao processar com Claude: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@app.post("/chat/stream")
@limiter.limit(RATE_LIMITS["chat_stream"])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat com streaming e sessão persistente."""
    # Validacao anti-injection (Debito #3)
    validation = validate_prompt(chat_request.message)
    if not validation.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Mensagem bloqueada: {validation.message}"
        )

    try:
        c = await get_client()

        async def generate():
            await c.query(chat_request.message)
            async for message in c.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            yield f"data: {block.text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except ProcessError as e:
        raise HTTPException(status_code=503, detail=f"Erro ao processar com Claude: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@app.post("/reset")
@limiter.limit("5/minute")
async def reset_session(request: Request, api_key: str = Depends(verify_api_key)):
    """Inicia nova sessão (novo JSONL)."""
    old_session_id = get_session_id()
    await reset_client()

    # Limpar arquivo de sessão atual (AgentFS)
    session_file_path = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file_path.exists():
        try:
            session_file_path.unlink()
        except Exception:
            pass

    # Aguardar nova sessão ser criada
    await asyncio.sleep(0.1)
    new_session_id = extract_session_id_from_jsonl()
    set_session_id(new_session_id)

    return {
        "status": "ok",
        "message": "Nova sessão iniciada!",
        "old_session_id": old_session_id,
        "new_session_id": new_session_id
    }


@app.get("/sessions")
async def list_sessions():
    """Lista todas as sessões disponíveis."""
    sessions = []

    if not SESSIONS_DIR.exists():
        return {"count": 0, "sessions": []}

    for file in sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            lines = file.read_text().strip().split('\n')
            message_count = len(lines)

            # Usar nome do arquivo como session_id único
            session_id = file.stem

            # Tentar extrair modelo da primeira mensagem
            model = "unknown"
            for line in lines[:5]:
                try:
                    data = json.loads(line)
                    if "message" in data and "model" in data.get("message", {}):
                        model = data["message"]["model"]
                        break
                except:
                    pass

            # Verificar se há outputs para esta sessão
            session_output_dir = RAG_OUTPUTS_DIR / session_id
            has_outputs = session_output_dir.exists()
            output_count = len(list(session_output_dir.iterdir())) if has_outputs else 0

            sessions.append({
                "session_id": session_id,
                "file_name": file.name,
                "file": str(file),
                "message_count": message_count,
                "model": model,
                "updated_at": file.stat().st_mtime * 1000,
                "has_outputs": has_outputs,
                "output_count": output_count
            })
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")

    return {"count": len(sessions), "sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retorna mensagens de uma sessão."""
    file_path = SESSIONS_DIR / f"{session_id}.jsonl"

    if not file_path.exists():
        return {"error": "Sessão não encontrada"}

    messages = []
    for line in file_path.read_text().strip().split('\n'):
        try:
            messages.append(json.loads(line))
        except:
            pass

    return {"count": len(messages), "messages": messages}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deleta uma sessão e sua pasta de outputs."""
    import shutil

    file_path = SESSIONS_DIR / f"{session_id}.jsonl"

    if not file_path.exists():
        return {"success": False, "error": "Sessão não encontrada"}

    try:
        # Deletar arquivo da sessão
        file_path.unlink()

        # Deletar pasta de outputs da sessão (se existir)
        outputs_dir = RAG_OUTPUTS_DIR / session_id
        if outputs_dir.exists() and outputs_dir.is_dir():
            shutil.rmtree(outputs_dir)
            print(f"🗑️ Pasta de outputs removida: {outputs_dir}")

        # Deletar arquivos do AgentFS da sessão (se existirem)
        agentfs_dir = Path.home() / ".claude" / ".agentfs"
        agentfs_files = [
            agentfs_dir / f"{session_id}.db",
            agentfs_dir / f"{session_id}.db-wal",
            agentfs_dir / f"{session_id}.db-shm",
            agentfs_dir / "audit" / f"{session_id}.jsonl"
        ]
        for f in agentfs_files:
            if f.exists():
                f.unlink()
                print(f"🗑️ AgentFS removido: {f.name}")

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/rag-outputs/{session_id}")
async def list_rag_outputs_by_session(session_id: str):
    """Lista arquivos de uma sessão específica do RAG agent."""
    session_dir = RAG_OUTPUTS_DIR / session_id

    if not session_dir.exists():
        return {"session_id": session_id, "files": [], "count": 0}

    files = []
    for file in session_dir.iterdir():
        if file.is_file():
            stat = file.stat()
            files.append({
                "name": file.name,
                "path": str(file.relative_to(RAG_OUTPUTS_DIR)),
                "size": stat.st_size,
                "modified": stat.st_mtime * 1000
            })

    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"session_id": session_id, "files": files, "count": len(files)}


@app.get("/sessions/{session_id}/rag-outputs")
async def get_session_rag_outputs_detailed(session_id: str):
    """Retorna informação detalhada dos outputs RAG de uma sessão."""
    session_dir = RAG_OUTPUTS_DIR / session_id

    if not session_dir.exists():
        return {
            "session_id": session_id,
            "exists": False,
            "files": [],
            "total_size": 0,
            "count": 0
        }

    files = []
    total_size = 0

    for file in session_dir.iterdir():
        if file.is_file():
            stat = file.stat()
            total_size += stat.st_size

            # Ler primeiras linhas para preview
            preview = ""
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    preview = f.read(200)
            except:
                preview = "[binary file]"

            files.append({
                "name": file.name,
                "path": str(file.relative_to(RAG_OUTPUTS_DIR)),
                "size": stat.st_size,
                "modified": stat.st_mtime * 1000,
                "preview": preview
            })

    files.sort(key=lambda f: f["modified"], reverse=True)

    return {
        "session_id": session_id,
        "exists": True,
        "files": files,
        "total_size": total_size,
        "count": len(files)
    }


@app.delete("/sessions/{session_id}/rag-outputs")
async def delete_session_rag_outputs(session_id: str):
    """Deleta todos os outputs RAG de uma sessão."""
    session_dir = RAG_OUTPUTS_DIR / session_id

    if not session_dir.exists():
        return {"success": False, "error": "Sessão não encontrada"}

    try:
        shutil.rmtree(session_dir)
        return {"success": True, "message": f"Outputs da sessão {session_id} deletados"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_session_outputs_dir(session_id: Optional[str] = None) -> Path:
    """Retorna diretório de outputs da sessão."""
    if session_id:
        return RAG_OUTPUTS_DIR / session_id

    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        current_session = session_file.read_text().strip()
        return RAG_OUTPUTS_DIR / current_session
    return RAG_OUTPUTS_DIR

@app.get("/outputs")
async def list_outputs(session_id: Optional[str] = None):
    """Lista arquivos da pasta outputs de uma sessão."""
    outputs_dir = _get_session_outputs_dir(session_id)

    if not outputs_dir.exists():
        return {"files": [], "directory": str(outputs_dir), "session_id": session_id}

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
    return {"files": files, "directory": str(outputs_dir), "session_id": session_id}

@app.get("/outputs/file/{filename}")
async def get_output_file(filename: str, session_id: Optional[str] = None):
    """Retorna conteúdo de um arquivo de output."""
    from fastapi.responses import FileResponse

    outputs_dir = _get_session_outputs_dir(session_id)
    file_path = outputs_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    return FileResponse(file_path, filename=filename)

@app.delete("/outputs/{filename}")
async def delete_output(filename: str, session_id: Optional[str] = None):
    """Deleta um arquivo da pasta outputs de uma sessão."""
    outputs_dir = _get_session_outputs_dir(session_id)
    file_path = outputs_dir / filename

    if not file_path.exists():
        return {"success": False, "error": "Arquivo nao encontrado"}

    try:
        file_path.unlink()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================================================================
# ENDPOINTS DE AUDITORIA - Tool Calls
# =============================================================================

AUDIT_DIR = Path.home() / ".claude" / ".agentfs" / "audit"
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/audit/tools")
async def get_audit_tools(limit: int = 100, session_id: Optional[str] = None):
    """
    Retorna histórico de tool calls da sessão.

    Args:
        limit: Número máximo de registros (padrão 100)
        session_id: ID da sessão (opcional, usa atual se não fornecido)
    """
    if not session_id:
        session_id = _get_current_session_id()

    if not session_id or session_id == "default":
        return {"error": "Nenhuma sessão ativa", "records": []}

    audit_file = AUDIT_DIR / f"{session_id}.jsonl"

    if not audit_file.exists():
        return {
            "session_id": session_id,
            "records": [],
            "count": 0,
            "message": "Nenhum registro de auditoria ainda"
        }

    records = []
    try:
        with open(audit_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Limitar e ordenar por mais recente
        records = records[-limit:]
        records.reverse()

        return {
            "session_id": session_id,
            "records": records,
            "count": len(records)
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@app.get("/audit/stats")
async def get_audit_stats(session_id: Optional[str] = None):
    """Retorna estatísticas de auditoria da sessão."""
    if not session_id:
        session_id = _get_current_session_id()

    if not session_id or session_id == "default":
        return {"error": "Nenhuma sessão ativa"}

    audit_file = AUDIT_DIR / f"{session_id}.jsonl"

    if not audit_file.exists():
        return {
            "session_id": session_id,
            "total_calls": 0,
            "by_tool": {},
            "errors": 0,
            "avg_duration_ms": 0
        }

    try:
        records = []
        with open(audit_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return {
                "session_id": session_id,
                "total_calls": 0,
                "by_tool": {},
                "errors": 0,
                "avg_duration_ms": 0
            }

        by_tool = {}
        total_duration = 0
        errors = 0

        for r in records:
            tool = r.get("tool_name", "unknown")
            by_tool[tool] = by_tool.get(tool, 0) + 1
            total_duration += r.get("duration_ms", 0)
            if r.get("error"):
                errors += 1

        return {
            "session_id": session_id,
            "total_calls": len(records),
            "by_tool": by_tool,
            "errors": errors,
            "avg_duration_ms": round(total_duration / len(records), 2),
            "first_call": records[0].get("started_at") if records else None,
            "last_call": records[-1].get("completed_at") if records else None
        }
    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@app.get("/audit/sessions")
async def list_audit_sessions():
    """Lista todas as sessões com dados de auditoria."""
    if not AUDIT_DIR.exists():
        return {"sessions": [], "count": 0}

    sessions = []
    for file in sorted(AUDIT_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            # Contar linhas (tool calls)
            with open(file, 'r') as f:
                lines = [l for l in f if l.strip()]

            sessions.append({
                "session_id": file.stem,
                "tool_calls": len(lines),
                "file_size": file.stat().st_size,
                "modified": file.stat().st_mtime * 1000
            })
        except Exception as e:
            print(f"Erro ao ler audit file {file}: {e}")

    return {
        "sessions": sessions,
        "count": len(sessions),
        "audit_dir": str(AUDIT_DIR)
    }


@app.get("/audit/dashboard")
async def get_audit_dashboard():
    """Serve o dashboard HTML de auditoria."""
    dashboard_path = STATIC_DIR / "audit_dashboard.html"

    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")

    return FileResponse(
        dashboard_path,
        media_type="text/html",
        filename="audit_dashboard.html"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
