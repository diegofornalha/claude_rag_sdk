"""Sessions endpoints."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

import app_state
from app_state import (
    AGENTFS_DIR,
    SESSIONS_DIR,
    clear_session,
    get_current_session_id,
    reset_session,
)
from claude_rag_sdk.core.auth import verify_api_key
from claude_rag_sdk.core.rate_limiter import get_limiter
from utils.validators import validate_session_id

router = APIRouter(tags=["Sessions"])
limiter = get_limiter()


@router.get("/session/current")
async def get_current_session():
    """Get current session info."""
    session_id = get_current_session_id()

    if not session_id:
        return {"active": False, "session_id": None, "message": "No active session"}

    session_db = AGENTFS_DIR / f"{session_id}.db"
    session_outputs = Path.cwd() / "outputs" / session_id

    if not session_db.exists() and not session_outputs.exists():
        return {"active": False, "session_id": None, "message": "No active session"}

    session_info = None
    output_files = []
    if app_state.agentfs:
        try:
            session_info = await app_state.agentfs.kv.get("session:info")
            output_files = await app_state.agentfs.fs.readdir("/outputs")
        except (OSError, IOError, KeyError):
            pass  # Session info is optional, continue without it

    return {
        "active": True,
        "session_id": session_id,
        "info": session_info,
        "has_outputs": len(output_files) > 0,
        "output_count": len(output_files),
        "outputs": output_files[:10],
    }


@router.post("/reset")
@limiter.limit("5/minute")
async def reset_endpoint(request: Request, api_key: str = Depends(verify_api_key)):
    """Start new session."""
    old_session = app_state.current_session_id
    await reset_session()

    # Obter projeto do header ou body
    project = request.headers.get("X-Client-Project", "chat-simples")
    try:
        body = await request.json()
        project = body.get("project", project)
    except Exception:
        pass

    # Armazenar projeto no AgentFS
    try:
        agentfs = await app_state.get_agentfs()
        await agentfs.kv.set("session:project", project)
    except Exception as e:
        print(f"[WARN] Could not save project: {e}")

    return {
        "status": "ok",
        "message": "New session started!",
        "old_session_id": old_session,
        "new_session_id": app_state.current_session_id,
        "project": project,
    }


@router.get("/sessions")
async def list_sessions():
    """List all sessions (AgentFS + old JSONL sessions)."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    sessions = []
    seen_session_ids = set()  # Para evitar duplicatas

    if AGENTFS_DIR.exists():
        db_files = list(AGENTFS_DIR.glob("*.db"))
        db_files = [
            f for f in db_files if not f.name.endswith(("-wal", "-shm", ".db-wal", ".db-shm"))
        ]

        # Filtrar arquivos que ainda existem (podem ter sido deletados)
        valid_db_files = []
        for f in db_files:
            try:
                f.stat()  # Verifica se existe
                # Filtrar sessões internas do Claude Code (agent-*)
                if f.stem.startswith("agent-"):
                    continue
                valid_db_files.append(f)
            except FileNotFoundError:
                continue

        for db_file in valid_db_files:
            session_id = db_file.stem

            if session_id.endswith("_rag"):
                session_id = session_id[:-4]

            message_count = 0
            output_count = 0

            jsonl_file = SESSIONS_DIR / f"{session_id}.jsonl"
            if jsonl_file.exists():
                try:
                    with open(jsonl_file, "r") as f:
                        message_count = len(f.readlines())
                except (OSError, IOError):
                    pass  # File read failed, continue with count=0

            session_agentfs = None
            project = "chat-simples"  # Default
            title = None  # Título customizado
            favorite = False  # Favorito
            assigned_project_id = None  # Projeto atribuído
            try:
                session_agentfs = await AgentFS.open(AgentFSOptions(id=session_id))

                # Ler projeto armazenado na sessão
                try:
                    stored_project = await session_agentfs.kv.get("session:project")
                    if stored_project:
                        project = stored_project
                except Exception:
                    pass

                # Ler título customizado
                try:
                    stored_title = await session_agentfs.kv.get("session:title")
                    if stored_title:
                        title = stored_title
                except Exception:
                    pass

                # Ler favorito
                try:
                    stored_favorite = await session_agentfs.kv.get("session:favorite")
                    if stored_favorite is not None:
                        favorite = stored_favorite
                except Exception:
                    pass

                # Ler projeto atribuído
                try:
                    stored_project_id = await session_agentfs.kv.get("session:project_id")
                    if stored_project_id:
                        assigned_project_id = stored_project_id
                except Exception:
                    pass

                if message_count == 0:
                    history = await session_agentfs.kv.get("conversation:history")
                    if history:
                        message_count = len(history)

                try:
                    outputs = await session_agentfs.fs.readdir("/outputs")
                    output_count = len(outputs) if outputs else 0
                except (OSError, IOError, FileNotFoundError):
                    output_count = 0  # No outputs directory or read failed
            except Exception as e:
                print(f"[WARN] Could not read session {session_id}: {e}")
            finally:
                if session_agentfs:
                    await session_agentfs.close()

            # Verificar se arquivo ainda existe (pode ter sido deletado durante a listagem)
            try:
                db_stat = db_file.stat()
                seen_session_ids.add(session_id)  # Marcar como já processado
                sessions.append(
                    {
                        "session_id": session_id,
                        "title": title,  # Título customizado (pode ser None)
                        "favorite": favorite,  # Favorito
                        "project_id": assigned_project_id,  # Projeto atribuído
                        "file": f"{project}/{session_id}",
                        "file_name": session_id,
                        "db_file": str(db_file),
                        "db_size": db_stat.st_size,
                        "updated_at": db_stat.st_mtime * 1000,
                        "is_current": session_id == app_state.current_session_id,
                        "message_count": message_count,
                        "has_outputs": output_count > 0,
                        "output_count": output_count,
                        "model": "claude-haiku-4-5",
                    }
                )
            except FileNotFoundError:
                # Arquivo foi deletado durante a listagem, ignorar
                continue

    if SESSIONS_DIR.exists():
        try:
            jsonl_files = list(SESSIONS_DIR.glob("*.jsonl"))
        except OSError:
            jsonl_files = []

        for jsonl_file in jsonl_files:
            try:
                session_id = jsonl_file.stem

                # Filtrar sessões internas do Claude Code (agent-*)
                if session_id.startswith("agent-"):
                    continue

                # Evitar duplicatas (já processado como AgentFS DB)
                if session_id in seen_session_ids:
                    continue

                # Verificar se é sessão do Claude Code (tem gitBranch no JSONL)
                # Sessões do chat-simples/angular NÃO têm gitBranch
                is_claude_code_session = False
                message_count = 0
                try:
                    with open(jsonl_file, "r") as f:
                        lines = f.readlines()
                        message_count = len(lines)
                        # Verificar qualquer linha para gitBranch (pode estar na 2ª linha)
                        for line in lines[:5]:  # Verificar primeiras 5 linhas
                            if '"gitBranch"' in line:
                                is_claude_code_session = True
                                break
                except (OSError, IOError):
                    pass  # File read failed, continue with count=0

                # Ignorar sessões do Claude Code que não têm DB correspondente
                if is_claude_code_session:
                    continue

                jsonl_stat = jsonl_file.stat()
                sessions.append(
                    {
                        "session_id": session_id,
                        "file": f"backend/{jsonl_file.name}",
                        "file_name": jsonl_file.name,
                        "db_file": str(jsonl_file),
                        "db_size": jsonl_stat.st_size,
                        "updated_at": jsonl_stat.st_mtime * 1000,
                        "is_current": False,
                        "message_count": message_count,
                        "has_outputs": False,
                        "output_count": 0,
                        "model": "claude-haiku-4-5",
                    }
                )
            except FileNotFoundError:
                # Arquivo foi deletado durante a listagem, ignorar
                continue

    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"count": len(sessions), "sessions": sessions}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (AgentFS or legacy JSONL)."""
    # Validate session_id to prevent path traversal
    validate_session_id(session_id)

    # Se for a sessão ativa, apenas limpar (não criar nova automaticamente)
    # Nova sessão será criada quando usuário enviar primeira mensagem
    was_active = session_id == app_state.current_session_id
    if was_active:
        await clear_session()

    deleted = []

    for pattern in [
        f"{session_id}.db",
        f"{session_id}.db-wal",
        f"{session_id}.db-shm",
        f"{session_id}_rag.db",
    ]:
        f = AGENTFS_DIR / pattern
        if f.exists():
            f.unlink()
            deleted.append(str(f))

    audit_file = AGENTFS_DIR / "audit" / f"{session_id}.jsonl"
    if audit_file.exists():
        audit_file.unlink()
        deleted.append(str(audit_file))

    # Deletar JSONL no SESSIONS_DIR
    sessions_jsonl = SESSIONS_DIR / f"{session_id}.jsonl"
    if sessions_jsonl.exists():
        sessions_jsonl.unlink()
        deleted.append(str(sessions_jsonl))

    old_sessions_dirs = [
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend",
        Path.home()
        / ".claude"
        / "projects"
        / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs",
    ]
    for old_dir in old_sessions_dirs:
        jsonl_file = old_dir / f"{session_id}.jsonl"
        if jsonl_file.exists():
            jsonl_file.unlink()
            deleted.append(str(jsonl_file))

    outputs_dir = Path.cwd() / "outputs" / session_id
    if outputs_dir.exists() and outputs_dir.is_dir():
        shutil.rmtree(outputs_dir)
        deleted.append(str(outputs_dir))

    return {
        "success": True,
        "deleted": deleted,
        "was_active": was_active,
        "new_session_id": app_state.current_session_id if was_active else None,
    }


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, request: Request):
    """Update session metadata (title, favorite, project_id)."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    validate_session_id(session_id)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Verificar se há pelo menos um campo para atualizar
    title = body.get("title")
    favorite = body.get("favorite")
    project_id = body.get("project_id")

    if title is None and favorite is None and project_id is None:
        raise HTTPException(
            status_code=400, detail="At least one field required: title, favorite, project_id"
        )

    session_agentfs = None
    result = {"success": True, "session_id": session_id}

    try:
        session_agentfs = await AgentFS.open(AgentFSOptions(id=session_id))

        # Atualizar título
        if title is not None:
            if not isinstance(title, str):
                raise HTTPException(status_code=400, detail="Title must be a string")
            title = title.strip()[:100]
            await session_agentfs.kv.set("session:title", title)
            result["title"] = title

        # Atualizar favorito
        if favorite is not None:
            if not isinstance(favorite, bool):
                raise HTTPException(status_code=400, detail="Favorite must be a boolean")
            await session_agentfs.kv.set("session:favorite", favorite)
            result["favorite"] = favorite

        # Atualizar projeto
        if project_id is not None:
            if project_id and not isinstance(project_id, str):
                raise HTTPException(status_code=400, detail="Project ID must be a string or null")
            if project_id:
                await session_agentfs.kv.set("session:project_id", project_id)
            else:
                # Remover do projeto (null)
                try:
                    await session_agentfs.kv.delete("session:project_id")
                except Exception:
                    pass  # Ignorar se não existir
            result["project_id"] = project_id

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {e}")
    finally:
        if session_agentfs:
            await session_agentfs.close()


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get session details with messages."""
    # Validate session_id to prevent path traversal
    validate_session_id(session_id)
    return await get_session_messages(session_id)


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get messages from a session (supports both AgentFS and legacy JSONL)."""
    # Validate session_id to prevent path traversal
    validate_session_id(session_id)
    import json

    old_sessions_dirs = [
        Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend",
        Path.home()
        / ".claude"
        / "projects"
        / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs",
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
            with open(jsonl_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("type") in ["user", "assistant"]:
                            msg = entry.get("message", {})
                            messages.append(
                                {
                                    "role": msg.get("role"),
                                    "content": msg.get("content"),
                                    "timestamp": entry.get("timestamp"),
                                }
                            )
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines
            return {"messages": messages, "count": len(messages), "type": "jsonl"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if app_state.agentfs and app_state.current_session_id == session_id:
        try:
            history = await app_state.agentfs.kv.get("conversation:history") or []
            return {"messages": history, "count": len(history), "type": "agentfs"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=404, detail="Session not found")
