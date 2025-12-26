"""Chat endpoints."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app_state import SESSIONS_DIR, get_agentfs, get_client
from claude_rag_sdk.core.auth import verify_api_key
from claude_rag_sdk.core.prompt_guard import PromptGuard
from claude_rag_sdk.core.rate_limiter import RATE_LIMITS, get_limiter
from utils.validators import validate_session_id

router = APIRouter(tags=["Chat"])


def append_to_jsonl(
    session_id: str, user_message: str, assistant_response: str, parent_uuid: Optional[str] = None
):
    """Append user and assistant messages to a session's JSONL file."""
    jsonl_file = SESSIONS_DIR / f"{session_id}.jsonl"

    if not jsonl_file.exists():
        print(f"[WARN] JSONL file not found for session {session_id}")
        return

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    user_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())

    # User message entry
    user_entry = {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "cwd": str(Path.cwd() / "outputs"),
        "sessionId": session_id,
        "version": "2.0.72",
        "type": "user",
        "message": {"role": "user", "content": user_message},
        "uuid": user_uuid,
        "timestamp": timestamp,
    }

    # Assistant message entry
    assistant_entry = {
        "parentUuid": user_uuid,
        "isSidechain": False,
        "userType": "external",
        "cwd": str(Path.cwd() / "outputs"),
        "sessionId": session_id,
        "version": "2.0.72",
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}],
        },
        "uuid": assistant_uuid,
        "timestamp": timestamp,
    }

    try:
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(user_entry, ensure_ascii=False) + "\n")
            f.write(json.dumps(assistant_entry, ensure_ascii=False) + "\n")
        print(f"[JSONL] Appended to {session_id}.jsonl")
    except Exception as e:
        print(f"[ERROR] Failed to append to JSONL: {e}")


# RAG Knowledge base path (separado do AgentFS para evitar conflitos)
RAG_DB_PATH = Path.cwd() / "data" / "rag_knowledge.db"
limiter = get_limiter()
prompt_guard = PromptGuard(strict_mode=False)


async def search_rag_context(query: str, top_k: int = 3) -> str:
    """Busca contexto relevante na base RAG."""
    if not RAG_DB_PATH.exists():
        return ""

    try:
        from claude_rag_sdk.search import SearchEngine

        engine = SearchEngine(
            db_path=str(RAG_DB_PATH),
            embedding_model="BAAI/bge-small-en-v1.5",
            enable_reranking=False,  # Mais rápido
        )
        results = await engine.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for r in results:
            context_parts.append(f"[Fonte: {r.source}]\n{r.content[:2000]}")

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")
        # Retornar mensagem de erro para o usuário saber que RAG falhou
        return "[AVISO: Busca na base de conhecimento falhou - respondendo sem contexto RAG]"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "haiku"  # haiku, sonnet, opus


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Chat with RAG-powered AI."""
    from agentfs_sdk import AgentFS, AgentFSOptions
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolResultBlock, ToolUseBlock

    import app_state

    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400, detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    try:
        c = await get_client(model=chat_request.model)

        session_specific_afs = None
        target_session_id = chat_request.session_id or app_state.current_session_id

        # Validate session_id to prevent path traversal
        if target_session_id:
            validate_session_id(target_session_id)

        if chat_request.session_id and chat_request.session_id != app_state.current_session_id:
            session_specific_afs = await AgentFS.open(AgentFSOptions(id=chat_request.session_id))
            afs = session_specific_afs
            print(f"[INFO] Usando sessão específica: {chat_request.session_id}")
        else:
            afs = await get_agentfs()

        call_id = await afs.tools.start("chat", {"message": chat_request.message[:100]})

        # Buscar contexto RAG
        rag_context = await search_rag_context(chat_request.message)

        outputs_path = str(Path.cwd() / "outputs" / target_session_id)

        # Construir mensagem com contexto RAG se disponível
        if rag_context:
            context_message = f"""[CONTEXTO DO SISTEMA - Session ID: {target_session_id}]
Ao criar arquivos, use EXATAMENTE este caminho: {outputs_path}/

[BASE DE CONHECIMENTO - Use estas informações para responder]
{rag_context}

[MENSAGEM DO USUÁRIO]
{chat_request.message}"""
            print(f"[RAG] Contexto encontrado: {len(rag_context)} chars")
        else:
            context_message = f"""[CONTEXTO DO SISTEMA - Session ID: {target_session_id}]
Ao criar arquivos, use EXATAMENTE este caminho: {outputs_path}/
Exemplo: {outputs_path}/meu_arquivo.txt

[MENSAGEM DO USUÁRIO]
{chat_request.message}"""

        await c.query(context_message)

        response_text = ""
        tool_calls = {}

        async for message in c.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        tool_call_id = await afs.tools.start(
                            block.name, {"input": str(block.input)[:500]}
                        )
                        tool_calls[block.id] = tool_call_id
                        print(f"[TOOL] {block.name} (id: {block.id})")
                    elif isinstance(block, ToolResultBlock):
                        if block.tool_use_id in tool_calls:
                            await afs.tools.success(
                                tool_calls[block.tool_use_id],
                                {"result": str(block.content)[:500]},
                            )

        for tool_use_id, tool_call_id in tool_calls.items():
            await afs.tools.success(tool_call_id, {"status": "completed_by_sdk"})

        await afs.tools.success(call_id, {"response_length": len(response_text)})

        history = await afs.kv.get("conversation:history") or []
        history.append({"role": "user", "content": chat_request.message})

        for tool_use_id, tool_call_id in tool_calls.items():
            history.append(
                {
                    "role": "assistant",
                    "content": f"[Tool Call: {tool_use_id}]",
                    "type": "tool_use",
                }
            )
            history.append(
                {
                    "role": "tool",
                    "content": f"[Tool Result: {tool_use_id}]",
                    "type": "tool_result",
                }
            )

        history.append({"role": "assistant", "content": response_text})
        await afs.kv.set("conversation:history", history[-100:])

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        await afs.fs.write_file(
            f"/outputs/chat_{timestamp}.json",
            json.dumps(
                {
                    "timestamp": timestamp,
                    "question": chat_request.message,
                    "answer": response_text,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

        import shutil

        outputs_root = Path.cwd() / "outputs"
        session_outputs = outputs_root / target_session_id
        session_outputs.mkdir(parents=True, exist_ok=True)

        if outputs_root.exists():
            for item in outputs_root.iterdir():
                if item.is_file():
                    target = session_outputs / item.name
                    shutil.move(str(item), str(target))

        for default_folder in ["default", "default_session"]:
            default_path = outputs_root / default_folder
            if default_path.exists() and default_path.is_dir():
                for item in default_path.iterdir():
                    if item.is_file():
                        target = session_outputs / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))

        tmp_outputs = Path("/tmp/outputs")
        if tmp_outputs.exists():
            for item in tmp_outputs.iterdir():
                # Validate filename to prevent path traversal attacks
                if item.is_file() and ".." not in item.name and "/" not in item.name:
                    target = session_outputs / item.name
                    shutil.move(str(item), str(target))

        # Quando session_id específico é fornecido, escrever no JSONL dessa sessão
        if chat_request.session_id:
            append_to_jsonl(
                session_id=chat_request.session_id,
                user_message=chat_request.message,
                assistant_response=response_text,
            )

            # Limpar sessão temporária que foi criada pelo client
            # A sessão atual do client não é a mesma que foi solicitada
            if (
                app_state.current_session_id
                and app_state.current_session_id != chat_request.session_id
            ):
                temp_session_id = app_state.current_session_id
                # Deletar arquivos da sessão temporária
                from app_state import AGENTFS_DIR, SESSIONS_DIR

                for pattern in [
                    f"{temp_session_id}.db",
                    f"{temp_session_id}.db-wal",
                    f"{temp_session_id}.db-shm",
                ]:
                    temp_file = AGENTFS_DIR / pattern
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except Exception:
                            pass
                temp_jsonl = SESSIONS_DIR / f"{temp_session_id}.jsonl"
                if temp_jsonl.exists():
                    try:
                        temp_jsonl.unlink()
                    except Exception:
                        pass
                print(f"[INFO] Sessão temporária {temp_session_id} removida")

        return ChatResponse(response=response_text)

    except Exception as e:
        print(f"[ERROR] Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")
    finally:
        if session_specific_afs:
            await session_specific_afs.close()


@router.post("/chat/stream")
@limiter.limit(RATE_LIMITS.get("chat_stream", "20/minute"))
async def chat_stream(
    request: Request, chat_request: ChatRequest, api_key: str = Depends(verify_api_key)
):
    """Chat with streaming response."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400, detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    # Validar session_id para prevenir path traversal
    if chat_request.session_id:
        validate_session_id(chat_request.session_id)

    r = None
    afs = None
    is_new_session = False
    try:
        import app_state
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

        # Determinar session_id a usar
        if chat_request.session_id:
            # Frontend enviou session_id específico - usar esse
            target_session_id = chat_request.session_id
        else:
            # Sem session_id do frontend - usar get_client() para criar/obter sessão
            # Isso garante que JSONL seja criado via ClaudeSDKClient (mesmo comportamento do /chat)
            await get_client(model=chat_request.model)
            target_session_id = app_state.current_session_id
            is_new_session = True
            print(f"[STREAM] Usando sessão do get_client: {target_session_id}")

        # Configurar system_prompt com caminho correto para outputs
        outputs_path = str(Path.cwd() / "outputs" / target_session_id)
        system_prompt = f"""Você é um assistente RAG especializado em responder perguntas usando uma base de conhecimento.

## Regras para criação de arquivos:
- SEMPRE salve arquivos em: {outputs_path}/
- Use nomes descritivos e extensões apropriadas (ex: relatorio.txt, dados.json)
- NUNCA use /tmp/ ou outros diretórios
- Confirme ao usuário o caminho completo do arquivo criado

Responda sempre em português brasileiro."""

        r = await ClaudeRAG.open(
            ClaudeRAGOptions(id=target_session_id, system_prompt=system_prompt)
        )

        # Abrir AgentFS para salvar histórico
        afs = await AgentFS.open(AgentFSOptions(id=target_session_id))

        # Para sessões novas, salvar projeto como chat-angular
        if is_new_session:
            try:
                await afs.kv.set("session:project", "chat-angular")
            except Exception:
                pass

        async def generate():
            nonlocal afs
            full_response = ""
            try:
                response = await r.query(chat_request.message)
                text = response.answer
                full_response = text
                chunk_size = 50

                # Enviar session_id no primeiro chunk
                yield f"data: {json.dumps({'session_id': target_session_id})}\n\n"

                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01)

                if response.citations:
                    yield f"data: {json.dumps({'citations': response.citations})}\n\n"

                # Salvar histórico no AgentFS
                try:
                    history = await afs.kv.get("conversation:history") or []
                    history.append({"role": "user", "content": chat_request.message})
                    history.append({"role": "assistant", "content": full_response})
                    await afs.kv.set("conversation:history", history[-100:])
                    print(f"[STREAM] Histórico salvo: {len(history)} mensagens")

                    # Auto-rename: definir título com primeiras 3 palavras na primeira mensagem
                    if len(history) <= 2:  # Primeira mensagem (user + assistant)
                        try:
                            existing_title = await afs.kv.get("session:title")
                            if not existing_title:
                                # Extrair primeiras 3 palavras da mensagem do usuário
                                words = chat_request.message.strip().split()[:3]
                                auto_title = " ".join(words)
                                if len(auto_title) > 50:
                                    auto_title = auto_title[:50]
                                await afs.kv.set("session:title", auto_title)
                                print(f"[STREAM] Auto-título definido: {auto_title}")
                        except Exception as title_err:
                            print(f"[WARN] Erro ao definir auto-título: {title_err}")
                except Exception as save_err:
                    print(f"[WARN] Erro ao salvar histórico: {save_err}")

                # Salvar no JSONL (tanto para sessões novas quanto continuadas)
                try:
                    append_to_jsonl(
                        session_id=target_session_id,
                        user_message=chat_request.message,
                        assistant_response=full_response,
                    )
                except Exception as jsonl_err:
                    print(f"[WARN] Erro ao salvar JSONL: {jsonl_err}")

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                await r.close()
                if afs:
                    await afs.close()

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        if r:
            await r.close()
        if afs:
            await afs.close()
        raise HTTPException(status_code=500, detail=str(e))
