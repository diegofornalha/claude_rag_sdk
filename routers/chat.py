"""Chat endpoints."""
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import asyncio
import time
import re

from claude_rag_sdk.core.rate_limiter import get_limiter, RATE_LIMITS
from claude_rag_sdk.core.prompt_guard import PromptGuard
from claude_rag_sdk.core.auth import verify_api_key

from app_state import get_client, get_agentfs

router = APIRouter(tags=["Chat"])


def validate_session_id(session_id: str) -> bool:
    """Validate session_id to prevent path traversal attacks.

    Returns True if valid, raises HTTPException if invalid.
    """
    if not session_id:
        return False
    # Only allow UUID-like patterns and alphanumeric with hyphens
    if not re.match(r'^[a-zA-Z0-9\-_]+$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    # Prevent path traversal
    if '..' in session_id or '/' in session_id or '\\' in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id: path traversal detected")
    return True

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
        print(f"[WARN] RAG search error: {e}")
        return ""


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = "haiku"  # haiku, sonnet, opus


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat with RAG-powered AI."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock
    from agentfs_sdk import AgentFS, AgentFSOptions
    import app_state

    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked: {scan_result.threat_level.value}"
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
                            block.name,
                            {"input": str(block.input)[:500]}
                        )
                        tool_calls[block.id] = tool_call_id
                        print(f"[TOOL] {block.name} (id: {block.id})")
                    elif isinstance(block, ToolResultBlock):
                        if block.tool_use_id in tool_calls:
                            await afs.tools.success(
                                tool_calls[block.tool_use_id],
                                {"result": str(block.content)[:500]}
                            )

        for tool_use_id, tool_call_id in tool_calls.items():
            await afs.tools.success(tool_call_id, {"status": "completed_by_sdk"})

        await afs.tools.success(call_id, {"response_length": len(response_text)})

        history = await afs.kv.get("conversation:history") or []
        history.append({"role": "user", "content": chat_request.message})

        for tool_use_id, tool_call_id in tool_calls.items():
            history.append({"role": "assistant", "content": f"[Tool Call: {tool_use_id}]", "type": "tool_use"})
            history.append({"role": "tool", "content": f"[Tool Result: {tool_use_id}]", "type": "tool_result"})

        history.append({"role": "assistant", "content": response_text})
        await afs.kv.set("conversation:history", history[-100:])

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        await afs.fs.write_file(
            f"/outputs/chat_{timestamp}.json",
            json.dumps({
                "timestamp": timestamp,
                "question": chat_request.message,
                "answer": response_text,
            }, indent=2, ensure_ascii=False)
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
                if item.is_file() and '..' not in item.name and '/' not in item.name:
                    target = session_outputs / item.name
                    shutil.move(str(item), str(target))

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
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat with streaming response."""
    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        import app_state

        r = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))

        async def generate():
            try:
                response = await r.query(chat_request.message)
                text = response.answer
                chunk_size = 50
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0.01)

                if response.citations:
                    yield f"data: {json.dumps({'citations': response.citations})}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                await r.close()

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
