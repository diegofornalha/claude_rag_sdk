"""Chat endpoints."""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app_state import SESSIONS_DIR, get_agentfs, get_client
from claude_rag_sdk.core.auth import verify_api_key
from claude_rag_sdk.core.cache import get_response_cache
from claude_rag_sdk.core.guest_limits import GuestLimitAction, get_guest_limit_manager
from claude_rag_sdk.core.logger import get_logger
from claude_rag_sdk.core.prompt_guard import PromptGuard
from claude_rag_sdk.core.rate_limiter import RATE_LIMITS, get_limiter
from claude_rag_sdk.core.sdk_hooks import set_current_session_id
from utils.validators import validate_session_id

# Cache global para contexto RAG
_rag_context_cache = get_response_cache()

router = APIRouter(tags=["Chat"])
logger = get_logger("chat")


# Padrões para detectar comandos de gerenciamento de sessão
SESSION_COMMANDS = {
    "favorite": [
        r"favorit[ae]r?\b",
        r"favorit[ae]\s+(esse|este|essa|esta)\s+(chat|conversa)",
        r"adiciona[r]?\s+(aos|nos)\s+favoritos",
        r"coloca[r]?\s+(nos|nos)\s+favoritos",
        r"marca[r]?\s+como\s+favorito",
    ],
    "unfavorite": [
        r"desfavorit[ae]r?\b",
        r"tir[ae]r?\s+(dos|de)\s+favoritos",
        r"remov[ae]r?\s+(dos|de)\s+favoritos",
        r"desmarca[r]?\s+favorito",
    ],
    "rename": [
        r"renomei?a?r?\s+(?:para\s+)?['\"]?(.+?)['\"]?\s*$",
        r"renome\w*\s+(?:para\s+)?['\"]?(.+?)['\"]?\s*$",
        r"muda[r]?\s+(?:o\s+)?nome\s+(?:para\s+)?['\"]?(.+?)['\"]?\s*$",
        r"(?:chama[r]?|nomea[r]?)\s+(?:de\s+)?['\"]?(.+?)['\"]?\s*$",
        r"(?:define|defina|coloca|coloque)\s+(?:o\s+)?(?:nome|título)\s+(?:como\s+)?['\"]?(.+?)['\"]?\s*$",
    ],
}


def detect_session_command(message: str) -> tuple[str | None, str | None]:
    """Detecta comandos de gerenciamento de sessão na mensagem.

    Returns:
        tuple: (command_type, extra_data) onde:
            - command_type: 'favorite', 'unfavorite', 'rename' ou None
            - extra_data: novo título para rename, ou None
    """
    msg_lower = message.lower().strip()

    # IMPORTANTE: Verificar comandos de DESFAVORITAR ANTES de favoritar
    # porque "desfavoritar" contém "favoritar"
    for pattern in SESSION_COMMANDS["unfavorite"]:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("unfavorite", None)

    # Verificar comandos de favoritar
    for pattern in SESSION_COMMANDS["favorite"]:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("favorite", None)

    # Verificar comandos de renomear (captura o novo nome)
    for pattern in SESSION_COMMANDS["rename"]:
        match = re.search(pattern, msg_lower, re.IGNORECASE)
        if match:
            # Extrair o novo nome do grupo de captura
            new_name = match.group(1).strip() if match.lastindex else None
            if new_name:
                # Limpar aspas e espaços extras
                new_name = new_name.strip("'\"").strip()
                if len(new_name) > 0:
                    return ("rename", new_name[:100])  # Limitar a 100 chars

    return (None, None)


async def execute_session_command(
    afs, session_id: str, command: str, extra_data: str | None
) -> str:
    """Executa um comando de gerenciamento de sessão.

    Returns:
        Mensagem de confirmação para o usuário
    """
    try:
        if command == "favorite":
            await afs.kv.set("session:favorite", True)
            return "✅ Chat adicionado aos favoritos!"

        elif command == "unfavorite":
            await afs.kv.set("session:favorite", False)
            return "✅ Chat removido dos favoritos."

        elif command == "rename" and extra_data:
            await afs.kv.set("session:title", extra_data)
            return f"✅ Chat renomeado para: **{extra_data}**"

        return "❌ Comando não reconhecido."

    except Exception as e:
        logger.error(
            "Erro ao executar comando de sessão", error_type="session_command", error=str(e)
        )
        return f"❌ Erro ao executar comando: {str(e)}"


def append_to_jsonl(
    session_id: str, user_message: str, assistant_response: str, parent_uuid: str | None = None
):
    """Append user and assistant messages to a session's JSONL file."""
    jsonl_file = SESSIONS_DIR / f"{session_id}.jsonl"

    # CORREÇÃO: Criar arquivo se não existir (em vez de retornar)
    if not jsonl_file.exists():
        logger.info("Criando arquivo JSONL para nova sessão", session_id=session_id)
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file.touch()  # Criar arquivo vazio

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    user_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())

    # User message entry
    user_entry = {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "cwd": str(Path.cwd() / "artifacts"),
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
        "cwd": str(Path.cwd() / "artifacts"),
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
        logger.debug("Mensagens salvas no JSONL", session_id=session_id)
    except Exception as e:
        logger.error("Falha ao salvar JSONL", session_id=session_id, error=str(e))


# Config centralizada para RAG
from claude_rag_sdk.core.config import get_config

limiter = get_limiter()
prompt_guard = PromptGuard(strict_mode=False)


async def search_rag_context(query: str, top_k: int = 5) -> str:
    """Busca contexto relevante na base RAG com cache.

    Cache evita recalcular embeddings e buscas para perguntas similares.
    """
    config = get_config()
    rag_db_path = config.rag_db_path

    if not rag_db_path.exists():
        return ""

    # Verificar cache primeiro
    cache_key = f"rag:{query[:200]}"  # Limita tamanho da chave
    cached = _rag_context_cache.get(cache_key, top_k)
    if cached:
        logger.debug("RAG cache hit", query_preview=query[:50])
        return cached.get("context", "")

    try:
        from claude_rag_sdk.search import SearchEngine

        engine = SearchEngine(
            db_path=str(rag_db_path),
            embedding_model=config.embedding_model_string,
            enable_reranking=True,  # Re-ranking para melhor relevância
        )
        results = await engine.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for r in results:
            context_parts.append(f"[Fonte: {r.source}]\n{r.content[:2000]}")

        context = "\n\n---\n\n".join(context_parts)

        # Salvar no cache (TTL do .env ou 30 min default)
        _rag_context_cache.set(cache_key, top_k, {"context": context})
        logger.debug("RAG cache miss - salvando", query_preview=query[:50])

        return context
    except Exception as e:
        logger.error("Busca RAG falhou", error=str(e))
        # Retornar mensagem de erro para o usuário saber que RAG falhou
        return "[AVISO: Busca na base de conhecimento falhou - respondendo sem contexto RAG]"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str | None = "haiku"  # haiku, sonnet, opus
    resume: bool | None = False  # Resume previous conversation context
    fork_session: bool | None = False  # Fork instead of continue


class ChatStreamRequest(BaseModel):
    """Request model for streaming chat endpoint."""

    message: str
    session_id: str | None = None
    model: str | None = "opus"  # haiku, sonnet, opus
    resume: bool | None = True  # Resume previous conversation context
    fork_session: str | None = None  # Fork from this session
    use_rag: bool = True  # Enable RAG context
    top_k: int = 5  # Number of RAG results


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Chat with RAG-powered AI."""
    from agentfs_sdk import AgentFS, AgentFSOptions
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

    import app_state

    scan_result = prompt_guard.scan(chat_request.message)
    if not scan_result.is_safe:
        raise HTTPException(
            status_code=400, detail=f"Message blocked: {scan_result.threat_level.value}"
        )

    # Obter projeto do header (chat-angular ou outro)
    project = request.headers.get("X-Client-Project", "default")

    try:
        # Configurar resume se solicitado
        resume_id = chat_request.session_id if chat_request.resume else None
        c = await get_client(
            model=chat_request.model,
            project=project,
            resume_session=resume_id,
            fork_session=chat_request.fork_session,
        )

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

        # Buscar contexto RAG
        rag_context = await search_rag_context(chat_request.message)

        artifacts_path = str(Path.cwd() / "artifacts" / target_session_id)

        # Construir mensagem com contexto RAG se disponível
        if rag_context:
            context_message = f"""[CONTEXTO DO SISTEMA - Session ID: {target_session_id}]
Ao criar arquivos, use EXATAMENTE este caminho: {artifacts_path}/

<contexto_interno>
{rag_context}
</contexto_interno>

INSTRUÇÕES: O contexto acima é APENAS para sua referência interna. NÃO repita, cite ou mostre esse contexto na sua resposta. Responda de forma natural e direta à pergunta do usuário, usando as informações do contexto quando relevante, mas sem expor a estrutura interna.

Pergunta do usuário: {chat_request.message}"""
            print(f"[RAG] Contexto encontrado: {len(rag_context)} chars")
        else:
            context_message = f"""[CONTEXTO DO SISTEMA - Session ID: {target_session_id}]
Ao criar arquivos, use EXATAMENTE este caminho: {artifacts_path}/
Exemplo: {artifacts_path}/meu_arquivo.txt

Pergunta do usuário: {chat_request.message}"""

        await c.query(context_message)

        response_text = ""

        async for message in c.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        print(f"[TOOL] {block.name} (id: {block.id})")
                        # Registrar tool call no AgentFS para auditoria
                        try:
                            tool_call_id = await afs.tools.start(
                                block.name, {"input": str(getattr(block, "input", {}))[:500]}
                            )
                            await afs.tools.success(tool_call_id, {"status": "completed"})
                        except Exception as tool_err:
                            print(f"[WARN] Erro ao registrar tool: {tool_err}")

        history = await afs.kv.get("conversation:history") or []
        history.append({"role": "user", "content": chat_request.message})
        history.append({"role": "assistant", "content": response_text})
        await afs.kv.set("conversation:history", history[-100:])

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        await afs.fs.write_file(
            f"/artifacts/chat_{timestamp}.json",
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

        artifacts_root = Path.cwd() / "artifacts"
        session_artifacts = artifacts_root / target_session_id
        session_artifacts.mkdir(parents=True, exist_ok=True)

        if artifacts_root.exists():
            for item in artifacts_root.iterdir():
                if item.is_file():
                    target = session_artifacts / item.name
                    shutil.move(str(item), str(target))

        for default_folder in ["default", "default_session"]:
            default_path = artifacts_root / default_folder
            if default_path.exists() and default_path.is_dir():
                for item in default_path.iterdir():
                    if item.is_file():
                        target = session_artifacts / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))

        tmp_artifacts = Path("/tmp/artifacts")
        if tmp_artifacts.exists():
            for item in tmp_artifacts.iterdir():
                # Validate filename to prevent path traversal attacks
                if item.is_file() and ".." not in item.name and "/" not in item.name:
                    target = session_artifacts / item.name
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
        raise HTTPException(status_code=500, detail="Chat processing failed") from e
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

    afs = None
    try:
        import app_state

        # Obter projeto do header (Angular envia X-Client-Project)
        project = request.headers.get("X-Client-Project", "default")

        # Determinar session_id a usar
        # IMPORTANTE: Guardamos referência ao client para NÃO criar sessões desnecessárias
        client_ref = None

        if chat_request.session_id:
            # Frontend enviou session_id específico - VALIDAR e MANTER essa sessão
            # Validar que é um UUID válido para prevenir criação de sessões indevidas
            validate_session_id(chat_request.session_id)
            target_session_id = chat_request.session_id
            # SÓ obter client se já existir, NÃO criar nova sessão
            if app_state.client is not None:
                client_ref = app_state.client
                print(f"[STREAM] Reutilizando client existente para sessão: {target_session_id}")
            else:
                # Criar client com resume se solicitado
                resume_id = target_session_id if chat_request.resume else None
                client_ref = await get_client(
                    model=chat_request.model,
                    project=project,
                    resume_session=resume_id,
                    fork_session=chat_request.fork_session,
                )
                # Se criou sessão diferente da solicitada, vamos limpar depois
                if app_state.current_session_id != target_session_id:
                    print(
                        f"[STREAM] Client criou sessão {app_state.current_session_id}, mas usando {target_session_id}"
                    )
        else:
            # Sem session_id do frontend - CRIAR NOVA SESSÃO
            # Isso acontece quando o usuário clica em "Novo bate-papo" e envia a primeira mensagem
            # IMPORTANTE: Forçar criação de nova sessão mesmo se já existir um client
            from app_state import reset_session

            await reset_session(project=project)
            client_ref = app_state.client
            target_session_id = app_state.current_session_id
            print(f"[STREAM] Nova sessão criada: {target_session_id} (projeto: {project})")

        # Abrir AgentFS para salvar histórico (não usar ClaudeRAG para evitar JSONL duplicado)
        afs = await AgentFS.open(AgentFSOptions(id=target_session_id))

        # =================================================================
        # VERIFICAR LIMITE DE GUEST (antes de processar o prompt)
        # =================================================================
        guest_manager = get_guest_limit_manager()
        guest_check = await guest_manager.check_limit(afs)

        if not guest_check.can_continue:
            # Usuário atingiu limite - retornar erro com detalhes para signup
            await afs.close()
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "SIGNUP_REQUIRED",
                    "message": guest_check.message,
                    "prompt_count": guest_check.prompt_count,
                    "action": guest_check.action.value,
                },
            )

        # =================================================================
        # CONFIGURAR SESSION_ID PARA VALIDAÇÃO DE PATH NOS HOOKS
        # =================================================================
        # Isso garante que tools Write/Edit só podem escrever em artifacts/{session_id}/
        set_current_session_id(target_session_id)

        # SEMPRE salvar projeto do header (Angular envia "chat-angular")
        # Isso garante que mesmo sessões criadas via /reset sejam marcadas corretamente
        try:
            current_project = await afs.kv.get("session:project")
            # Só atualiza se ainda não foi definido ou é diferente
            if not current_project or current_project != project:
                await afs.kv.set("session:project", project)
                print(f"[STREAM] Projeto definido: {project} (anterior: {current_project})")
        except Exception as e:
            print(f"[WARN] Erro ao salvar projeto: {e}")

        # Detectar comandos de gerenciamento de sessão
        command, extra_data = detect_session_command(chat_request.message)
        if command:
            print(f"[STREAM] Comando de sessão detectado: {command}, extra: {extra_data}")

            async def generate_command_response():
                nonlocal afs
                try:
                    # Executar o comando
                    response_text = await execute_session_command(
                        afs, target_session_id, command, extra_data
                    )

                    # Enviar session_id no primeiro chunk
                    yield f"data: {json.dumps({'session_id': target_session_id})}\n\n"

                    # Enviar resposta
                    yield f"data: {json.dumps({'text': response_text})}\n\n"

                    # Sinalizar que sessões precisam ser recarregadas
                    yield f"data: {json.dumps({'refresh_sessions': True, 'command': command})}\n\n"

                    # Salvar no histórico
                    try:
                        history = await afs.kv.get("conversation:history") or []
                        history.append({"role": "user", "content": chat_request.message})
                        history.append({"role": "assistant", "content": response_text})
                        await afs.kv.set("conversation:history", history[-100:])
                    except Exception as save_err:
                        print(f"[WARN] Erro ao salvar histórico: {save_err}")

                    # Salvar no JSONL
                    try:
                        append_to_jsonl(
                            session_id=target_session_id,
                            user_message=chat_request.message,
                            assistant_response=response_text,
                        )
                    except Exception as jsonl_err:
                        print(f"[WARN] Erro ao salvar JSONL: {jsonl_err}")

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    if afs:
                        await afs.close()

            return StreamingResponse(generate_command_response(), media_type="text/event-stream")

        async def generate():
            nonlocal afs, client_ref
            full_response = ""
            call_id = None
            try:
                from claude_agent_sdk import AssistantMessage, TextBlock

                # Registrar tool call para auditoria (mesmo comportamento do /chat)
                call_id = await afs.tools.start("chat", {"message": chat_request.message[:100]})

                # IMPORTANTE: Usar client_ref já obtido, NÃO chamar get_client() novamente
                # Isso evita criar sessões duplicadas
                client = client_ref

                # Buscar contexto RAG para enriquecer a resposta
                rag_context = await search_rag_context(chat_request.message)

                # Construir mensagem com contexto RAG (se disponível)
                artifacts_path = str(Path.cwd() / "artifacts" / target_session_id)

                if rag_context and not rag_context.startswith("[AVISO"):
                    # Incluir contexto RAG como instrução interna
                    query_message = f"""Ao criar arquivos, use: {artifacts_path}/

<base_conhecimento>
{rag_context}
</base_conhecimento>

IMPORTANTE: Use a base de conhecimento acima para responder, mas NÃO mostre, cite ou mencione que você está usando uma base de conhecimento. Responda naturalmente.

{chat_request.message}"""
                    print(f"[RAG] Contexto incluído: {len(rag_context)} chars")
                else:
                    query_message = chat_request.message

                await client.query(query_message)

                # Enviar session_id no primeiro chunk
                yield f"data: {json.dumps({'session_id': target_session_id})}\n\n"

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text = block.text
                                full_response += text
                                # Enviar em chunks menores para streaming mais suave
                                chunk_size = 50
                                for i in range(0, len(text), chunk_size):
                                    chunk = text[i : i + chunk_size]
                                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                                    await asyncio.sleep(0.001)  # Reduzido para menor latência
                            # Registrar tool calls no AgentFS para auditoria
                            elif hasattr(block, "name") and hasattr(block, "id"):
                                # É um ToolUseBlock
                                tool_name = block.name
                                tool_input = getattr(block, "input", {})
                                try:
                                    tool_call_id = await afs.tools.start(
                                        tool_name, {"input": str(tool_input)[:500]}
                                    )
                                    # Marcar como sucesso (não temos o resultado aqui)
                                    await afs.tools.success(tool_call_id, {"status": "completed"})
                                    print(f"[AUDIT] Tool registrada: {tool_name}")
                                except Exception as tool_err:
                                    print(f"[WARN] Erro ao registrar tool {tool_name}: {tool_err}")

                # Salvar histórico no AgentFS
                try:
                    history = await afs.kv.get("conversation:history") or []
                    history.append({"role": "user", "content": chat_request.message})
                    history.append({"role": "assistant", "content": full_response})
                    await afs.kv.set("conversation:history", history[-100:])
                    print(f"[STREAM] Histórico salvo: {len(history)} mensagens")

                    # Auto-rename: gerar título inteligente em BACKGROUND (não bloqueia resposta)
                    try:
                        existing_title = await afs.kv.get("session:title")
                        if not existing_title:
                            # Função async para rodar em background
                            async def generate_title_background(
                                session_id: str, user_msg: str, assistant_resp: str
                            ):
                                try:
                                    from agentfs_sdk import AgentFS, AgentFSOptions
                                    from agents.title_generator import get_smart_title

                                    auto_title = await get_smart_title(
                                        user_message=user_msg,
                                        assistant_response=assistant_resp,
                                    )
                                    # Abrir nova conexão para salvar (afs original já fechou)
                                    bg_afs = await AgentFS.open(AgentFSOptions(id=session_id))
                                    await bg_afs.kv.set("session:title", auto_title)
                                    await bg_afs.close()
                                    print(f"[BG] Auto-título: {auto_title}")
                                except Exception as e:
                                    print(f"[BG] Erro auto-título: {e}")

                            # Disparar em background - NÃO aguarda
                            asyncio.create_task(
                                generate_title_background(
                                    target_session_id, chat_request.message, full_response
                                )
                            )
                    except Exception as title_err:
                        print(f"[WARN] Erro ao configurar auto-título: {title_err}")
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

                # Marcar tool call como sucesso (para auditoria)
                if call_id:
                    await afs.tools.success(call_id, {"response_length": len(full_response)})

                # Incrementar contador de prompts (para guest limits)
                try:
                    guest_result = await guest_manager.check_and_increment(afs)
                    if guest_result.action == GuestLimitAction.SOFT_LIMIT:
                        # Enviar aviso ao frontend sobre limite próximo
                        yield f"data: {json.dumps({'guest_warning': guest_result.message, 'prompts_remaining': guest_result.prompts_remaining})}\n\n"
                except Exception as guest_err:
                    print(f"[WARN] Erro ao incrementar contador guest: {guest_err}")

                # Limpar sessão temporária se o client criou uma diferente da solicitada
                if (
                    chat_request.session_id
                    and app_state.current_session_id
                    and app_state.current_session_id != target_session_id
                ):
                    temp_session_id = app_state.current_session_id
                    from app_state import AGENTFS_DIR, SESSIONS_DIR

                    # Deletar arquivos da sessão temporária
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
                    print(f"[STREAM] Sessão temporária {temp_session_id} removida")

                yield "data: [DONE]\n\n"
            except Exception as e:
                # Marcar tool call como erro (para auditoria)
                if call_id:
                    try:
                        await afs.tools.error(call_id, str(e))
                    except Exception:
                        pass
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if afs:
                    await afs.close()

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        if afs:
            await afs.close()
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# V2 Endpoints - Using ChatAgent Abstraction
# =============================================================================


@router.post("/v2/chat/stream")
async def chat_stream_v2(
    chat_request: ChatStreamRequest,
    request: Request,
    _: None = Depends(verify_api_key),
):
    """Chat streaming endpoint V2 - Usa ChatAgent abstraction.

    Esta versão usa a abstração ChatAgent que encapsula:
    - Gerenciamento de sessões
    - Comandos de sessão (favoritar, renomear)
    - Integração com RAG
    - Streaming SSE
    - Auditoria
    - Persistência JSONL

    Exemplo de uso:
        POST /v2/chat/stream
        {
            "message": "olá",
            "session_id": "uuid-opcional",
            "model": "opus"
        }
    """
    from agents.chat_agent import ChatRequest as AgentChatRequest
    from agents.chat_agent import create_chat_agent

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    try:
        await RATE_LIMITS["chat"].check(client_ip)
    except Exception as rate_err:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {rate_err}",
        )

    # Obter projeto do header
    project = request.headers.get("X-Client-Project", "default")

    # Criar request para o agent
    agent_request = AgentChatRequest(
        message=chat_request.message,
        session_id=chat_request.session_id,
        model=chat_request.model or "opus",
        resume=chat_request.resume if chat_request.resume is not None else True,
        fork_session=chat_request.fork_session,
        project=project,
    )

    # Criar agent com função de busca RAG
    agent = create_chat_agent(rag_search_fn=search_rag_context)

    async def generate():
        async for chunk in agent.stream(agent_request):
            yield chunk.to_sse()

    return StreamingResponse(generate(), media_type="text/event-stream")
