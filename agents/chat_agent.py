"""ChatAgent - Abstração de alto nível para chat usando Claude Agent SDK.

Este módulo encapsula toda a lógica de chat:
- Gerenciamento de sessões
- Streaming SSE
- Comandos de sessão (favoritar, renomear)
- Integração com RAG
- Auditoria de tool calls
- Persistência em JSONL
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from agentfs_sdk import AgentFS, AgentFSOptions

from app_state import SESSIONS_DIR, get_client, reset_session
from claude_rag_sdk.core.logger import get_logger
from claude_rag_sdk.core.sdk_hooks import set_current_session_id
from utils.validators import validate_session_id
from agents.metrics import get_metrics_manager, estimate_tokens

logger = get_logger("chat_agent")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StreamChunk:
    """Representa um chunk de streaming."""

    text: str | None = None
    session_id: str | None = None
    error: str | None = None
    done: bool = False
    refresh_sessions: bool = False
    command: str | None = None
    tool_call: dict | None = None
    metrics: dict | None = None  # Métricas de tokens/custo

    def to_sse(self) -> str:
        """Converte para formato SSE."""
        if self.done:
            return "data: [DONE]\n\n"

        data = {}
        if self.text is not None:
            data["text"] = self.text
        if self.session_id is not None:
            data["session_id"] = self.session_id
        if self.error is not None:
            data["error"] = self.error
        if self.refresh_sessions:
            data["refresh_sessions"] = True
            if self.command:
                data["command"] = self.command
        if self.tool_call:
            data["tool_call"] = self.tool_call
        if self.metrics:
            data["metrics"] = self.metrics

        return f"data: {json.dumps(data)}\n\n"


@dataclass
class ChatRequest:
    """Request para chat."""

    message: str
    session_id: str | None = None
    model: str = "opus"
    resume: bool = True
    fork_session: str | None = None
    project: str = "default"


@dataclass
class ChatContext:
    """Contexto interno durante processamento do chat."""

    session_id: str
    afs: AgentFS
    client: object  # ClaudeSDKClient
    project: str
    rag_context: str | None = None


# =============================================================================
# Session Commands
# =============================================================================

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
    """Detecta comandos de gerenciamento de sessão na mensagem."""
    msg_lower = message.lower().strip()

    # Verificar comandos de DESFAVORITAR ANTES de favoritar
    for pattern in SESSION_COMMANDS["unfavorite"]:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("unfavorite", None)

    for pattern in SESSION_COMMANDS["favorite"]:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("favorite", None)

    for pattern in SESSION_COMMANDS["rename"]:
        match = re.search(pattern, msg_lower, re.IGNORECASE)
        if match:
            new_name = match.group(1).strip() if match.lastindex else None
            if new_name:
                new_name = new_name.strip("'\"").strip()
                if len(new_name) > 0:
                    return ("rename", new_name[:100])

    return (None, None)


async def execute_session_command(
    afs: AgentFS, session_id: str, command: str, extra_data: str | None
) -> str:
    """Executa um comando de gerenciamento de sessão."""
    try:
        if command == "favorite":
            await afs.kv.set("session:favorite", True)
            return "Chat adicionado aos favoritos!"

        elif command == "unfavorite":
            await afs.kv.set("session:favorite", False)
            return "Chat removido dos favoritos."

        elif command == "rename" and extra_data:
            await afs.kv.set("session:title", extra_data)
            return f"Chat renomeado para: **{extra_data}**"

        return "Comando nao reconhecido."

    except Exception as e:
        logger.error("Erro ao executar comando de sessao", error=str(e))
        return f"Erro ao executar comando: {str(e)}"


# =============================================================================
# JSONL Persistence
# =============================================================================


def append_to_jsonl(
    session_id: str, user_message: str, assistant_response: str, parent_uuid: str | None = None
):
    """Salva mensagens no arquivo JSONL da sessão."""
    jsonl_file = SESSIONS_DIR / f"{session_id}.jsonl"

    if not jsonl_file.exists():
        logger.info("Criando arquivo JSONL para nova sessao", session_id=session_id)
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file.touch()

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    user_uuid = str(uuid.uuid4())
    assistant_uuid = str(uuid.uuid4())

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
            f.write(json.dumps(user_entry) + "\n")
            f.write(json.dumps(assistant_entry) + "\n")
        logger.debug("Mensagens salvas no JSONL", session_id=session_id)
    except Exception as e:
        logger.error("Falha ao salvar JSONL", session_id=session_id, error=str(e))


# =============================================================================
# ChatAgent - Main Class
# =============================================================================


class ChatAgent:
    """Agente de chat que encapsula toda a lógica de interação com Claude.

    Uso:
        agent = ChatAgent()
        async for chunk in agent.stream(request):
            yield chunk.to_sse()

    Funcionalidades:
        - Gerenciamento automático de sessões
        - Detecção e execução de comandos (favoritar, renomear)
        - Integração com RAG para contexto
        - Streaming SSE
        - Auditoria de tool calls
        - Persistência em JSONL
    """

    def __init__(self, rag_search_fn=None):
        """
        Args:
            rag_search_fn: Função async para buscar contexto RAG.
                          Assinatura: async def search(query: str) -> str | None
        """
        self.rag_search_fn = rag_search_fn

    async def stream(self, request: ChatRequest) -> AsyncGenerator[StreamChunk, None]:
        """Processa mensagem e retorna stream de chunks.

        Args:
            request: ChatRequest com mensagem e configurações

        Yields:
            StreamChunk com texto, session_id, erros, etc.
        """
        import app_state

        afs = None
        try:
            # Validar session_id se fornecido
            if request.session_id:
                validate_session_id(request.session_id)

            # Resolver sessão e obter client
            ctx = await self._resolve_session(request)
            afs = ctx.afs

            # Configurar session_id para hooks de validação de path
            set_current_session_id(ctx.session_id)

            # Salvar projeto na sessão
            await self._save_project(ctx)

            # Detectar comandos de sessão
            command, extra_data = detect_session_command(request.message)
            if command:
                async for chunk in self._handle_command(ctx, command, extra_data, request.message):
                    yield chunk
                return

            # Processar chat normal
            async for chunk in self._process_chat(ctx, request):
                yield chunk

        except Exception as e:
            logger.error("Erro no ChatAgent.stream", error=str(e))
            yield StreamChunk(error=str(e))
        finally:
            if afs:
                await afs.close()

    async def _resolve_session(self, request: ChatRequest) -> ChatContext:
        """Resolve sessão e obtém client."""
        import app_state

        client_ref = None

        if request.session_id:
            # Sessão existente
            target_session_id = request.session_id

            if app_state.client is not None:
                client_ref = app_state.client
                logger.debug(f"Reutilizando client para sessao: {target_session_id}")
            else:
                resume_id = target_session_id if request.resume else None
                client_ref = await get_client(
                    model=request.model,
                    project=request.project,
                    resume_session=resume_id,
                    fork_session=request.fork_session,
                )
        else:
            # Nova sessão
            await reset_session(project=request.project)
            client_ref = app_state.client
            target_session_id = app_state.current_session_id
            logger.info(f"Nova sessao criada: {target_session_id}")

        # Abrir AgentFS
        afs = await AgentFS.open(AgentFSOptions(id=target_session_id))

        return ChatContext(
            session_id=target_session_id,
            afs=afs,
            client=client_ref,
            project=request.project,
        )

    async def _save_project(self, ctx: ChatContext):
        """Salva projeto na sessão."""
        try:
            current_project = await ctx.afs.kv.get("session:project")
            if not current_project or current_project != ctx.project:
                await ctx.afs.kv.set("session:project", ctx.project)
        except Exception as e:
            logger.warning(f"Erro ao salvar projeto: {e}")

    async def _handle_command(
        self, ctx: ChatContext, command: str, extra_data: str | None, message: str
    ) -> AsyncGenerator[StreamChunk, None]:
        """Processa comando de sessão."""
        logger.info(f"Comando de sessao detectado: {command}")

        # Executar comando
        response_text = await execute_session_command(ctx.afs, ctx.session_id, command, extra_data)

        # Enviar session_id
        yield StreamChunk(session_id=ctx.session_id)

        # Enviar resposta
        yield StreamChunk(text=response_text)

        # Sinalizar refresh
        yield StreamChunk(refresh_sessions=True, command=command)

        # Salvar no histórico
        try:
            history = await ctx.afs.kv.get("conversation:history") or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_text})
            await ctx.afs.kv.set("conversation:history", history[-100:])
        except Exception as e:
            logger.warning(f"Erro ao salvar historico: {e}")

        # Salvar no JSONL
        append_to_jsonl(ctx.session_id, message, response_text)

        yield StreamChunk(done=True)

    async def _process_chat(
        self, ctx: ChatContext, request: ChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Processa chat normal com streaming."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        full_response = ""
        call_id = None
        tool_call_count = 0

        # Iniciar métricas
        metrics_manager = get_metrics_manager()
        request_metrics = metrics_manager.start_request(
            request_id=str(uuid.uuid4()),
            session_id=ctx.session_id,
            model=request.model,
        )

        try:
            # Registrar tool call para auditoria
            call_id = await ctx.afs.tools.start("chat", {"message": request.message[:100]})

            # Buscar contexto RAG
            rag_context = None
            if self.rag_search_fn:
                try:
                    rag_context = await self.rag_search_fn(request.message)
                except Exception as e:
                    logger.warning(f"Erro ao buscar RAG: {e}")

            # Construir mensagem com contexto
            artifacts_path = str(Path.cwd() / "artifacts" / ctx.session_id)
            query_message = self._build_query_message(request.message, rag_context, artifacts_path)

            # Enviar query
            await ctx.client.query(query_message)

            # Enviar session_id primeiro
            yield StreamChunk(session_id=ctx.session_id)

            # Stream da resposta
            async for message in ctx.client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text
                            full_response += text

                            # Streaming em chunks menores
                            chunk_size = 50
                            for i in range(0, len(text), chunk_size):
                                chunk = text[i : i + chunk_size]
                                yield StreamChunk(text=chunk)
                                await asyncio.sleep(0.01)

                        # Registrar tool calls
                        elif hasattr(block, "name") and hasattr(block, "id"):
                            tool_name = block.name
                            tool_input = getattr(block, "input", {})
                            tool_call_count += 1
                            try:
                                await ctx.afs.tools.start(tool_name, {"input": str(tool_input)[:500]})
                            except Exception:
                                pass

            # Finalizar auditoria
            if call_id:
                await ctx.afs.tools.finish(call_id, {"response_length": len(full_response)})

            # Calcular e finalizar métricas
            input_tokens = estimate_tokens(request.message)
            output_tokens = estimate_tokens(full_response)
            metrics_manager.finish_request(
                request_metrics,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_call_count,
            )

            # Persistir métricas no AgentFS
            try:
                await metrics_manager.persist_to_agentfs(ctx.afs, ctx.session_id)
            except Exception as e:
                logger.warning(f"Erro ao persistir metricas: {e}")

            # Salvar no histórico
            try:
                history = await ctx.afs.kv.get("conversation:history") or []
                history.append({"role": "user", "content": request.message})
                history.append({"role": "assistant", "content": full_response})
                await ctx.afs.kv.set("conversation:history", history[-100:])
            except Exception as e:
                logger.warning(f"Erro ao salvar historico: {e}")

            # Salvar no JSONL
            append_to_jsonl(ctx.session_id, request.message, full_response)

            # Enviar métricas antes do done
            yield StreamChunk(metrics=request_metrics.to_dict())
            yield StreamChunk(done=True)

        except Exception as e:
            logger.error("Erro no processamento do chat", error=str(e))
            # Finalizar métricas com erro
            metrics_manager.finish_request(request_metrics, error=str(e))
            if call_id:
                try:
                    await ctx.afs.tools.finish(call_id, {"error": str(e)})
                except Exception:
                    pass
            yield StreamChunk(error=str(e))

    def _build_query_message(
        self, message: str, rag_context: str | None, artifacts_path: str
    ) -> str:
        """Constrói mensagem com contexto RAG."""
        if rag_context and not rag_context.startswith("[AVISO"):
            return f"""Ao criar arquivos, use: {artifacts_path}/

<base_conhecimento>
{rag_context}
</base_conhecimento>

IMPORTANTE: Use a base de conhecimento acima para responder, mas NAO mostre, cite ou mencione que voce esta usando uma base de conhecimento. Responda naturalmente.

{message}"""
        return message


# =============================================================================
# Factory Function
# =============================================================================


def create_chat_agent(rag_search_fn=None) -> ChatAgent:
    """Cria instância do ChatAgent.

    Args:
        rag_search_fn: Função opcional para buscar contexto RAG

    Returns:
        ChatAgent configurado
    """
    return ChatAgent(rag_search_fn=rag_search_fn)
