"""Core module - shared state and helper functions."""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTFS_DIR = Path.cwd() / ".agentfs"

# Sessions directory (Claude Code uses cwd-based path)
SESSIONS_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs"
)

# Global client and AgentFS instances
client: Optional["ClaudeSDKClient"] = None
agentfs: Optional["AgentFS"] = None
current_session_id: Optional[str] = None
current_model: str = "haiku"  # Modelo atual: haiku, sonnet, opus


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


def extract_session_id_from_jsonl() -> Optional[str]:
    """Extrai session_id do arquivo JSONL mais recente."""
    if not SESSIONS_DIR.exists():
        return None

    jsonl_files = sorted(
        SESSIONS_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True
    )

    if not jsonl_files:
        return None

    latest_jsonl = jsonl_files[0]
    try:
        with open(latest_jsonl, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return data.get("sessionId", latest_jsonl.stem)
    except (OSError, IOError, json.JSONDecodeError):
        return latest_jsonl.stem  # Fallback to filename if parse fails

    return None


def _get_agent_model(model_name: str) -> tuple["AgentModel", str]:
    """Converte nome do modelo para AgentModel enum.

    Returns:
        Tuple of (AgentModel enum, normalized model name string)
    """
    from claude_rag_sdk import AgentModel

    model_map = {
        "haiku": AgentModel.HAIKU,
        "sonnet": AgentModel.SONNET,
        "opus": AgentModel.OPUS,
    }
    normalized = model_name.lower()
    if normalized in model_map:
        return model_map[normalized], normalized
    else:
        # Invalid model - fallback to haiku
        print(f"[WARN] Invalid model '{model_name}', using 'haiku' as fallback")
        return AgentModel.HAIKU, "haiku"


async def get_client(model: Optional[str] = None) -> "ClaudeSDKClient":
    """Get ClaudeSDKClient instance (manages sessions automatically).

    Args:
        model: Optional model name (haiku, sonnet, opus). If different from
               current model, recreates client with new model.
    """
    global client, agentfs, current_session_id, current_model

    requested_model = (model or current_model).lower()  # Normalize case

    # Verificar se modelo mudou - se sim, criar NOVA sessão
    if client is not None and requested_model != current_model.lower():
        print(f"[INFO] Modelo mudou: {current_model} -> {requested_model}, criando nova sessão...")
        # Don't try to close client - causes RuntimeError due to task scope issues
        client = None
        # Resetar sessão também - modelo diferente requer nova sessão
        if agentfs is not None:
            try:
                await agentfs.close()
            except Exception as e:
                print(f"[WARN] Error closing agentfs: {e}")
            agentfs = None
        current_session_id = None
        current_model = requested_model

    # Verificar se sessão atual ainda existe, se não, resetar
    if client is not None and current_session_id:
        session_db = AGENTFS_DIR / f"{current_session_id}.db"
        if not session_db.exists():
            print(f"[WARN] Sessão {current_session_id} foi deletada, resetando...")
            # Don't try to close client - causes RuntimeError due to task scope issues
            client = None
            if agentfs is not None:
                try:
                    await agentfs.close()
                except Exception as e:
                    print(f"[WARN] Error closing agentfs: {e}")
                agentfs = None
            current_session_id = None

    if client is None:
        from claude_agent_sdk import ClaudeSDKClient

        from claude_rag_sdk import ClaudeRAGOptions
        from claude_rag_sdk.agent import AgentEngine

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

        # Usar modelo solicitado
        agent_model, normalized_model = _get_agent_model(requested_model)
        current_model = normalized_model

        temp_options = ClaudeRAGOptions(
            id="temp", agent_model=agent_model, system_prompt=system_prompt
        )
        engine = AgentEngine(options=temp_options, mcp_server_path=None)
        client_options = engine._get_agent_options()

        client = ClaudeSDKClient(options=client_options)
        await client.__aenter__()

        await asyncio.sleep(0.3)

        # SEMPRE extrair novo session_id do client recém-criado
        # Não reutilizar sessões antigas pois podem ter outro modelo
        new_session_id = extract_session_id_from_jsonl()
        if not new_session_id:
            raise RuntimeError("Failed to extract session_id from ClaudeSDKClient")

        # Fechar agentfs antigo se existir
        if agentfs is not None:
            await agentfs.close()
            agentfs = None

        current_session_id = new_session_id

        from agentfs_sdk import AgentFS, AgentFSOptions

        agentfs = await AgentFS.open(AgentFSOptions(id=current_session_id))

        await agentfs.kv.set(
            "session:info",
            {
                "id": current_session_id,
                "model": current_model,
                "created_at": time.time(),
            },
        )

        await agentfs.fs.write_file(
            "/logs/session_start.txt",
            f"Session {current_session_id} | Model: {current_model} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        session_file = AGENTFS_DIR / "current_session"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(current_session_id)

        print(f"[INFO] Nova sessão: {current_session_id} | Modelo: {current_model}")

    return client


async def get_agentfs() -> "AgentFS":
    """Get AgentFS instance."""
    global agentfs
    if agentfs is None:
        await get_client()
    return agentfs


async def reset_session():
    """Reset session (creates new ClaudeSDKClient + AgentFS).

    Note: We don't try to close the old client because it causes issues
    when called from a different asyncio task. The garbage collector
    will handle cleanup.
    """
    global client, agentfs, current_session_id

    old_session = current_session_id

    # Don't try to close client - causes RuntimeError due to task scope issues
    # Just set to None and let GC handle cleanup
    client = None

    if agentfs is not None:
        try:
            await agentfs.close()
        except Exception as e:
            print(f"[WARN] Error closing agentfs: {e}")
        agentfs = None

    current_session_id = None

    await get_client()
    print(f"[INFO] Session reset: {old_session} -> {current_session_id}")


def get_current_session_id() -> Optional[str]:
    """Get current session ID."""
    global current_session_id

    if current_session_id:
        return current_session_id

    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        try:
            return session_file.read_text().strip()
        except (OSError, IOError):
            pass  # File read failed, return None

    return None


async def cleanup():
    """Cleanup resources on shutdown.

    Note: We don't try to close the client because it causes issues
    when called from a different asyncio task during shutdown.
    """
    global client, agentfs
    # Don't try to close client - causes RuntimeError due to task scope issues
    if client is not None:
        client = None
        print("[INFO] ClaudeSDKClient released (GC will cleanup)")
    if agentfs is not None:
        try:
            await agentfs.close()
            print("[INFO] AgentFS closed!")
        except Exception as e:
            print(f"[WARN] Error closing agentfs: {e}")
