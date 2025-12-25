"""Core module - shared state and helper functions."""
from pathlib import Path
from typing import Optional
import asyncio
import json
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTFS_DIR = Path.cwd() / ".agentfs"

# Sessions directory (Claude Code uses cwd-based path)
SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-Users-2a--claude-hello-agent-chat-simples-backend-outputs"

# Global client and AgentFS instances
client: Optional["ClaudeSDKClient"] = None
agentfs: Optional["AgentFS"] = None
current_session_id: Optional[str] = None


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
            print(f"[WARN] Sessão {current_session_id} foi deletada, resetando...")
            if client is not None:
                await client.__aexit__(None, None, None)
                client = None
            if agentfs is not None:
                await agentfs.close()
                agentfs = None
            current_session_id = None

    if client is None:
        from claude_rag_sdk.agent import AgentEngine
        from claude_rag_sdk import ClaudeRAGOptions, AgentModel
        from claude_agent_sdk import ClaudeSDKClient

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

        client = ClaudeSDKClient(options=client_options)
        await client.__aenter__()

        await asyncio.sleep(0.2)

        current_session_id = extract_session_id_from_jsonl()
        if not current_session_id:
            raise RuntimeError("Failed to extract session_id from ClaudeSDKClient")

        from agentfs_sdk import AgentFS, AgentFSOptions
        agentfs = await AgentFS.open(AgentFSOptions(id=current_session_id))

        await agentfs.kv.set("session:info", {
            "id": current_session_id,
            "created_at": time.time(),
        })

        await agentfs.fs.write_file(
            f"/logs/session_start.txt",
            f"Session {current_session_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        session_file = AGENTFS_DIR / "current_session"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(current_session_id)

        print(f"[INFO] Session created: {current_session_id}")

    return client


async def get_agentfs() -> "AgentFS":
    """Get AgentFS instance."""
    global agentfs
    if agentfs is None:
        await get_client()
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
        except:
            pass

    return None


async def cleanup():
    """Cleanup resources on shutdown."""
    global client, agentfs
    if client is not None:
        await client.__aexit__(None, None, None)
        print("[INFO] ClaudeSDKClient closed!")
    if agentfs is not None:
        await agentfs.close()
        print("[INFO] AgentFS closed!")
