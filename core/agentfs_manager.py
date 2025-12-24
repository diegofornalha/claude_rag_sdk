"""
=============================================================================
AGENTFS MANAGER - Gerenciador Singleton de AgentFS para RAG Agent
=============================================================================
Gerencia lifecycle do AgentFS, garantindo uma instância por sessão
=============================================================================
"""

import sys
from typing import Optional
from pathlib import Path

# Adicionar caminho do SDK do AgentFS
agentfs_sdk_path = Path(__file__).parent.parent / "agentfs" / "sdk" / "python"
sys.path.insert(0, str(agentfs_sdk_path))

from agentfs_sdk import AgentFS, AgentFSOptions
from .logger import logger, get_session_id

# Instância global
_agentfs: Optional[AgentFS] = None
_current_session_id: Optional[str] = None


async def init_agentfs(session_id: str) -> AgentFS:
    """
    Inicializa AgentFS para a sessão atual.

    Args:
        session_id: ID único da sessão

    Returns:
        Instância do AgentFS

    Raises:
        RuntimeError: Se não conseguir inicializar
    """
    global _agentfs, _current_session_id

    # Se já existe e é a mesma sessão, retorna
    if _agentfs and _current_session_id == session_id:
        logger.info("AgentFS já inicializado para esta sessão", session_id=session_id)
        return _agentfs

    # Se existe mas é sessão diferente, fecha anterior
    if _agentfs:
        logger.info("Fechando AgentFS da sessão anterior", old_session=_current_session_id, new_session=session_id)
        await close_agentfs()

    # Cria diretório .agentfs se não existe
    agentfs_dir = Path.home() / ".claude" / ".agentfs"
    agentfs_dir.mkdir(parents=True, exist_ok=True)

    # Abre AgentFS para a sessão
    db_path = agentfs_dir / f"{session_id}.db"

    try:
        _agentfs = await AgentFS.open(AgentFSOptions(
            id=session_id,
            path=str(db_path)
        ))
        _current_session_id = session_id

        logger.info(
            "AgentFS inicializado com sucesso",
            session_id=session_id,
            db_path=str(db_path),
            exists=db_path.exists()
        )

        return _agentfs

    except Exception as e:
        logger.error(f"Erro ao inicializar AgentFS: {e}", session_id=session_id, error=str(e))
        raise RuntimeError(f"Falha ao inicializar AgentFS: {e}")


def get_agentfs() -> AgentFS:
    """
    Retorna instância AgentFS atual.

    Returns:
        Instância do AgentFS

    Raises:
        RuntimeError: Se AgentFS não foi inicializado
    """
    if _agentfs is None:
        raise RuntimeError(
            "AgentFS não inicializado. "
            "Chame init_agentfs(session_id) primeiro."
        )
    return _agentfs


async def close_agentfs() -> None:
    """
    Fecha AgentFS e libera recursos.

    Este método é idempotente - pode ser chamado múltiplas vezes.
    """
    global _agentfs, _current_session_id

    if _agentfs:
        session_id = _current_session_id
        try:
            await _agentfs.close()
            logger.info("AgentFS fechado", session_id=session_id)
        except Exception as e:
            logger.error(f"Erro ao fechar AgentFS: {e}", session_id=session_id, error=str(e))
        finally:
            _agentfs = None
            _current_session_id = None
    else:
        logger.debug("AgentFS já estava fechado ou nunca foi inicializado")


def is_initialized() -> bool:
    """Verifica se AgentFS está inicializado."""
    return _agentfs is not None


def get_current_session_id() -> Optional[str]:
    """Retorna o session_id atual (se houver)."""
    return _current_session_id


async def ensure_agentfs() -> AgentFS:
    """
    Garante que AgentFS está inicializado, inicializando se necessário.

    Lê o session_id do arquivo compartilhado se não estiver inicializado.

    Returns:
        Instância do AgentFS

    Raises:
        RuntimeError: Se não conseguir inicializar
    """
    global _agentfs, _current_session_id

    # Se já está inicializado, retorna
    if _agentfs is not None:
        return _agentfs

    # Tenta ler session_id do arquivo compartilhado
    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"

    if session_file.exists():
        session_id = session_file.read_text().strip()
        if session_id:
            logger.info("Inicializando AgentFS a partir do arquivo de sessão", session_id=session_id)
            return await init_agentfs(session_id)

    raise RuntimeError(
        "AgentFS não inicializado e não foi possível determinar session_id. "
        "Certifique-se de que o servidor iniciou uma sessão primeiro."
    )
