"""Artifacts endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

import app_state
from app_state import get_agentfs
from claude_rag_sdk.core.rate_limiter import RATE_LIMITS, limiter
from utils.validators import validate_directory_path, validate_filename, validate_session_id

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


@router.get("")
@limiter.limit(RATE_LIMITS.get("default", "60/minute"))
async def list_artifacts(request: Request, directory: str = "artifacts", session_id: str = None):
    """List artifact files from physical filesystem.

    When session_id is provided, lists files from artifacts/{session_id}/.
    When session_id is not provided, lists ALL files from ALL sessions recursively.
    """
    try:
        # Validate inputs
        validate_directory_path(directory, allowed_prefixes=["artifacts", "/artifacts"])
        if session_id:
            validate_session_id(session_id)
            artifacts_dir = Path.cwd() / directory / session_id
        else:
            artifacts_dir = Path.cwd() / directory

        if not artifacts_dir.exists():
            return {
                "files": [],
                "directory": str(artifacts_dir),
                "count": 0,
                "session_id": session_id,
            }

        files = []

        if session_id:
            # List files from specific session directory
            for file in artifacts_dir.iterdir():
                if file.is_file():
                    stat = file.stat()
                    files.append(
                        {
                            "name": file.name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime * 1000,
                            "session_id": session_id,
                            "path": f"{session_id}/{file.name}",
                        }
                    )
        else:
            # List ALL files from ALL session subdirectories recursively
            for session_dir in artifacts_dir.iterdir():
                if session_dir.is_dir():
                    sid = session_dir.name
                    for file in session_dir.iterdir():
                        if file.is_file():
                            stat = file.stat()
                            files.append(
                                {
                                    "name": file.name,
                                    "size": stat.st_size,
                                    "modified": stat.st_mtime * 1000,
                                    "session_id": sid,
                                    "path": f"{sid}/{file.name}",
                                }
                            )
                elif session_dir.is_file():
                    # Also include files in root artifacts/ directory
                    stat = session_dir.stat()
                    files.append(
                        {
                            "name": session_dir.name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime * 1000,
                            "session_id": None,
                            "path": session_dir.name,
                        }
                    )

        files.sort(key=lambda f: f["modified"], reverse=True)
        return {
            "files": files,
            "directory": str(artifacts_dir),
            "count": len(files),
            "session_id": session_id,
        }
    except Exception as e:
        print(f"[ERROR] Failed to list artifacts: {e}")
        return {
            "files": [],
            "error": "Failed to list artifacts",
            "session_id": session_id,
        }


@router.get("/file/{filename:path}")
@limiter.limit(RATE_LIMITS.get("default", "60/minute"))
async def get_artifact_file(request: Request, filename: str):
    """Get artifact file content from AgentFS filesystem."""
    # Validate filename to prevent path traversal
    validate_filename(filename)

    afs = await get_agentfs()

    try:
        filepath = f"/artifacts/{filename}" if not filename.startswith("/") else filename
        content = await afs.fs.read_file(filepath)
        return {
            "filename": filename,
            "content": content,
            "session_id": app_state.current_session_id,
        }
    except Exception as e:
        print(f"[ERROR] Failed to read file {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found") from e


@router.post("/write")
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
async def write_artifact_file(
    request: Request, filename: str, content: str, directory: str = "/artifacts"
):
    """Write file to AgentFS filesystem."""
    # Validate inputs to prevent path traversal
    validate_filename(filename)
    validate_directory_path(directory, allowed_prefixes=["/artifacts", "/logs"])

    afs = await get_agentfs()

    try:
        filepath = f"{directory}/{filename}"
        await afs.fs.write_file(filepath, content)
        return {
            "success": True,
            "filepath": filepath,
            "session_id": app_state.current_session_id,
        }
    except Exception as e:
        print(f"[ERROR] Failed to write file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to write file") from e


@router.delete("/{filename:path}")
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
async def delete_artifact(request: Request, filename: str):
    """Delete file from physical filesystem (artifacts/ directory)."""
    # Validate filename to prevent path traversal
    validate_filename(filename)

    try:
        # Deletar do filesystem físico (artifacts/{session_id}/{filename})
        artifacts_dir = Path.cwd() / "artifacts"
        filepath = artifacts_dir / filename

        # Verificar se o caminho está dentro de artifacts/ (segurança)
        if not filepath.resolve().is_relative_to(artifacts_dir.resolve()):
            print(f"[SECURITY] Tentativa de path traversal: {filename}")
            return {"success": False, "error": "Invalid path"}

        if filepath.exists() and filepath.is_file():
            # Guardar pasta pai para verificar se ficou vazia
            parent_dir = filepath.parent

            # Deletar arquivo
            filepath.unlink()
            print(f"[DELETE] Arquivo deletado: {filepath}")

            # Verificar se pasta ficou vazia e deletar se for o caso
            deleted_dir = None
            if parent_dir.exists() and parent_dir != artifacts_dir:
                remaining_files = list(parent_dir.iterdir())
                if len(remaining_files) == 0:
                    parent_dir.rmdir()
                    deleted_dir = str(parent_dir)
                    print(f"[DELETE] Pasta vazia removida: {parent_dir}")

            return {"success": True, "deleted": str(filepath), "deleted_dir": deleted_dir}
        else:
            print(f"[WARN] Arquivo não encontrado: {filepath}")
            return {"success": False, "error": "File not found"}
    except Exception as e:
        print(f"[ERROR] Failed to delete file {filename}: {e}")
        return {"success": False, "error": f"Failed to delete file: {e}"}
