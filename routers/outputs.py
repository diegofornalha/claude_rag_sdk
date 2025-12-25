"""Outputs endpoints."""
from fastapi import APIRouter, HTTPException
from pathlib import Path

from app_state import get_agentfs
import app_state

router = APIRouter(prefix="/outputs", tags=["Outputs"])


@router.get("")
async def list_outputs(directory: str = "outputs", session_id: str = None):
    """List output files from physical filesystem."""
    try:
        if session_id:
            outputs_dir = Path.cwd() / directory / session_id
        else:
            outputs_dir = Path.cwd() / directory

        if not outputs_dir.exists():
            return {"files": [], "directory": str(outputs_dir), "count": 0, "session_id": session_id}

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
        return {
            "files": files,
            "directory": str(outputs_dir),
            "count": len(files),
            "session_id": session_id,
        }
    except Exception as e:
        return {"files": [], "error": str(e), "session_id": session_id}


@router.get("/file/{filename:path}")
async def get_output_file(filename: str):
    """Get output file content from AgentFS filesystem."""
    afs = await get_agentfs()

    try:
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        content = await afs.fs.read_file(filepath)
        return {
            "filename": filename,
            "content": content,
            "session_id": app_state.current_session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")


@router.post("/write")
async def write_output_file(filename: str, content: str, directory: str = "/outputs"):
    """Write file to AgentFS filesystem."""
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
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename:path}")
async def delete_output(filename: str):
    """Delete file from AgentFS filesystem."""
    try:
        afs = await get_agentfs()
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        await afs.fs.unlink(filepath)
        return {"success": True, "deleted": filepath}
    except Exception as e:
        return {"success": False, "error": str(e)}
