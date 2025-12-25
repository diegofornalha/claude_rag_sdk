"""Outputs endpoints."""
from fastapi import APIRouter, HTTPException
from pathlib import Path

from app_state import get_agentfs
import app_state
from utils.validators import validate_session_id, validate_filename, validate_directory_path

router = APIRouter(prefix="/outputs", tags=["Outputs"])


@router.get("")
async def list_outputs(directory: str = "outputs", session_id: str = None):
    """List output files from physical filesystem."""
    try:
        # Validate inputs
        validate_directory_path(directory, allowed_prefixes=['outputs', '/outputs'])
        if session_id:
            validate_session_id(session_id)
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
        print(f"[ERROR] Failed to list outputs: {e}")
        return {"files": [], "error": "Failed to list outputs", "session_id": session_id}


@router.get("/file/{filename:path}")
async def get_output_file(filename: str):
    """Get output file content from AgentFS filesystem."""
    # Validate filename to prevent path traversal
    validate_filename(filename)

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
        print(f"[ERROR] Failed to read file {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")


@router.post("/write")
async def write_output_file(filename: str, content: str, directory: str = "/outputs"):
    """Write file to AgentFS filesystem."""
    # Validate inputs to prevent path traversal
    validate_filename(filename)
    validate_directory_path(directory, allowed_prefixes=['/outputs', '/logs'])

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
        raise HTTPException(status_code=500, detail="Failed to write file")


@router.delete("/{filename:path}")
async def delete_output(filename: str):
    """Delete file from AgentFS filesystem."""
    # Validate filename to prevent path traversal
    validate_filename(filename)

    try:
        afs = await get_agentfs()
        filepath = f"/outputs/{filename}" if not filename.startswith("/") else filename
        await afs.fs.unlink(filepath)
        return {"success": True, "deleted": filepath}
    except Exception as e:
        print(f"[ERROR] Failed to delete file {filename}: {e}")
        return {"success": False, "error": "Failed to delete file"}
