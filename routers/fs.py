"""Filesystem and KV endpoints."""
from fastapi import APIRouter, HTTPException

from app_state import get_agentfs
import app_state

router = APIRouter(tags=["FileSystem"])


@router.get("/fs/tree")
async def get_filesystem_tree(path: str = "/"):
    """Get filesystem tree from AgentFS."""
    afs = await get_agentfs()

    try:
        async def list_tree(dir_path: str, depth: int = 0) -> list:
            if depth > 3:
                return []
            items = []
            try:
                entries = await afs.fs.readdir(dir_path)
                for entry in entries:
                    full_path = f"{dir_path}/{entry}".replace("//", "/")
                    item = {"name": entry, "path": full_path}
                    try:
                        children = await list_tree(full_path, depth + 1)
                        if children:
                            item["children"] = children
                            item["type"] = "directory"
                        else:
                            item["type"] = "file"
                    except (OSError, IOError, PermissionError):
                        item["type"] = "file"  # Fallback if can't list children
                    items.append(item)
            except (OSError, IOError, PermissionError):
                pass  # Skip inaccessible directories
            return items

        tree = await list_tree(path)
        return {
            "path": path,
            "tree": tree,
            "session_id": app_state.current_session_id,
        }
    except Exception as e:
        print(f"[ERROR] Failed to read tree: {e}")
        return {"error": "Failed to read filesystem tree"}


@router.get("/kv/list")
async def list_kv(prefix: str = ""):
    """List KV store keys."""
    afs = await get_agentfs()

    try:
        keys = await afs.kv.list(prefix=prefix if prefix else None)
        return {"keys": keys, "count": len(keys)}
    except Exception as e:
        print(f"[ERROR] Failed to list KV keys: {e}")
        return {"keys": [], "error": "Failed to list keys"}


@router.get("/kv/{key}")
async def get_kv(key: str):
    """Get KV value."""
    afs = await get_agentfs()

    try:
        value = await afs.kv.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"key": key, "value": value}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
