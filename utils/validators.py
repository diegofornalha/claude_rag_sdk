"""Input validators for security."""
import re
from fastapi import HTTPException


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


def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks.

    Returns True if valid, raises HTTPException if invalid.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Prevent path traversal
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")

    # Prevent absolute paths
    if ':' in filename:  # Windows drive letters
        raise HTTPException(status_code=400, detail="Invalid filename: absolute path not allowed")

    return True


def validate_directory_path(directory: str, allowed_prefixes: list[str] = None) -> bool:
    """Validate directory path to prevent traversal.

    Args:
        directory: Directory path to validate
        allowed_prefixes: List of allowed directory prefixes (e.g., ['/outputs', '/logs'])

    Returns True if valid, raises HTTPException if invalid.
    """
    if not directory:
        raise HTTPException(status_code=400, detail="Directory cannot be empty")

    # Prevent path traversal
    if '..' in directory:
        raise HTTPException(status_code=400, detail="Invalid directory: path traversal detected")

    # Validate against whitelist if provided
    if allowed_prefixes:
        if not any(directory.startswith(prefix) for prefix in allowed_prefixes):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid directory: must start with {allowed_prefixes}"
            )

    return True
