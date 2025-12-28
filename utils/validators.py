"""Input validators for security."""

import re
from pathlib import Path
from typing import Optional

from claude_rag_sdk.core.exceptions import (
    InvalidInputError,
    InvalidSessionIdError,
    PathTraversalError,
    ValidationError,
)


def validate_session_id(session_id: str) -> bool:
    """Validate session_id to prevent path traversal attacks.

    Returns True if valid, raises InvalidSessionIdError if invalid.
    """
    if not session_id:
        return False
    # Only allow UUID-like patterns and alphanumeric with hyphens
    if not re.match(r"^[a-zA-Z0-9\-_]+$", session_id):
        raise InvalidSessionIdError(
            message="Formato de session_id inválido",
            details={"session_id": session_id[:20]},
        )
    # Prevent path traversal
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise PathTraversalError(
            message="Tentativa de path traversal detectada no session_id",
            details={"session_id": session_id[:20]},
        )
    return True


def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal attacks.

    Returns True if valid, raises ValidationError if invalid.
    """
    if not filename:
        raise InvalidInputError(message="Nome do arquivo não pode ser vazio")

    # Prevent path traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise PathTraversalError(
            message="Tentativa de path traversal detectada no nome do arquivo",
            details={"filename": filename[:50]},
        )

    # Prevent absolute paths
    if ":" in filename:  # Windows drive letters
        raise PathTraversalError(
            message="Caminho absoluto não permitido",
            details={"filename": filename[:50]},
        )

    return True


def validate_directory_path(
    directory: str,
    allowed_prefixes: Optional[list[str]] = None,
    base_path: Optional[Path] = None,
) -> bool:
    """Validate directory path to prevent traversal.

    Args:
        directory: Directory path to validate
        allowed_prefixes: List of allowed directory prefixes (e.g., ['/artifacts', '/logs'])
        base_path: Base path to resolve against (for additional security)

    Returns True if valid, raises ValidationError if invalid.
    """
    if not directory:
        raise InvalidInputError(message="Diretório não pode ser vazio")

    # Prevent path traversal characters
    if ".." in directory:
        raise PathTraversalError(
            message="Tentativa de path traversal detectada no diretório",
            details={"directory": directory[:100]},
        )

    # Use Path.resolve() for additional security
    if base_path:
        try:
            resolved = (base_path / directory).resolve()
            base_resolved = base_path.resolve()
            # Ensure the resolved path is still within base_path
            if not str(resolved).startswith(str(base_resolved)):
                raise PathTraversalError(
                    message="Caminho resolvido fora do diretório permitido",
                    details={"directory": directory[:100]},
                )
        except (ValueError, OSError) as e:
            raise ValidationError(
                message="Caminho inválido",
                details={"directory": directory[:100], "error": str(e)},
            )

    # Validate against whitelist if provided
    if allowed_prefixes:
        if not any(directory.startswith(prefix) for prefix in allowed_prefixes):
            raise ValidationError(
                message=f"Diretório deve começar com: {allowed_prefixes}",
                details={"directory": directory[:100], "allowed": allowed_prefixes},
            )

    return True


def validate_file_path(file_path: str, base_path: Path) -> Path:
    """Validate and resolve file path securely.

    Args:
        file_path: File path to validate
        base_path: Base directory the file must be within

    Returns:
        Resolved Path if valid

    Raises:
        PathTraversalError if path escapes base directory
    """
    try:
        resolved = (base_path / file_path).resolve()
        base_resolved = base_path.resolve()

        if not str(resolved).startswith(str(base_resolved)):
            raise PathTraversalError(
                message="Acesso fora do diretório permitido",
                details={"path": file_path[:100], "base": str(base_path)},
            )

        return resolved
    except (ValueError, OSError) as e:
        raise ValidationError(
            message="Caminho de arquivo inválido",
            details={"path": file_path[:100], "error": str(e)},
        )
