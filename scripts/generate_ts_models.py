#!/usr/bin/env python3
"""Generate TypeScript interfaces from Pydantic models.

Usage:
    python scripts/generate_ts_models.py
    python scripts/generate_ts_models.py --output ../claude_front_sdk_angular/projects/claude-front-sdk/src/lib/models/api.models.ts
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, get_origin

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def python_type_to_ts(python_type: Any) -> str:
    """Convert Python type to TypeScript type."""
    if python_type is None or python_type is type(None):
        return "null"

    origin = get_origin(python_type)

    # Handle Optional[X] -> X | null
    if origin is type(None):
        return "null"

    # Handle Union types (including Optional)
    if hasattr(python_type, "__origin__") and python_type.__origin__ is type(None):
        return "null"

    # String representation for complex types
    type_str = str(python_type)

    # Basic types
    type_mapping = {
        "str": "string",
        "int": "number",
        "float": "number",
        "bool": "boolean",
        "Any": "any",
        "dict": "Record<string, any>",
        "Dict": "Record<string, any>",
        "list": "any[]",
        "List": "any[]",
        "None": "null",
        "NoneType": "null",
    }

    # Check basic mappings
    for py_type, ts_type in type_mapping.items():
        if py_type in type_str:
            return ts_type

    # Handle list[X]
    if "list[" in type_str.lower():
        inner = type_str.split("[")[1].rstrip("]")
        inner_ts = python_type_to_ts_str(inner)
        return f"{inner_ts}[]"

    # Handle dict[K, V]
    if "dict[" in type_str.lower():
        return "Record<string, any>"

    # Handle Optional[X]
    if "Optional[" in type_str:
        inner = type_str.replace("Optional[", "").rstrip("]")
        inner_ts = python_type_to_ts_str(inner)
        return f"{inner_ts} | null"

    # Default
    return "any"


def python_type_to_ts_str(type_str: str) -> str:
    """Convert Python type string to TypeScript."""
    mapping = {
        "str": "string",
        "int": "number",
        "float": "number",
        "bool": "boolean",
        "Any": "any",
        "None": "null",
    }
    return mapping.get(type_str.strip(), "any")


def generate_interface(model_class: type) -> str:
    """Generate TypeScript interface from Pydantic model."""
    from pydantic import BaseModel

    if not issubclass(model_class, BaseModel):
        return ""

    lines = [f"export interface {model_class.__name__} {{"]

    # Get model fields
    for field_name, field_info in model_class.model_fields.items():
        ts_type = python_type_to_ts(field_info.annotation)
        optional = "?" if not field_info.is_required() else ""
        lines.append(f"  {field_name}{optional}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


def collect_models() -> list[tuple[str, type]]:
    """Collect all Pydantic models from routers."""
    models = []

    # Import routers
    try:
        from routers.chat import ChatRequest, ChatResponse

        models.extend([("chat", ChatRequest), ("chat", ChatResponse)])
    except ImportError as e:
        print(f"Warning: Could not import chat models: {e}")

    try:
        from routers.mcp_ingest import (
            AdapterInfo,
            IngestRequest,
            IngestResult,
            MCPStatus,
        )

        models.extend(
            [
                ("mcp", AdapterInfo),
                ("mcp", IngestRequest),
                ("mcp", IngestResult),
                ("mcp", MCPStatus),
            ]
        )
    except ImportError as e:
        print(f"Warning: Could not import mcp models: {e}")

    # Add common response models
    return models


def generate_common_types() -> str:
    """Generate common TypeScript types used across the API."""
    return """/**
 * Common API response wrapper
 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

/**
 * Session info
 */
export interface SessionInfo {
  session_id: string;
  title: string;
  favorite: boolean;
  project_id: string | null;
  user_id: string | null;
  is_guest: boolean;
  message_count: number;
  last_message_at: string | null;
  created_at: string;
}

/**
 * Chat message
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  tool_calls?: ToolCall[];
}

/**
 * Tool call info
 */
export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, any>;
  output?: any;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration_ms?: number;
}

/**
 * RAG search result
 */
export interface SearchResult {
  content: string;
  source: string;
  score: number;
  metadata?: Record<string, any>;
}

/**
 * Artifact/Output file
 */
export interface ArtifactFile {
  name: string;
  path: string;
  size: number;
  modified: number;
  session_id?: string;
}

/**
 * API error response
 */
export interface ApiError {
  error: string;
  message: string;
  code?: string;
  details?: Record<string, any>;
}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate TypeScript interfaces from Pydantic models"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../claude_front_sdk_angular/projects/claude-front-sdk/src/lib/models/api.models.ts",
        help="Output file path",
    )
    args = parser.parse_args()

    output_path = Path(__file__).parent.parent / args.output

    # Header
    content = f"""/**
 * API Models - Auto-generated from Pydantic models
 * Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 *
 * DO NOT EDIT MANUALLY - Run `python scripts/generate_ts_models.py` to regenerate
 */

"""

    # Common types
    content += generate_common_types()
    content += "\n\n"

    # Collect and generate models
    models = collect_models()
    generated = set()

    for module, model in models:
        if model.__name__ not in generated:
            content += f"// From {module}\n"
            content += generate_interface(model)
            content += "\n\n"
            generated.add(model.__name__)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"Generated TypeScript models: {output_path}")
    print(f"Total interfaces: {len(generated) + 8}")  # +8 for common types


if __name__ == "__main__":
    main()
