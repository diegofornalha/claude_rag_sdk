"""
MCP Server for Claude RAG SDK

Exposes RAG tools via Model Context Protocol (MCP).
This server can be used with Claude Agent SDK or any MCP-compatible client.
"""

import time
from pathlib import Path
from typing import Optional
import os

from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("claude-rag-tools")

# Global RAG instance (initialized lazily)
_rag = None
_session_id = None


def _get_session_id() -> str:
    """Get current session ID."""
    global _session_id

    if _session_id:
        return _session_id

    # Try to read from file
    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        try:
            return session_file.read_text().strip()
        except:
            pass

    return "default"


async def _get_rag():
    """Get or create RAG instance."""
    global _rag, _session_id

    if _rag is None:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

        _session_id = _get_session_id()

        options = ClaudeRAGOptions(
            id=_session_id,
            enable_reranking=True,
            enable_prompt_guard=True,
        )

        _rag = await ClaudeRAG.open(options)
        print(f"🚀 MCP RAG initialized: {_session_id}")

    return _rag


# =============================================================================
# RAG SEARCH TOOLS
# =============================================================================

@mcp.tool()
async def search_documents(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True
) -> list:
    """
    Semantic search in indexed documents.

    Args:
        query: Search query
        top_k: Number of results (default 5)
        use_reranking: Apply reranking for better precision (default True)

    Returns:
        List of relevant documents with source, content and score
    """
    start_time = time.perf_counter()

    try:
        rag = await _get_rag()
        results = await rag.search(query, top_k=top_k, use_reranking=use_reranking)

        response = [
            {
                "doc_id": r.doc_id,
                "source": r.source,
                "type": r.doc_type,
                "content": r.content[:1000] if r.content else "",
                "similarity": r.similarity,
                "rerank_score": r.rerank_score,
                "rank": r.rank,
            }
            for r in results
        ]

        # Track tool call
        await rag.tools.record(
            "search_documents",
            started_at=start_time,
            completed_at=time.perf_counter(),
            parameters={"query": query[:100], "top_k": top_k},
            result={"count": len(response)},
        )

        return response

    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def search_hybrid(
    query: str,
    top_k: int = 5,
    vector_weight: float = 0.7
) -> list:
    """
    Hybrid search combining BM25 (lexical) and vector search.

    Args:
        query: Search query
        top_k: Number of results (default 5)
        vector_weight: Weight for vector search (0-1, default 0.7)

    Returns:
        List of documents with hybrid scores
    """
    try:
        rag = await _get_rag()
        results = await rag.search_hybrid(
            query,
            top_k=top_k,
            vector_weight=vector_weight,
        )

        return [
            {
                "doc_id": r.doc_id,
                "source": r.source,
                "type": r.doc_type,
                "content": r.content[:1000] if r.content else "",
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
                "hybrid_score": r.hybrid_score,
                "rank": r.rank,
            }
            for r in results
        ]

    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def get_document(doc_id: int) -> dict:
    """
    Get full document by ID.

    Args:
        doc_id: Document ID

    Returns:
        Complete document with all fields
    """
    try:
        rag = await _get_rag()
        doc = await rag.get_document(doc_id)

        if doc is None:
            return {"error": f"Document {doc_id} not found"}

        return doc

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def list_sources() -> list:
    """
    List all available document sources.

    Returns:
        List of documents with name and type
    """
    try:
        rag = await _get_rag()
        return await rag.list_sources()

    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def count_documents() -> dict:
    """
    Count documents and embeddings.

    Returns:
        Database statistics
    """
    try:
        rag = await _get_rag()
        stats = await rag.stats()
        return stats.get("documents", {})

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# AGENTFS FILESYSTEM TOOLS
# =============================================================================

@mcp.tool()
async def create_file(path: str, content: str) -> dict:
    """
    Create file in agent filesystem.

    Args:
        path: File name (e.g., "report.txt")
        content: File content

    Returns:
        Info about created file
    """
    try:
        rag = await _get_rag()
        await rag.fs.write_file(f"/{path}", content)

        # Also save to physical filesystem organized by session
        session_id = _get_session_id()
        if session_id and session_id != "default":
            # Get backend directory (assuming it's 3 levels up from this file)
            backend_dir = Path(__file__).parent.parent.parent
            outputs_dir = backend_dir / "outputs" / session_id
            outputs_dir.mkdir(parents=True, exist_ok=True)

            # Extract filename from path (remove leading / if present)
            filename = path.lstrip("/")
            physical_file = outputs_dir / filename
            physical_file.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "path": path,
            "size": len(content),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def read_file(path: str) -> dict:
    """
    Read file from agent filesystem.

    Args:
        path: File name

    Returns:
        File content
    """
    try:
        rag = await _get_rag()
        content = await rag.fs.read_file(f"/{path}")

        return {
            "success": True,
            "path": path,
            "content": content,
            "size": len(content) if content else 0,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_files(directory: str = "/") -> dict:
    """
    List files in agent filesystem.

    Args:
        directory: Directory to list (default: root)

    Returns:
        List of files and directories
    """
    try:
        rag = await _get_rag()
        entries = await rag.fs.readdir(directory)

        files = []
        for entry in entries:
            files.append({
                "name": entry.name,
                "size": entry.size if hasattr(entry, 'size') else 0,
                "is_dir": entry.is_dir() if hasattr(entry, 'is_dir') else False,
            })

        return {
            "success": True,
            "directory": directory,
            "files": files,
            "count": len(files),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# AGENTFS KV STORE TOOLS
# =============================================================================

@mcp.tool()
async def set_state(key: str, value: str) -> dict:
    """
    Save state in KV store.

    Args:
        key: Unique key (e.g., "user_preferences")
        value: Value to save (string or JSON)

    Returns:
        Confirmation
    """
    try:
        rag = await _get_rag()
        await rag.kv.set(key, value)

        return {
            "success": True,
            "key": key,
            "size": len(value),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_state(key: str) -> dict:
    """
    Get state from KV store.

    Args:
        key: State key

    Returns:
        Stored value or error
    """
    try:
        rag = await _get_rag()
        value = await rag.kv.get(key)

        if value is None:
            return {"success": False, "key": key, "error": "Key not found"}

        return {
            "success": True,
            "key": key,
            "value": value,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_states(prefix: str = "") -> dict:
    """
    List keys in KV store.

    Args:
        prefix: Optional prefix to filter keys

    Returns:
        List of keys
    """
    try:
        rag = await _get_rag()
        keys = await rag.kv.list(prefix=prefix if prefix else None)

        return {
            "success": True,
            "prefix": prefix or "(all)",
            "keys": keys,
            "count": len(keys),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# METRICS & HEALTH TOOLS
# =============================================================================

@mcp.tool()
async def get_metrics() -> dict:
    """
    Get RAG system metrics.

    Returns:
        Statistics including cache, queries, and errors
    """
    try:
        rag = await _get_rag()
        return await rag.stats()

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_health() -> dict:
    """
    Get system health status.

    Returns:
        Health check with component status
    """
    try:
        rag = await _get_rag()
        stats = await rag.stats()

        return {
            "status": "healthy",
            "session_id": _session_id,
            "documents": stats.get("documents", {}).get("total_documents", 0),
            "cache_enabled": stats.get("cache", {}).get("enabled", False),
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
