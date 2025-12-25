"""RAG endpoints."""
from fastapi import APIRouter, Request, Depends, HTTPException

from claude_rag_sdk.core.auth import verify_api_key

import app_state

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/search")
async def rag_search(
    request: Request,
    query: str,
    top_k: int = 5,
    api_key: str = Depends(verify_api_key)
):
    """Search documents using RAG."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        results = await temp_rag.search(query, top_k=top_k)
        await temp_rag.close()
        return {
            "query": query,
            "results": [res.to_dict() for res in results],
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def rag_ingest(
    request: Request,
    content: str,
    source: str,
    api_key: str = Depends(verify_api_key)
):
    """Add document to RAG."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        result = await temp_rag.add_text(content, source)
        await temp_rag.close()
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def rag_stats():
    """Get RAG statistics."""
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        stats = await temp_rag.stats()
        await temp_rag.close()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
