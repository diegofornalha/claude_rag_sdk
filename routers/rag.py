"""RAG endpoints."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

import app_state
from claude_rag_sdk.core.auth import verify_api_key
from utils.file_watcher import get_watcher

router = APIRouter(prefix="/rag", tags=["RAG"])

# RAG Knowledge base path
RAG_DB_PATH = Path.cwd() / "data" / "rag_knowledge.db"


@router.post("/search")
async def rag_search(
    request: Request, query: str, top_k: int = 5, api_key: str = Depends(verify_api_key)
):
    """Search documents using RAG."""
    from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

    temp_rag = None
    try:
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        results = await temp_rag.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [res.to_dict() for res in results],
            "count": len(results),
        }
    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")
    finally:
        if temp_rag:
            await temp_rag.close()


@router.get("/search/test")
async def search_test(query: str, top_k: int = 5):
    """Test search endpoint (no auth required for testing)."""
    if not RAG_DB_PATH.exists():
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": "Database not found",
        }

    engine = None
    try:
        from claude_rag_sdk.search import SearchEngine

        engine = SearchEngine(
            db_path=str(RAG_DB_PATH),
            embedding_model="BAAI/bge-small-en-v1.5",
            enable_reranking=False,
        )

        results = await engine.search(query, top_k=top_k)

        return {
            "query": query,
            "results": [
                {
                    "id": r.doc_id,
                    "source": r.source,
                    "content": r.content,
                    "score": round(r.similarity, 4),
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        print(f"[ERROR] RAG search test failed: {e}")
        return {"query": query, "results": [], "count": 0, "error": "Search failed"}
    finally:
        # SearchEngine doesn't hold persistent connections, but cleanup if needed
        del engine


@router.post("/ingest")
async def rag_ingest(
    request: Request, content: str, source: str, api_key: str = Depends(verify_api_key)
):
    """Add document to RAG."""
    from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

    temp_rag = None
    try:
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        result = await temp_rag.add_text(content, source)
        return result.to_dict()
    except Exception as e:
        print(f"[ERROR] RAG ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Ingest failed")
    finally:
        if temp_rag:
            await temp_rag.close()


@router.get("/stats")
async def rag_stats():
    """Get RAG statistics."""
    from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

    temp_rag = None
    try:
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        stats = await temp_rag.stats()
        return stats
    except Exception as e:
        print(f"[ERROR] RAG stats failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed")
    finally:
        if temp_rag:
            await temp_rag.close()


@router.post("/reingest")
async def reingest_backend(api_key: str = Depends(verify_api_key)):
    """Reingest backend files (run ingest_backend.py)."""
    import asyncio
    from pathlib import Path

    script_path = Path(__file__).parent.parent / "scripts" / "ingest_backend.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Ingest script not found")

    try:
        # Use asyncio subprocess for better compatibility with uvicorn
        process = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent.parent),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,  # 5 min timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise HTTPException(status_code=504, detail="Ingest timeout (5 min)")

        return {
            "success": process.returncode == 0,
            "output": stdout.decode() if stdout else "",
            "error": stderr.decode() if stderr and process.returncode != 0 else "",
            "returncode": process.returncode,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Reingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reingest/sdk")
async def reingest_sdk(api_key: str = Depends(verify_api_key)):
    """Reingest Claude Agent SDK files (run ingest_sdk.py)."""
    import asyncio
    from pathlib import Path

    script_path = Path(__file__).parent.parent / "scripts" / "ingest_sdk.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail="SDK ingest script not found")

    try:
        # Use asyncio subprocess for better compatibility with uvicorn
        process = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent.parent),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,  # 5 min timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            raise HTTPException(status_code=504, detail="Ingest timeout (5 min)")

        return {
            "success": process.returncode == 0,
            "output": stdout.decode() if stdout else "",
            "error": stderr.decode() if stderr and process.returncode != 0 else "",
            "returncode": process.returncode,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] SDK reingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def rag_config():
    """Get RAG configuration and statistics."""

    db_exists = RAG_DB_PATH.exists()
    db_size = RAG_DB_PATH.stat().st_size if db_exists else 0

    stats = {
        "total_documents": 0,
        "total_embeddings": 0,
        "total_size_bytes": 0,
        "status": "empty",
    }

    if db_exists and db_size > 0:
        engine = None
        try:
            from claude_rag_sdk.ingest import IngestEngine

            engine = IngestEngine(
                db_path=str(RAG_DB_PATH), embedding_model="BAAI/bge-small-en-v1.5"
            )
            stats = engine.stats
        except Exception as e:
            print(f"[WARN] Could not get RAG stats: {e}")
        finally:
            # IngestEngine doesn't hold persistent connections, but cleanup if needed
            del engine

    return {
        "db_path": str(RAG_DB_PATH),
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "db_size_human": _format_size(db_size),
        "stats": stats,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunk_size": 500,
        "chunk_overlap": 50,
    }


@router.get("/watcher/status")
async def watcher_status():
    """Get file watcher status."""
    watcher = get_watcher()
    return watcher.get_status()


@router.post("/watcher/start")
async def watcher_start(api_key: str = Depends(verify_api_key)):
    """Start automatic file watching and reindexing."""
    watcher = get_watcher()
    try:
        watcher.start()
        return {
            "success": True,
            "message": "File watcher started",
            "status": watcher.get_status(),
        }
    except Exception as e:
        print(f"[ERROR] Failed to start watcher: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start watcher: {str(e)}")


@router.post("/watcher/stop")
async def watcher_stop(api_key: str = Depends(verify_api_key)):
    """Stop automatic file watching."""
    watcher = get_watcher()
    try:
        watcher.stop()
        return {
            "success": True,
            "message": "File watcher stopped",
            "status": watcher.get_status(),
        }
    except Exception as e:
        print(f"[ERROR] Failed to stop watcher: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop watcher: {str(e)}")


@router.delete("/reset")
async def rag_reset(api_key: str = Depends(verify_api_key)):
    """Delete RAG database and reset to empty state."""

    if not RAG_DB_PATH.exists():
        return {
            "success": True,
            "message": "Database already empty",
            "deleted_files": [],
        }

    deleted = []
    try:
        # Delete main DB file
        if RAG_DB_PATH.exists():
            os.remove(RAG_DB_PATH)
            deleted.append(str(RAG_DB_PATH))

        # Delete WAL and SHM files if they exist
        wal_path = RAG_DB_PATH.with_suffix(".db-wal")
        shm_path = RAG_DB_PATH.with_suffix(".db-shm")

        if wal_path.exists():
            os.remove(wal_path)
            deleted.append(str(wal_path))

        if shm_path.exists():
            os.remove(shm_path)
            deleted.append(str(shm_path))

        return {
            "success": True,
            "message": "RAG database deleted successfully",
            "deleted_files": deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete RAG database: {str(e)}")


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# === DOCUMENT LISTING ===


@router.get("/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List all documents in RAG database."""
    import json
    import sqlite3

    if not RAG_DB_PATH.exists():
        return {"documents": [], "total": 0}

    try:
        with sqlite3.connect(str(RAG_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM documentos")
            total = cursor.fetchone()[0]

            # Get documents with pagination
            cursor.execute(
                """
                SELECT
                    id,
                    nome,
                    tipo,
                    LENGTH(conteudo) as content_length,
                    caminho,
                    hash,
                    metadata,
                    criado_em
                FROM documentos
                ORDER BY criado_em DESC
                LIMIT ? OFFSET ?
            """,
                [limit, offset],
            )

            documents = []
            for row in cursor.fetchall():
                doc = dict(row)
                # Parse metadata if JSON
                if doc.get("metadata"):
                    try:
                        doc["metadata"] = json.loads(doc["metadata"])
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Failed to parse metadata JSON: {e}")
                        pass  # Keep raw metadata if not valid JSON
                documents.append(doc)

            return {
                "documents": documents,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    except Exception as e:
        print(f"[ERROR] Failed to list documents: {e}")
        return {"documents": [], "total": 0, "error": "Failed to list documents"}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get document details including content and chunks."""
    import json
    import sqlite3

    if not RAG_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="RAG database not found")

    try:
        with sqlite3.connect(str(RAG_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get document
            cursor.execute(
                """
                SELECT * FROM documentos WHERE id = ?
            """,
                [doc_id],
            )

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document not found")

            doc = dict(row)

            # Parse metadata
            if doc.get("metadata"):
                try:
                    doc["metadata"] = json.loads(doc["metadata"])
                except json.JSONDecodeError as e:
                    print(f"[WARN] Failed to parse metadata JSON: {e}")
                    pass  # Keep raw metadata if not valid JSON

            doc["has_embedding"] = True  # Assume yes (embedding is created with doc)

            return doc

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, api_key: str = Depends(verify_api_key)):
    """Delete a specific document from RAG."""
    import apsw
    import sqlite_vec

    if not RAG_DB_PATH.exists():
        raise HTTPException(status_code=404, detail="RAG database not found")

    conn = None
    try:
        conn = apsw.Connection(str(RAG_DB_PATH))
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT nome FROM documentos WHERE id = ?", [doc_id])
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        nome = row[0]

        # Delete from vec_documentos
        cursor.execute("DELETE FROM vec_documentos WHERE doc_id = ?", [doc_id])

        # Delete from documentos
        cursor.execute("DELETE FROM documentos WHERE id = ?", [doc_id])

        return {
            "success": True,
            "message": f"Document '{nome}' deleted",
            "doc_id": doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Delete document failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")
    finally:
        if conn:
            conn.close()
