"""RAG endpoints."""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

import app_state
from claude_rag_sdk.core.auth import verify_api_key
from claude_rag_sdk.core.config import get_config, reload_config
from claude_rag_sdk.core.rate_limiter import RATE_LIMITS, limiter
from utils.file_watcher import get_watcher

router = APIRouter(prefix="/rag", tags=["RAG"])


def _get_rag_db_path() -> Path:
    """Obtém caminho do RAG database da config centralizada."""
    return get_config().rag_db_path


@router.post("/search")
@limiter.limit(RATE_LIMITS.get("search", "60/minute"))
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
        raise HTTPException(status_code=500, detail="Search failed") from e
    finally:
        if temp_rag:
            await temp_rag.close()


@router.get("/search/test")
@limiter.limit(RATE_LIMITS.get("search", "60/minute"))
async def search_test(request: Request, query: str, top_k: int = 5):
    """Test search endpoint (no auth required for testing)."""
    config = get_config()
    rag_db_path = config.rag_db_path

    if not rag_db_path.exists():
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
            db_path=str(rag_db_path),
            embedding_model=config.embedding_model_string,
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
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
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
        raise HTTPException(status_code=500, detail="Ingest failed") from e
    finally:
        if temp_rag:
            await temp_rag.close()


@router.get("/stats")
@limiter.limit(RATE_LIMITS.get("default", "60/minute"))
async def rag_stats(request: Request):
    """Get RAG statistics."""
    from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

    temp_rag = None
    try:
        temp_rag = await ClaudeRAG.open(ClaudeRAGOptions(id=app_state.current_session_id or "temp"))
        stats = await temp_rag.stats()
        return stats
    except Exception as e:
        print(f"[ERROR] RAG stats failed: {e}")
        raise HTTPException(status_code=500, detail="Stats retrieval failed") from e
    finally:
        if temp_rag:
            await temp_rag.close()


@router.post("/reingest/atlantyx")
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
async def reingest_atlantyx(request: Request, reingest: bool = False, api_key: str = Depends(verify_api_key)):
    """Reingest Atlantyx documents (6 docs for evaluation).

    Args:
        reingest: If True, clear database before ingesting
    """
    import asyncio
    from pathlib import Path

    script_path = Path(__file__).parent.parent / "scripts" / "ingest_atlantyx.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Atlantyx ingest script not found")

    try:
        args = ["python", str(script_path)]
        if reingest:
            args.append("--reingest")

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_path.parent.parent),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,
            )
        except asyncio.TimeoutError as e:
            process.kill()
            raise HTTPException(status_code=504, detail="Ingest timeout (5 min)") from e

        return {
            "success": process.returncode == 0,
            "output": stdout.decode() if stdout else "",
            "error": stderr.decode() if stderr and process.returncode != 0 else "",
            "returncode": process.returncode,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Atlantyx reingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reingest")
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
async def reingest_backend(request: Request, api_key: str = Depends(verify_api_key)):
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
        except asyncio.TimeoutError as e:
            process.kill()
            raise HTTPException(status_code=504, detail="Ingest timeout (5 min)") from e

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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reingest/sdk")
@limiter.limit(RATE_LIMITS.get("ingest", "10/minute"))
async def reingest_sdk(request: Request, api_key: str = Depends(verify_api_key)):
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
        except asyncio.TimeoutError as e:
            process.kill()
            raise HTTPException(status_code=504, detail="Ingest timeout (5 min)") from e

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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/config")
@limiter.limit(RATE_LIMITS.get("default", "60/minute"))
async def rag_config(request: Request):
    """Get RAG configuration and statistics."""
    config = get_config()
    rag_db_path = config.rag_db_path

    db_exists = rag_db_path.exists()
    db_size = rag_db_path.stat().st_size if db_exists else 0

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
                db_path=str(rag_db_path),
                embedding_model=config.embedding_model_string,
            )
            stats = engine.stats
        except Exception as e:
            print(f"[WARN] Could not get RAG stats: {e}")
        finally:
            # IngestEngine doesn't hold persistent connections, but cleanup if needed
            del engine

    return {
        "db_path": str(rag_db_path),
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "db_size_human": _format_size(db_size),
        "stats": stats,
        "embedding_model": config.embedding_model_string,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


@router.post("/config/reload")
async def reload_rag_config(api_key: str = Depends(verify_api_key)):
    """Reload RAG configuration from environment variables."""
    from dotenv import load_dotenv

    # Recarrega variáveis de ambiente do .env
    load_dotenv(override=True)

    # Recarrega a configuração singleton
    new_config = reload_config()

    return {
        "success": True,
        "message": "Configuration reloaded from .env",
        "config": {
            "chunk_size": new_config.chunk_size,
            "chunk_overlap": new_config.chunk_overlap,
            "embedding_model": new_config.embedding_model_string,
        },
    }


@router.get("/embedding-models")
async def list_embedding_models():
    """Lista todos os modelos de embedding disponíveis."""
    from claude_rag_sdk.core.config import EmbeddingModel

    models = []
    for model in EmbeddingModel:
        models.append({
            "value": model.value,
            "short_name": model.short_name,
            "display_name": model.display_name,
            "language": model.language,
            "dimensions": model.dimensions,
        })

    current = get_config()
    return {
        "models": models,
        "current": current.embedding_model.short_name,
    }


@router.post("/embedding-model")
async def change_embedding_model(
    model: str,
    api_key: str = Depends(verify_api_key),
):
    """Muda o modelo de embedding e requer reingestão dos documentos.

    Args:
        model: short_name do modelo (ex: 'bge-small', 'bertimbau-base')

    Returns:
        Confirmação da mudança. Documentos precisam ser reingeridos!
    """
    import os
    from pathlib import Path

    from dotenv import set_key

    from claude_rag_sdk.core.config import EmbeddingModel, reload_config

    # Validar modelo
    valid_models = {m.short_name: m for m in EmbeddingModel}
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo inválido: {model}. Válidos: {list(valid_models.keys())}"
        )

    new_model = valid_models[model]

    # Atualizar .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        set_key(str(env_path), "EMBEDDING_MODEL", model)

    # Também definir no ambiente atual
    os.environ["EMBEDDING_MODEL"] = model

    # Recarregar config
    reload_config()

    return {
        "success": True,
        "message": f"Modelo alterado para {new_model.display_name}. IMPORTANTE: Execute reingestão dos documentos!",
        "model": {
            "value": new_model.value,
            "short_name": new_model.short_name,
            "display_name": new_model.display_name,
            "language": new_model.language,
            "dimensions": new_model.dimensions,
        },
        "requires_reingest": True,
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
        raise HTTPException(status_code=500, detail=f"Failed to start watcher: {str(e)}") from e


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
        raise HTTPException(status_code=500, detail=f"Failed to stop watcher: {str(e)}") from e


@router.delete("/reset")
async def rag_reset(api_key: str = Depends(verify_api_key)):
    """Delete RAG database and reset to empty state."""
    rag_db_path = get_config().rag_db_path

    if not rag_db_path.exists():
        return {
            "success": True,
            "message": "Database already empty",
            "deleted_files": [],
        }

    deleted = []
    try:
        # Delete main DB file
        if rag_db_path.exists():
            os.remove(rag_db_path)
            deleted.append(str(rag_db_path))

        # Delete WAL and SHM files if they exist
        wal_path = rag_db_path.with_suffix(".db-wal")
        shm_path = rag_db_path.with_suffix(".db-shm")

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
        raise HTTPException(
            status_code=500, detail=f"Failed to delete RAG database: {str(e)}"
        ) from e


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

    rag_db_path = get_config().rag_db_path

    if not rag_db_path.exists():
        return {"documents": [], "total": 0}

    try:
        with sqlite3.connect(str(rag_db_path)) as conn:
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

    rag_db_path = get_config().rag_db_path

    if not rag_db_path.exists():
        raise HTTPException(status_code=404, detail="RAG database not found")

    try:
        with sqlite3.connect(str(rag_db_path)) as conn:
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
        raise HTTPException(status_code=500, detail=str(e)) from e


# === ASK ENDPOINT (Pergunta com citações, confiança e métricas) ===


@router.get("/ask/test")
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def rag_ask_test(
    request: Request,
    question: str,
    top_k: int = 5,
    user_role: str = "viewer",
):
    """Endpoint de teste sem autenticação para avaliação."""
    return await _rag_ask_internal(question, top_k, user_role)


@router.post("/ask")
@limiter.limit(RATE_LIMITS.get("chat", "30/minute"))
async def rag_ask(
    request: Request,
    question: str,
    top_k: int = 5,
    user_role: str = "viewer",
    api_key: str = Depends(verify_api_key),
):
    """Endpoint com autenticação."""
    return await _rag_ask_internal(question, top_k, user_role)


async def _rag_ask_internal(
    question: str,
    top_k: int = 5,
    user_role: str = "viewer",
):
    """
    Faz uma pergunta ao RAG e retorna resposta estruturada.

    Returns:
        - answer: Resposta gerada
        - citations: Lista de citações [{source, quote}]
        - confidence: Score de confiança (0-1)
        - metrics: {tokens, latency_ms, sources_count}
    """
    import time

    from claude_rag_sdk.core.config import get_config

    config = get_config()
    start_time = time.time()

    # 1. Buscar documentos relevantes
    search_results = []
    try:
        from claude_rag_sdk.search import SearchEngine

        engine = SearchEngine(
            db_path=str(config.rag_db_path),
            embedding_model=config.embedding_model_string,
            enable_reranking=True,
        )
        search_results = await engine.search(question, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")

    # 2. Preparar contexto
    if not search_results:
        return {
            "answer": "Não encontrei informações relevantes nos documentos fornecidos.",
            "citations": [],
            "confidence": 0.0,
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "sources_count": 0,
            },
        }

    context_parts = []
    sources_used = set()
    for r in search_results:
        context_parts.append(f"[Fonte: {r.source}]\n{r.content}")
        sources_used.add(r.source)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Chamar Claude para gerar resposta
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import query as sdk_query

        system_prompt = f"""Você é um assistente que responde perguntas baseado EXCLUSIVAMENTE nos documentos fornecidos.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com informações dos documentos fornecidos
2. SEMPRE inclua citações no formato: [Fonte: nome_do_documento]
3. Se não houver evidência suficiente, diga: "Não encontrei essa informação nos documentos fornecidos"
4. Seja preciso e cite trechos relevantes
5. User role: {user_role}

CONTEXTO DOS DOCUMENTOS:
{context}"""

        options = ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",
            system_prompt=system_prompt,
        )

        answer = ""
        async for message in sdk_query(prompt=question, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        answer += block.text

        # 4. Extrair citações da resposta
        citations = _extract_citations(answer, search_results)

        # 5. Calcular confiança
        confidence = _calculate_confidence(answer, citations, search_results)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Estimar tokens (aproximado)
        input_tokens = len(system_prompt + question) // 4
        output_tokens = len(answer) // 4

        return {
            "answer": answer.strip(),
            "citations": citations,
            "confidence": round(confidence, 2),
            "metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "sources_count": len(sources_used),
            },
        }

    except Exception as e:
        print(f"[ERROR] RAG ask failed: {e}")
        return {
            "answer": f"Erro ao processar pergunta: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "sources_count": len(search_results),
            },
        }


def _extract_citations(answer: str, search_results: list) -> list:
    """Extrai citações da resposta baseado nas fontes encontradas."""
    import re

    citations = []
    seen_sources = set()

    # Padrão: [Fonte: nome_do_arquivo]
    fonte_pattern = r"\[Fonte:\s*([^\]]+)\]"
    matches = re.findall(fonte_pattern, answer, re.IGNORECASE)

    for source_name in matches:
        source_name = source_name.strip()
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)

        # Encontrar trecho correspondente
        for r in search_results:
            if source_name.lower() in r.source.lower():
                # Pegar primeiro trecho relevante (até 200 chars)
                quote = r.content[:200].strip()
                if len(r.content) > 200:
                    quote += "..."
                citations.append({
                    "source": r.source,
                    "quote": quote,
                })
                break

    return citations


def _calculate_confidence(answer: str, citations: list, search_results: list) -> float:
    """Calcula score de confiança baseado em heurísticas."""
    # Base: 0.4
    confidence = 0.4

    # +0.2 se tem citações
    if citations:
        confidence += 0.2

    # +0.1 por cada citação adicional (max +0.3)
    confidence += min(len(citations) - 1, 3) * 0.1

    # -0.3 se resposta indica que não encontrou
    negative_phrases = [
        "não encontrei",
        "não há informação",
        "não localizei",
        "não consta",
        "sem evidência",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in negative_phrases):
        confidence = max(0.1, confidence - 0.3)

    # +0.1 se usou múltiplas fontes
    unique_sources = {c["source"] for c in citations}
    if len(unique_sources) >= 2:
        confidence += 0.1

    return min(confidence, 1.0)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, api_key: str = Depends(verify_api_key)):
    """Delete a specific document from RAG."""
    import apsw
    import sqlite_vec

    rag_db_path = get_config().rag_db_path

    if not rag_db_path.exists():
        raise HTTPException(status_code=404, detail="RAG database not found")

    conn = None
    try:
        conn = apsw.Connection(str(rag_db_path))
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
        raise HTTPException(status_code=500, detail="Failed to delete document") from e
    finally:
        if conn:
            conn.close()
