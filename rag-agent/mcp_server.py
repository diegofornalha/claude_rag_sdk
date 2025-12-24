# =============================================================================
# MCP SERVER - RAG Tools para Desafio Atlantyx
# =============================================================================
# Ferramentas de busca semantica usando FastEmbed + sqlite-vec
# Com todas as integrações: logging, métricas, cache, hybrid search,
# reranking, circuit breaker, prompt guard, RBAC
# =============================================================================

import time
from mcp.server.fastmcp import FastMCP
from fastembed import TextEmbedding
import apsw
import sqlite_vec
from pathlib import Path
from typing import Optional

# Imports do core
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.logger import logger, set_request_id, set_conversation_id
from core.cache import get_embedding_cache, get_response_cache
from core.circuit_breaker import get_or_create_circuit_breaker, CircuitBreakerError
from core.prompt_guard import get_prompt_guard, ThreatLevel
from core.reranker import LightweightReranker
from core.config import get_config
from core.adaptive_search import apply_adaptive_topk
from core.sync_audit import audit_sync_tool, get_audit_queue
from api.metrics import get_metrics

# Carregar configuração
config = get_config()

# Caminho do banco de dados
DB_PATH = config.db_path

# Inicializar MCP Server
mcp = FastMCP("rag-tools")

# Modelo de embeddings (carregado da config)
model = TextEmbedding(config.embedding_model.value)

# Coletor de métricas
metrics = get_metrics()

# Cache de embeddings
embedding_cache = get_embedding_cache()

# Cache de respostas
response_cache = get_response_cache()

# Circuit breaker para operações de DB
db_circuit = get_or_create_circuit_breaker("database", failure_threshold=3, timeout=30.0)

# Prompt guard
prompt_guard = get_prompt_guard(strict_mode=False)

# Reranker
reranker = LightweightReranker()


def get_connection():
    """Cria conexao com sqlite-vec carregado usando apsw."""
    conn = apsw.Connection(str(DB_PATH))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def serialize_embedding(embedding: list) -> bytes:
    """Converte lista de floats para bytes usando sqlite_vec."""
    return sqlite_vec.serialize_float32(embedding)


def get_embedding_cached(text: str) -> list[float]:
    """Obtém embedding com cache."""
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached

    embeddings = list(model.embed([text]))
    embedding = embeddings[0].tolist()
    embedding_cache.set(text, embedding)
    return embedding


@mcp.tool()
@audit_sync_tool("search_documents")
def search_documents(query: str, top_k: int = 5, use_reranking: bool = True, use_adaptive: bool = True) -> list:
    """
    Busca semantica nos documentos indexados.

    Args:
        query: Pergunta ou texto para buscar
        top_k: Numero de resultados (padrao 5)
        use_reranking: Aplicar re-ranking para melhor precisao (padrao True)
        use_adaptive: Usar top-k adaptativo baseado em confidence (padrao True)

    Returns:
        Lista de documentos relevantes com source, content e score
    """
    request_id = set_request_id()
    start_time = time.perf_counter()

    try:
        # Verificar prompt injection
        scan_result = prompt_guard.scan(query)
        if not scan_result.is_safe:
            logger.warning(
                "prompt_blocked",
                query=query[:100],
                threat_level=scan_result.threat_level.value,
                threats=scan_result.threats_detected[:3],
            )
            metrics.record_error("PromptInjectionBlocked")
            return [{
                "error": "Query blocked by security filter",
                "threat_level": scan_result.threat_level.value,
            }]

        # Verificar cache de resposta (incluindo use_reranking na chave)
        cached_response = response_cache.get(query, top_k, use_reranking=use_reranking)
        if cached_response:
            logger.info("cache_hit", query=query[:50], use_reranking=use_reranking)
            return cached_response

        # Usar circuit breaker para operação de DB
        def do_search():
            # Gerar embedding com cache
            embedding = get_embedding_cached(query)
            query_vec = serialize_embedding(embedding)

            conn = get_connection()
            cursor = conn.cursor()

            # Buscar mais resultados para re-ranking
            fetch_k = top_k * 2 if use_reranking else top_k

            results = []
            for row in cursor.execute("""
                SELECT v.doc_id, v.distance, d.nome, d.conteudo, d.tipo
                FROM vec_documentos v
                JOIN documentos d ON d.id = v.doc_id
                WHERE v.embedding MATCH ? AND k = ?
            """, (query_vec, fetch_k)):
                doc_id, distance, nome, conteudo, tipo = row
                similarity = max(0, 1 - distance)

                results.append({
                    "doc_id": doc_id,
                    "source": nome,
                    "type": tipo,
                    "content": conteudo[:1000] if conteudo else "",
                    "similarity": round(similarity, 3)
                })

            conn.close()
            return results

        # Executar com circuit breaker
        try:
            results = db_circuit.call(do_search)
        except CircuitBreakerError as e:
            logger.log_error("CircuitBreakerOpen", str(e))
            metrics.record_error("CircuitBreakerOpen")
            return [{"error": "Service temporarily unavailable", "retry_after": 30}]

        # Aplicar top-k adaptativo ANTES do reranking
        if use_adaptive and len(results) > 0:
            # Criar objetos mock com similarity para adaptive
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                similarity: float

            mock_results = [MockResult(similarity=r["similarity"]) for r in results]
            _, adaptive_decision = apply_adaptive_topk(mock_results, top_k, enabled=True)

            # Ajustar fetch_k baseado na decisão adaptativa
            adaptive_k = adaptive_decision.adjusted_k

            # Log da decisão
            logger.info(
                "adaptive_topk_decision",
                original_k=top_k,
                adjusted_k=adaptive_k,
                reason=adaptive_decision.reason,
                confidence=adaptive_decision.confidence_level,
                top_similarity=adaptive_decision.top_similarity,
            )

            # Aplicar o ajuste
            results = results[:adaptive_k]
        else:
            adaptive_k = top_k
            results = results[:top_k]

        # Aplicar re-ranking se habilitado
        if use_reranking and len(results) > 0:
            docs_for_rerank = [
                (r["doc_id"], r["content"], r["similarity"], {"source": r["source"], "type": r["type"]})
                for r in results
            ]
            # Usar adaptive_k como limite do reranking
            reranked = reranker.rerank(query, docs_for_rerank, top_k=adaptive_k)

            results = [
                {
                    "doc_id": r.doc_id,
                    "source": r.metadata["source"],
                    "type": r.metadata["type"],
                    "content": r.content,
                    "similarity": r.original_score,
                    "rerank_score": r.rerank_score,
                    "rank": r.final_rank,
                }
                for r in reranked
            ]

        # Calcular latencia
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Registrar metricas
        metrics.record_query(latency_ms, len(results))

        # Log estruturado
        doc_ids = [r["doc_id"] for r in results]
        similarities = [r["similarity"] for r in results]
        logger.log_query(query, top_k, len(results), latency_ms)
        logger.log_retrieval(doc_ids, similarities, latency_ms)

        # Salvar em cache (incluindo use_reranking na chave)
        response_cache.set(query, top_k, results, use_reranking=use_reranking)

        return results

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_error(type(e).__name__)
        logger.log_error(type(e).__name__, str(e), query=query[:100])
        raise


@mcp.tool()
@audit_sync_tool("search_hybrid")
def search_hybrid(query: str, top_k: int = 5, vector_weight: float = 0.7) -> list:
    """
    Busca hibrida combinando BM25 (lexica) e vetorial.

    Args:
        query: Pergunta ou texto para buscar
        top_k: Numero de resultados (padrao 5)
        vector_weight: Peso da busca vetorial (0-1, padrao 0.7)

    Returns:
        Lista de documentos com scores hibridos
    """
    from core.hybrid_search import HybridSearch

    request_id = set_request_id()
    start_time = time.perf_counter()

    try:
        # Verificar prompt injection
        scan_result = prompt_guard.scan(query)
        if not scan_result.is_safe:
            metrics.record_error("PromptInjectionBlocked")
            return [{"error": "Query blocked by security filter"}]

        # Busca hibrida
        hybrid = HybridSearch(
            str(DB_PATH),
            vector_weight=vector_weight,
            bm25_weight=1 - vector_weight,
        )

        results = hybrid.search(query, top_k=top_k)

        # Converter para formato de resposta
        response = [
            {
                "doc_id": r.doc_id,
                "source": r.nome,
                "type": r.tipo,
                "content": r.content,
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
                "hybrid_score": r.hybrid_score,
                "rank": r.rank,
            }
            for r in results
        ]

        # Metricas e logs
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_query(latency_ms, len(response))
        logger.log_query(query, top_k, len(response), latency_ms)

        return response

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_error(type(e).__name__)
        logger.log_error(type(e).__name__, str(e), query=query[:100])
        raise


@mcp.tool()
@audit_sync_tool("get_document")
def get_document(doc_id: int) -> dict:
    """
    Recupera documento completo pelo ID.

    Args:
        doc_id: ID do documento

    Returns:
        Documento completo com todos os campos
    """
    def fetch_doc():
        conn = get_connection()
        cursor = conn.cursor()

        row = None
        for r in cursor.execute("""
            SELECT id, nome, tipo, conteudo, caminho, criado_em
            FROM documentos
            WHERE id = ?
        """, (doc_id,)):
            row = r
            break

        conn.close()
        return row

    try:
        row = db_circuit.call(fetch_doc)
    except CircuitBreakerError:
        return {"error": "Service temporarily unavailable"}

    if not row:
        return {"error": f"Documento {doc_id} nao encontrado"}

    return {
        "id": row[0],
        "nome": row[1],
        "tipo": row[2],
        "conteudo": row[3],
        "caminho": row[4],
        "criado_em": str(row[5]) if row[5] else None
    }


@mcp.tool()
@audit_sync_tool("list_sources")
def list_sources() -> list:
    """
    Lista todas as fontes/documentos disponiveis no banco.

    Returns:
        Lista de documentos com nome e tipo
    """
    def fetch_sources():
        conn = get_connection()
        cursor = conn.cursor()

        results = [
            {"id": r[0], "nome": r[1], "tipo": r[2], "tamanho": r[3]}
            for r in cursor.execute("""
                SELECT id, nome, tipo, LENGTH(conteudo) as tamanho
                FROM documentos
                ORDER BY nome
            """)
        ]

        conn.close()
        return results

    try:
        return db_circuit.call(fetch_sources)
    except CircuitBreakerError:
        return [{"error": "Service temporarily unavailable"}]


@mcp.tool()
@audit_sync_tool("count_documents")
def count_documents() -> dict:
    """
    Conta documentos e embeddings no banco.

    Returns:
        Estatisticas do banco
    """
    def count():
        conn = get_connection()
        cursor = conn.cursor()

        total_docs = 0
        for r in cursor.execute("SELECT COUNT(*) FROM documentos"):
            total_docs = r[0]

        total_embeddings = 0
        for r in cursor.execute("SELECT COUNT(*) FROM vec_documentos"):
            total_embeddings = r[0]

        conn.close()

        return {
            "total_documentos": total_docs,
            "total_embeddings": total_embeddings,
            "status": "ok" if total_docs == total_embeddings else "incompleto"
        }

    try:
        return db_circuit.call(count)
    except CircuitBreakerError:
        return {"error": "Service temporarily unavailable", "status": "unavailable"}


@mcp.tool()
@audit_sync_tool("get_metrics_summary")
def get_metrics_summary() -> dict:
    """
    Retorna metricas do sistema RAG.

    Returns:
        Estatisticas de uso, latencia, custos e erros
    """
    all_metrics = metrics.get_all_metrics()

    # Incluir stats de cache
    emb_cache_stats = embedding_cache.stats
    resp_cache_stats = response_cache.stats

    return {
        "uptime_seconds": all_metrics["uptime_seconds"],
        "queries": {
            "total": all_metrics["rag"]["queries_total"],
            "latency_avg_ms": all_metrics["rag"]["query_latency"]["avg"],
            "latency_p95_ms": all_metrics["rag"]["query_latency"]["p95"],
        },
        "cache": {
            "embedding": {
                "hits": emb_cache_stats.hits,
                "misses": emb_cache_stats.misses,
                "hit_rate": round(emb_cache_stats.hit_rate, 2),
            },
            "response": {
                "hits": resp_cache_stats.hits,
                "misses": resp_cache_stats.misses,
                "hit_rate": round(resp_cache_stats.hit_rate, 2),
            },
        },
        "circuit_breaker": {
            "state": db_circuit.state.value,
            "stats": {
                "total": db_circuit.stats.total_calls,
                "failures": db_circuit.stats.failed_calls,
                "rejected": db_circuit.stats.rejected_calls,
            },
        },
        "errors": all_metrics["rag"]["errors_by_type"],
    }


@mcp.tool()
@audit_sync_tool("get_health")
def get_health() -> dict:
    """
    Retorna status de saude do sistema.

    Returns:
        Health check com status de todos os componentes
    """
    from api.health import HealthChecker

    checker = HealthChecker(str(DB_PATH))
    report = checker.check_health(include_details=False)

    return {
        "status": report.status.value,
        "uptime_seconds": round(report.uptime_seconds, 2),
        "components": [
            {
                "name": c.name,
                "status": c.status.value,
                "latency_ms": round(c.latency_ms, 2),
            }
            for c in report.components
        ],
    }


@mcp.tool()
@audit_sync_tool("get_config_info")
def get_config_info() -> dict:
    """
    Retorna configuração atual do sistema RAG.

    Returns:
        Configuração completa incluindo modelo, chunking e parâmetros
    """
    return config.to_dict()


@mcp.tool()
@audit_sync_tool("clear_cache")
def clear_cache(cache_type: str = "all") -> dict:
    """
    Limpa cache de embeddings ou respostas.

    Args:
        cache_type: Tipo de cache ("embedding", "response", ou "all")

    Returns:
        Estatisticas de limpeza
    """
    result = {"cleared": []}

    if cache_type in ("embedding", "all"):
        emb_stats_before = embedding_cache.stats
        embedding_cache.clear()
        result["cleared"].append({
            "type": "embedding",
            "entries_cleared": emb_stats_before.size,
            "memory_freed_mb": round(emb_stats_before.memory_bytes / 1024 / 1024, 2),
        })

    if cache_type in ("response", "all"):
        resp_stats_before = response_cache.stats
        response_cache.clear()
        result["cleared"].append({
            "type": "response",
            "entries_cleared": resp_stats_before.size,
            "memory_freed_mb": round(resp_stats_before.memory_bytes / 1024 / 1024, 2),
        })

    result["status"] = "success"
    return result


# =============================================================================
# AGENTFS FILESYSTEM TOOLS
# =============================================================================

@mcp.tool()
@audit_sync_tool("create_file")
async def create_file(path: str, content: str) -> dict:
    """
    Cria arquivo no filesystem do agent.

    Args:
        path: Nome do arquivo (ex: "resumo.txt")
        content: Conteúdo do arquivo

    Returns:
        Informações sobre o arquivo criado
    """
    from pathlib import Path
    import os

    storage_type = "local"

    try:
        # Determina pasta de outputs da sessão
        session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
        if session_file.exists():
            session_id = session_file.read_text().strip()
            outputs_dir = Path(__file__).parent / "outputs" / session_id
        else:
            outputs_dir = Path(__file__).parent / "outputs"

        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Escreve no diretório de outputs da sessão
        local_path = outputs_dir / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding='utf-8')

        logger.info("Arquivo criado", path=str(local_path), size=len(content))

        return {
            "success": True,
            "path": str(local_path),
            "size": len(content),
            "storage": storage_type
        }
    except Exception as e:
        logger.error(f"Erro ao criar arquivo: {e}", path=path, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
@audit_sync_tool("read_file")
async def read_file(path: str) -> dict:
    """
    Lê arquivo do filesystem do agent.

    Args:
        path: Nome do arquivo

    Returns:
        Conteúdo do arquivo
    """
    from pathlib import Path

    try:
        # Primeiro tenta ler como caminho absoluto
        file_path = Path(path)
        if not file_path.is_absolute():
            # Tenta na pasta de outputs da sessão
            session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
            if session_file.exists():
                session_id = session_file.read_text().strip()
                file_path = Path(__file__).parent / "outputs" / session_id / path
            else:
                file_path = Path(__file__).parent / "outputs" / path

        if not file_path.exists():
            return {"success": False, "error": f"Arquivo não encontrado: {path}"}

        content = file_path.read_text(encoding='utf-8')
        logger.info("Arquivo lido", path=str(file_path), size=len(content))

        return {
            "success": True,
            "path": str(file_path),
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"Erro ao ler arquivo: {e}", path=path, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
@audit_sync_tool("list_files")
async def list_files(directory: str = "/") -> dict:
    """
    Lista arquivos no filesystem do agent.

    Args:
        directory: Diretório a listar (padrão: raiz)

    Returns:
        Lista de arquivos e diretórios
    """
    from pathlib import Path
    import os

    try:
        # Determina pasta de outputs da sessão
        session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
        if session_file.exists():
            session_id = session_file.read_text().strip()
            base_dir = Path(__file__).parent / "outputs" / session_id
        else:
            base_dir = Path(__file__).parent / "outputs"

        # Se directory é "/", lista a raiz dos outputs
        if directory == "/" or directory == "":
            list_dir = base_dir
        else:
            list_dir = base_dir / directory.lstrip("/")

        if not list_dir.exists():
            list_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for entry in list_dir.iterdir():
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "size": stat.st_size,
                "is_dir": entry.is_dir(),
                "modified": stat.st_mtime
            })

        logger.info("Diretório listado", directory=str(list_dir), count=len(files))

        return {
            "success": True,
            "directory": str(list_dir),
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Erro ao listar diretório: {e}", directory=directory, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
@audit_sync_tool("delete_file")
async def delete_file(path: str) -> dict:
    """
    Remove arquivo do filesystem.

    Args:
        path: Nome do arquivo

    Returns:
        Status da operação
    """
    from pathlib import Path

    try:
        # Determina pasta de outputs da sessão
        session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
        if session_file.exists():
            session_id = session_file.read_text().strip()
            file_path = Path(__file__).parent / "outputs" / session_id / path
        else:
            file_path = Path(__file__).parent / "outputs" / path

        if not file_path.exists():
            return {"success": False, "error": f"Arquivo não encontrado: {path}"}

        file_path.unlink()
        logger.info("Arquivo deletado", path=str(file_path))

        return {"success": True, "path": str(file_path)}
    except Exception as e:
        logger.error(f"Erro ao deletar arquivo: {e}", path=path, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
@audit_sync_tool("get_file_info")
async def get_file_info(path: str) -> dict:
    """
    Obtém informações sobre um arquivo.

    Args:
        path: Nome do arquivo

    Returns:
        Metadados do arquivo
    """
    from pathlib import Path

    try:
        # Determina pasta de outputs da sessão
        session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
        if session_file.exists():
            session_id = session_file.read_text().strip()
            file_path = Path(__file__).parent / "outputs" / session_id / path
        else:
            file_path = Path(__file__).parent / "outputs" / path

        if not file_path.exists():
            return {"success": False, "error": f"Arquivo não encontrado: {path}"}

        stat = file_path.stat()
        logger.info("Info do arquivo obtida", path=str(file_path))

        return {
            "success": True,
            "path": str(file_path),
            "size": stat.st_size,
            "is_dir": file_path.is_dir(),
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime
        }
    except Exception as e:
        logger.error(f"Erro ao obter info do arquivo: {e}", path=path, error=str(e))
        return {"success": False, "error": str(e)}


# =============================================================================
# AGENTFS KV STORE TOOLS - State Management
# =============================================================================

@mcp.tool()
async def set_state(key: str, value: str) -> dict:
    """
    Salva um estado/valor no KV Store do agent.

    Args:
        key: Chave única para o estado (ex: "user_preferences", "last_search")
        value: Valor a salvar (string, JSON será serializado)

    Returns:
        Confirmação do salvamento
    """
    try:
        from core.agentfs_manager import ensure_agentfs

        agentfs = await ensure_agentfs()
        await agentfs.kv.set(key, value)

        logger.info("Estado salvo", key=key, size=len(value))

        return {
            "success": True,
            "key": key,
            "size": len(value),
            "storage": "agentfs_kv"
        }
    except Exception as e:
        logger.error(f"Erro ao salvar estado: {e}", key=key, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_state(key: str) -> dict:
    """
    Recupera um estado/valor do KV Store.

    Args:
        key: Chave do estado

    Returns:
        Valor armazenado ou erro se não existir
    """
    try:
        from core.agentfs_manager import ensure_agentfs

        agentfs = await ensure_agentfs()
        value = await agentfs.kv.get(key)

        if value is None:
            return {
                "success": False,
                "key": key,
                "error": "Key not found"
            }

        logger.info("Estado recuperado", key=key)

        return {
            "success": True,
            "key": key,
            "value": value,
            "size": len(value) if value else 0
        }
    except Exception as e:
        logger.error(f"Erro ao recuperar estado: {e}", key=key, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_state(key: str) -> dict:
    """
    Remove um estado do KV Store.

    Args:
        key: Chave do estado a remover

    Returns:
        Confirmação da remoção
    """
    try:
        from core.agentfs_manager import ensure_agentfs

        agentfs = await ensure_agentfs()
        await agentfs.kv.delete(key)

        logger.info("Estado removido", key=key)

        return {"success": True, "key": key}
    except Exception as e:
        logger.error(f"Erro ao remover estado: {e}", key=key, error=str(e))
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_states(prefix: str = "") -> dict:
    """
    Lista todas as chaves no KV Store.

    Args:
        prefix: Prefixo opcional para filtrar chaves (ex: "user_" lista apenas chaves que começam com "user_")

    Returns:
        Lista de chaves disponíveis
    """
    try:
        from core.agentfs_manager import ensure_agentfs

        agentfs = await ensure_agentfs()
        keys = await agentfs.kv.list(prefix=prefix if prefix else None)

        logger.info("Estados listados", prefix=prefix, count=len(keys))

        return {
            "success": True,
            "prefix": prefix or "(all)",
            "keys": keys,
            "count": len(keys)
        }
    except Exception as e:
        logger.error(f"Erro ao listar estados: {e}", prefix=prefix, error=str(e))
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
