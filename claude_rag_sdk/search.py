"""Search engine for ClaudeRAG SDK - Semantic and Hybrid search."""

import time
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import apsw
import sqlite_vec
from fastembed import TextEmbedding


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        doc_id: Document ID in the database
        source: Source file name
        content: Document content (truncated)
        similarity: Cosine similarity score (0-1)
        doc_type: Document type (pdf, docx, html, etc)
        rerank_score: Score after reranking (if enabled)
        rank: Final rank position
        metadata: Additional metadata
    """
    doc_id: int
    source: str
    content: str
    similarity: float
    doc_type: Optional[str] = None
    rerank_score: Optional[float] = None
    rank: Optional[int] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "content": self.content,
            "similarity": self.similarity,
            "type": self.doc_type,
            "rerank_score": self.rerank_score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class HybridSearchResult(SearchResult):
    """Search result with hybrid scores."""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0


class SearchEngine:
    """Semantic and hybrid search engine.

    Example:
        >>> engine = SearchEngine(db_path='rag.db', embedding_model='BAAI/bge-small-en-v1.5')
        >>> results = await engine.search('What is RAG?', top_k=5)
        >>> for r in results:
        ...     print(f"{r.source}: {r.similarity:.2f}")
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        enable_reranking: bool = True,
        enable_adaptive_topk: bool = True,
        enable_prompt_guard: bool = True,
        cache_embeddings: bool = True,
    ):
        """Initialize search engine.

        Args:
            db_path: Path to sqlite-vec database
            embedding_model: FastEmbed model name
            enable_reranking: Enable cross-encoder reranking
            enable_adaptive_topk: Enable adaptive top-k
            enable_prompt_guard: Enable prompt injection detection
            cache_embeddings: Cache embeddings in memory
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.enable_reranking = enable_reranking
        self.enable_adaptive_topk = enable_adaptive_topk
        self.enable_prompt_guard = enable_prompt_guard
        self.cache_embeddings = cache_embeddings

        # Lazy load
        self._model: Optional[TextEmbedding] = None
        self._embedding_cache: dict[str, list[float]] = {}
        self._reranker = None
        self._prompt_guard = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy load embedding model."""
        if self._model is None:
            self._model = TextEmbedding(self.embedding_model_name)
        return self._model

    def _get_connection(self) -> apsw.Connection:
        """Create connection with sqlite-vec loaded."""
        conn = apsw.Connection(str(self.db_path))
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        return conn

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding with optional caching."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        embeddings = list(self.model.embed([text]))
        embedding = embeddings[0].tolist()

        if self.cache_embeddings:
            self._embedding_cache[text] = embedding

        return embedding

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Convert embedding to bytes for sqlite-vec."""
        return sqlite_vec.serialize_float32(embedding)

    def _check_prompt_safety(self, query: str) -> tuple[bool, Optional[str]]:
        """Check if query is safe (no prompt injection)."""
        if not self.enable_prompt_guard:
            return True, None

        # Lazy load prompt guard
        if self._prompt_guard is None:
            try:
                from .core.prompt_guard import PromptGuard
                self._prompt_guard = PromptGuard(strict_mode=False)
            except ImportError:
                return True, None

        result = self._prompt_guard.scan(query)
        if not result.is_safe:
            return False, f"Query blocked: {result.threat_level.value}"
        return True, None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: Optional[bool] = None,
        use_adaptive: Optional[bool] = None,
        content_max_length: int = 1000,
    ) -> list[SearchResult]:
        """Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Override reranking setting
            use_adaptive: Override adaptive top-k setting
            content_max_length: Maximum content length in results

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If query is blocked by prompt guard
        """
        # Check safety
        is_safe, error = self._check_prompt_safety(query)
        if not is_safe:
            raise ValueError(error)

        # Settings
        reranking = use_reranking if use_reranking is not None else self.enable_reranking
        adaptive = use_adaptive if use_adaptive is not None else self.enable_adaptive_topk

        # Get embedding
        embedding = self._get_embedding(query)
        query_vec = self._serialize_embedding(embedding)

        # Fetch more results if reranking
        fetch_k = top_k * 2 if reranking else top_k

        # Query database
        conn = self._get_connection()
        cursor = conn.cursor()

        results = []
        for row in cursor.execute("""
            SELECT v.doc_id, v.distance, d.nome, d.conteudo, d.tipo
            FROM vec_documentos v
            JOIN documentos d ON d.id = v.doc_id
            WHERE v.embedding MATCH ? AND k = ?
        """, (query_vec, fetch_k)):
            doc_id, distance, nome, conteudo, tipo = row
            similarity = max(0, 1 - distance)

            results.append(SearchResult(
                doc_id=doc_id,
                source=nome,
                content=conteudo[:content_max_length] if conteudo else "",
                similarity=round(similarity, 4),
                doc_type=tipo,
            ))

        conn.close()

        # Apply adaptive top-k
        if adaptive and results:
            results = self._apply_adaptive_topk(results, top_k)
        else:
            results = results[:top_k]

        # Apply reranking
        if reranking and results:
            results = self._apply_reranking(query, results, top_k)

        # Set final ranks
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        content_max_length: int = 1000,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search (semantic + BM25).

        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector search (0-1), BM25 gets 1-vector_weight
            content_max_length: Maximum content length

        Returns:
            List of HybridSearchResult objects
        """
        # Check safety
        is_safe, error = self._check_prompt_safety(query)
        if not is_safe:
            raise ValueError(error)

        try:
            from .core.hybrid_search import HybridSearch
        except ImportError:
            # Fallback to regular search if hybrid not available
            results = await self.search(query, top_k, content_max_length=content_max_length)
            return [
                HybridSearchResult(
                    doc_id=r.doc_id,
                    source=r.source,
                    content=r.content,
                    similarity=r.similarity,
                    doc_type=r.doc_type,
                    vector_score=r.similarity,
                    bm25_score=0.0,
                    hybrid_score=r.similarity,
                    rank=r.rank,
                )
                for r in results
            ]

        hybrid = HybridSearch(
            str(self.db_path),
            vector_weight=vector_weight,
            bm25_weight=1 - vector_weight,
        )

        raw_results = hybrid.search(query, top_k=top_k)

        results = []
        for i, r in enumerate(raw_results):
            results.append(HybridSearchResult(
                doc_id=r.doc_id,
                source=r.nome,
                content=r.content[:content_max_length] if r.content else "",
                similarity=r.vector_score,
                doc_type=r.tipo,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                hybrid_score=r.hybrid_score,
                rank=i + 1,
            ))

        return results

    async def get_document(self, doc_id: int) -> Optional[dict]:
        """Get full document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None if not found
        """
        conn = self._get_connection()
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

        if not row:
            return None

        return {
            "id": row[0],
            "nome": row[1],
            "tipo": row[2],
            "conteudo": row[3],
            "caminho": row[4],
            "criado_em": str(row[5]) if row[5] else None,
        }

    async def list_sources(self) -> list[dict]:
        """List all document sources.

        Returns:
            List of source info dicts
        """
        conn = self._get_connection()
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

    async def count_documents(self) -> dict:
        """Count documents and embeddings.

        Returns:
            Stats dict with counts
        """
        conn = self._get_connection()
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
            "status": "ok" if total_docs == total_embeddings else "incompleto",
        }

    def _apply_adaptive_topk(
        self,
        results: list[SearchResult],
        requested_k: int,
    ) -> list[SearchResult]:
        """Apply adaptive top-k based on confidence scores."""
        if not results:
            return results

        top_similarity = results[0].similarity

        # Adjust k based on confidence
        if top_similarity > 0.7:
            # High confidence - fewer results needed
            adjusted_k = min(requested_k, max(2, requested_k // 2))
        elif top_similarity < 0.5:
            # Low confidence - more results needed
            adjusted_k = min(len(results), int(requested_k * 1.5))
        else:
            adjusted_k = requested_k

        return results[:adjusted_k]

    def _apply_reranking(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Apply reranking to results."""
        if not results:
            return results

        # Lazy load reranker
        if self._reranker is None:
            try:
                from .core.reranker import LightweightReranker
                self._reranker = LightweightReranker()
            except ImportError:
                # No reranking available
                return results[:top_k]

        docs_for_rerank = [
            (r.doc_id, r.content, r.similarity, {"source": r.source, "type": r.doc_type})
            for r in results
        ]

        reranked = self._reranker.rerank(query, docs_for_rerank, top_k=top_k)

        return [
            SearchResult(
                doc_id=r.doc_id,
                source=r.metadata["source"],
                content=r.content,
                similarity=r.original_score,
                doc_type=r.metadata.get("type"),
                rerank_score=r.rerank_score,
                rank=r.final_rank,
            )
            for r in reranked
        ]

    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()

    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._embedding_cache),
            "enabled": self.cache_embeddings,
        }
