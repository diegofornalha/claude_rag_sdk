"""Main ClaudeRAG class - unified interface for RAG operations."""

import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union

from agentfs_sdk import AgentFS, AgentFSOptions

from .options import ClaudeRAGOptions
from .search import SearchEngine, SearchResult, HybridSearchResult
from .ingest import IngestEngine, IngestResult
from .agent import AgentEngine, AgentResponse, StreamChunk


class ClaudeRAG:
    """ClaudeRAG - RAG SDK for Claude with AgentFS integration.

    Provides a unified interface for:
    - Document ingestion and indexing
    - Semantic and hybrid search
    - AI-powered Q&A with citations
    - AgentFS filesystem and KV store

    Example:
        >>> from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
        >>>
        >>> async with await ClaudeRAG.open(ClaudeRAGOptions(id='my-agent')) as rag:
        ...     # AgentFS features
        ...     await rag.kv.set('config', {'theme': 'dark'})
        ...     await rag.fs.write_file('/notes.txt', 'Hello')
        ...
        ...     # RAG features
        ...     await rag.ingest.add_document('manual.pdf')
        ...     results = await rag.search('What is RAG?')
        ...
        ...     # Agent features
        ...     response = await rag.query('Explain RAG in detail')
        ...     print(response.answer)
    """

    def __init__(
        self,
        agentfs: AgentFS,
        options: ClaudeRAGOptions,
        search_engine: SearchEngine,
        ingest_engine: IngestEngine,
        agent_engine: Optional[AgentEngine] = None,
    ):
        """Private constructor - use ClaudeRAG.open() instead."""
        self._agentfs = agentfs
        self._options = options
        self._search = search_engine
        self._ingest = ingest_engine
        self._agent = agent_engine

    @staticmethod
    async def open(options: ClaudeRAGOptions) -> "ClaudeRAG":
        """Open a ClaudeRAG instance.

        Args:
            options: Configuration options

        Returns:
            Fully initialized ClaudeRAG instance

        Example:
            >>> rag = await ClaudeRAG.open(ClaudeRAGOptions(id='my-agent'))
            >>> # or with custom options
            >>> rag = await ClaudeRAG.open(ClaudeRAGOptions(
            ...     id='my-agent',
            ...     embedding_model=EmbeddingModel.BGE_BASE,
            ...     enable_reranking=True,
            ... ))
        """
        # Open AgentFS
        agentfs_path = options.get_agentfs_path()
        agentfs = await AgentFS.open(AgentFSOptions(
            id=options.id,
            path=agentfs_path if options.path else None,
        ))

        # Initialize search engine
        search_engine = SearchEngine(
            db_path=options.db_path,
            embedding_model=options.embedding_model.value,
            enable_reranking=options.enable_reranking,
            enable_adaptive_topk=options.enable_adaptive_topk,
            enable_prompt_guard=options.enable_prompt_guard,
        )

        # Initialize ingest engine
        ingest_engine = IngestEngine(
            db_path=options.db_path,
            embedding_model=options.embedding_model.value,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            chunking_strategy=options.chunking_strategy,
        )

        # Initialize agent engine (optional, may not have MCP server)
        agent_engine = AgentEngine(
            options=options,
            mcp_server_path=None,  # Will use SimpleAgent fallback
        )

        return ClaudeRAG(
            agentfs=agentfs,
            options=options,
            search_engine=search_engine,
            ingest_engine=ingest_engine,
            agent_engine=agent_engine,
        )

    # =========================================================================
    # AgentFS Passthrough Properties
    # =========================================================================

    @property
    def kv(self):
        """AgentFS KV Store for state management.

        Example:
            >>> await rag.kv.set('user:123', {'name': 'Alice'})
            >>> user = await rag.kv.get('user:123')
            >>> keys = await rag.kv.list('user:')
        """
        return self._agentfs.kv

    @property
    def fs(self):
        """AgentFS Filesystem for file operations.

        Example:
            >>> await rag.fs.write_file('/output/report.pdf', content)
            >>> files = await rag.fs.readdir('/output')
            >>> content = await rag.fs.read_file('/output/report.pdf')
        """
        return self._agentfs.fs

    @property
    def tools(self):
        """AgentFS Tool tracking for audit.

        Example:
            >>> call_id = await rag.tools.start('search', {'query': 'RAG'})
            >>> await rag.tools.success(call_id, {'results': [...]})
            >>> stats = await rag.tools.get_stats()
        """
        return self._agentfs.tools

    # =========================================================================
    # RAG Properties
    # =========================================================================

    @property
    def ingest(self) -> IngestEngine:
        """Ingestion engine for adding documents.

        Example:
            >>> result = await rag.ingest.add_document('manual.pdf')
            >>> result = await rag.ingest.add_text('Content...', source='note.txt')
        """
        return self._ingest

    @property
    def search_engine(self) -> SearchEngine:
        """Direct access to search engine."""
        return self._search

    @property
    def options(self) -> ClaudeRAGOptions:
        """Current configuration options."""
        return self._options

    # =========================================================================
    # Search Methods
    # =========================================================================

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
    ) -> list[SearchResult]:
        """Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results (default from options)
            use_reranking: Override reranking setting

        Returns:
            List of SearchResult objects

        Example:
            >>> results = await rag.search('What is RAG?')
            >>> for r in results:
            ...     print(f"{r.source}: {r.similarity:.2f}")
            ...     print(r.content[:200])
        """
        k = top_k or self._options.default_top_k
        return await self._search.search(
            query=query,
            top_k=k,
            use_reranking=use_reranking,
        )

    async def search_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search (semantic + BM25).

        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector search (0-1)

        Returns:
            List of HybridSearchResult objects

        Example:
            >>> results = await rag.search_hybrid('What is RAG?', vector_weight=0.7)
            >>> for r in results:
            ...     print(f"{r.source}: vector={r.vector_score:.2f}, bm25={r.bm25_score:.2f}")
        """
        k = top_k or self._options.default_top_k
        weight = vector_weight or self._options.vector_weight
        return await self._search.search_hybrid(
            query=query,
            top_k=k,
            vector_weight=weight,
        )

    async def get_document(self, doc_id: int) -> Optional[dict]:
        """Get full document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None
        """
        return await self._search.get_document(doc_id)

    async def list_sources(self) -> list[dict]:
        """List all document sources.

        Returns:
            List of source info dicts
        """
        return await self._search.list_sources()

    # =========================================================================
    # Agent Methods
    # =========================================================================

    async def query(self, question: str) -> AgentResponse:
        """Ask a question using RAG + Claude.

        Args:
            question: Question to ask

        Returns:
            AgentResponse with answer, citations, and confidence

        Example:
            >>> response = await rag.query('What are the benefits of RAG?')
            >>> print(response.answer)
            >>> for citation in response.citations:
            ...     print(f"Source: {citation['source']}")
        """
        # Track tool call
        call_id = await self.tools.start("query", {"question": question[:100]})

        try:
            # Use AgentEngine with Claude Agent SDK (uses Claude Code subscription)
            if self._agent is None:
                from .agent import AgentEngine
                self._agent = AgentEngine(
                    options=self._options,
                    mcp_server_path=None,  # No MCP server needed
                )
            response = await self._agent.query(question)

            await self.tools.success(call_id, {
                "citations": len(response.citations),
                "confidence": response.confidence,
            })

            return response

        except Exception as e:
            await self.tools.error(call_id, str(e))
            raise

    async def query_stream(self, question: str) -> AsyncIterator[StreamChunk]:
        """Ask a question with streaming response.

        Args:
            question: Question to ask

        Yields:
            StreamChunk objects

        Example:
            >>> async for chunk in rag.query_stream('Explain RAG'):
            ...     if chunk.text:
            ...         print(chunk.text, end='')
        """
        if self._agent:
            async for chunk in self._agent.query_stream(question):
                yield chunk

    # =========================================================================
    # Stats & Health
    # =========================================================================

    async def stats(self) -> dict:
        """Get comprehensive statistics.

        Returns:
            Stats dict with documents, embeddings, cache info
        """
        ingest_stats = self._ingest.stats
        cache_stats = self._search.cache_stats
        tool_stats = await self.tools.get_stats()

        return {
            "documents": ingest_stats,
            "cache": cache_stats,
            "tools": [
                {"name": s.name, "calls": s.total_calls, "avg_ms": s.avg_duration_ms}
                for s in tool_stats
            ],
            "options": self._options.to_dict(),
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close all connections."""
        await self._agentfs.close()

    async def __aenter__(self) -> "ClaudeRAG":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Context manager exit."""
        await self.close()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def add_document(
        self,
        path: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """Convenience method to add a document.

        Args:
            path: Path to document
            metadata: Optional metadata

        Returns:
            IngestResult
        """
        return await self._ingest.add_document(path, metadata)

    async def add_text(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """Convenience method to add text content.

        Args:
            content: Text content
            source: Source identifier
            metadata: Optional metadata

        Returns:
            IngestResult
        """
        return await self._ingest.add_text(content, source, metadata=metadata)

    def clear_cache(self):
        """Clear all caches."""
        self._search.clear_cache()
