"""Claude RAG SDK - RAG capabilities for Claude with AgentFS integration.

A powerful SDK for building RAG (Retrieval-Augmented Generation) applications
with Claude, built on top of AgentFS for state management and audit trails.

Example:
    >>> from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
    >>>
    >>> async def main():
    ...     async with await ClaudeRAG.open(ClaudeRAGOptions(id='my-agent')) as rag:
    ...         # Add documents
    ...         await rag.ingest.add_document('manual.pdf')
    ...
    ...         # Search
    ...         results = await rag.search('What is RAG?')
    ...
    ...         # Ask questions
    ...         response = await rag.query('Explain RAG in detail')
    ...         print(response.answer)
    ...
    ...         # Use AgentFS features
    ...         await rag.kv.set('last_query', 'What is RAG?')
    ...         await rag.fs.write_file('/output/report.txt', response.answer)
"""

from .rag import ClaudeRAG
from .options import (
    ClaudeRAGOptions,
    EmbeddingModel,
    ChunkingStrategy,
    AgentModel,
)
from .search import (
    SearchEngine,
    SearchResult,
    HybridSearchResult,
)
from .ingest import (
    IngestEngine,
    IngestResult,
    Document,
)
from .agent import (
    AgentEngine,
    AgentResponse,
    StreamChunk,
    SimpleAgent,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "ClaudeRAG",
    "ClaudeRAGOptions",

    # Options/Config
    "EmbeddingModel",
    "ChunkingStrategy",
    "AgentModel",

    # Search
    "SearchEngine",
    "SearchResult",
    "HybridSearchResult",

    # Ingest
    "IngestEngine",
    "IngestResult",
    "Document",

    # Agent
    "AgentEngine",
    "AgentResponse",
    "StreamChunk",
    "SimpleAgent",
]
