"""Configuration options for ClaudeRAG SDK."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os


class EmbeddingModel(Enum):
    """Available embedding models."""
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # 384 dims, fastest
    BGE_BASE = "BAAI/bge-base-en-v1.5"    # 768 dims, balanced
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # 1024 dims, best quality


class ChunkingStrategy(Enum):
    """Document chunking strategies."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class AgentModel(Enum):
    """Claude models for agent queries."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


@dataclass
class ClaudeRAGOptions:
    """Configuration options for opening a ClaudeRAG instance.

    Attributes:
        id: Unique identifier for the agent. Creates storage at `.agentfs/{id}.db`
        path: Explicit path to the AgentFS database file
        db_path: Path to the RAG vector database (sqlite-vec)
        embedding_model: Model for generating embeddings
        chunk_size: Size of document chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        chunking_strategy: Strategy for splitting documents
        agent_model: Claude model for agent queries
        cache_ttl: Cache time-to-live in seconds
        cache_max_size: Maximum cache entries
        enable_reranking: Enable cross-encoder reranking
        enable_adaptive_topk: Enable adaptive top-k based on confidence
        enable_prompt_guard: Enable prompt injection detection
        circuit_breaker_threshold: Failures before circuit opens
        circuit_breaker_timeout: Seconds before circuit half-opens

    Example:
        >>> options = ClaudeRAGOptions(id='my-agent')
        >>> options = ClaudeRAGOptions(
        ...     id='my-agent',
        ...     embedding_model=EmbeddingModel.BGE_BASE,
        ...     enable_reranking=True
        ... )
    """
    # Identity
    id: Optional[str] = None
    path: Optional[str] = None

    # RAG Database
    db_path: Optional[str] = None

    # Embedding
    embedding_model: EmbeddingModel = EmbeddingModel.BGE_SMALL

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED

    # Agent
    agent_model: AgentModel = AgentModel.HAIKU
    system_prompt: Optional[str] = None

    # Cache
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000

    # Features
    enable_reranking: bool = True
    enable_adaptive_topk: bool = True
    enable_prompt_guard: bool = True
    enable_hybrid_search: bool = True

    # Resilience
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: float = 30.0

    # Search defaults
    default_top_k: int = 5
    vector_weight: float = 0.7  # For hybrid search

    def __post_init__(self):
        """Validate and set defaults."""
        if not self.id and not self.path:
            raise ValueError("ClaudeRAGOptions requires at least 'id' or 'path'")

        # Set default db_path based on id
        if not self.db_path and self.id:
            agentfs_dir = Path.home() / ".claude" / ".agentfs"
            agentfs_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(agentfs_dir / f"{self.id}_rag.db")

    def get_agentfs_path(self) -> str:
        """Get the AgentFS database path."""
        if self.path:
            return self.path
        agentfs_dir = Path.home() / ".claude" / ".agentfs"
        agentfs_dir.mkdir(parents=True, exist_ok=True)
        return str(agentfs_dir / f"{self.id}.db")

    def to_dict(self) -> dict:
        """Convert options to dictionary."""
        return {
            "id": self.id,
            "path": self.path,
            "db_path": self.db_path,
            "embedding_model": self.embedding_model.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_strategy": self.chunking_strategy.value,
            "agent_model": self.agent_model.value,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "enable_reranking": self.enable_reranking,
            "enable_adaptive_topk": self.enable_adaptive_topk,
            "enable_prompt_guard": self.enable_prompt_guard,
            "enable_hybrid_search": self.enable_hybrid_search,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "default_top_k": self.default_top_k,
            "vector_weight": self.vector_weight,
        }

    @classmethod
    def from_env(cls, prefix: str = "CLAUDE_RAG_") -> "ClaudeRAGOptions":
        """Create options from environment variables.

        Example:
            CLAUDE_RAG_ID=my-agent
            CLAUDE_RAG_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
            CLAUDE_RAG_ENABLE_RERANKING=true
        """
        def get_env(key: str, default=None):
            return os.getenv(f"{prefix}{key}", default)

        def get_bool(key: str, default: bool) -> bool:
            val = get_env(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            val = get_env(key)
            return int(val) if val else default

        def get_float(key: str, default: float) -> float:
            val = get_env(key)
            return float(val) if val else default

        embedding_model_str = get_env("EMBEDDING_MODEL")
        embedding_model = EmbeddingModel.BGE_SMALL
        if embedding_model_str:
            for model in EmbeddingModel:
                if model.value == embedding_model_str:
                    embedding_model = model
                    break

        return cls(
            id=get_env("ID"),
            path=get_env("PATH"),
            db_path=get_env("DB_PATH"),
            embedding_model=embedding_model,
            chunk_size=get_int("CHUNK_SIZE", 500),
            chunk_overlap=get_int("CHUNK_OVERLAP", 50),
            agent_model=AgentModel(get_env("AGENT_MODEL", "haiku")),
            cache_ttl=get_int("CACHE_TTL", 3600),
            cache_max_size=get_int("CACHE_MAX_SIZE", 1000),
            enable_reranking=get_bool("ENABLE_RERANKING", True),
            enable_adaptive_topk=get_bool("ENABLE_ADAPTIVE_TOPK", True),
            enable_prompt_guard=get_bool("ENABLE_PROMPT_GUARD", True),
            enable_hybrid_search=get_bool("ENABLE_HYBRID_SEARCH", True),
            default_top_k=get_int("DEFAULT_TOP_K", 5),
            vector_weight=get_float("VECTOR_WEIGHT", 0.7),
        )
