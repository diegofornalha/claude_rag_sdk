# =============================================================================
# CONFIG - Configuração Centralizada do RAG Agent
# =============================================================================
# Gerencia modelos de embedding, chunking e outros parâmetros
# =============================================================================

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


class EmbeddingModel(str, Enum):
    """Modelos de embedding disponíveis."""
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    # Futuro: suporte a OpenAI
    # TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    # TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    @property
    def dimensions(self) -> int:
        """Retorna dimensionalidade do modelo."""
        dims = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            # "text-embedding-3-small": 1536,
            # "text-embedding-3-large": 3072,
        }
        return dims.get(self.value, 384)

    @property
    def short_name(self) -> str:
        """Retorna nome curto do modelo."""
        names = {
            "BAAI/bge-small-en-v1.5": "bge-small",
            "BAAI/bge-base-en-v1.5": "bge-base",
            "BAAI/bge-large-en-v1.5": "bge-large",
        }
        return names.get(self.value, "unknown")


class ChunkingStrategy(str, Enum):
    """Estratégias de chunking (re-export from chunker)."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class RAGConfig:
    """Configuração completa do sistema RAG."""

    # --- Embedding Model ---
    embedding_model: EmbeddingModel
    embedding_dimensions: int

    # --- Database ---
    db_path: Path

    # --- Chunking ---
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int

    # --- Search ---
    default_top_k: int
    fetch_k_multiplier: float  # Para reranking
    vector_weight: float  # Para hybrid search
    bm25_weight: float

    # --- Adaptive Top-K ---
    adaptive_topk_enabled: bool
    high_confidence_threshold: float
    low_confidence_threshold: float
    high_confidence_k: int
    low_confidence_multiplier: float

    # --- Cache ---
    embedding_cache_size: int
    embedding_cache_ttl: int  # segundos
    response_cache_size: int
    response_cache_ttl: int

    # --- Performance ---
    max_concurrent_embeddings: int

    @classmethod
    def from_env(cls, db_path: Optional[Path] = None) -> "RAGConfig":
        """
        Cria configuração a partir de variáveis de ambiente.

        Environment Variables:
            EMBEDDING_MODEL: Nome do modelo (default: bge-large)
            CHUNKING_STRATEGY: Estratégia de chunking (default: semantic)
            CHUNK_SIZE: Tamanho do chunk em tokens (default: 500)
            CHUNK_OVERLAP: Overlap em tokens (default: 50)
            DEFAULT_TOP_K: Top-k padrão (default: 5)
            ADAPTIVE_TOPK_ENABLED: Habilitar top-k adaptativo (default: true)
            EMBEDDING_CACHE_SIZE: Tamanho do cache de embeddings (default: 10000)
            RESPONSE_CACHE_SIZE: Tamanho do cache de respostas (default: 1000)
        """
        # Embedding model
        model_name = os.getenv("EMBEDDING_MODEL", "bge-small")
        model_map = {
            "bge-small": EmbeddingModel.BGE_SMALL,
            "bge-base": EmbeddingModel.BGE_BASE,
            "bge-large": EmbeddingModel.BGE_LARGE,
        }
        embedding_model = model_map.get(model_name, EmbeddingModel.BGE_SMALL)

        # Chunking
        chunking_strategy_name = os.getenv("CHUNKING_STRATEGY", "semantic")
        chunking_strategy = ChunkingStrategy(chunking_strategy_name)

        # Database
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "teste" / "documentos.db"

        return cls(
            # Embedding
            embedding_model=embedding_model,
            embedding_dimensions=embedding_model.dimensions,

            # Database
            db_path=db_path,

            # Chunking
            chunking_strategy=chunking_strategy,
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),

            # Search
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            fetch_k_multiplier=float(os.getenv("FETCH_K_MULTIPLIER", "2.0")),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),

            # Adaptive Top-K
            adaptive_topk_enabled=os.getenv("ADAPTIVE_TOPK_ENABLED", "true").lower() == "true",
            high_confidence_threshold=float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.7")),
            low_confidence_threshold=float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.5")),
            high_confidence_k=int(os.getenv("HIGH_CONFIDENCE_K", "2")),
            low_confidence_multiplier=float(os.getenv("LOW_CONFIDENCE_MULTIPLIER", "1.5")),

            # Cache
            embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "10000")),
            embedding_cache_ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600")),
            response_cache_size=int(os.getenv("RESPONSE_CACHE_SIZE", "1000")),
            response_cache_ttl=int(os.getenv("RESPONSE_CACHE_TTL", "300")),

            # Performance
            max_concurrent_embeddings=int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "10")),
        )

    def to_dict(self) -> dict:
        """Converte configuração para dict."""
        return {
            "embedding": {
                "model": self.embedding_model.value,
                "short_name": self.embedding_model.short_name,
                "dimensions": self.embedding_dimensions,
            },
            "chunking": {
                "strategy": self.chunking_strategy.value,
                "chunk_size": self.chunk_size,
                "overlap": self.chunk_overlap,
                "min_size": self.min_chunk_size,
            },
            "search": {
                "default_top_k": self.default_top_k,
                "fetch_k_multiplier": self.fetch_k_multiplier,
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight,
            },
            "adaptive_topk": {
                "enabled": self.adaptive_topk_enabled,
                "high_confidence_threshold": self.high_confidence_threshold,
                "low_confidence_threshold": self.low_confidence_threshold,
                "high_confidence_k": self.high_confidence_k,
                "low_confidence_multiplier": self.low_confidence_multiplier,
            },
            "cache": {
                "embedding_size": self.embedding_cache_size,
                "embedding_ttl": self.embedding_cache_ttl,
                "response_size": self.response_cache_size,
                "response_ttl": self.response_cache_ttl,
            },
        }


# Instância global
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Retorna configuração global (singleton)."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def reload_config():
    """Recarrega configuração do ambiente."""
    global _config
    _config = RAGConfig.from_env()
    return _config


if __name__ == "__main__":
    # Teste da configuração
    import json

    config = get_config()
    print("=== Configuração RAG ===\n")
    print(json.dumps(config.to_dict(), indent=2))
