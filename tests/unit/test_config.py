# =============================================================================
# TESTES - Config Module
# =============================================================================
# Testes unitários para configuração centralizada
# =============================================================================

import os
from unittest.mock import patch


class TestEmbeddingModel:
    """Testes para EmbeddingModel enum."""

    def test_bge_small_dimensions(self):
        """Verifica dimensões do BGE Small."""
        from claude_rag_sdk.core.config import EmbeddingModel

        assert EmbeddingModel.BGE_SMALL.dimensions == 384

    def test_bge_base_dimensions(self):
        """Verifica dimensões do BGE Base."""
        from claude_rag_sdk.core.config import EmbeddingModel

        assert EmbeddingModel.BGE_BASE.dimensions == 768

    def test_bge_large_dimensions(self):
        """Verifica dimensões do BGE Large."""
        from claude_rag_sdk.core.config import EmbeddingModel

        assert EmbeddingModel.BGE_LARGE.dimensions == 1024

    def test_short_name(self):
        """Verifica short_name dos modelos."""
        from claude_rag_sdk.core.config import EmbeddingModel

        assert EmbeddingModel.BGE_SMALL.short_name == "bge-small"
        assert EmbeddingModel.BGE_BASE.short_name == "bge-base"
        assert EmbeddingModel.BGE_LARGE.short_name == "bge-large"


class TestChunkingStrategy:
    """Testes para ChunkingStrategy enum."""

    def test_all_strategies_exist(self):
        """Verifica todas as estratégias existem."""
        from claude_rag_sdk.core.config import ChunkingStrategy

        assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"


class TestRAGConfig:
    """Testes para RAGConfig dataclass."""

    def test_from_env_defaults(self):
        """Verifica valores padrão do from_env."""
        from claude_rag_sdk.core.config import ChunkingStrategy, EmbeddingModel, RAGConfig

        with patch.dict(os.environ, {}, clear=True):
            config = RAGConfig.from_env()

        assert config.embedding_model == EmbeddingModel.BGE_SMALL
        assert config.chunking_strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.default_top_k == 5

    def test_from_env_custom_values(self):
        """Verifica valores customizados via env vars."""
        from claude_rag_sdk.core.config import EmbeddingModel, RAGConfig

        env_vars = {
            "EMBEDDING_MODEL": "bge-large",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "100",
            "DEFAULT_TOP_K": "10",
            "ADAPTIVE_TOPK_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = RAGConfig.from_env()

        assert config.embedding_model == EmbeddingModel.BGE_LARGE
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.default_top_k == 10
        assert config.adaptive_topk_enabled is False

    def test_to_dict_structure(self):
        """Verifica estrutura do to_dict."""
        from claude_rag_sdk.core.config import RAGConfig

        config = RAGConfig.from_env()
        data = config.to_dict()

        # Verificar seções existem
        assert "embedding" in data
        assert "chunking" in data
        assert "search" in data
        assert "adaptive_topk" in data
        assert "cache" in data

        # Verificar campos
        assert "model" in data["embedding"]
        assert "dimensions" in data["embedding"]
        assert "chunk_size" in data["chunking"]

    def test_embedding_dimensions_match_model(self):
        """Verifica dimensões correspondem ao modelo."""
        from claude_rag_sdk.core.config import RAGConfig

        config = RAGConfig.from_env()

        assert config.embedding_dimensions == config.embedding_model.dimensions


class TestGetConfig:
    """Testes para get_config singleton."""

    def test_returns_same_instance(self):
        """Verifica singleton retorna mesma instância."""
        from claude_rag_sdk.core.config import get_config, reload_config

        # Recarregar para limpar cache
        reload_config()

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reload_creates_new_instance(self):
        """Verifica reload cria nova instância."""
        from claude_rag_sdk.core.config import get_config, reload_config

        get_config()
        config2 = reload_config()

        # Após reload, get_config deve retornar a nova
        config3 = get_config()

        assert config2 is config3


class TestConfigValidation:
    """Testes de validação de configuração."""

    def test_invalid_embedding_model_falls_back(self):
        """Verifica fallback para modelo inválido."""
        from claude_rag_sdk.core.config import EmbeddingModel, RAGConfig

        env_vars = {"EMBEDDING_MODEL": "invalid-model-name"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = RAGConfig.from_env()

        # Deve usar fallback (bge-small)
        assert config.embedding_model == EmbeddingModel.BGE_SMALL

    def test_numeric_env_vars_parsed(self):
        """Verifica parsing de números das env vars."""
        from claude_rag_sdk.core.config import RAGConfig

        env_vars = {
            "CHUNK_SIZE": "750",
            "DEFAULT_TOP_K": "7",
            "VECTOR_WEIGHT": "0.8",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = RAGConfig.from_env()

        assert config.chunk_size == 750
        assert config.default_top_k == 7
        assert config.vector_weight == 0.8

    def test_boolean_env_vars_parsed(self):
        """Verifica parsing de booleanos das env vars."""
        from claude_rag_sdk.core.config import RAGConfig

        # Teste true
        with patch.dict(os.environ, {"ADAPTIVE_TOPK_ENABLED": "true"}, clear=True):
            config = RAGConfig.from_env()
            assert config.adaptive_topk_enabled is True

        # Teste false
        with patch.dict(os.environ, {"ADAPTIVE_TOPK_ENABLED": "false"}, clear=True):
            config = RAGConfig.from_env()
            assert config.adaptive_topk_enabled is False

        # Teste TRUE (case insensitive)
        with patch.dict(os.environ, {"ADAPTIVE_TOPK_ENABLED": "TRUE"}, clear=True):
            config = RAGConfig.from_env()
            assert config.adaptive_topk_enabled is True
