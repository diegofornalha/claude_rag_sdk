# =============================================================================
# TESTES - Reranker Module
# =============================================================================
# Testes unitários para re-ranking de resultados de busca
# =============================================================================

from unittest.mock import MagicMock, patch

import pytest


class TestRerankResult:
    """Testes para RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Verifica criação de RerankResult."""
        from claude_rag_sdk.core.reranker import RerankResult

        result = RerankResult(
            doc_id=1,
            content="Test content",
            original_score=0.8,
            rerank_score=0.9,
            final_rank=1,
            metadata={"source": "test"},
        )

        assert result.doc_id == 1
        assert result.content == "Test content"
        assert result.original_score == 0.8
        assert result.rerank_score == 0.9
        assert result.final_rank == 1
        assert result.metadata == {"source": "test"}


class TestLightweightReranker:
    """Testes para LightweightReranker."""

    def test_initialization(self):
        """Verifica inicialização."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        assert reranker.exact_match_weight == 0.3
        assert reranker.term_coverage_weight == 0.2
        assert reranker.position_weight == 0.1

    def test_empty_documents(self):
        """Verifica com lista vazia."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()
        results = reranker.rerank("test query", [], top_k=5)

        assert results == []

    def test_basic_reranking(self):
        """Verifica reranking básico."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "This is about apples and oranges", 0.7, {"name": "doc1"}),
            (2, "Apples are delicious fruits", 0.8, {"name": "doc2"}),
            (3, "Nothing related here", 0.9, {"name": "doc3"}),
        ]

        results = reranker.rerank("apples", documents, top_k=3)

        assert len(results) == 3
        # Resultados devem ter ranks atribuídos
        assert all(r.final_rank > 0 for r in results)
        # Ranks devem ser sequenciais
        ranks = [r.final_rank for r in results]
        assert ranks == [1, 2, 3]

    def test_exact_match_boost(self):
        """Verifica boost por match exato."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "Document about machine learning basics", 0.7, {}),
            (2, "machine learning is a subfield of AI", 0.6, {}),  # exact match
        ]

        results = reranker.rerank("machine learning", documents, top_k=2)

        # Doc com match exato deve ter score maior
        doc2_result = next(r for r in results if r.doc_id == 2)
        next(r for r in results if r.doc_id == 1)

        assert doc2_result.rerank_score > doc2_result.original_score

    def test_term_coverage_boost(self):
        """Verifica boost por cobertura de termos."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "Only contains python", 0.5, {}),
            (2, "Contains python programming language", 0.5, {}),  # mais termos
        ]

        results = reranker.rerank("python programming language", documents, top_k=2)

        # Doc com mais termos deve ter score maior
        doc2_result = next(r for r in results if r.doc_id == 2)
        doc1_result = next(r for r in results if r.doc_id == 1)

        assert doc2_result.rerank_score >= doc1_result.rerank_score

    def test_top_k_limiting(self):
        """Verifica limite de top_k."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [(i, f"Document {i} content", 0.5, {}) for i in range(10)]

        results = reranker.rerank("document", documents, top_k=3)

        assert len(results) == 3

    def test_preserves_metadata(self):
        """Verifica preservação de metadata."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "Test content", 0.8, {"source": "file.txt", "page": 5}),
        ]

        results = reranker.rerank("test", documents, top_k=1)

        assert results[0].metadata == {"source": "file.txt", "page": 5}

    def test_case_insensitive_matching(self):
        """Verifica matching case-insensitive."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "PYTHON PROGRAMMING", 0.5, {}),
            (2, "python programming", 0.5, {}),
        ]

        results = reranker.rerank("Python Programming", documents, top_k=2)

        # Ambos devem ter boost similar
        scores = [r.rerank_score for r in results]
        assert scores[0] == pytest.approx(scores[1], rel=0.01)


class TestCrossEncoderReranker:
    """Testes para CrossEncoderReranker."""

    def test_initialization(self):
        """Verifica inicialização."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="test-model")

        assert reranker.model_name == "test-model"
        assert reranker._model is None
        assert reranker._load_attempted is False

    def test_lazy_loading(self):
        """Verifica carregamento sob demanda."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Modelo não deve ser carregado até necessário
        assert reranker._model is None
        assert reranker._load_attempted is False

    def test_fallback_without_sentence_transformers(self):
        """Verifica fallback quando sentence-transformers não está instalado."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        documents = [
            (1, "Test about python programming", 0.8, {}),
            (2, "Another document about programming", 0.7, {}),
        ]

        # Deve funcionar mesmo sem modelo (usa fallback)
        results = reranker.rerank("python programming", documents, top_k=2)

        assert len(results) == 2
        assert all(r.final_rank > 0 for r in results)

    def test_empty_documents(self):
        """Verifica com lista vazia."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test query", [], top_k=5)

        assert results == []

    @patch("claude_rag_sdk.core.reranker.CrossEncoderReranker._load_model")
    def test_with_mocked_model(self, mock_load_model):
        """Verifica com modelo mockado."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.6]
        mock_load_model.return_value = mock_model

        reranker = CrossEncoderReranker()

        documents = [
            (1, "Document one", 0.5, {}),
            (2, "Document two", 0.5, {}),
        ]

        reranker.rerank("query", documents, top_k=2)

        # Verificar que predict foi chamado
        mock_load_model.assert_called_once()

    def test_fallback_scoring_term_boost(self):
        """Verifica boost de termos no fallback."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        # Forçar uso de fallback
        reranker._load_attempted = True
        reranker._model = None

        documents = [
            (1, "Contains all search terms python machine learning", 0.5, {}),
            (2, "No matching terms here", 0.5, {}),
        ]

        results = reranker.rerank("python machine learning", documents, top_k=2)

        # Doc com termos deve ter score maior
        doc1 = next(r for r in results if r.doc_id == 1)
        doc2 = next(r for r in results if r.doc_id == 2)

        assert doc1.rerank_score > doc2.rerank_score

    def test_fallback_exact_phrase_boost(self):
        """Verifica boost de frase exata no fallback."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._load_attempted = True
        reranker._model = None

        documents = [
            (1, "machine learning is powerful", 0.5, {}),  # exact phrase
            (2, "learning about machine something", 0.5, {}),  # terms but not phrase
        ]

        results = reranker.rerank("machine learning", documents, top_k=2)

        doc1 = next(r for r in results if r.doc_id == 1)
        doc2 = next(r for r in results if r.doc_id == 2)

        assert doc1.rerank_score > doc2.rerank_score


class TestCreateReranker:
    """Testes para factory function."""

    def test_create_lightweight(self):
        """Verifica criação de LightweightReranker."""
        from claude_rag_sdk.core.reranker import LightweightReranker, create_reranker

        reranker = create_reranker(use_cross_encoder=False)

        assert isinstance(reranker, LightweightReranker)

    def test_create_cross_encoder(self):
        """Verifica criação de CrossEncoderReranker."""
        from claude_rag_sdk.core.reranker import CrossEncoderReranker, create_reranker

        reranker = create_reranker(use_cross_encoder=True)

        assert isinstance(reranker, CrossEncoderReranker)

    def test_default_is_lightweight(self):
        """Verifica que padrão é LightweightReranker."""
        from claude_rag_sdk.core.reranker import LightweightReranker, create_reranker

        reranker = create_reranker()

        assert isinstance(reranker, LightweightReranker)


class TestRerankingSorting:
    """Testes para ordenação de resultados."""

    def test_results_sorted_by_score(self):
        """Verifica que resultados são ordenados por score."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "No match here", 0.5, {}),
            (2, "Exact match for query here", 0.5, {}),
            (3, "query is mentioned once", 0.5, {}),
        ]

        results = reranker.rerank("query", documents, top_k=3)

        # Scores devem estar em ordem decrescente
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self):
        """Verifica que ranks são sequenciais."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [(i, f"Document {i}", 0.5, {}) for i in range(5)]

        results = reranker.rerank("document", documents, top_k=5)

        ranks = [r.final_rank for r in results]
        assert ranks == [1, 2, 3, 4, 5]


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_single_document(self):
        """Verifica com único documento."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [(1, "Single document", 0.8, {})]

        results = reranker.rerank("single", documents, top_k=1)

        assert len(results) == 1
        assert results[0].final_rank == 1

    def test_top_k_larger_than_documents(self):
        """Verifica quando top_k > número de documentos."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "Doc one", 0.8, {}),
            (2, "Doc two", 0.7, {}),
        ]

        results = reranker.rerank("doc", documents, top_k=10)

        assert len(results) == 2

    def test_empty_content(self):
        """Verifica com conteúdo vazio."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "", 0.8, {}),
            (2, "Has content", 0.7, {}),
        ]

        results = reranker.rerank("content", documents, top_k=2)

        assert len(results) == 2

    def test_special_characters_in_query(self):
        """Verifica query com caracteres especiais."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "C++ programming language", 0.8, {}),
            (2, "JavaScript framework", 0.7, {}),
        ]

        # Não deve quebrar com caracteres especiais
        results = reranker.rerank("C++ language", documents, top_k=2)

        assert len(results) == 2

    def test_unicode_content(self):
        """Verifica conteúdo com unicode."""
        from claude_rag_sdk.core.reranker import LightweightReranker

        reranker = LightweightReranker()

        documents = [
            (1, "Programação em português com acentos", 0.8, {}),
            (2, "日本語テキスト", 0.7, {}),
        ]

        results = reranker.rerank("português", documents, top_k=2)

        assert len(results) == 2


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_reranker_tests():
    """Executa testes manualmente."""

    print("\n" + "=" * 60)
    print("TESTES - RERANKER")
    print("=" * 60 + "\n")

    exit_code = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
        ]
    )

    return exit_code == 0


if __name__ == "__main__":
    import sys

    success = run_reranker_tests()
    sys.exit(0 if success else 1)
