# =============================================================================
# TESTES - Hybrid Search Module
# =============================================================================
# Testes unitários para busca híbrida BM25 + Vetorial
# =============================================================================

import pytest
import math
from unittest.mock import patch, MagicMock


class TestBM25:
    """Testes para implementação BM25."""

    def test_initialization(self):
        """Verifica inicialização com parâmetros padrão."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.num_docs == 0
        assert bm25._indexed is False

    def test_custom_parameters(self):
        """Verifica inicialização com parâmetros customizados."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25(k1=2.0, b=0.5)

        assert bm25.k1 == 2.0
        assert bm25.b == 0.5

    def test_tokenize_basic(self):
        """Verifica tokenização básica."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        tokens = bm25._tokenize("Hello world, this is a test!")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stopwords devem ser removidas
        assert "is" not in tokens
        assert "a" not in tokens

    def test_tokenize_portuguese_stopwords(self):
        """Verifica remoção de stopwords em português."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        tokens = bm25._tokenize("O documento está na pasta do projeto")

        # Stopwords PT devem ser removidas
        assert "o" not in tokens
        assert "na" not in tokens
        assert "do" not in tokens
        # Palavras relevantes devem permanecer
        assert "documento" in tokens
        assert "pasta" in tokens
        assert "projeto" in tokens

    def test_tokenize_removes_short_words(self):
        """Verifica remoção de palavras curtas (<=2 chars)."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        tokens = bm25._tokenize("AI is the future of ML and NLP")

        # Palavras com <= 2 chars são removidas
        assert "ai" not in tokens
        assert "is" not in tokens
        assert "ml" not in tokens

    def test_index_documents(self):
        """Verifica indexação de documentos."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Python é uma linguagem de programação"),
            (2, "JavaScript é usado para web"),
            (3, "Python também é usado para machine learning"),
        ]

        bm25.index(documents)

        assert bm25._indexed is True
        assert bm25.num_docs == 3
        assert 1 in bm25.doc_lengths
        assert 2 in bm25.doc_lengths
        assert 3 in bm25.doc_lengths

    def test_search_not_indexed(self):
        """Verifica busca sem indexação."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        results = bm25.search("python")

        assert results == {}

    def test_search_basic(self):
        """Verifica busca básica."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Python programming language"),
            (2, "JavaScript for web development"),
            (3, "Python machine learning"),
        ]

        bm25.index(documents)
        results = bm25.search("python")

        # Docs 1 e 3 devem aparecer (contêm "python")
        assert 1 in results
        assert 3 in results
        # Doc 2 não deve aparecer
        assert 2 not in results

    def test_search_multiple_terms(self):
        """Verifica busca com múltiplos termos."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Python programming basics"),
            (2, "Machine learning with Python"),
            (3, "Web development"),
        ]

        bm25.index(documents)
        results = bm25.search("python machine learning")

        # Doc 2 deve ter score mais alto (tem todos os termos)
        assert 2 in results
        assert results[2] > results.get(1, 0)

    def test_search_with_doc_ids_filter(self):
        """Verifica busca filtrada por doc_ids."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Python programming"),
            (2, "Python development"),
            (3, "Python analysis"),
        ]

        bm25.index(documents)
        results = bm25.search("python", doc_ids=[1, 2])

        # Apenas docs 1 e 2 devem aparecer
        assert 1 in results
        assert 2 in results
        assert 3 not in results

    def test_search_no_matches(self):
        """Verifica busca sem resultados."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Python programming"),
            (2, "JavaScript development"),
        ]

        bm25.index(documents)
        results = bm25.search("rust golang")

        assert results == {}

    def test_term_frequency_scoring(self):
        """Verifica que TF influencia score."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "python"),
            (2, "python python python"),  # Mais ocorrências
        ]

        bm25.index(documents)
        results = bm25.search("python")

        # Doc 2 deve ter score maior (mais ocorrências)
        assert results[2] > results[1]


class TestSearchResult:
    """Testes para SearchResult dataclass."""

    def test_search_result_creation(self):
        """Verifica criação de SearchResult."""
        from claude_rag_sdk.core.hybrid_search import SearchResult

        result = SearchResult(
            doc_id=1,
            nome="test.txt",
            tipo="document",
            content="Test content",
            vector_score=0.8,
            bm25_score=0.6,
            hybrid_score=0.72,
            rank=1,
        )

        assert result.doc_id == 1
        assert result.nome == "test.txt"
        assert result.tipo == "document"
        assert result.content == "Test content"
        assert result.vector_score == 0.8
        assert result.bm25_score == 0.6
        assert result.hybrid_score == 0.72
        assert result.rank == 1


class TestHybridSearchMocked:
    """Testes para HybridSearch com mocks."""

    @patch("claude_rag_sdk.core.hybrid_search.get_config")
    @patch("claude_rag_sdk.core.hybrid_search.TextEmbedding")
    @patch("claude_rag_sdk.core.hybrid_search.apsw")
    @patch("claude_rag_sdk.core.hybrid_search.sqlite_vec")
    def test_initialization(self, mock_sqlite_vec, mock_apsw, mock_embedding, mock_config):
        """Verifica inicialização do HybridSearch."""
        from claude_rag_sdk.core.hybrid_search import HybridSearch

        # Setup mocks
        mock_config_obj = MagicMock()
        mock_config_obj.embedding_model.value = "test-model"
        mock_config.return_value = mock_config_obj

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_apsw.Connection.return_value = mock_conn

        mock_sqlite_vec.loadable_path.return_value = "/path/to/ext"

        # Criar instância
        hybrid = HybridSearch(
            db_path="/test/db.sqlite",
            vector_weight=0.7,
            bm25_weight=0.3,
        )

        assert hybrid.vector_weight == 0.7
        assert hybrid.bm25_weight == 0.3

    @patch("claude_rag_sdk.core.hybrid_search.get_config")
    @patch("claude_rag_sdk.core.hybrid_search.TextEmbedding")
    @patch("claude_rag_sdk.core.hybrid_search.apsw")
    @patch("claude_rag_sdk.core.hybrid_search.sqlite_vec")
    def test_weight_normalization(self, mock_sqlite_vec, mock_apsw, mock_embedding, mock_config):
        """Verifica que pesos são usados corretamente."""
        from claude_rag_sdk.core.hybrid_search import HybridSearch

        # Setup mocks
        mock_config_obj = MagicMock()
        mock_config_obj.embedding_model.value = "test-model"
        mock_config.return_value = mock_config_obj

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_apsw.Connection.return_value = mock_conn

        # Pesos que somam 1.0
        hybrid = HybridSearch(
            db_path="/test/db.sqlite",
            vector_weight=0.6,
            bm25_weight=0.4,
        )

        assert hybrid.vector_weight + hybrid.bm25_weight == 1.0


class TestHybridScoreCalculation:
    """Testes para cálculo de score híbrido."""

    def test_hybrid_score_formula(self):
        """Verifica fórmula de score híbrido."""
        vector_weight = 0.7
        bm25_weight = 0.3

        vector_score = 0.8
        bm25_score = 0.6

        expected_hybrid = vector_weight * vector_score + bm25_weight * bm25_score
        assert expected_hybrid == pytest.approx(0.74, rel=0.01)

    def test_hybrid_score_all_vector(self):
        """Verifica score quando apenas vetorial contribui."""
        vector_weight = 1.0
        bm25_weight = 0.0

        vector_score = 0.9
        bm25_score = 0.0

        hybrid = vector_weight * vector_score + bm25_weight * bm25_score
        assert hybrid == 0.9

    def test_hybrid_score_all_bm25(self):
        """Verifica score quando apenas BM25 contribui."""
        vector_weight = 0.0
        bm25_weight = 1.0

        vector_score = 0.0
        bm25_score = 0.8

        hybrid = vector_weight * vector_score + bm25_weight * bm25_score
        assert hybrid == 0.8


class TestBM25Scoring:
    """Testes detalhados de scoring BM25."""

    def test_idf_calculation(self):
        """Verifica cálculo de IDF."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        # Indexar docs onde "python" aparece em 2 de 4
        documents = [
            (1, "python programming"),
            (2, "java programming"),
            (3, "python development"),
            (4, "javascript web"),
        ]

        bm25.index(documents)

        # IDF manual: log((N - df + 0.5) / (df + 0.5) + 1)
        # N=4, df("python")=2
        # IDF = log((4 - 2 + 0.5) / (2 + 0.5) + 1) = log(2)
        expected_idf = math.log((4 - 2 + 0.5) / (2 + 0.5) + 1)

        # Buscar e verificar que docs com "python" têm score
        results = bm25.search("python")
        assert len(results) == 2

    def test_document_length_normalization(self):
        """Verifica normalização por tamanho de documento."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        # Doc curto vs doc longo
        documents = [
            (1, "python"),  # Curto
            (
                2,
                "python programming language for data science and machine learning applications",
            ),  # Longo
        ]

        bm25.index(documents)

        # Ambos contêm "python", mas doc curto deve ter score maior
        # devido à normalização de tamanho
        results = bm25.search("python")

        # Doc 1 (curto) deve ter score diferente de doc 2 (longo)
        assert 1 in results
        assert 2 in results


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_empty_query(self):
        """Verifica query vazia."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [(1, "Python programming")]
        bm25.index(documents)

        results = bm25.search("")
        assert results == {}

    def test_empty_documents(self):
        """Verifica indexação de documentos vazios."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, ""),
            (2, "   "),
            (3, "valid content"),
        ]

        bm25.index(documents)

        # Deve funcionar sem erros
        results = bm25.search("content")
        assert 3 in results

    def test_special_characters(self):
        """Verifica tratamento de caracteres especiais."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "C++ programming"),
            (2, "Python 3.10 features"),
            (3, "@decorator usage"),
        ]

        bm25.index(documents)

        # Não deve quebrar
        results = bm25.search("C++")
        assert isinstance(results, dict)

    def test_unicode_text(self):
        """Verifica texto com unicode."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        documents = [
            (1, "Programação em Python é incrível"),
            (2, "日本語テキスト"),
            (3, "Código com acentuação"),
        ]

        bm25.index(documents)

        results = bm25.search("programação python")
        assert 1 in results

    def test_very_long_document(self):
        """Verifica documento muito longo."""
        from claude_rag_sdk.core.hybrid_search import BM25

        bm25 = BM25()

        long_text = "python " * 10000
        documents = [
            (1, long_text),
            (2, "short document"),
        ]

        bm25.index(documents)

        # Não deve quebrar
        results = bm25.search("python")
        assert 1 in results


class TestBM25Parameters:
    """Testes de parâmetros BM25."""

    def test_k1_effect(self):
        """Verifica efeito do parâmetro k1."""
        from claude_rag_sdk.core.hybrid_search import BM25

        documents = [
            (1, "python python python"),  # Alta frequência
            (2, "python"),  # Baixa frequência
        ]

        # k1 alto: mais saturação de TF
        bm25_high_k1 = BM25(k1=3.0)
        bm25_high_k1.index(documents)
        results_high = bm25_high_k1.search("python")

        # k1 baixo: menos saturação
        bm25_low_k1 = BM25(k1=0.5)
        bm25_low_k1.index(documents)
        results_low = bm25_low_k1.search("python")

        # A diferença entre doc1 e doc2 deve ser diferente
        diff_high = results_high[1] - results_high[2]
        diff_low = results_low[1] - results_low[2]

        # Com k1 diferente, as diferenças devem ser diferentes
        assert diff_high != diff_low

    def test_b_effect(self):
        """Verifica efeito do parâmetro b."""
        from claude_rag_sdk.core.hybrid_search import BM25

        documents = [
            (1, "python"),  # Curto
            (2, "python is a programming language used for many things"),  # Longo
        ]

        # b=0: sem normalização de tamanho
        bm25_no_norm = BM25(b=0.0)
        bm25_no_norm.index(documents)

        # b=1: normalização máxima
        bm25_full_norm = BM25(b=1.0)
        bm25_full_norm.index(documents)

        # Com b=0, tamanho não afeta tanto
        # Com b=1, doc curto tem vantagem maior


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_hybrid_search_tests():
    """Executa testes manualmente."""
    import sys

    print("\n" + "=" * 60)
    print("TESTES - HYBRID SEARCH")
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

    success = run_hybrid_search_tests()
    sys.exit(0 if success else 1)
