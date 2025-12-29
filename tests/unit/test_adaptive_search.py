# =============================================================================
# TESTES - Adaptive Search Module
# =============================================================================
# Testes unitários para busca adaptativa com top-k dinâmico
# =============================================================================

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class MockSearchResult:
    """Mock de resultado de busca."""

    similarity: float


class TestAdaptiveDecision:
    """Testes para AdaptiveDecision dataclass."""

    def test_adaptive_decision_creation(self):
        """Verifica criação de AdaptiveDecision."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveDecision

        decision = AdaptiveDecision(
            original_k=5,
            adjusted_k=2,
            reason="high_confidence",
            top_similarity=0.85,
            confidence_level="high",
        )

        assert decision.original_k == 5
        assert decision.adjusted_k == 2
        assert decision.reason == "high_confidence"
        assert decision.top_similarity == 0.85
        assert decision.confidence_level == "high"

    def test_adaptive_decision_all_fields(self):
        """Verifica todos os campos de AdaptiveDecision."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveDecision

        decision = AdaptiveDecision(
            original_k=10,
            adjusted_k=15,
            reason="low_confidence",
            top_similarity=0.35,
            confidence_level="low",
        )

        # Verificar que todos os campos são acessíveis
        assert hasattr(decision, "original_k")
        assert hasattr(decision, "adjusted_k")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "top_similarity")
        assert hasattr(decision, "confidence_level")


class TestAdaptiveTopK:
    """Testes para AdaptiveTopK."""

    def test_default_initialization(self):
        """Verifica inicialização com valores padrão."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        assert adapter.high_confidence_threshold == 0.7
        assert adapter.low_confidence_threshold == 0.5
        assert adapter.high_confidence_k == 2
        assert adapter.low_confidence_multiplier == 1.5

    def test_custom_initialization(self):
        """Verifica inicialização com valores customizados."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(
            high_confidence_threshold=0.8,
            low_confidence_threshold=0.4,
            high_confidence_k=3,
            low_confidence_multiplier=2.0,
        )

        assert adapter.high_confidence_threshold == 0.8
        assert adapter.low_confidence_threshold == 0.4
        assert adapter.high_confidence_k == 3
        assert adapter.low_confidence_multiplier == 2.0

    def test_calculate_optimal_k_empty_results(self):
        """Verifica cálculo com lista vazia."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()
        decision = adapter.calculate_optimal_k([], base_top_k=5)

        assert decision.adjusted_k == 5
        assert decision.reason == "no_results"
        assert decision.top_similarity == 0.0
        assert decision.confidence_level == "none"

    def test_calculate_optimal_k_high_confidence(self):
        """Verifica cálculo para alta confiança."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(
            high_confidence_threshold=0.7,
            high_confidence_k=2,
        )

        results = [
            MockSearchResult(similarity=0.85),
            MockSearchResult(similarity=0.78),
            MockSearchResult(similarity=0.65),
        ]

        decision = adapter.calculate_optimal_k(results, base_top_k=5)

        assert decision.adjusted_k == 2
        assert decision.reason == "high_confidence"
        assert decision.confidence_level == "high"
        assert decision.top_similarity == 0.85

    def test_calculate_optimal_k_low_confidence(self):
        """Verifica cálculo para baixa confiança."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(
            low_confidence_threshold=0.5,
            low_confidence_multiplier=1.5,
        )

        results = [
            MockSearchResult(similarity=0.42),
            MockSearchResult(similarity=0.38),
            MockSearchResult(similarity=0.35),
            MockSearchResult(similarity=0.30),
            MockSearchResult(similarity=0.25),
            MockSearchResult(similarity=0.20),
            MockSearchResult(similarity=0.15),
            MockSearchResult(similarity=0.10),
        ]

        decision = adapter.calculate_optimal_k(results, base_top_k=5)

        assert decision.adjusted_k == 7  # 5 * 1.5 = 7.5 -> 7
        assert decision.reason == "low_confidence"
        assert decision.confidence_level == "low"

    def test_calculate_optimal_k_medium_confidence(self):
        """Verifica cálculo para confiança média."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(
            high_confidence_threshold=0.7,
            low_confidence_threshold=0.5,
        )

        results = [
            MockSearchResult(similarity=0.62),
            MockSearchResult(similarity=0.58),
            MockSearchResult(similarity=0.52),
        ]

        decision = adapter.calculate_optimal_k(results, base_top_k=5)

        assert decision.adjusted_k == 3  # min(5, len(results))
        assert decision.reason == "medium_confidence"
        assert decision.confidence_level == "medium"

    def test_calculate_optimal_k_respects_results_length(self):
        """Verifica que adjusted_k não excede número de resultados."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        # Apenas 2 resultados, mesmo com baixa confiança
        results = [
            MockSearchResult(similarity=0.30),
            MockSearchResult(similarity=0.25),
        ]

        decision = adapter.calculate_optimal_k(results, base_top_k=10)

        # Não pode exceder len(results)
        assert decision.adjusted_k <= len(results)

    def test_should_fetch_more_empty_results(self):
        """Verifica should_fetch_more com lista vazia."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        assert adapter.should_fetch_more([]) is True

    def test_should_fetch_more_weak_results(self):
        """Verifica should_fetch_more com resultados fracos."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        results = [MockSearchResult(similarity=0.3)]

        assert adapter.should_fetch_more(results, threshold=0.5) is True

    def test_should_fetch_more_strong_results(self):
        """Verifica should_fetch_more com resultados fortes."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        results = [MockSearchResult(similarity=0.8)]

        assert adapter.should_fetch_more(results, threshold=0.5) is False


class TestApplyAdaptiveTopK:
    """Testes para função apply_adaptive_topk."""

    def test_apply_disabled(self):
        """Verifica comportamento quando desabilitado."""
        from claude_rag_sdk.core.adaptive_search import apply_adaptive_topk

        results = [
            MockSearchResult(similarity=0.9),
            MockSearchResult(similarity=0.8),
            MockSearchResult(similarity=0.7),
        ]

        filtered, decision = apply_adaptive_topk(results, base_top_k=2, enabled=False)

        assert len(filtered) == 2
        assert decision.reason == "adaptive_disabled"

    def test_apply_empty_results(self):
        """Verifica com lista vazia."""
        from claude_rag_sdk.core.adaptive_search import apply_adaptive_topk

        filtered, decision = apply_adaptive_topk([], base_top_k=5, enabled=True)

        assert len(filtered) == 0
        assert decision.reason == "no_results"

    @patch("claude_rag_sdk.core.adaptive_search.get_config")
    def test_apply_with_config(self, mock_get_config):
        """Verifica integração com config."""
        from claude_rag_sdk.core.adaptive_search import apply_adaptive_topk

        # Mock config
        mock_config = MagicMock()
        mock_config.high_confidence_threshold = 0.7
        mock_config.low_confidence_threshold = 0.5
        mock_config.high_confidence_k = 2
        mock_config.low_confidence_multiplier = 1.5
        mock_get_config.return_value = mock_config

        results = [
            MockSearchResult(similarity=0.85),
            MockSearchResult(similarity=0.75),
            MockSearchResult(similarity=0.65),
        ]

        filtered, decision = apply_adaptive_topk(results, base_top_k=5, enabled=True)

        assert decision.confidence_level == "high"
        assert len(filtered) == 2


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_single_result_high_confidence(self):
        """Verifica com único resultado de alta confiança."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(high_confidence_k=2)
        results = [MockSearchResult(similarity=0.95)]

        decision = adapter.calculate_optimal_k(results, base_top_k=10)

        # Não pode retornar mais do que existe
        assert decision.adjusted_k == 1

    def test_exact_threshold_boundary(self):
        """Verifica comportamento nos limites exatos dos thresholds."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK(
            high_confidence_threshold=0.7,
            low_confidence_threshold=0.5,
        )

        # Exatamente no threshold de alta confiança
        results_high = [MockSearchResult(similarity=0.7)]
        decision = adapter.calculate_optimal_k(results_high, base_top_k=5)
        assert decision.confidence_level == "high"

        # Exatamente no threshold de baixa confiança (não inclusivo)
        results_medium = [MockSearchResult(similarity=0.5)]
        decision = adapter.calculate_optimal_k(results_medium, base_top_k=5)
        assert decision.confidence_level == "medium"

        # Logo abaixo do threshold de baixa confiança
        results_low = [MockSearchResult(similarity=0.49)]
        decision = adapter.calculate_optimal_k(results_low, base_top_k=5)
        assert decision.confidence_level == "low"

    def test_very_large_base_top_k(self):
        """Verifica com base_top_k muito grande."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()
        results = [MockSearchResult(similarity=0.3) for _ in range(5)]

        decision = adapter.calculate_optimal_k(results, base_top_k=1000)

        # Deve ser limitado ao tamanho da lista
        assert decision.adjusted_k <= len(results)


class TestSearchResultProtocol:
    """Testes para verificar compatibilidade com o Protocol."""

    def test_protocol_compliance(self):
        """Verifica que objetos com 'similarity' funcionam."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        # Classe simples que segue o protocol
        class SimpleResult:
            def __init__(self, sim):
                self.similarity = sim

        adapter = AdaptiveTopK()
        results = [SimpleResult(0.8), SimpleResult(0.6)]

        decision = adapter.calculate_optimal_k(results, base_top_k=5)

        assert decision.top_similarity == 0.8

    def test_dict_like_objects_fail(self):
        """Verifica que dicts não funcionam (precisam de atributo)."""
        from claude_rag_sdk.core.adaptive_search import AdaptiveTopK

        adapter = AdaptiveTopK()

        # Dict não tem atributo similarity
        results = [{"similarity": 0.8}]

        with pytest.raises(AttributeError):
            adapter.calculate_optimal_k(results, base_top_k=5)


# =============================================================================
# RUNNER PARA EXECUÇÃO MANUAL
# =============================================================================


def run_adaptive_search_tests():
    """Executa testes manualmente."""

    print("\n" + "=" * 60)
    print("TESTES - ADAPTIVE SEARCH")
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

    success = run_adaptive_search_tests()
    sys.exit(0 if success else 1)
