# =============================================================================
# ADAPTIVE SEARCH - Top-K Dinâmico Baseado em Confidence
# =============================================================================
# Ajusta automaticamente número de resultados baseado em similarity scores
# =============================================================================

from dataclasses import dataclass
from typing import List, Protocol
from core.config import get_config


class SearchResultProtocol(Protocol):
    """Protocol para objetos de resultado de busca."""
    similarity: float


@dataclass
class AdaptiveDecision:
    """Decisão do top-k adaptativo."""
    original_k: int
    adjusted_k: int
    reason: str
    top_similarity: float
    confidence_level: str


class AdaptiveTopK:
    """Calcula top-k dinâmico baseado em confidence dos resultados."""

    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.5,
        high_confidence_k: int = 2,
        low_confidence_multiplier: float = 1.5,
    ):
        """
        Inicializa adaptador de top-k.

        Args:
            high_confidence_threshold: Score acima do qual considera alta confiança
            low_confidence_threshold: Score abaixo do qual considera baixa confiança
            high_confidence_k: Número de resultados para alta confiança
            low_confidence_multiplier: Multiplicador para baixa confiança
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.high_confidence_k = high_confidence_k
        self.low_confidence_multiplier = low_confidence_multiplier

    @classmethod
    def from_config(cls) -> "AdaptiveTopK":
        """Cria instância a partir da configuração global."""
        config = get_config()
        return cls(
            high_confidence_threshold=config.high_confidence_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
            high_confidence_k=config.high_confidence_k,
            low_confidence_multiplier=config.low_confidence_multiplier,
        )

    def calculate_optimal_k(
        self,
        results: List[SearchResultProtocol],
        base_top_k: int,
    ) -> AdaptiveDecision:
        """
        Calcula top-k ótimo baseado nos resultados.

        Args:
            results: Lista de resultados de busca (ordenados por similarity DESC)
            base_top_k: Top-k base configurado

        Returns:
            AdaptiveDecision com k ajustado e razão
        """
        if not results:
            return AdaptiveDecision(
                original_k=base_top_k,
                adjusted_k=base_top_k,
                reason="no_results",
                top_similarity=0.0,
                confidence_level="none",
            )

        top_similarity = results[0].similarity

        # Alta confiança: reduzir número de resultados
        if top_similarity >= self.high_confidence_threshold:
            adjusted_k = min(self.high_confidence_k, len(results))
            return AdaptiveDecision(
                original_k=base_top_k,
                adjusted_k=adjusted_k,
                reason="high_confidence",
                top_similarity=top_similarity,
                confidence_level="high",
            )

        # Baixa confiança: aumentar número de resultados
        elif top_similarity < self.low_confidence_threshold:
            adjusted_k = min(
                int(base_top_k * self.low_confidence_multiplier),
                len(results)
            )
            return AdaptiveDecision(
                original_k=base_top_k,
                adjusted_k=adjusted_k,
                reason="low_confidence",
                top_similarity=top_similarity,
                confidence_level="low",
            )

        # Confiança média: manter base_top_k
        else:
            adjusted_k = min(base_top_k, len(results))
            return AdaptiveDecision(
                original_k=base_top_k,
                adjusted_k=adjusted_k,
                reason="medium_confidence",
                top_similarity=top_similarity,
                confidence_level="medium",
            )

    def should_fetch_more(
        self,
        results: List[SearchResultProtocol],
        threshold: float = 0.5,
    ) -> bool:
        """
        Determina se deve buscar mais resultados.

        Args:
            results: Resultados atuais
            threshold: Threshold de similarity

        Returns:
            True se deve buscar mais resultados
        """
        if not results:
            return True

        # Se top result é fraco, buscar mais
        return results[0].similarity < threshold


def apply_adaptive_topk(
    results: list,
    base_top_k: int,
    enabled: bool = True,
) -> tuple[list, AdaptiveDecision]:
    """
    Aplica top-k adaptativo a uma lista de resultados.

    Args:
        results: Lista de resultados com atributo 'similarity'
        base_top_k: Top-k base
        enabled: Se False, retorna resultados sem modificação

    Returns:
        Tupla (resultados filtrados, decisão)
    """
    # Se desabilitado ou sem resultados, retornar como está
    if not enabled or not results:
        decision = AdaptiveDecision(
            original_k=base_top_k,
            adjusted_k=base_top_k,
            reason="adaptive_disabled" if not enabled else "no_results",
            top_similarity=0.0,
            confidence_level="none",
        )
        return results[:base_top_k], decision

    # Criar adapter
    adapter = AdaptiveTopK.from_config()

    # Calcular top-k ótimo
    decision = adapter.calculate_optimal_k(results, base_top_k)

    # Retornar resultados ajustados
    return results[:decision.adjusted_k], decision


# Aliases para compatibilidade
def get_adaptive_topk() -> AdaptiveTopK:
    """Retorna instância do adaptador baseado na config."""
    return AdaptiveTopK.from_config()


if __name__ == "__main__":
    # Teste do adaptive top-k
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        similarity: float

    adapter = AdaptiveTopK(
        high_confidence_threshold=0.7,
        low_confidence_threshold=0.5,
        high_confidence_k=2,
        low_confidence_multiplier=1.5,
    )

    test_cases = [
        ("Alta confiança", [MockResult(0.85), MockResult(0.78), MockResult(0.65)]),
        ("Média confiança", [MockResult(0.62), MockResult(0.58), MockResult(0.52)]),
        ("Baixa confiança", [MockResult(0.42), MockResult(0.38), MockResult(0.35)]),
        ("Sem resultados", []),
    ]

    print("=== Teste de Top-K Adaptativo ===\n")

    for name, results in test_cases:
        decision = adapter.calculate_optimal_k(results, base_top_k=5)
        print(f"{name}:")
        print(f"  Top similarity: {decision.top_similarity:.2f}")
        print(f"  Confidence: {decision.confidence_level}")
        print(f"  K ajustado: {decision.original_k} → {decision.adjusted_k}")
        print(f"  Razão: {decision.reason}")
        print()
