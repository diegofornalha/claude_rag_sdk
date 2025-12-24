# =============================================================================
# RERANKER - Re-ranking com Cross-Encoder
# =============================================================================
# Melhora a precisão reordenando candidatos com modelo cross-encoder
# =============================================================================

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class RerankResult:
    """Resultado após re-ranking."""
    doc_id: int
    content: str
    original_score: float
    rerank_score: float
    final_rank: int
    metadata: dict


class CrossEncoderReranker:
    """Re-ranker usando cross-encoder (sentence-transformers)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Inicializa o cross-encoder.

        Args:
            model_name: Nome do modelo cross-encoder
        """
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        """Carrega modelo sob demanda."""
        if self._load_attempted:
            return self._model

        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            # Fallback: usar scoring simples se não tiver sentence-transformers
            self._model = None
        except Exception:
            self._model = None

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[tuple[int, str, float, dict]],  # (doc_id, content, score, metadata)
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Re-rankeia documentos usando cross-encoder.

        Args:
            query: Query de busca
            documents: Lista de (doc_id, content, original_score, metadata)
            top_k: Número de resultados finais

        Returns:
            Lista de RerankResult ordenada por relevância
        """
        if not documents:
            return []

        model = self._load_model()

        if model is not None:
            # Usar cross-encoder
            pairs = [(query, doc[1][:512]) for doc in documents]  # Truncar para 512 chars
            scores = model.predict(pairs)

            results = []
            for i, (doc_id, content, original_score, metadata) in enumerate(documents):
                results.append(RerankResult(
                    doc_id=doc_id,
                    content=content,
                    original_score=original_score,
                    rerank_score=float(scores[i]),
                    final_rank=0,
                    metadata=metadata,
                ))
        else:
            # Fallback: usar score original com boost por match exato
            results = []
            query_lower = query.lower()
            query_terms = set(query_lower.split())

            for doc_id, content, original_score, metadata in documents:
                content_lower = content.lower()

                # Boost por termos exatos encontrados
                term_matches = sum(1 for term in query_terms if term in content_lower)
                term_boost = term_matches / max(len(query_terms), 1) * 0.2

                # Boost por frase exata
                phrase_boost = 0.3 if query_lower in content_lower else 0

                rerank_score = original_score + term_boost + phrase_boost

                results.append(RerankResult(
                    doc_id=doc_id,
                    content=content,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    final_rank=0,
                    metadata=metadata,
                ))

        # Ordenar por rerank_score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Atribuir ranks
        for i, r in enumerate(results[:top_k]):
            r.final_rank = i + 1

        return results[:top_k]


class LightweightReranker:
    """Re-ranker leve sem dependências externas."""

    def __init__(self):
        # Pesos para diferentes sinais
        self.exact_match_weight = 0.3
        self.term_coverage_weight = 0.2
        self.position_weight = 0.1

    def rerank(
        self,
        query: str,
        documents: list[tuple[int, str, float, dict]],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Re-rankeia usando heurísticas simples.

        Args:
            query: Query de busca
            documents: Lista de (doc_id, content, original_score, metadata)
            top_k: Número de resultados

        Returns:
            Lista de RerankResult
        """
        if not documents:
            return []

        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) > 2]

        results = []
        for doc_id, content, original_score, metadata in documents:
            content_lower = content.lower()

            # 1. Match exato da query
            exact_match = self.exact_match_weight if query_lower in content_lower else 0

            # 2. Cobertura de termos
            matched_terms = sum(1 for term in query_terms if term in content_lower)
            term_coverage = (matched_terms / max(len(query_terms), 1)) * self.term_coverage_weight

            # 3. Posição dos termos (termos no início = mais relevante)
            position_score = 0
            for term in query_terms:
                pos = content_lower.find(term)
                if pos >= 0:
                    # Score maior para termos no início
                    position_score += (1 - pos / max(len(content_lower), 1)) * 0.1

            position_score = min(position_score, self.position_weight)

            # Score final
            rerank_score = original_score + exact_match + term_coverage + position_score

            results.append(RerankResult(
                doc_id=doc_id,
                content=content,
                original_score=original_score,
                rerank_score=round(rerank_score, 4),
                final_rank=0,
                metadata=metadata,
            ))

        # Ordenar por rerank_score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Atribuir ranks
        for i, r in enumerate(results[:top_k]):
            r.final_rank = i + 1

        return results[:top_k]


# Factory para criar reranker apropriado
def create_reranker(use_cross_encoder: bool = False) -> CrossEncoderReranker | LightweightReranker:
    """
    Cria reranker apropriado.

    Args:
        use_cross_encoder: Se True, tenta usar cross-encoder

    Returns:
        Instância de reranker
    """
    if use_cross_encoder:
        return CrossEncoderReranker()
    return LightweightReranker()


if __name__ == "__main__":
    # Teste do reranker
    print("=== Teste de Re-ranking ===\n")

    # Documentos de teste
    documents = [
        (1, "A política de IA estabelece princípios obrigatórios para uso de inteligência artificial.", 0.8, {"nome": "Doc1"}),
        (2, "Arquitetura RAG enterprise com componentes de busca vetorial e reranking.", 0.75, {"nome": "Doc2"}),
        (3, "Métricas de monitoramento incluem latência, throughput e taxa de erro.", 0.7, {"nome": "Doc3"}),
        (4, "Os princípios obrigatórios da política incluem transparência e responsabilidade.", 0.65, {"nome": "Doc4"}),
    ]

    query = "princípios obrigatórios da política"

    # Testar reranker leve
    print("--- Lightweight Reranker ---")
    reranker = LightweightReranker()
    results = reranker.rerank(query, documents, top_k=3)

    for r in results:
        print(f"  [{r.final_rank}] Doc {r.doc_id}: {r.rerank_score:.3f} (original: {r.original_score:.3f})")
        print(f"      {r.content[:60]}...")

    # Testar cross-encoder (se disponível)
    print("\n--- Cross-Encoder Reranker ---")
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, documents, top_k=3)

    for r in results:
        print(f"  [{r.final_rank}] Doc {r.doc_id}: {r.rerank_score:.3f} (original: {r.original_score:.3f})")
