# =============================================================================
# HYBRID SEARCH - Busca Híbrida BM25 + Vetorial
# =============================================================================
# Combina busca léxica (BM25) com busca semântica (vetorial) para melhor recall
# =============================================================================

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import apsw
import sqlite_vec
from fastembed import TextEmbedding
from pathlib import Path
import sys

# Adicionar parent ao path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_config


@dataclass
class SearchResult:
    """Resultado de busca híbrida."""
    doc_id: int
    nome: str
    tipo: str
    content: str
    vector_score: float      # Score da busca vetorial (0-1)
    bm25_score: float        # Score BM25 normalizado (0-1)
    hybrid_score: float      # Score combinado
    rank: int


class BM25:
    """Implementação simples de BM25 para busca léxica."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length: float = 0
        self.doc_freqs: dict[str, int] = {}  # Quantos docs contêm cada termo
        self.term_freqs: dict[int, Counter] = {}  # Frequência de termos por doc
        self.num_docs: int = 0
        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        """Tokenização simples."""
        text = text.lower()
        # Remover pontuação e dividir em palavras
        tokens = re.findall(r'\b\w+\b', text)
        # Remover stopwords básicas (português)
        stopwords = {'de', 'da', 'do', 'e', 'a', 'o', 'que', 'em', 'para', 'com', 'um', 'uma', 'os', 'as', 'na', 'no', 'por', 'se', 'ao'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def index(self, documents: list[tuple[int, str]]) -> None:
        """
        Indexa documentos para BM25.

        Args:
            documents: Lista de (doc_id, texto)
        """
        self.doc_lengths = {}
        self.doc_freqs = Counter()
        self.term_freqs = {}
        self.num_docs = len(documents)

        total_length = 0

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Contar frequência de termos neste doc
            tf = Counter(tokens)
            self.term_freqs[doc_id] = tf

            # Atualizar document frequency
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        self._indexed = True

    def search(self, query: str, doc_ids: Optional[list[int]] = None) -> dict[int, float]:
        """
        Busca BM25.

        Args:
            query: Query de busca
            doc_ids: IDs de documentos para buscar (None = todos)

        Returns:
            Dict de doc_id -> score
        """
        if not self._indexed:
            return {}

        query_tokens = self._tokenize(query)
        scores: dict[int, float] = {}

        search_docs = doc_ids if doc_ids else list(self.doc_lengths.keys())

        for doc_id in search_docs:
            if doc_id not in self.term_freqs:
                continue

            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            tf_doc = self.term_freqs[doc_id]

            for term in query_tokens:
                if term not in tf_doc:
                    continue

                # IDF
                df = self.doc_freqs.get(term, 0)
                idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

                # TF com saturação
                tf = tf_doc[term]
                tf_normalized = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))

                score += idf * tf_normalized

            if score > 0:
                scores[doc_id] = score

        return scores


class HybridSearch:
    """Busca híbrida combinando vetorial e BM25."""

    def __init__(
        self,
        db_path: str,
        vector_weight: float = 0.7,     # Peso da busca vetorial
        bm25_weight: float = 0.3,       # Peso do BM25
    ):
        self.db_path = db_path
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Carregar configuração e modelo de embeddings
        config = get_config()
        self.model = TextEmbedding(config.embedding_model.value)

        # Inicializar BM25
        self.bm25 = BM25()
        self._index_bm25()

    def _get_connection(self):
        """Cria conexão com sqlite-vec."""
        conn = apsw.Connection(self.db_path)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        return conn

    def _index_bm25(self) -> None:
        """Indexa documentos para BM25."""
        conn = self._get_connection()
        cursor = conn.cursor()

        documents = []
        for row in cursor.execute("SELECT id, conteudo FROM documentos WHERE conteudo IS NOT NULL"):
            documents.append((row[0], row[1]))

        conn.close()
        self.bm25.index(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int = 20,         # Buscar mais no vetorial para combinar
    ) -> list[SearchResult]:
        """
        Busca híbrida.

        Args:
            query: Query de busca
            top_k: Número de resultados finais
            vector_top_k: Número de candidatos da busca vetorial

        Returns:
            Lista de SearchResult ordenada por score híbrido
        """
        # 1. Busca vetorial
        embeddings = list(self.model.embed([query]))
        query_vec = sqlite_vec.serialize_float32(embeddings[0].tolist())

        conn = self._get_connection()
        cursor = conn.cursor()

        vector_results = {}
        for row in cursor.execute("""
            SELECT v.doc_id, v.distance, d.nome, d.conteudo, d.tipo
            FROM vec_documentos v
            JOIN documentos d ON d.id = v.doc_id
            WHERE v.embedding MATCH ? AND k = ?
        """, (query_vec, vector_top_k)):
            doc_id, distance, nome, conteudo, tipo = row
            similarity = max(0, 1 - distance)
            vector_results[doc_id] = {
                "vector_score": similarity,
                "nome": nome,
                "conteudo": conteudo,
                "tipo": tipo,
            }

        conn.close()

        # 2. Busca BM25 nos mesmos documentos + extras
        bm25_scores = self.bm25.search(query)

        # 3. Combinar resultados
        all_doc_ids = set(vector_results.keys()) | set(bm25_scores.keys())

        # Normalizar scores BM25
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        normalized_bm25 = {k: v / max_bm25 for k, v in bm25_scores.items()}

        results = []
        for doc_id in all_doc_ids:
            vector_score = vector_results.get(doc_id, {}).get("vector_score", 0)
            bm25_score = normalized_bm25.get(doc_id, 0)

            # Score híbrido ponderado
            hybrid_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score
            )

            # Buscar dados do documento se não tiver
            if doc_id in vector_results:
                doc_data = vector_results[doc_id]
            else:
                conn = self._get_connection()
                cursor = conn.cursor()
                row = None
                for r in cursor.execute(
                    "SELECT nome, conteudo, tipo FROM documentos WHERE id = ?",
                    (doc_id,)
                ):
                    row = r
                    break
                conn.close()
                if row:
                    doc_data = {"nome": row[0], "conteudo": row[1], "tipo": row[2]}
                else:
                    continue

            results.append(SearchResult(
                doc_id=doc_id,
                nome=doc_data["nome"],
                tipo=doc_data["tipo"],
                content=doc_data["conteudo"][:1000] if doc_data["conteudo"] else "",
                vector_score=round(vector_score, 3),
                bm25_score=round(bm25_score, 3),
                hybrid_score=round(hybrid_score, 3),
                rank=0,
            ))

        # Ordenar por score híbrido
        results.sort(key=lambda x: x.hybrid_score, reverse=True)

        # Atribuir ranks
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1

        return results[:top_k]


if __name__ == "__main__":
    from pathlib import Path

    db_path = str(Path(__file__).parent.parent.parent / "teste" / "documentos.db")

    print("=== Teste de Busca Híbrida ===\n")

    hybrid = HybridSearch(db_path)

    queries = [
        "princípios obrigatórios da política de IA",
        "componentes arquitetura RAG",
        "métricas de monitoramento",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        results = hybrid.search(query, top_k=3)
        for r in results:
            print(f"  [{r.rank}] {r.nome}")
            print(f"      Hybrid: {r.hybrid_score:.3f} (vec: {r.vector_score:.3f}, bm25: {r.bm25_score:.3f})")
