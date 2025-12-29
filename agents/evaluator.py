"""RAG Evaluator - Avaliação de qualidade das respostas do agente RAG.

Métricas implementadas:
- Groundedness: Resposta baseada em evidências (citações)
- Keyword Coverage: Keywords esperadas presentes na resposta
- Source Accuracy: Fontes corretas utilizadas
- Answer Similarity: Similaridade semântica com resposta esperada
- Citation Quality: Qualidade e quantidade de citações

Uso:
    evaluator = RAGEvaluator()
    results = await evaluator.run_evaluation()
    print(results.summary())
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_rag_sdk.core.logger import get_logger

logger = get_logger("evaluator")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationMetrics:
    """Métricas de avaliação de uma resposta."""

    question_id: int
    question: str

    # Scores (0.0 - 1.0)
    groundedness: float = 0.0  # Resposta tem citações válidas
    keyword_coverage: float = 0.0  # Keywords esperadas presentes
    source_accuracy: float = 0.0  # Fontes corretas
    answer_relevance: float = 0.0  # Resposta relevante à pergunta
    citation_quality: float = 0.0  # Qualidade das citações

    # Detalhes
    expected_keywords: list[str] = field(default_factory=list)
    found_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    found_sources: list[str] = field(default_factory=list)
    citations_count: int = 0

    # Response
    actual_answer: str = ""
    expected_answer: str = ""
    confidence: float = 0.0

    # Timing
    latency_ms: float = 0.0

    @property
    def overall_score(self) -> float:
        """Score geral ponderado."""
        weights = {
            "groundedness": 0.25,
            "keyword_coverage": 0.25,
            "source_accuracy": 0.20,
            "answer_relevance": 0.15,
            "citation_quality": 0.15,
        }
        return (
            self.groundedness * weights["groundedness"] +
            self.keyword_coverage * weights["keyword_coverage"] +
            self.source_accuracy * weights["source_accuracy"] +
            self.answer_relevance * weights["answer_relevance"] +
            self.citation_quality * weights["citation_quality"]
        )

    @property
    def passed(self) -> bool:
        """Passou no teste (score >= 0.6)."""
        return self.overall_score >= 0.6

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question[:100] + "..." if len(self.question) > 100 else self.question,
            "scores": {
                "overall": round(self.overall_score, 3),
                "groundedness": round(self.groundedness, 3),
                "keyword_coverage": round(self.keyword_coverage, 3),
                "source_accuracy": round(self.source_accuracy, 3),
                "answer_relevance": round(self.answer_relevance, 3),
                "citation_quality": round(self.citation_quality, 3),
            },
            "passed": self.passed,
            "keywords": {
                "expected": self.expected_keywords,
                "found": self.found_keywords,
                "missing": self.missing_keywords,
            },
            "sources": {
                "expected": self.expected_sources,
                "found": self.found_sources,
            },
            "citations_count": self.citations_count,
            "confidence": round(self.confidence, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class EvaluationReport:
    """Relatório completo de avaliação."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_questions: int = 0
    passed_count: int = 0
    failed_count: int = 0

    # Scores médios
    avg_overall: float = 0.0
    avg_groundedness: float = 0.0
    avg_keyword_coverage: float = 0.0
    avg_source_accuracy: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_citation_quality: float = 0.0

    # Timing
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Detalhes
    results: list[EvaluationMetrics] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.passed_count / self.total_questions

    def add_result(self, metrics: EvaluationMetrics):
        """Adiciona resultado e recalcula médias."""
        self.results.append(metrics)
        self.total_questions = len(self.results)

        if metrics.passed:
            self.passed_count += 1
        else:
            self.failed_count += 1

        # Recalcular médias
        n = self.total_questions
        self.avg_overall = sum(r.overall_score for r in self.results) / n
        self.avg_groundedness = sum(r.groundedness for r in self.results) / n
        self.avg_keyword_coverage = sum(r.keyword_coverage for r in self.results) / n
        self.avg_source_accuracy = sum(r.source_accuracy for r in self.results) / n
        self.avg_answer_relevance = sum(r.answer_relevance for r in self.results) / n
        self.avg_citation_quality = sum(r.citation_quality for r in self.results) / n

        self.total_latency_ms = sum(r.latency_ms for r in self.results)
        self.avg_latency_ms = self.total_latency_ms / n

    def generate_recommendations(self):
        """Gera recomendações baseadas nos resultados."""
        self.recommendations = []

        if self.avg_groundedness < 0.6:
            self.recommendations.append(
                "⚠️ Groundedness baixo: Melhorar citações nas respostas. "
                "Verificar se o agente está usando search_documents corretamente."
            )

        if self.avg_keyword_coverage < 0.6:
            self.recommendations.append(
                "⚠️ Keyword coverage baixo: Respostas não contêm termos esperados. "
                "Verificar qualidade dos documentos indexados."
            )

        if self.avg_source_accuracy < 0.6:
            self.recommendations.append(
                "⚠️ Source accuracy baixo: Fontes incorretas sendo citadas. "
                "Verificar se os documentos corretos estão na base."
            )

        if self.avg_citation_quality < 0.6:
            self.recommendations.append(
                "⚠️ Citation quality baixo: Citações incompletas ou mal formatadas. "
                "Ajustar system prompt para exigir citações estruturadas."
            )

        if self.avg_latency_ms > 4000:
            self.recommendations.append(
                "⚠️ Latência alta (>4s): Considerar cache, modelo menor ou top_k menor."
            )

        if self.pass_rate >= 0.8:
            self.recommendations.append(
                "✅ Excelente! Taxa de acerto >= 80%. Sistema pronto para produção."
            )
        elif self.pass_rate >= 0.6:
            self.recommendations.append(
                "🔶 Bom, mas pode melhorar. Taxa de acerto entre 60-80%."
            )
        else:
            self.recommendations.append(
                "❌ Atenção: Taxa de acerto < 60%. Revisar documentos e configurações."
            )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_questions": self.total_questions,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "pass_rate": round(self.pass_rate, 3),
            },
            "scores": {
                "overall": round(self.avg_overall, 3),
                "groundedness": round(self.avg_groundedness, 3),
                "keyword_coverage": round(self.avg_keyword_coverage, 3),
                "source_accuracy": round(self.avg_source_accuracy, 3),
                "answer_relevance": round(self.avg_answer_relevance, 3),
                "citation_quality": round(self.avg_citation_quality, 3),
            },
            "latency": {
                "total_ms": round(self.total_latency_ms, 2),
                "avg_ms": round(self.avg_latency_ms, 2),
            },
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Retorna resumo textual."""
        lines = [
            "=" * 60,
            "📊 RELATÓRIO DE AVALIAÇÃO RAG",
            "=" * 60,
            f"Data: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📈 RESUMO",
            f"  Total de perguntas: {self.total_questions}",
            f"  Aprovadas: {self.passed_count} ({self.pass_rate:.1%})",
            f"  Reprovadas: {self.failed_count}",
            "",
            "📊 SCORES MÉDIOS",
            f"  Overall:          {self.avg_overall:.1%}",
            f"  Groundedness:     {self.avg_groundedness:.1%}",
            f"  Keyword Coverage: {self.avg_keyword_coverage:.1%}",
            f"  Source Accuracy:  {self.avg_source_accuracy:.1%}",
            f"  Answer Relevance: {self.avg_answer_relevance:.1%}",
            f"  Citation Quality: {self.avg_citation_quality:.1%}",
            "",
            f"⏱️ LATÊNCIA: {self.avg_latency_ms:.0f}ms média",
            "",
            "💡 RECOMENDAÇÕES",
        ]
        for rec in self.recommendations:
            lines.append(f"  {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# RAG Evaluator
# =============================================================================


class RAGEvaluator:
    """Avaliador de qualidade do RAG Agent."""

    def __init__(self, config_path: str | Path | None = None):
        """Inicializa o avaliador.

        Args:
            config_path: Caminho para atlantyx_agent.json (opcional)
        """
        self.config = self._load_config(config_path)
        self.qa_pairs = self.config.get("qa_pairs", [])
        self.api_url = "http://localhost:8001"
        self.api_key: str | None = None

    def _load_config(self, config_path: str | Path | None) -> dict:
        """Carrega configuração do agente."""
        paths = [
            config_path,
            Path(__file__).parent.parent / "config" / "atlantyx_agent.json",
            Path.cwd() / "config" / "atlantyx_agent.json",
        ]

        for path in paths:
            if path and Path(path).exists():
                try:
                    return json.loads(Path(path).read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load config: {e}")

        return {}

    def set_api_key(self, api_key: str):
        """Define API key para autenticação."""
        self.api_key = api_key

    def _load_api_key_from_env(self) -> str | None:
        """Carrega API key do .env."""
        env_paths = [
            Path(__file__).parent.parent / "claude_rag_sdk" / ".env",
            Path(__file__).parent.parent / ".env",
            Path.cwd() / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                try:
                    content = env_path.read_text()
                    for line in content.split("\n"):
                        if line.startswith("RAG_API_KEY="):
                            return line.split("=", 1)[1].strip().strip("'\"")
                except OSError:
                    pass
        return None

    async def ask_question(self, question: str) -> tuple[dict, float]:
        """Envia pergunta para o RAG e retorna resposta + latência.

        Returns:
            (response_data, latency_ms)
        """
        import httpx

        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Usar endpoint de teste sem autenticação
            response = await client.get(
                f"{self.api_url}/rag/ask/test",
                params={"question": question, "top_k": 5},
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return {
                    "answer": data.get("answer", ""),
                    "citations": data.get("citations", []),
                    "confidence": data.get("confidence", 0.0),
                    "metrics": data.get("metrics", {}),
                }, latency_ms
            else:
                return {"answer": f"ERROR: {response.status_code} - {response.text}"}, latency_ms

    def _extract_citations(self, text: str) -> list[dict]:
        """Extrai citações do texto da resposta."""
        citations = []

        # Padrão JSON: {"source": "...", "quote": "..."}
        json_pattern = r'\{[^{}]*"source"[^{}]*"quote"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                citation = json.loads(match)
                if "source" in citation:
                    citations.append(citation)
            except json.JSONDecodeError:
                pass

        # Padrão textual: [Fonte: arquivo]
        text_pattern = r'\[Fonte:\s*([^\]]+)\]'
        for match in re.findall(text_pattern, text):
            citations.append({"source": match.strip(), "quote": ""})

        return citations

    def _extract_confidence(self, text: str) -> float:
        """Extrai confidence da resposta."""
        # Procurar no JSON
        conf_pattern = r'"confidence"\s*:\s*(0\.\d+|1\.0|1)'
        match = re.search(conf_pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5  # Default

    def _calculate_keyword_coverage(
        self,
        answer: str,
        keywords: list[str]
    ) -> tuple[float, list[str], list[str]]:
        """Calcula cobertura de keywords."""
        answer_lower = answer.lower()
        found = []
        missing = []

        for kw in keywords:
            # Normalizar keyword
            kw_normalized = kw.lower().strip()
            if kw_normalized in answer_lower:
                found.append(kw)
            else:
                # Tentar match parcial (palavras individuais)
                words = kw_normalized.split()
                if len(words) > 1 and all(w in answer_lower for w in words):
                    found.append(kw)
                else:
                    missing.append(kw)

        coverage = len(found) / len(keywords) if keywords else 1.0
        return coverage, found, missing

    def _calculate_source_accuracy(
        self,
        citations: list[dict],
        expected_sources: str
    ) -> tuple[float, list[str]]:
        """Calcula acurácia das fontes."""
        # Parse expected sources (formato: "Doc1.docx | Seção 2 ; Doc2.pdf")
        expected = []
        for part in expected_sources.split(";"):
            source = part.split("|")[0].strip()
            if source:
                expected.append(source.lower())

        if not expected:
            return 1.0, []

        # Verificar citações
        found_sources = []
        for citation in citations:
            source = citation.get("source", "").lower()
            for exp in expected:
                # Match parcial (nome do arquivo pode variar)
                exp_base = exp.split(".")[0]  # Remove extensão
                if exp_base in source or source in exp:
                    found_sources.append(citation.get("source", ""))
                    break

        accuracy = len(set(found_sources)) / len(expected) if expected else 1.0
        return min(accuracy, 1.0), list(set(found_sources))

    def _calculate_answer_relevance(self, answer: str, expected: str) -> float:
        """Calcula relevância da resposta (similaridade simples)."""
        if not answer or not expected:
            return 0.0

        # Tokenização simples
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())

        # Jaccard similarity
        intersection = len(answer_words & expected_words)
        union = len(answer_words | expected_words)

        return intersection / union if union > 0 else 0.0

    def _calculate_citation_quality(self, citations: list[dict], answer: str) -> float:
        """Calcula qualidade das citações."""
        if not citations:
            # Penalizar ausência de citações
            return 0.2 if "não encontr" in answer.lower() else 0.0

        quality_score = 0.0

        for citation in citations:
            score = 0.0

            # Tem source
            if citation.get("source"):
                score += 0.5

            # Tem quote
            if citation.get("quote") and len(citation["quote"]) > 10:
                score += 0.5

            quality_score += score

        return min(quality_score / len(citations), 1.0)

    async def evaluate_question(self, qa_pair: dict) -> EvaluationMetrics:
        """Avalia uma pergunta específica."""
        question_id = qa_pair.get("id", 0)
        question = qa_pair.get("question", "")
        expected_answer = qa_pair.get("expected_answer", "")
        expected_sources = qa_pair.get("expected_sources", "")
        keywords = qa_pair.get("evidence_keywords", [])

        # Normalizar keywords se for string
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(";")]

        logger.info(f"Evaluating Q{question_id}: {question[:50]}...")

        # Fazer pergunta
        try:
            response, latency_ms = await self.ask_question(question)
            actual_answer = response.get("answer", "")
        except Exception as e:
            logger.error(f"Failed to get answer: {e}")
            return EvaluationMetrics(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                actual_answer=f"ERROR: {e}",
            )

        # Usar citações diretas do endpoint (se disponíveis)
        citations = response.get("citations", [])
        if not citations:
            # Fallback: extrair do texto
            citations = self._extract_citations(actual_answer)

        confidence = response.get("confidence", 0.0)
        if confidence == 0.0:
            # Fallback: extrair do texto
            confidence = self._extract_confidence(actual_answer)

        # Calcular métricas
        keyword_coverage, found_kw, missing_kw = self._calculate_keyword_coverage(
            actual_answer, keywords
        )

        source_accuracy, found_sources = self._calculate_source_accuracy(
            citations, expected_sources
        )

        answer_relevance = self._calculate_answer_relevance(actual_answer, expected_answer)
        citation_quality = self._calculate_citation_quality(citations, actual_answer)

        # Groundedness = tem citações válidas
        groundedness = min(len(citations) / 2, 1.0) if citations else 0.0

        return EvaluationMetrics(
            question_id=question_id,
            question=question,
            groundedness=groundedness,
            keyword_coverage=keyword_coverage,
            source_accuracy=source_accuracy,
            answer_relevance=answer_relevance,
            citation_quality=citation_quality,
            expected_keywords=keywords,
            found_keywords=found_kw,
            missing_keywords=missing_kw,
            expected_sources=[s.strip() for s in expected_sources.split(";") if s.strip()],
            found_sources=found_sources,
            citations_count=len(citations),
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    async def run_evaluation(
        self,
        question_ids: list[int] | None = None,
        max_questions: int | None = None,
    ) -> EvaluationReport:
        """Executa avaliação completa.

        Args:
            question_ids: IDs específicos para avaliar (None = todas)
            max_questions: Limite de perguntas (None = todas)

        Returns:
            EvaluationReport com resultados
        """
        report = EvaluationReport()

        # Filtrar perguntas
        qa_to_evaluate = self.qa_pairs

        if question_ids:
            qa_to_evaluate = [q for q in qa_to_evaluate if q.get("id") in question_ids]

        if max_questions:
            qa_to_evaluate = qa_to_evaluate[:max_questions]

        logger.info(f"Starting evaluation of {len(qa_to_evaluate)} questions...")

        for qa_pair in qa_to_evaluate:
            metrics = await self.evaluate_question(qa_pair)
            report.add_result(metrics)

            status = "✅" if metrics.passed else "❌"
            logger.info(
                f"  Q{metrics.question_id}: {status} "
                f"(score={metrics.overall_score:.1%}, latency={metrics.latency_ms:.0f}ms)"
            )

        report.generate_recommendations()

        logger.info(f"Evaluation complete. Pass rate: {report.pass_rate:.1%}")

        return report


# =============================================================================
# Singleton
# =============================================================================

_evaluator: RAGEvaluator | None = None


def get_evaluator() -> RAGEvaluator:
    """Retorna instância singleton do avaliador."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator
