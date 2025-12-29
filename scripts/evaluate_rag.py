#!/usr/bin/env python3
"""
Script de Avaliação Automática do RAG.

Executa perguntas do CSV de teste e gera relatório com métricas.

Uso:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --csv path/to/questions.csv
    python scripts/evaluate_rag.py --output reports/evaluation.json
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QuestionResult:
    """Resultado de uma pergunta avaliada."""

    id: int
    question: str
    expected_sources: list[str]
    expected_keywords: list[str]
    answer: str
    citations: list[dict]
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    sources_found: list[str]
    keywords_found: list[str]
    source_match: bool
    keyword_match: bool
    has_citations: bool
    success: bool
    error: str | None = None


@dataclass
class EvaluationReport:
    """Relatório completo de avaliação."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0

    # Métricas de citação
    questions_with_citations: int = 0
    total_citations: int = 0
    avg_citations_per_question: float = 0.0

    # Métricas de fonte
    source_matches: int = 0
    source_match_rate: float = 0.0

    # Métricas de keywords
    keyword_matches: int = 0
    keyword_match_rate: float = 0.0

    # Métricas de confiança
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0

    # Métricas de performance
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Métricas de tokens
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # Resultados detalhados
    results: list = field(default_factory=list)


def parse_csv(csv_path: Path) -> list[dict]:
    """Lê e parseia o CSV de perguntas."""
    questions = []

    # Tentar diferentes encodings
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    content = None

    for encoding in encodings:
        try:
            with open(csv_path, encoding=encoding) as f:
                content = f.read()
                break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Could not decode CSV file with any encoding: {encodings}")

    # Detectar delimitador
    if ";" in content[:1024]:
        delimiter = ";"
    else:
        delimiter = ","

    # Usar StringIO para processar como arquivo
    from io import StringIO
    reader = csv.DictReader(StringIO(content), delimiter=delimiter)

    for row in reader:
        # Normalizar nomes de colunas (remover espaços, lowercase)
        normalized = {k.strip().lower(): v for k, v in row.items()}

        # Extrair campos
        q_id = normalized.get("id", "0")
        question = normalized.get("question", "")
        sources = normalized.get("sources", "")
        keywords = normalized.get("evidence_keywords", "")

        if question:
            questions.append({
                "id": int(q_id) if q_id.isdigit() else 0,
                "question": question.strip(),
                "sources": [s.strip() for s in sources.split("|") if s.strip()],
                "keywords": [k.strip() for k in keywords.split(";") if k.strip()],
            })

    return questions


async def ask_question(
    client: httpx.AsyncClient,
    question: str,
    api_url: str,
    api_key: str,
) -> dict:
    """Envia pergunta para o endpoint /rag/ask."""
    try:
        # Tentar primeiro o endpoint de teste (sem auth)
        response = await client.get(
            f"{api_url}/rag/ask/test",
            params={"question": question, "top_k": 5},
            timeout=120.0,
        )

        # Se falhar, tentar com autenticação
        if response.status_code == 404:
            response = await client.post(
                f"{api_url}/rag/ask",
                params={"question": question, "top_k": 5},
                headers={"X-API-Key": api_key},
                timeout=120.0,
            )

        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
                "sources_count": 0,
            },
            "error": str(e),
        }


def check_source_match(citations: list[dict], expected_sources: list[str]) -> tuple[bool, list[str]]:
    """Verifica se as citações correspondem às fontes esperadas."""
    found_sources = []

    for citation in citations:
        source = citation.get("source", "")
        found_sources.append(source)

    if not expected_sources:
        return True, found_sources

    # Verificar se pelo menos uma fonte esperada foi citada
    for expected in expected_sources:
        expected_lower = expected.lower()
        for found in found_sources:
            if expected_lower in found.lower():
                return True, found_sources

    return False, found_sources


def check_keyword_match(answer: str, expected_keywords: list[str]) -> tuple[bool, list[str]]:
    """Verifica se a resposta contém as keywords esperadas."""
    found_keywords = []
    answer_lower = answer.lower()

    for keyword in expected_keywords:
        keyword_lower = keyword.lower().strip()
        if keyword_lower and keyword_lower in answer_lower:
            found_keywords.append(keyword)

    if not expected_keywords:
        return True, found_keywords

    # Sucesso se encontrou pelo menos metade das keywords
    match = len(found_keywords) >= len(expected_keywords) / 2
    return match, found_keywords


async def evaluate_questions(
    questions: list[dict],
    api_url: str,
    api_key: str,
    verbose: bool = True,
) -> EvaluationReport:
    """Executa avaliação de todas as perguntas."""
    report = EvaluationReport()
    report.total_questions = len(questions)

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions):
            if verbose:
                print(f"\n[{i+1}/{len(questions)}] Pergunta: {q['question'][:60]}...")

            # Fazer pergunta
            start_time = time.time()
            result = await ask_question(client, q["question"], api_url, api_key)

            # Extrair dados
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            confidence = result.get("confidence", 0.0)
            metrics = result.get("metrics", {})
            error = result.get("error")

            latency_ms = metrics.get("latency_ms", (time.time() - start_time) * 1000)
            input_tokens = metrics.get("input_tokens", 0)
            output_tokens = metrics.get("output_tokens", 0)

            # Verificar matches
            source_match, sources_found = check_source_match(citations, q["sources"])
            keyword_match, keywords_found = check_keyword_match(answer, q["keywords"])

            # Criar resultado
            question_result = QuestionResult(
                id=q["id"],
                question=q["question"],
                expected_sources=q["sources"],
                expected_keywords=q["keywords"],
                answer=answer,
                citations=citations,
                confidence=confidence,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                sources_found=sources_found,
                keywords_found=keywords_found,
                source_match=source_match,
                keyword_match=keyword_match,
                has_citations=len(citations) > 0,
                success=not error,
                error=error,
            )

            # Atualizar relatório
            if question_result.success:
                report.successful_questions += 1
            else:
                report.failed_questions += 1

            if question_result.has_citations:
                report.questions_with_citations += 1
                report.total_citations += len(citations)

            if source_match:
                report.source_matches += 1

            if keyword_match:
                report.keyword_matches += 1

            # Métricas de confiança
            report.avg_confidence += confidence
            report.min_confidence = min(report.min_confidence, confidence)
            report.max_confidence = max(report.max_confidence, confidence)

            # Métricas de latência
            report.total_latency_ms += latency_ms
            report.min_latency_ms = min(report.min_latency_ms, latency_ms)
            report.max_latency_ms = max(report.max_latency_ms, latency_ms)

            # Métricas de tokens
            report.total_input_tokens += input_tokens
            report.total_output_tokens += output_tokens

            # Adicionar resultado
            report.results.append({
                "id": question_result.id,
                "question": question_result.question,
                "answer": question_result.answer[:500] + "..." if len(question_result.answer) > 500 else question_result.answer,
                "citations": question_result.citations,
                "confidence": question_result.confidence,
                "latency_ms": round(question_result.latency_ms, 2),
                "source_match": question_result.source_match,
                "keyword_match": question_result.keyword_match,
                "has_citations": question_result.has_citations,
                "success": question_result.success,
                "error": question_result.error,
            })

            if verbose:
                status = "✅" if question_result.success and question_result.has_citations else "⚠️"
                print(f"   {status} Confiança: {confidence:.2f} | Citações: {len(citations)} | Latência: {latency_ms:.0f}ms")

    # Calcular médias
    n = report.total_questions
    if n > 0:
        report.avg_confidence /= n
        report.avg_latency_ms = report.total_latency_ms / n
        report.avg_input_tokens = report.total_input_tokens / n
        report.avg_output_tokens = report.total_output_tokens / n
        report.avg_citations_per_question = report.total_citations / n
        report.source_match_rate = report.source_matches / n
        report.keyword_match_rate = report.keyword_matches / n

    return report


def generate_report(report: EvaluationReport, output_path: Path) -> None:
    """Gera relatório em JSON e texto."""
    # JSON report
    json_data = {
        "timestamp": report.timestamp,
        "summary": {
            "total_questions": report.total_questions,
            "successful_questions": report.successful_questions,
            "failed_questions": report.failed_questions,
            "success_rate": round(report.successful_questions / max(report.total_questions, 1), 4),
        },
        "citations": {
            "questions_with_citations": report.questions_with_citations,
            "total_citations": report.total_citations,
            "avg_citations_per_question": round(report.avg_citations_per_question, 2),
            "citation_rate": round(report.questions_with_citations / max(report.total_questions, 1), 4),
        },
        "accuracy": {
            "source_matches": report.source_matches,
            "source_match_rate": round(report.source_match_rate, 4),
            "keyword_matches": report.keyword_matches,
            "keyword_match_rate": round(report.keyword_match_rate, 4),
        },
        "confidence": {
            "avg_confidence": round(report.avg_confidence, 4),
            "min_confidence": round(report.min_confidence, 4),
            "max_confidence": round(report.max_confidence, 4),
        },
        "performance": {
            "avg_latency_ms": round(report.avg_latency_ms, 2),
            "min_latency_ms": round(report.min_latency_ms, 2),
            "max_latency_ms": round(report.max_latency_ms, 2),
            "total_latency_ms": round(report.total_latency_ms, 2),
        },
        "tokens": {
            "total_input_tokens": report.total_input_tokens,
            "total_output_tokens": report.total_output_tokens,
            "avg_input_tokens": round(report.avg_input_tokens, 2),
            "avg_output_tokens": round(report.avg_output_tokens, 2),
        },
        "results": report.results,
    }

    # Salvar JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Gerar texto legível
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE AVALIAÇÃO DO RAG\n")
        f.write(f"Gerado em: {report.timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("RESUMO\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de perguntas: {report.total_questions}\n")
        f.write(f"Sucesso: {report.successful_questions} ({report.successful_questions/max(report.total_questions,1)*100:.1f}%)\n")
        f.write(f"Falhas: {report.failed_questions}\n\n")

        f.write("CITAÇÕES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Perguntas com citações: {report.questions_with_citations} ({report.questions_with_citations/max(report.total_questions,1)*100:.1f}%)\n")
        f.write(f"Total de citações: {report.total_citations}\n")
        f.write(f"Média por pergunta: {report.avg_citations_per_question:.2f}\n\n")

        f.write("PRECISÃO\n")
        f.write("-" * 40 + "\n")
        f.write(f"Match de fontes: {report.source_matches}/{report.total_questions} ({report.source_match_rate*100:.1f}%)\n")
        f.write(f"Match de keywords: {report.keyword_matches}/{report.total_questions} ({report.keyword_match_rate*100:.1f}%)\n\n")

        f.write("CONFIANÇA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Média: {report.avg_confidence:.2f}\n")
        f.write(f"Mínima: {report.min_confidence:.2f}\n")
        f.write(f"Máxima: {report.max_confidence:.2f}\n\n")

        f.write("PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Latência média: {report.avg_latency_ms:.0f}ms\n")
        f.write(f"Latência mínima: {report.min_latency_ms:.0f}ms\n")
        f.write(f"Latência máxima: {report.max_latency_ms:.0f}ms\n")
        f.write(f"Latência total: {report.total_latency_ms/1000:.1f}s\n\n")

        f.write("TOKENS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input total: {report.total_input_tokens:,}\n")
        f.write(f"Output total: {report.total_output_tokens:,}\n")
        f.write(f"Input médio: {report.avg_input_tokens:.0f}\n")
        f.write(f"Output médio: {report.avg_output_tokens:.0f}\n\n")

        f.write("=" * 60 + "\n")
        f.write("RESULTADOS DETALHADOS\n")
        f.write("=" * 60 + "\n\n")

        for r in report.results:
            status = "✅" if r["success"] and r["has_citations"] else "⚠️" if r["success"] else "❌"
            f.write(f"{status} Pergunta {r['id']}: {r['question'][:60]}...\n")
            f.write(f"   Confiança: {r['confidence']:.2f} | Citações: {len(r['citations'])} | Latência: {r['latency_ms']:.0f}ms\n")
            f.write(f"   Source match: {'Sim' if r['source_match'] else 'Não'} | Keyword match: {'Sim' if r['keyword_match'] else 'Não'}\n")
            if r["error"]:
                f.write(f"   ERRO: {r['error']}\n")
            f.write("\n")

    print("\n📊 Relatórios gerados:")
    print(f"   JSON: {output_path}")
    print(f"   TXT:  {txt_path}")


async def main():
    parser = argparse.ArgumentParser(description="Avaliação automática do RAG")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent / "claude_rag_sdk" / "ingest" / "Prova_Atlantyx_IA_Senior_10_Perguntas a serem respondidas pela IA com base nos documentos.csv",
        help="Caminho do CSV com perguntas",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "reports" / "evaluation.json",
        help="Caminho do relatório de saída",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8001"),
        help="URL base da API",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="API key para autenticação",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso (sem output verbose)",
    )

    args = parser.parse_args()

    # Carregar API key do .env se não fornecida
    if not args.api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("RAG_API_KEY=") or line.startswith("API_KEY="):
                        args.api_key = line.split("=", 1)[1].strip().strip('"')
                        break

    if not args.api_key:
        print("❌ API_KEY não encontrada. Use --api-key ou defina no .env")
        sys.exit(1)

    # Verificar CSV
    if not args.csv.exists():
        print(f"❌ CSV não encontrado: {args.csv}")
        sys.exit(1)

    print("=" * 60)
    print("AVALIAÇÃO AUTOMÁTICA DO RAG")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"API: {args.api_url}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Parsear perguntas
    questions = parse_csv(args.csv)
    print(f"\n📋 {len(questions)} perguntas carregadas do CSV")

    # Executar avaliação
    report = await evaluate_questions(
        questions,
        args.api_url,
        args.api_key,
        verbose=not args.quiet,
    )

    # Gerar relatório
    generate_report(report, args.output)

    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"✅ Sucesso: {report.successful_questions}/{report.total_questions}")
    print(f"📝 Citações: {report.questions_with_citations}/{report.total_questions} perguntas")
    print(f"🎯 Match fontes: {report.source_match_rate*100:.1f}%")
    print(f"🔑 Match keywords: {report.keyword_match_rate*100:.1f}%")
    print(f"📊 Confiança média: {report.avg_confidence:.2f}")
    print(f"⏱️ Latência média: {report.avg_latency_ms:.0f}ms")
    print(f"🔢 Tokens totais: {report.total_input_tokens + report.total_output_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
