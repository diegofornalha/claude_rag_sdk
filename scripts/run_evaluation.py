#!/usr/bin/env python3
"""Script para executar avaliação do RAG Agent.

Uso:
    # Avaliar todas as perguntas
    python scripts/run_evaluation.py

    # Avaliar perguntas específicas
    python scripts/run_evaluation.py --questions 1,2,3

    # Avaliar com limite
    python scripts/run_evaluation.py --max 5

    # Salvar relatório em JSON
    python scripts/run_evaluation.py --output results.json

    # Modo verbose
    python scripts/run_evaluation.py --verbose
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.evaluator import RAGEvaluator


async def main():
    parser = argparse.ArgumentParser(
        description="Executa avaliação do RAG Agent Atlantyx"
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        help="IDs das perguntas separados por vírgula (ex: 1,2,3)",
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        help="Número máximo de perguntas",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Arquivo de saída para relatório JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose (mostra respostas completas)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8001",
        help="URL da API (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API Key (opcional, lê do .env se não fornecida)",
    )

    args = parser.parse_args()

    # Inicializar avaliador
    evaluator = RAGEvaluator()
    evaluator.api_url = args.api_url

    if args.api_key:
        evaluator.set_api_key(args.api_key)

    # Parsear IDs das perguntas
    question_ids = None
    if args.questions:
        try:
            question_ids = [int(q.strip()) for q in args.questions.split(",")]
        except ValueError:
            print("❌ Erro: IDs das perguntas devem ser números separados por vírgula")
            sys.exit(1)

    print("=" * 60)
    print("🧪 AVALIAÇÃO DO RAG AGENT - ATLANTYX")
    print("=" * 60)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API: {evaluator.api_url}")
    print(f"📝 Perguntas: {question_ids or 'Todas'}")
    print(f"🔢 Máximo: {args.max or 'Sem limite'}")
    print("=" * 60)
    print()

    # Executar avaliação
    try:
        report = await evaluator.run_evaluation(
            question_ids=question_ids,
            max_questions=args.max,
        )
    except Exception as e:
        print(f"❌ Erro na avaliação: {e}")
        sys.exit(1)

    # Mostrar resumo
    print()
    print(report.summary())

    # Modo verbose: mostrar detalhes
    if args.verbose:
        print()
        print("=" * 60)
        print("📋 DETALHES POR PERGUNTA")
        print("=" * 60)

        for result in report.results:
            status = "✅ PASSOU" if result.passed else "❌ FALHOU"
            print(f"\n--- Q{result.question_id}: {status} ---")
            print(f"Pergunta: {result.question[:80]}...")
            print(f"Score: {result.overall_score:.1%}")
            print(f"  - Groundedness: {result.groundedness:.1%}")
            print(f"  - Keywords: {result.keyword_coverage:.1%} ({len(result.found_keywords)}/{len(result.expected_keywords)})")
            print(f"  - Sources: {result.source_accuracy:.1%}")
            print(f"  - Relevance: {result.answer_relevance:.1%}")
            print(f"  - Citations: {result.citation_quality:.1%} ({result.citations_count} citações)")
            print(f"Latência: {result.latency_ms:.0f}ms")

            if result.missing_keywords:
                print(f"Keywords faltando: {', '.join(result.missing_keywords[:5])}")

            print(f"\nResposta (primeiros 300 chars):")
            print(f"  {result.actual_answer[:300]}...")

    # Salvar relatório
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n📄 Relatório salvo em: {output_path}")

    # Exit code baseado no resultado
    if report.pass_rate < 0.6:
        print("\n⚠️ Taxa de aprovação abaixo de 60%!")
        sys.exit(1)

    print("\n✅ Avaliação concluída!")


if __name__ == "__main__":
    asyncio.run(main())
