#!/usr/bin/env python3
# =============================================================================
# RECHUNK DOCUMENTS - Script de Re-chunking de Documentos
# =============================================================================
# Reindexa documentos com nova estratégia de chunking (ex: SEMANTIC)
# =============================================================================

import sys
import argparse
from pathlib import Path

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.ingestion_pipeline import IngestionPipeline
from ingest.chunker import ChunkingStrategy
from core.config import get_config


def main():
    parser = argparse.ArgumentParser(
        description="Re-chunkar documentos com nova estratégia"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["fixed_size", "sentence", "paragraph", "semantic"],
        help="Nova estratégia de chunking (padrão: semantic)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Tamanho do chunk (padrão: config)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap em tokens (padrão: config)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Caminho do banco de dados (padrão: config)",
    )

    args = parser.parse_args()

    # DB path
    if args.db_path:
        db_path = args.db_path
    else:
        config = get_config()
        db_path = str(config.db_path)

    print("=" * 60)
    print("🔄 RE-CHUNKING DE DOCUMENTOS")
    print("=" * 60)
    print(f"Banco de dados: {db_path}")
    print(f"Nova estratégia: {args.strategy}")
    if args.chunk_size:
        print(f"Chunk size: {args.chunk_size}")
    if args.overlap:
        print(f"Overlap: {args.overlap}")
    print("=" * 60)

    confirm = input("\n⚠️  Isso irá reprocessar TODOS os documentos. Continuar? (sim/não): ")
    if confirm.lower() != "sim":
        print("❌ Operação cancelada.")
        return

    # Criar pipeline
    pipeline = IngestionPipeline(
        db_path=db_path,
        chunking_strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    # Processar todos os documentos
    print("\n🚀 Iniciando re-chunking...\n")
    results = pipeline.ingest_all_documents()

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📊 RESUMO DO RE-CHUNKING")
    print("=" * 60)

    success_count = 0
    failed_count = 0
    total_chunks = 0

    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} [{result.doc_id}] {result.doc_name}")
        if result.success:
            print(f"   Chunks: {result.chunks_created} | Embeddings: {result.embeddings_created}")
            success_count += 1
            total_chunks += result.chunks_created
        else:
            print(f"   Erro: {result.error}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"Total processado: {len(results)}")
    print(f"Sucesso: {success_count}")
    print(f"Falhas: {failed_count}")
    print(f"Total de chunks criados: {total_chunks}")
    print("=" * 60)

    if success_count == len(results):
        print("\n🎉 Re-chunking completo com sucesso!")
    else:
        print(f"\n⚠️  Re-chunking completo com {failed_count} falhas.")


if __name__ == "__main__":
    main()
