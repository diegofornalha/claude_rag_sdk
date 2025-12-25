#!/usr/bin/env python3
"""Script para ingerir o ebook.txt e criar banco vetorial."""

import asyncio
import sys
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_rag_sdk.ingest import IngestEngine
from claude_rag_sdk.options import ChunkingStrategy


async def main():
    # Caminho do ebook
    ebook_path = Path(__file__).parent.parent / "claude_rag_sdk" / "ingest" / "ebook.txt"

    # Banco de dados de saída (separado do AgentFS para evitar conflitos)
    db_path = Path(__file__).parent.parent / "data" / "rag_knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📚 Ingerindo: {ebook_path}")
    print(f"💾 Banco de dados: {db_path}")

    if not ebook_path.exists():
        print(f"❌ Arquivo não encontrado: {ebook_path}")
        return

    # Criar engine de ingestão
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=500,
        chunk_overlap=50,
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
    )

    print("🔄 Processando documento...")
    result = await engine.add_document(ebook_path)

    if result.success:
        print("✅ Sucesso!")
        print(f"   - Document ID: {result.doc_id}")
        print(f"   - Chunks criados: {result.chunks}")
        print(f"   - Fonte: {result.source}")
        if result.error:
            print(f"   - Nota: {result.error}")
    else:
        print(f"❌ Erro: {result.error}")

    # Mostrar estatísticas
    print("\n📊 Estatísticas do banco:")
    stats = engine.stats
    print(f"   - Total documentos: {stats['total_documents']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    print(f"   - Tamanho total: {stats['total_size_bytes']:,} bytes")
    print(f"   - Status: {stats['status']}")


if __name__ == "__main__":
    asyncio.run(main())
