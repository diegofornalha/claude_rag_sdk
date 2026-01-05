#!/usr/bin/env python3
"""Ingest do regulamento - Script standalone."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from claude_rag_sdk.ingest import IngestEngine
from claude_rag_sdk.options import ChunkingStrategy

async def main():
    print("=" * 70)
    print("INGESTÃO DO PDF - REGULAMENTO TON")
    print("=" * 70)

    pdf_path = Path(__file__).parent.parent / "ingest/rendaextra-todos-regulamentos.pdf"
    db_path = Path(__file__).parent.parent / "data/regulamento.db"

    print(f"📄 PDF: {pdf_path.name}")
    print(f"💾 DB:  {db_path}")
    print(f"🧠 Modelo: BAAI/bge-small-en-v1.5")
    print(f"📏 Chunk: 350 palavras (~450 tokens) | Overlap: 70 (20%) | Strategy: FIXED")
    print(f"🎯 Chunks esperados: ~50 | Overlap 20% para contexto ideal")
    print("=" * 70)

    # Criar dir
    db_path.parent.mkdir(exist_ok=True)

    # Engine - Configuração balanceada para regulamentos
    # 350 palavras ≈ 450 tokens | 70 palavras overlap (20%)
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=350,
        chunk_overlap=70,
        chunking_strategy=ChunkingStrategy.FIXED
    )

    # Ingerir
    print(f"\n📥 Ingerindo...")
    result = await engine.add_document(pdf_path)

    if result.success:
        print(f"✅ SUCESSO!")
        print(f"   Doc ID: {result.doc_id}")
        print(f"   Chunks: {result.chunks}")
        print(f"   Source: {result.source}")
    else:
        print(f"❌ ERRO: {result.error}")

    # Stats
    stats = engine.stats
    print(f"\n📊 Estatísticas do Banco:")
    print(f"   Documentos: {stats.get('total_documents', 0)}")
    print(f"   Chunks:     {stats.get('total_embeddings', stats.get('total_chunks', 0))}")
    print(f"   Tamanho:    {stats.get('total_size_bytes', 0):,} bytes")

    print("\n🎉 Ingestão concluída!")
    print(f"   Banco RAG pronto em: {db_path}")

if __name__ == "__main__":
    asyncio.run(main())
