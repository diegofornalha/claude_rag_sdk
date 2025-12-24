#!/usr/bin/env python3
"""Script para adicionar documentos de meta-documentação ao banco."""

import sqlite3
from pathlib import Path

# Paths
db_path = Path(__file__).parent / "documentos.db"
meta_docs_dir = Path(__file__).parent / "meta_docs"

print(f"📂 Banco: {db_path}")
print(f"📂 Meta-docs: {meta_docs_dir}")
print()

# Conectar ao banco
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Processar cada documento
docs_added = 0
for doc_file in sorted(meta_docs_dir.glob("*.md")):
    content = doc_file.read_text(encoding='utf-8')
    nome = doc_file.name
    tipo = "markdown"
    caminho = str(doc_file.absolute())

    print(f"➕ Adicionando: {nome}")
    print(f"   Tamanho: {len(content)} bytes")

    cursor.execute("""
        INSERT INTO documentos (nome, tipo, conteudo, caminho)
        VALUES (?, ?, ?, ?)
    """, (nome, tipo, content, caminho))

    docs_added += 1

conn.commit()
conn.close()

print()
print(f"✅ {docs_added} documentos adicionados com sucesso!")
print()
print("📊 Próximo passo:")
print("   cd /Users/2a/.claude/hello-agent/chat-simples/backend/rag-agent")
print("   python3 -c \"from ingest.ingestion_pipeline import IngestionPipeline; IngestionPipeline().ingest_all_documents()\"")
