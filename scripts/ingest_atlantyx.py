#!/usr/bin/env python3
"""Script para ingerir documentos do desafio Atlantyx.

Uso:
    # Ingerir todos os documentos
    python scripts/ingest_atlantyx.py

    # Reingerir (limpa base antes)
    python scripts/ingest_atlantyx.py --reingest

    # Verificar status
    python scripts/ingest_atlantyx.py --status

    # Listar documentos na pasta
    python scripts/ingest_atlantyx.py --list
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pasta padrão dos documentos Atlantyx
ATLANTYX_DOCS_PATH = Path(__file__).parent.parent / "claude_rag_sdk" / "ingest"

# Documentos esperados
EXPECTED_DOCUMENTS = [
    "Doc1_Politica_IA_Grandes_Empresas_v1_2.docx",
    "Doc2_Playbook_Implantacao_IA_Enterprise_v0_9.docx",
    "PDF1_Arquitetura_Referencia_RAG_Enterprise.pdf",
    "PDF2_Matriz_Riscos_Controles_IA.pdf",
    "HTML1_FAQ_Glossario_IA_Grandes_Empresas.html",
    "HTML2_Caso_Uso_Roadmap_IA_Empresa_X.html",
]


def list_documents():
    """Lista documentos na pasta de ingestão."""
    print(f"\n📁 Pasta: {ATLANTYX_DOCS_PATH}\n")

    if not ATLANTYX_DOCS_PATH.exists():
        print("❌ Pasta não existe!")
        return

    found = []
    missing = []

    for doc in EXPECTED_DOCUMENTS:
        doc_path = ATLANTYX_DOCS_PATH / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            found.append((doc, size))
        else:
            missing.append(doc)

    print("✅ Documentos encontrados:")
    for doc, size in found:
        print(f"   - {doc} ({size:,} bytes)")

    if missing:
        print("\n❌ Documentos faltando:")
        for doc in missing:
            print(f"   - {doc}")

    # Outros arquivos na pasta
    other_files = [
        f.name for f in ATLANTYX_DOCS_PATH.iterdir()
        if f.is_file() and f.name not in EXPECTED_DOCUMENTS
    ]
    if other_files:
        print("\n📄 Outros arquivos na pasta:")
        for f in other_files:
            print(f"   - {f}")

    print(f"\n📊 Status: {len(found)}/{len(EXPECTED_DOCUMENTS)} documentos")


async def check_status():
    """Verifica status da base RAG."""
    from claude_rag_sdk.core.config import get_config

    config = get_config()
    db_path = config.rag_db_path

    print(f"\n📊 Status da Base RAG")
    print(f"   DB Path: {db_path}")
    print(f"   Existe: {'✅' if db_path.exists() else '❌'}")

    if db_path.exists():
        size = db_path.stat().st_size
        print(f"   Tamanho: {size:,} bytes ({size / 1024:.1f} KB)")

        # Contar documentos
        import sqlite3
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documentos")
                count = cursor.fetchone()[0]
                print(f"   Documentos: {count}")

                # Listar fontes
                cursor.execute("SELECT DISTINCT nome FROM documentos LIMIT 10")
                sources = [row[0] for row in cursor.fetchall()]
                print(f"   Fontes:")
                for s in sources:
                    print(f"      - {s}")
        except Exception as e:
            print(f"   ⚠️ Erro ao ler DB: {e}")


async def clear_database():
    """Limpa a base RAG."""
    from claude_rag_sdk.core.config import get_config
    import os

    config = get_config()
    db_path = config.rag_db_path

    if not db_path.exists():
        print("   ℹ️ Base já está vazia")
        return

    # Deletar arquivos
    deleted = []
    for suffix in ["", "-wal", "-shm"]:
        path = Path(str(db_path) + suffix)
        if path.exists():
            os.remove(path)
            deleted.append(path.name)

    print(f"   🗑️ Deletados: {', '.join(deleted)}")


async def ingest_documents(reingest: bool = False):
    """Ingere documentos do Atlantyx."""
    from claude_rag_sdk.ingest import IngestEngine
    from claude_rag_sdk.core.config import get_config

    config = get_config()

    print("\n🚀 Ingestão de Documentos Atlantyx")
    print("=" * 50)

    # Verificar pasta
    if not ATLANTYX_DOCS_PATH.exists():
        print(f"❌ Pasta não encontrada: {ATLANTYX_DOCS_PATH}")
        return False

    # Listar arquivos para ingerir
    files_to_ingest = []
    for f in ATLANTYX_DOCS_PATH.iterdir():
        if f.is_file() and f.suffix.lower() in [".docx", ".pdf", ".html", ".txt", ".md"]:
            files_to_ingest.append(f)

    if not files_to_ingest:
        print("❌ Nenhum documento encontrado para ingerir")
        return False

    print(f"📄 {len(files_to_ingest)} documentos para ingerir:")
    for f in files_to_ingest:
        print(f"   - {f.name}")

    # Reingestão: limpar base primeiro
    if reingest:
        print("\n🗑️ Limpando base existente...")
        await clear_database()

    # Criar engine de ingestão
    print("\n⚙️ Inicializando engine de ingestão...")
    engine = IngestEngine(
        db_path=str(config.rag_db_path),
        embedding_model=config.embedding_model_string,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # Ingerir cada arquivo
    print("\n📥 Ingerindo documentos...")
    success_count = 0
    error_count = 0

    for file_path in files_to_ingest:
        try:
            print(f"\n   📄 {file_path.name}...")

            # Usar add_document que detecta tipo automaticamente
            result = await engine.add_document(str(file_path))

            if result and result.success:
                print(f"      ✅ Ingerido (doc_id: {result.doc_id})")
                success_count += 1
            else:
                error_msg = result.error if result else "Sem resultado"
                print(f"      ❌ Erro: {error_msg}")
                error_count += 1

        except Exception as e:
            print(f"      ❌ Erro: {e}")
            error_count += 1

    # Resumo
    print("\n" + "=" * 50)
    print(f"📊 Resumo da Ingestão")
    print(f"   ✅ Sucesso: {success_count}")
    print(f"   ❌ Erros: {error_count}")

    # Verificar status final
    await check_status()

    return error_count == 0


async def main():
    parser = argparse.ArgumentParser(
        description="Ingestão de documentos Atlantyx para RAG"
    )
    parser.add_argument(
        "--reingest", "-r",
        action="store_true",
        help="Limpar base e reingerir tudo",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Verificar status da base",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Listar documentos na pasta",
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Apenas limpar a base (sem reingerir)",
    )

    args = parser.parse_args()

    if args.list:
        list_documents()
        return

    if args.status:
        await check_status()
        return

    if args.clear:
        print("\n🗑️ Limpando base RAG...")
        await clear_database()
        print("✅ Base limpa!")
        return

    # Ingestão padrão
    success = await ingest_documents(reingest=args.reingest)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
