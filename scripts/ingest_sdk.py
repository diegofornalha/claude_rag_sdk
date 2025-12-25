#!/usr/bin/env python3
"""
Script para ingerir o Claude Agent SDK Python no RAG.

Uso:
    python scripts/ingest_sdk.py
"""

import asyncio
import sys
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_rag_sdk.ingest import IngestEngine
from claude_rag_sdk.options import ChunkingStrategy


# Use Path.home() for portability across different environments
SDK_PATH = Path.home() / ".claude" / "claude-agent-sdk-python"
BACKEND_PATH = Path(__file__).parent.parent

IGNORE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    ".pytest_cache", ".ruff_cache", "node_modules",
    ".egg-info", "build", "dist", ".mypy_cache",
}


def should_ignore(path: Path) -> bool:
    """Verifica se deve ignorar arquivo/diretório."""
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    if path.name.startswith('.') and path.name not in ['.env.example']:
        return True
    if path.suffix in ['.pyc', '.pyo', '.db', '.db-wal', '.db-shm']:
        return True
    return False


def collect_sdk_files() -> list[tuple[Path, str]]:
    """Coleta arquivos Python do SDK."""
    files = []

    if not SDK_PATH.exists():
        print(f"⚠️  SDK path não encontrado: {SDK_PATH}")
        return files

    # Coletar todos os .py
    for py_file in SDK_PATH.rglob("*.py"):
        if should_ignore(py_file):
            continue

        rel_path = py_file.relative_to(SDK_PATH)
        source = f"Claude Agent SDK - {rel_path}"
        files.append((py_file, source))

    # Adicionar README e CHANGELOG se existirem
    for doc in ["README.md", "CHANGELOG.md", "CLAUDE.md"]:
        doc_path = SDK_PATH / doc
        if doc_path.exists():
            files.append((doc_path, f"Claude Agent SDK - {doc}"))

    return files


async def main():
    db_path = BACKEND_PATH / "data" / "rag_knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📦 Claude Agent SDK Python")
    print(f"📂 Caminho: {SDK_PATH}")
    print(f"💾 Banco de dados: {db_path}")

    # Coletar arquivos
    all_files = collect_sdk_files()
    print(f"📄 Arquivos no SDK: {len(all_files)}")

    if not all_files:
        print("\n⚠️  Nenhum arquivo encontrado no SDK")
        return

    print(f"🔄 Ingerindo {len(all_files)} arquivos do SDK...\n")

    # Criar engine
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=1500,
        chunk_overlap=150,
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
    )

    success_count = 0
    error_count = 0
    total_chunks = 0

    for file_path, source in all_files:
        try:
            content = file_path.read_text(encoding="utf-8")

            # Adicionar header
            header = f"# Arquivo: {source}\n# Path: {file_path.relative_to(SDK_PATH)}\n\n"
            content_with_header = header + content

            result = await engine.add_text(
                content=content_with_header,
                source=source,
                doc_type=file_path.suffix[1:] if file_path.suffix else "txt",
                metadata={
                    "project": "claude-agent-sdk",
                    "file_path": str(file_path.relative_to(SDK_PATH)),
                    "file_type": file_path.suffix,
                }
            )

            if result.success:
                success_count += 1
                total_chunks += result.chunks
                print(f"✅ {source} ({result.chunks} chunks)")
            else:
                if "already exists" in str(result.error).lower():
                    print(f"⏭️  {source} (já existe)")
                else:
                    error_count += 1
                    print(f"❌ {source}: {result.error}")

        except Exception as e:
            error_count += 1
            print(f"❌ {source}: {e}")

    print()
    print("=" * 50)
    print(f"📊 Resumo da Ingestão SDK")
    print(f"   - Arquivos processados: {len(all_files)}")
    print(f"   - Novos/Atualizados: {success_count}")
    print(f"   - Erros: {error_count}")
    print(f"   - Total chunks: {total_chunks}")
    print()

    # Stats do banco
    stats = engine.stats
    print(f"📊 Estatísticas do banco:")
    print(f"   - Total documentos: {stats['total_documents']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    print(f"   - Status: {stats['status']}")


if __name__ == "__main__":
    asyncio.run(main())
