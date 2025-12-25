#!/usr/bin/env python3
"""
Script para ingerir o Claude Agent SDK Python no RAG.

Uso:
    python scripts/ingest_claude_agent_sdk.py          # Ingestão incremental (só novos/modificados)
    python scripts/ingest_claude_agent_sdk.py --full   # Ingestão completa (apaga e reingere)
    python scripts/ingest_claude_agent_sdk.py --stats  # Mostra estatísticas sem ingerir
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_rag_sdk.ingest import IngestEngine
from claude_rag_sdk.options import ChunkingStrategy

SDK_PATH = Path.home() / ".claude" / "claude-agent-sdk-python"
CACHE_FILE = Path(__file__).parent.parent / "data" / ".sdk_ingest_cache.json"


def collect_sdk_files() -> list[tuple[Path, str]]:
    """Coleta arquivos relevantes do SDK."""
    files = []

    # Documentação principal
    docs = [
        ("README.md", "Claude Agent SDK - README"),
        ("CHANGELOG.md", "Claude Agent SDK - Changelog"),
        ("CLAUDE.md", "Claude Agent SDK - CLAUDE.md"),
    ]

    for filename, source in docs:
        path = SDK_PATH / filename
        if path.exists():
            files.append((path, source))

    # Código fonte principal
    src_path = SDK_PATH / "src"
    if src_path.exists():
        for py_file in src_path.rglob("*.py"):
            rel_path = py_file.relative_to(SDK_PATH)
            files.append((py_file, f"Claude Agent SDK - {rel_path}"))

    # Exemplos
    examples_path = SDK_PATH / "examples"
    if examples_path.exists():
        for py_file in examples_path.rglob("*.py"):
            rel_path = py_file.relative_to(SDK_PATH)
            files.append((py_file, f"Claude Agent SDK - {rel_path}"))

    # Tests (úteis para entender comportamento)
    tests_path = SDK_PATH / "tests"
    if tests_path.exists():
        for py_file in tests_path.rglob("*.py"):
            rel_path = py_file.relative_to(SDK_PATH)
            files.append((py_file, f"Claude Agent SDK - {rel_path}"))

    return files


def load_cache() -> dict:
    """Carrega cache de arquivos já ingeridos."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    """Salva cache de arquivos ingeridos."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_file_hash(file_path: Path) -> str:
    """Calcula hash MD5 do arquivo."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def filter_modified_files(files: list[tuple[Path, str]], cache: dict) -> list[tuple[Path, str]]:
    """Filtra apenas arquivos novos ou modificados."""
    modified = []
    for file_path, source in files:
        file_hash = get_file_hash(file_path)
        cached_hash = cache.get(str(file_path))

        if cached_hash != file_hash:
            modified.append((file_path, source))

    return modified


async def main():
    # Processar argumentos
    full_mode = "--full" in sys.argv
    stats_only = "--stats" in sys.argv

    # Banco de dados de saída
    db_path = Path(__file__).parent.parent / "data" / "rag_knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("📦 Claude Agent SDK Python")
    print(f"📂 Caminho: {SDK_PATH}")
    print(f"💾 Banco de dados: {db_path}")

    if not SDK_PATH.exists():
        print(f"❌ SDK não encontrado: {SDK_PATH}")
        return

    # Coletar todos os arquivos
    all_files = collect_sdk_files()
    print(f"📄 Arquivos no SDK: {len(all_files)}")

    # Carregar cache
    cache = load_cache()

    if stats_only:
        # Apenas mostrar estatísticas
        modified = filter_modified_files(all_files, cache)
        print("\n📊 Status:")
        print(f"   - Arquivos no cache: {len(cache)}")
        print(f"   - Arquivos modificados/novos: {len(modified)}")
        if modified:
            print("\n📝 Arquivos para atualizar:")
            for f, s in modified[:10]:
                print(f"   - {s}")
            if len(modified) > 10:
                print(f"   ... e mais {len(modified) - 10}")
        return

    # Determinar quais arquivos processar
    if full_mode:
        print("🔄 Modo FULL: reingerindo todos os arquivos")
        files_to_process = all_files
        cache = {}  # Reset cache
    else:
        files_to_process = filter_modified_files(all_files, cache)
        if not files_to_process:
            print("\n✅ Nenhum arquivo modificado. Base está atualizada!")
            return
        print(f"🔄 Modo INCREMENTAL: {len(files_to_process)} arquivos novos/modificados")

    print()

    # Criar engine de ingestão
    # chunk_size maior (1500) para ter mais contexto de código no RAG
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=1500,
        chunk_overlap=150,
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
    )

    # Ingerir cada arquivo
    success_count = 0
    error_count = 0
    skipped_count = 0
    total_chunks = 0

    for file_path, source in files_to_process:
        try:
            content = file_path.read_text(encoding="utf-8")

            # Adicionar header com informações do arquivo
            header = f"# Arquivo: {source}\n# Path: {file_path.relative_to(SDK_PATH)}\n\n"
            content_with_header = header + content

            result = await engine.add_text(
                content=content_with_header,
                source=source,
                doc_type=file_path.suffix[1:] if file_path.suffix else "txt",
                metadata={
                    "sdk": "claude-agent-sdk-python",
                    "file_path": str(file_path.relative_to(SDK_PATH)),
                    "file_type": file_path.suffix,
                },
            )

            if result.success:
                success_count += 1
                total_chunks += result.chunks
                # Salvar hash no cache
                cache[str(file_path)] = get_file_hash(file_path)
                print(f"✅ {source} ({result.chunks} chunks)")
            else:
                if "UNIQUE constraint" in str(result.error):
                    # Arquivo já existe mas conteúdo mudou - atualizar cache
                    cache[str(file_path)] = get_file_hash(file_path)
                    skipped_count += 1
                    print(f"⏭️  {source} (conteúdo idêntico)")
                else:
                    error_count += 1
                    print(f"❌ {source}: {result.error}")

        except Exception as e:
            error_count += 1
            print(f"❌ {source}: {e}")

    # Salvar cache atualizado
    save_cache(cache)

    print()
    print("=" * 50)
    print("📊 Resumo da Ingestão")
    print(f"   - Arquivos processados: {len(files_to_process)}")
    print(f"   - Novos/Atualizados: {success_count}")
    print(f"   - Sem alteração: {skipped_count}")
    print(f"   - Erros: {error_count}")
    print(f"   - Total chunks criados: {total_chunks}")
    print()

    # Mostrar estatísticas do banco
    print("📊 Estatísticas do banco:")
    stats = engine.stats
    print(f"   - Total documentos: {stats['total_documents']}")
    print(f"   - Total embeddings: {stats['total_embeddings']}")
    print(f"   - Tamanho total: {stats['total_size_bytes']:,} bytes")
    print(f"   - Status: {stats['status']}")

    print("\n💡 Dica: Use --stats para ver arquivos pendentes sem ingerir")


if __name__ == "__main__":
    asyncio.run(main())
