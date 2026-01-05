#!/usr/bin/env python3
"""
Script para ingerir o backend do chat-simples no RAG.

Uso:
    python scripts/ingest_backend.py          # Ingestão incremental (só novos/modificados)
    python scripts/ingest_backend.py --full   # Ingestão completa (apaga e reingere)
    python scripts/ingest_backend.py --stats  # Mostra estatísticas sem ingerir
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

BACKEND_PATH = Path(__file__).parent.parent
CACHE_FILE = BACKEND_PATH / "data" / ".backend_ingest_cache.json"

# Diretórios e arquivos a ignorar
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "data",  # Ignora banco de dados e cache
    ".egg-info",
    "build",
    "dist",
}

IGNORE_FILES = {
    ".DS_Store",
    ".gitignore",
    "*.pyc",
    "*.pyo",
    "*.db",
    "*.db-wal",
    "*.db-shm",
}


def should_ignore(path: Path) -> bool:
    """Verifica se arquivo/diretório deve ser ignorado."""
    # Ignorar diretórios
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True

    # Ignorar arquivos específicos
    if path.name in IGNORE_FILES:
        return True
    if path.suffix in {".pyc", ".pyo", ".db"}:
        return True

    return False


def collect_backend_files() -> list[tuple[Path, str]]:
    """Coleta arquivos relevantes do backend."""
    files = []

    # Arquivos de configuração na raiz
    root_files = [
        ("server.py", "Backend - server.py (FastAPI principal)"),
        ("app_state.py", "Backend - app_state.py"),
        ("pyproject.toml", "Backend - pyproject.toml"),
        ("requirements.txt", "Backend - requirements.txt"),
        ("CLAUDE.md", "Backend - CLAUDE.md"),
    ]

    for filename, source in root_files:
        path = BACKEND_PATH / filename
        if path.exists():
            files.append((path, source))

    # Claude RAG SDK
    sdk_path = BACKEND_PATH / "claude_rag_sdk"
    if sdk_path.exists():
        for py_file in sdk_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    # Routers
    routers_path = BACKEND_PATH / "routers"
    if routers_path.exists():
        for py_file in routers_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    # Scripts (exceto este próprio)
    scripts_path = BACKEND_PATH / "scripts"
    if scripts_path.exists():
        for py_file in scripts_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            # Não ingerir scripts de ingestão para evitar confusão
            if "ingest" in py_file.name:
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    # Tests
    tests_path = BACKEND_PATH / "tests"
    if tests_path.exists():
        for py_file in tests_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    # Utils
    utils_path = BACKEND_PATH / "utils"
    if utils_path.exists():
        for py_file in utils_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    # AgentFS SDK (submodule)
    agentfs_path = BACKEND_PATH / "agentfs"
    if agentfs_path.exists():
        for py_file in agentfs_path.rglob("*.py"):
            if should_ignore(py_file):
                continue
            rel_path = py_file.relative_to(BACKEND_PATH)
            files.append((py_file, f"Backend - {rel_path}"))

    return files


def load_cache() -> dict:
    """Carrega cache de arquivos já ingeridos."""
    if CACHE_FILE.exists():
        try:
            content = CACHE_FILE.read_text()
            if content.strip():
                return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Cache corrompido, resetando: {e}")
            CACHE_FILE.unlink()
    return {}


def save_cache(cache: dict):
    """Salva cache de arquivos ingeridos."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_file_hash(file_path: Path) -> str:
    """Calcula hash MD5 do arquivo."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def filter_modified_files(
    files: list[tuple[Path, str]], cache: dict, existing_sources: set = None
) -> list[tuple[Path, str]]:
    """Filtra apenas arquivos novos ou modificados.

    Args:
        files: Lista de (Path, source)
        cache: Cache de hashes
        existing_sources: Set de sources que já existem no RAG (opcional)
    """
    modified = []
    for file_path, source in files:
        file_hash = get_file_hash(file_path)
        cached_hash = cache.get(str(file_path))

        # Arquivo novo ou modificado
        if cached_hash != file_hash:
            modified.append((file_path, source))
        # Arquivo no cache mas não no RAG (foi deletado via UI)
        elif existing_sources is not None and source not in existing_sources:
            modified.append((file_path, source))
            print(f"🔄 Detectado deletado via UI: {source}")

    return modified


async def cleanup_deleted_files(
    engine: IngestEngine, current_files: list[tuple[Path, str]], cache: dict
) -> int:
    """Remove documentos do RAG cujos arquivos foram deletados."""
    # Criar set de paths atuais
    current_paths = {str(f[0]) for f in current_files}

    # Identificar arquivos no cache que não existem mais
    deleted_paths = [path for path in cache.keys() if path not in current_paths]

    if not deleted_paths:
        return 0

    # Buscar documentos no RAG para deletar
    deleted_count = 0
    for deleted_path in deleted_paths:
        # Buscar documento por metadata (file_path)
        rel_path = (
            Path(deleted_path).relative_to(BACKEND_PATH)
            if Path(deleted_path).is_absolute()
            else Path(deleted_path)
        )

        # Tentar deletar via source name matching
        try:
            from claude_rag_sdk.search import SearchEngine

            SearchEngine(db_path=str(BACKEND_PATH / "data" / "regulamento.db.db"))

            # Buscar por source exato
            import sqlite3

            with sqlite3.connect(str(BACKEND_PATH / "data" / "regulamento.db.db")) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT id FROM documentos WHERE nome LIKE ?", (f"%{rel_path}%",))
                docs_to_delete = [row[0] for row in cursor.fetchall()]

                for doc_id in docs_to_delete:
                    await engine.delete_document(doc_id)
                    deleted_count += 1
                    print(f"🗑️  Removido: {rel_path}")

            # Remover do cache
            del cache[deleted_path]

        except Exception as e:
            print(f"⚠️  Erro ao remover {rel_path}: {e}")

    return deleted_count


async def main():
    # Processar argumentos
    full_mode = "--full" in sys.argv
    stats_only = "--stats" in sys.argv

    # Banco de dados de saída
    db_path = BACKEND_PATH / "data" / "regulamento.db.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("📦 Backend Chat-Simples")
    print(f"📂 Caminho: {BACKEND_PATH}")
    print(f"💾 Banco de dados: {db_path}")

    # Coletar todos os arquivos
    all_files = collect_backend_files()
    print(f"📄 Arquivos no backend: {len(all_files)}")

    # Carregar cache
    cache = load_cache()

    # Criar engine para cleanup
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=1500,
        chunk_overlap=150,
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
    )

    # Cleanup de arquivos deletados (exceto em modo full)
    if not full_mode and cache:
        deleted_count = await cleanup_deleted_files(engine, all_files, cache)
        if deleted_count > 0:
            print(f"🗑️  {deleted_count} documento(s) removido(s) (arquivos deletados)")
            print()

    # Buscar sources existentes no RAG para detectar deleções via UI
    existing_sources = set()
    if not full_mode:
        try:
            sources_list = (
                await engine.search_engine.list_sources()
                if hasattr(engine, "search_engine")
                else []
            )
            if not sources_list:
                # Buscar diretamente do banco
                import sqlite3

                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.cursor()
                    existing_sources = {
                        row[0] for row in cursor.execute("SELECT nome FROM documentos")
                    }
            else:
                existing_sources = {s["nome"] for s in sources_list}
        except Exception as e:
            print(f"⚠️  Não foi possível verificar sources existentes: {e}")

    if stats_only:
        # Apenas mostrar estatísticas
        modified = filter_modified_files(all_files, cache, existing_sources)
        print("\n📊 Status:")
        print(f"   - Arquivos no cache: {len(cache)}")
        print(f"   - Arquivos modificados/novos: {len(modified)}")
        if modified:
            print("\n📝 Arquivos para atualizar:")
            for _f, s in modified[:10]:
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
        files_to_process = filter_modified_files(all_files, cache, existing_sources)
        has_deletions = any(path not in {str(f[0]) for f in all_files} for path in cache.keys())

        if not files_to_process and not has_deletions:
            print("\n✅ Nenhum arquivo modificado. Base está atualizada!")
            return

        if files_to_process:
            print(f"🔄 Modo INCREMENTAL: {len(files_to_process)} arquivos novos/modificados")

    print()

    # Ingerir cada arquivo
    success_count = 0
    error_count = 0
    skipped_count = 0
    total_chunks = 0

    for file_path, source in files_to_process:
        try:
            content = file_path.read_text(encoding="utf-8")

            # Adicionar header com informações do arquivo
            header = f"# Arquivo: {source}\n# Path: {file_path.relative_to(BACKEND_PATH)}\n\n"
            content_with_header = header + content

            result = await engine.add_text(
                content=content_with_header,
                source=source,
                doc_type=file_path.suffix[1:] if file_path.suffix else "txt",
                metadata={
                    "project": "chat-simples-backend",
                    "file_path": str(file_path.relative_to(BACKEND_PATH)),
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
