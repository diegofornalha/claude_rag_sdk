#!/usr/bin/env python3
"""
Script de ingestao de documentos para o RAG.

Uso:
    python scripts/ingest.py <arquivo>
    python scripts/ingest.py <arquivo1> <arquivo2> ...
    python scripts/ingest.py --dir <diretorio>
    python scripts/ingest.py --clear  # Limpa o banco

Formatos suportados:
    - .txt (texto puro)
    - .pdf (requer pypdf)
    - .docx (requer python-docx)
    - .html/.htm (requer beautifulsoup4)
    - .md (markdown)
    - .json

Exemplos:
    python scripts/ingest.py documento.pdf
    python scripts/ingest.py *.txt
    python scripts/ingest.py --dir ./documentos
    python scripts/ingest.py --chunk-size 300 --strategy paragraph documento.txt
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Adicionar o diretorio pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_rag_sdk.ingest import IngestEngine
from claude_rag_sdk.options import ChunkingStrategy


# Configuracoes padrao
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "rag_knowledge.db"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_STRATEGY = ChunkingStrategy.PARAGRAPH

# Formatos suportados
SUPPORTED_FORMATS = {'.txt', '.pdf', '.docx', '.html', '.htm', '.md', '.json'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingestao de documentos para RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='Arquivos para ingerir'
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Diretorio para ingerir (todos os arquivos suportados)'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Buscar recursivamente em subdiretorios'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Limpar banco RAG antes de ingerir'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f'Tamanho do chunk em palavras (default: {DEFAULT_CHUNK_SIZE})'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f'Overlap entre chunks (default: {DEFAULT_CHUNK_OVERLAP})'
    )

    parser.add_argument(
        '--strategy', '-s',
        choices=['fixed', 'sentence', 'paragraph'],
        default='paragraph',
        help='Estrategia de chunking (default: paragraph)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Modelo de embedding (default: {DEFAULT_EMBEDDING_MODEL})'
    )

    parser.add_argument(
        '--db',
        type=str,
        default=str(DEFAULT_DB_PATH),
        help='Caminho do banco de dados'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Mostrar apenas estatisticas do banco'
    )

    return parser.parse_args()


def get_strategy(name: str) -> ChunkingStrategy:
    strategies = {
        'fixed': ChunkingStrategy.FIXED,
        'sentence': ChunkingStrategy.SENTENCE,
        'paragraph': ChunkingStrategy.PARAGRAPH,
    }
    return strategies.get(name, ChunkingStrategy.PARAGRAPH)


def find_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Encontra todos os arquivos suportados em um diretorio."""
    files = []
    pattern = '**/*' if recursive else '*'

    for fmt in SUPPORTED_FORMATS:
        if recursive:
            files.extend(directory.rglob(f'*{fmt}'))
        else:
            files.extend(directory.glob(f'*{fmt}'))

    return sorted(files)


async def main():
    args = parse_args()

    # Garantir que o diretorio do banco existe
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Criar engine
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=get_strategy(args.strategy),
    )

    # Mostrar apenas stats
    if args.stats:
        stats = engine.stats
        print("\n📊 Estatisticas do Banco RAG")
        print("=" * 40)
        print(f"   Documentos:  {stats['total_documents']}")
        print(f"   Embeddings:  {stats['total_embeddings']}")
        print(f"   Conteudo:    {stats['total_size_bytes']:,} bytes")
        print(f"   Status:      {stats['status']}")
        print(f"   Banco:       {db_path}")
        return

    # Limpar banco se solicitado
    if args.clear:
        count = await engine.clear_all()
        print(f"🗑️  Banco limpo: {count} documentos removidos")
        if not args.files and not args.dir:
            return

    # Coletar arquivos
    files_to_ingest = []

    # Arquivos passados diretamente
    for f in args.files:
        path = Path(f)
        if path.exists():
            if path.suffix.lower() in SUPPORTED_FORMATS:
                files_to_ingest.append(path)
            else:
                print(f"⚠️  Formato nao suportado: {path.name}")
        else:
            print(f"⚠️  Arquivo nao encontrado: {f}")

    # Arquivos de um diretorio
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.exists() and dir_path.is_dir():
            found = find_files(dir_path, args.recursive)
            files_to_ingest.extend(found)
            print(f"📁 Encontrados {len(found)} arquivos em {args.dir}")
        else:
            print(f"❌ Diretorio nao encontrado: {args.dir}")
            return

    if not files_to_ingest:
        print("❌ Nenhum arquivo para ingerir")
        print("\nUso: python scripts/ingest.py <arquivo> [<arquivo2> ...]")
        print("     python scripts/ingest.py --dir <diretorio>")
        print("     python scripts/ingest.py --stats")
        print("     python scripts/ingest.py --help")
        return

    # Remover duplicatas mantendo ordem
    files_to_ingest = list(dict.fromkeys(files_to_ingest))

    print(f"\n📚 Ingerindo {len(files_to_ingest)} arquivo(s)")
    print(f"💾 Banco: {db_path}")
    print(f"🧠 Modelo: {args.model}")
    print(f"📏 Chunk: {args.chunk_size} palavras | Overlap: {args.chunk_overlap}")
    print(f"📐 Estrategia: {args.strategy}")
    print("=" * 50)

    success_count = 0
    error_count = 0

    for i, file_path in enumerate(files_to_ingest, 1):
        print(f"\n[{i}/{len(files_to_ingest)}] {file_path.name}")

        result = await engine.add_document(file_path)

        if result.success:
            success_count += 1
            if result.error and "duplicate" in result.error.lower():
                print(f"   ⏭️  Ja existe (duplicado)")
            else:
                print(f"   ✅ OK | ID: {result.doc_id} | Chunks: {result.chunks}")
        else:
            error_count += 1
            print(f"   ❌ Erro: {result.error}")

    # Resumo final
    print("\n" + "=" * 50)
    print("📊 Resumo")
    print(f"   ✅ Sucesso: {success_count}")
    print(f"   ❌ Erros:   {error_count}")

    stats = engine.stats
    print(f"\n📈 Banco atualizado:")
    print(f"   Documentos:  {stats['total_documents']}")
    print(f"   Embeddings:  {stats['total_embeddings']}")
    print(f"   Conteudo:    {stats['total_size_bytes']:,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
