#!/usr/bin/env python3
"""
Script para ingestão de documentação Angular via MCP.

Este script é um PLUGIN OPCIONAL. Pode ser removido sem afetar o sistema.

Uso:
    # Ingestão completa (docs + examples + best practices)
    python scripts/ingest_angular_mcp.py --all

    # Apenas documentação com queries específicas
    python scripts/ingest_angular_mcp.py --docs "signals" "standalone components"

    # Apenas exemplos
    python scripts/ingest_angular_mcp.py --examples

    # Apenas best practices
    python scripts/ingest_angular_mcp.py --practices

    # Ver status
    python scripts/ingest_angular_mcp.py --status

    # Modo dry-run (não ingere, apenas mostra o que faria)
    python scripts/ingest_angular_mcp.py --all --dry-run

Requisitos:
    - Node.js instalado (para npx @angular/cli mcp)
    - Adapter angular-cli habilitado na config ou via env var
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Queries padrão para documentação Angular
DEFAULT_DOC_QUERIES = [
    "standalone components",
    "signals",
    "control flow @if @for @switch",
    "dependency injection providers",
    "routing guards",
    "lazy loading routes",
    "reactive forms",
    "form validation",
    "http client interceptors",
    "change detection onpush",
    "zoneless applications",
    "defer loading",
    "testing components",
    "ng generate schematics",
]

# Tópicos padrão para exemplos
DEFAULT_EXAMPLE_TOPICS = [
    "standalone components",
    "signals",
    "forms",
    "http",
    "routing",
]


def print_header(text: str) -> None:
    """Imprime header formatado."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_status(label: str, value: str, ok: bool = True) -> None:
    """Imprime status formatado."""
    status = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {label}: {value}")


async def check_prerequisites() -> bool:
    """Verifica se os pré-requisitos estão instalados."""
    print_header("Verificando Pré-requisitos")

    # Verifica Node.js
    try:
        proc = await asyncio.create_subprocess_exec(
            "node",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        node_version = stdout.decode().strip()
        print_status("Node.js", node_version, True)
    except FileNotFoundError:
        print_status("Node.js", "não encontrado", False)
        print("\n  ERRO: Node.js é necessário para o Angular CLI MCP.")
        print("  Instale em: https://nodejs.org/")
        return False

    # Verifica se o módulo MCP está disponível
    try:
        from claude_rag_sdk.mcp_client import get_mcp_config  # noqa: F401

        print_status("MCP Module", "disponível", True)
    except ImportError as e:
        print_status("MCP Module", f"erro: {e}", False)
        return False

    # Verifica configuração
    config = get_mcp_config()
    angular_enabled = config.is_adapter_enabled("angular-cli")

    if angular_enabled:
        print_status("Angular CLI Adapter", "habilitado", True)
    else:
        print_status("Angular CLI Adapter", "desabilitado", False)
        print("\n  Para habilitar, defina a variável de ambiente:")
        print("  export MCP_ANGULAR_CLI_ENABLED=true")
        print("\n  Ou use: python scripts/ingest_angular_mcp.py --enable")
        return False

    return True


async def show_status() -> None:
    """Mostra status do sistema MCP."""
    print_header("Status do Sistema MCP")

    try:
        from claude_rag_sdk.mcp_client import get_mcp_registry

        registry = get_mcp_registry()

        print(f"  Adapters registrados: {len(registry.list_registered())}")
        print(f"  Adapters habilitados: {len(registry.list_enabled())}")
        print(f"  Adapters conectados:  {len(registry.list_active())}")

        print("\n  Adapters disponíveis:")
        for info in registry.get_all_adapters_info():
            status = "✓ habilitado" if info.enabled else "✗ desabilitado"
            print(f"    - {info.name}: {status}")
            print(f"      {info.description}")

    except Exception as e:
        print(f"  ERRO: {e}")


async def enable_angular_adapter() -> None:
    """Habilita o adapter Angular CLI."""
    from claude_rag_sdk.mcp_client import get_mcp_config

    config = get_mcp_config()
    config.enable_adapter("angular-cli")
    print_status("Angular CLI Adapter", "habilitado", True)
    print("\n  Adapter habilitado para esta sessão.")
    print("  Para persistir, defina: export MCP_ANGULAR_CLI_ENABLED=true")


async def run_ingest(
    queries: list[str],
    include_examples: bool,
    include_practices: bool,
    dry_run: bool = False,
) -> None:
    """Executa a ingestão de documentos."""
    from claude_rag_sdk.ingest import IngestEngine
    from claude_rag_sdk.mcp_client import get_adapter

    print_header("Iniciando Ingestão")

    print(f"  Queries de documentação: {len(queries)}")
    print(f"  Incluir exemplos: {include_examples}")
    print(f"  Incluir best practices: {include_practices}")
    print(f"  Modo dry-run: {dry_run}")

    if dry_run:
        print("\n  [DRY-RUN] Mostrando queries que seriam executadas:\n")
        for i, q in enumerate(queries, 1):
            print(f"    {i}. {q}")
        if include_examples:
            print("\n  Exemplos que seriam buscados:")
            for topic in DEFAULT_EXAMPLE_TOPICS:
                print(f"    - {topic}")
        if include_practices:
            print("\n    + Angular Best Practices Guide")
        return

    # Conecta ao adapter
    print("\n  Conectando ao Angular CLI MCP...")
    try:
        adapter = await get_adapter("angular-cli")
        print_status("Conexão", "estabelecida", True)
    except Exception as e:
        print_status("Conexão", f"falhou: {e}", False)
        return

    # Obtém documentos
    print("\n  Coletando documentos do MCP...")
    start_time = datetime.now()

    documents = await adapter.get_documents_for_ingest(
        queries=queries,
        include_examples=include_examples,
        include_best_practices=include_practices,
    )

    print(f"  Documentos coletados: {len(documents)}")

    if not documents:
        print("\n  Nenhum documento coletado. Verifique a conexão com o MCP.")
        return

    # Configura engine de ingestão
    db_path = Path(__file__).parent.parent / "data" / "regulamento.db.db"
    engine = IngestEngine(
        db_path=str(db_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=1500,
        chunk_overlap=150,
    )

    # CORREÇÃO: Deletar documentos Angular antigos ANTES de reingerir
    print("\n  Deletando documentos Angular antigos para evitar duplicação...\n")
    try:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

        # Criar opções para o RAG
        rag_options = ClaudeRAGOptions(id="angular-ingest", db_path=str(db_path))

        # Abrir RAG para buscar e deletar
        async with ClaudeRAG.open(rag_options) as rag:
            # Buscar todos os documentos Angular existentes
            all_docs = await rag.list_documents(limit=1000)
            angular_docs = [
                d
                for d in all_docs
                if "angular" in (d.get("source", "") or "").lower()
                or "angular-cli-mcp" in (d.get("source", "") or "").lower()
            ]

            deleted_count = 0
            for doc in angular_docs:
                doc_id = doc.get("id")
                if doc_id:
                    success = await rag.ingest.delete_document(doc_id)
                    if success:
                        deleted_count += 1

            print(f"    ✓ {deleted_count} documento(s) Angular antigo(s) deletado(s)\n")
    except Exception as e:
        print(f"    ⚠ Aviso ao deletar antigos: {e}\n")

    # Ingere documentos
    print("  Ingerindo novos documentos no RAG...\n")

    success_count = 0
    error_count = 0

    for i, doc in enumerate(documents, 1):
        try:
            # Formata conteúdo
            content = f"# {doc.title}\n\n"
            if doc.url:
                content += f"_Fonte: {doc.url}_\n\n"
            content += "---\n\n"
            content += doc.content

            result = await engine.add_text(
                content=content,
                source=doc.source,
                doc_type=doc.doc_type,
                metadata=doc.metadata,
            )

            if result.success:
                success_count += 1
                print(f"    [{i}/{len(documents)}] ✓ {doc.title[:50]}...")
            else:
                error_count += 1
                print(f"    [{i}/{len(documents)}] ✗ {doc.title[:50]} - {result.error}")

            # Rate limiting
            await asyncio.sleep(0.1)

        except Exception as e:
            error_count += 1
            print(f"    [{i}/{len(documents)}] ✗ {doc.title[:50]} - {e}")

    # Resumo
    duration = (datetime.now() - start_time).total_seconds()
    print_header("Resumo da Ingestão")
    print(f"  Total de documentos: {len(documents)}")
    print(f"  Sucesso: {success_count}")
    print(f"  Erros: {error_count}")
    print(f"  Duração: {duration:.2f}s")
    print(f"  Banco de dados: {db_path}")


async def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Ingestão de documentação Angular via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s --all                    # Ingestão completa
  %(prog)s --docs "signals"         # Busca específica
  %(prog)s --examples               # Apenas exemplos
  %(prog)s --status                 # Ver status do MCP
  %(prog)s --enable                 # Habilitar adapter
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Ingestão completa (docs + examples + practices)"
    )
    parser.add_argument(
        "--docs", nargs="*", metavar="QUERY", help="Queries para buscar documentação"
    )
    parser.add_argument("--examples", action="store_true", help="Incluir exemplos de código")
    parser.add_argument("--practices", action="store_true", help="Incluir best practices")
    parser.add_argument("--status", action="store_true", help="Mostrar status do sistema MCP")
    parser.add_argument("--enable", action="store_true", help="Habilitar adapter Angular CLI")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito")

    args = parser.parse_args()

    # Status
    if args.status:
        await show_status()
        return

    # Enable
    if args.enable:
        await enable_angular_adapter()
        return

    # Verifica pré-requisitos
    if not await check_prerequisites():
        # Tenta habilitar automaticamente se solicitado ingestão
        if args.all or args.docs or args.examples or args.practices:
            print("\n  Tentando habilitar adapter automaticamente...")
            await enable_angular_adapter()
            if not await check_prerequisites():
                sys.exit(1)
        else:
            sys.exit(1)

    # Define o que ingerir
    if args.all:
        queries = DEFAULT_DOC_QUERIES
        include_examples = True
        include_practices = True
    else:
        queries = args.docs if args.docs else DEFAULT_DOC_QUERIES
        include_examples = args.examples or args.all
        include_practices = args.practices or args.all

    # Se nenhuma opção específica, mostra ajuda
    if not (args.all or args.docs or args.examples or args.practices):
        parser.print_help()
        return

    # Executa ingestão
    await run_ingest(
        queries=queries,
        include_examples=include_examples,
        include_practices=include_practices,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
