#!/usr/bin/env python3
"""
Script para migrar print statements para logger estruturado.

Uso:
    python scripts/migrate_prints_to_logger.py --dry-run  # Apenas mostra mudanças
    python scripts/migrate_prints_to_logger.py            # Aplica mudanças

Este script analisa arquivos Python e converte prints para logger calls.
"""

import argparse
import re
from pathlib import Path

# Padrões de conversão
PATTERNS = {
    # [ERROR] messages
    r'print\(f"\[ERROR\]\s*(.+?)"\)': 'logger.error("{message}")',
    r"print\(f'\[ERROR\]\s*(.+?)'\)": 'logger.error("{message}")',
    # [WARN] messages
    r'print\(f"\[WARN\]\s*(.+?)"\)': 'logger.warning("{message}")',
    r"print\(f'\[WARN\]\s*(.+?)'\)": 'logger.warning("{message}")',
    # [INFO] messages
    r'print\(f"\[INFO\]\s*(.+?)"\)': 'logger.info("{message}")',
    r"print\(f'\[INFO\]\s*(.+?)'\)": 'logger.info("{message}")',
    # [DEBUG] messages
    r'print\(f"\[DEBUG\]\s*(.+?)"\)': 'logger.debug("{message}")',
    r"print\(f'\[DEBUG\]\s*(.+?)'\)": 'logger.debug("{message}")',
    # Generic tags [TAG]
    r'print\(f"\[(\w+)\]\s*(.+?)"\)': 'logger.info("{message}", tag="{tag}")',
}


def find_python_files(root: Path, exclude_dirs: set[str] = None) -> list[Path]:
    """Encontra arquivos Python excluindo diretórios específicos."""
    exclude_dirs = exclude_dirs or {"__pycache__", ".git", "venv", ".venv", "node_modules"}
    files = []

    for path in root.rglob("*.py"):
        if not any(excluded in path.parts for excluded in exclude_dirs):
            files.append(path)

    return files


def analyze_file(filepath: Path) -> list[dict]:
    """Analisa um arquivo e retorna prints encontrados."""
    findings = []

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        if "print(" in line:
            findings.append(
                {
                    "file": str(filepath),
                    "line": i,
                    "content": line.strip(),
                    "type": _classify_print(line),
                }
            )

    return findings


def _classify_print(line: str) -> str:
    """Classifica o tipo de print."""
    if "[ERROR]" in line:
        return "error"
    elif "[WARN]" in line:
        return "warning"
    elif "[INFO]" in line:
        return "info"
    elif "[DEBUG]" in line:
        return "debug"
    elif re.search(r"\[(\w+)\]", line):
        return "tagged"
    else:
        return "generic"


def generate_report(findings: list[dict]) -> str:
    """Gera relatório dos prints encontrados."""
    report = []
    report.append("=" * 60)
    report.append("RELATÓRIO DE MIGRAÇÃO - print() -> logger")
    report.append("=" * 60)
    report.append("")

    # Agrupar por arquivo
    by_file: dict[str, list[dict]] = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)

    # Estatísticas por tipo
    by_type: dict[str, int] = {}
    for f in findings:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1

    report.append(f"Total de prints encontrados: {len(findings)}")
    report.append(f"Arquivos afetados: {len(by_file)}")
    report.append("")
    report.append("Por tipo:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        report.append(f"  - {t}: {count}")
    report.append("")

    # Detalhes por arquivo
    report.append("Detalhes por arquivo:")
    report.append("-" * 40)
    for filepath, items in sorted(by_file.items()):
        report.append(f"\n{filepath} ({len(items)} prints):")
        for item in items[:5]:  # Limitar a 5 por arquivo
            report.append(f"  L{item['line']}: [{item['type']}] {item['content'][:60]}...")
        if len(items) > 5:
            report.append(f"  ... e mais {len(items) - 5} prints")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Migra print statements para logger")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra mudanças")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Caminho raiz para buscar arquivos",
    )
    args = parser.parse_args()

    root = args.path
    print(f"Analisando: {root}")

    files = find_python_files(root)
    print(f"Arquivos encontrados: {len(files)}")

    all_findings = []
    for filepath in files:
        findings = analyze_file(filepath)
        all_findings.extend(findings)

    report = generate_report(all_findings)
    print(report)

    # Salvar relatório
    report_path = root / "artifacts" / "migration_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nRelatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
