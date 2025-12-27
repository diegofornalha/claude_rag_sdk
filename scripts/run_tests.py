#!/usr/bin/env python3
# =============================================================================
# SCRIPT RUNNER DE TESTES
# =============================================================================
# Executa todos os testes do projeto com diferentes níveis de detalhe
# =============================================================================

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent
TESTS_DIR = ROOT_DIR / "tests"


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Executa comando e retorna resultado."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Executa testes unitários."""
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v" if verbose else "-q"]

    if coverage:
        cmd.extend(["--cov=claude_rag_sdk", "--cov-report=term-missing"])

    success, _ = run_command(cmd, "TESTES UNITÁRIOS")
    return success


def run_integration_tests(verbose: bool = False) -> bool:
    """Executa testes de integração."""
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else "-q"]

    success, _ = run_command(cmd, "TESTES DE INTEGRAÇÃO")
    return success


def run_e2e_tests(verbose: bool = False) -> bool:
    """Executa testes E2E."""
    cmd = ["python", "-m", "pytest", "tests/e2e/", "-v" if verbose else "-q"]

    success, _ = run_command(cmd, "TESTES E2E")
    return success


def run_performance_tests(verbose: bool = False) -> bool:
    """Executa testes de performance."""
    cmd = ["python", "-m", "pytest", "tests/performance/", "-v" if verbose else "-q", "-s"]

    success, _ = run_command(cmd, "TESTES DE PERFORMANCE")
    return success


def run_all_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Executa todos os testes."""
    cmd = ["python", "-m", "pytest", "tests/", "-v" if verbose else "-q"]

    if coverage:
        cmd.extend(["--cov=claude_rag_sdk", "--cov-report=term-missing", "--cov-report=html"])

    success, _ = run_command(cmd, "TODOS OS TESTES")
    return success


def run_specific_test(test_path: str, verbose: bool = False) -> bool:
    """Executa um teste específico."""
    cmd = ["python", "-m", "pytest", test_path, "-v" if verbose else "-q"]

    success, _ = run_command(cmd, f"TESTE: {test_path}")
    return success


def check_test_environment() -> bool:
    """Verifica ambiente de testes."""
    print("\n" + "="*60)
    print("  VERIFICANDO AMBIENTE")
    print("="*60 + "\n")

    # Verificar pytest
    try:
        import pytest
        print(f"✓ pytest versão {pytest.__version__}")
    except ImportError:
        print("✗ pytest não instalado")
        return False

    # Verificar diretórios de teste
    if not TESTS_DIR.exists():
        print("✗ Diretório tests/ não encontrado")
        return False
    print(f"✓ Diretório tests/ encontrado")

    # Contar testes
    test_files = list(TESTS_DIR.rglob("test_*.py"))
    print(f"✓ {len(test_files)} arquivos de teste encontrados")

    # Verificar conftest
    if (ROOT_DIR / "conftest.py").exists():
        print("✓ conftest.py encontrado")
    else:
        print("⚠ conftest.py não encontrado")

    return True


def main():
    parser = argparse.ArgumentParser(description="Runner de testes do Chat Simples")

    parser.add_argument(
        "type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "e2e", "performance", "check"],
        help="Tipo de teste a executar",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verboso",
    )

    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Gerar relatório de cobertura",
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Executar arquivo de teste específico",
    )

    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Parar no primeiro erro",
    )

    args = parser.parse_args()

    # Adicionar ROOT_DIR ao PYTHONPATH
    os.environ["PYTHONPATH"] = str(ROOT_DIR) + ":" + os.environ.get("PYTHONPATH", "")

    # Header
    print("\n" + "="*60)
    print("  CHAT SIMPLES - TEST RUNNER")
    print("="*60)

    # Verificar ambiente
    if args.type == "check":
        success = check_test_environment()
        return 0 if success else 1

    if not check_test_environment():
        print("\n✗ Falha na verificação do ambiente")
        return 1

    # Executar teste específico
    if args.file:
        success = run_specific_test(args.file, args.verbose)
        return 0 if success else 1

    # Executar por tipo
    results = {}

    if args.type == "all":
        results["unit"] = run_unit_tests(args.verbose, args.coverage)
        results["integration"] = run_integration_tests(args.verbose)
        results["e2e"] = run_e2e_tests(args.verbose)
        results["performance"] = run_performance_tests(args.verbose)
    elif args.type == "unit":
        results["unit"] = run_unit_tests(args.verbose, args.coverage)
    elif args.type == "integration":
        results["integration"] = run_integration_tests(args.verbose)
    elif args.type == "e2e":
        results["e2e"] = run_e2e_tests(args.verbose)
    elif args.type == "performance":
        results["performance"] = run_performance_tests(args.verbose)

    # Sumário
    print("\n" + "="*60)
    print("  SUMÁRIO")
    print("="*60 + "\n")

    all_passed = True
    for test_type, passed in results.items():
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        print(f"  {test_type.upper()}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)

    if all_passed:
        print("  ✓ TODOS OS TESTES PASSARAM!")
    else:
        print("  ✗ ALGUNS TESTES FALHARAM")

    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
