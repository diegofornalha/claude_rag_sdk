# =============================================================================
# MCP SERVER - Tools customizadas para o Hello Agent
# =============================================================================

from fastmcp import FastMCP
from pathlib import Path
import os

# Pasta de outputs
OUTPUTS_DIR = Path(__file__).parent / "outputs"

mcp = FastMCP("hello-agent-tools")


@mcp.tool()
def create_file(filename: str, content: str) -> str:
    """Cria um arquivo na pasta outputs.

    Args:
        filename: Nome do arquivo (ex: script.py, nota.txt)
        content: Conteudo do arquivo
    """
    filepath = OUTPUTS_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    return f"Arquivo criado: {filepath}"


@mcp.tool()
def list_outputs() -> str:
    """Lista todos os arquivos na pasta outputs."""
    if not OUTPUTS_DIR.exists():
        return "Pasta outputs vazia"

    files = list(OUTPUTS_DIR.glob("**/*"))
    if not files:
        return "Pasta outputs vazia"

    return "\n".join([f"- {f.relative_to(OUTPUTS_DIR)}" for f in files if f.is_file()])


@mcp.tool()
def read_output(filename: str) -> str:
    """Le um arquivo da pasta outputs.

    Args:
        filename: Nome do arquivo para ler
    """
    filepath = OUTPUTS_DIR / filename
    if not filepath.exists():
        return f"Arquivo nao encontrado: {filename}"
    return filepath.read_text()


@mcp.tool()
def delete_output(filename: str) -> str:
    """Deleta um arquivo da pasta outputs.

    Args:
        filename: Nome do arquivo para deletar
    """
    filepath = OUTPUTS_DIR / filename
    if not filepath.exists():
        return f"Arquivo nao encontrado: {filename}"
    filepath.unlink()
    return f"Arquivo deletado: {filename}"


if __name__ == "__main__":
    mcp.run()
