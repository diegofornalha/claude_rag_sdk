# =============================================================================
# TOOL VALIDATOR - Validador de Tools para Agente Restrito
# =============================================================================
# Whitelist rigorosa que permite APENAS tools de RAG
# Bloqueia qualquer tentativa de acessar filesystem, bash, web, etc.
# =============================================================================

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BlockReason(str, Enum):
    """Razao do bloqueio de tool."""
    NOT_IN_WHITELIST = "not_in_whitelist"
    BLOCKED_KEYWORD = "blocked_keyword"
    SUSPICIOUS_INPUT = "suspicious_input"
    INVALID_NAMESPACE = "invalid_namespace"


@dataclass
class ValidationResult:
    """Resultado da validacao de tool."""
    is_valid: bool
    tool_name: str
    block_reason: Optional[BlockReason] = None
    details: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "tool_name": self.tool_name,
            "block_reason": self.block_reason.value if self.block_reason else None,
            "details": self.details,
        }


class ToolValidator:
    """
    Validador de tools com whitelist rigorosa.

    Apenas permite tools de RAG do MCP server rag-tools.
    Bloqueia qualquer tentativa de usar outras tools.
    """

    # Tools permitidas (whitelist)
    ALLOWED_TOOLS: set[str] = {
        # RAG Tools
        "mcp__rag-tools__search_documents",
        "mcp__rag-tools__search_hybrid",
        "mcp__rag-tools__get_document",
        "mcp__rag-tools__list_sources",
        # Opcionais de monitoramento
        "mcp__rag-tools__count_documents",
        "mcp__rag-tools__get_metrics_summary",
        "mcp__rag-tools__get_health",
    }

    # Namespace permitido
    ALLOWED_NAMESPACE = "mcp__rag-tools__"

    # Keywords bloqueadas em nomes de tools
    BLOCKED_TOOL_KEYWORDS: set[str] = {
        # Shell/Command execution
        "bash", "shell", "sh", "cmd", "powershell", "terminal",
        "exec", "execute", "run", "spawn", "subprocess",

        # Filesystem
        "file", "read", "write", "delete", "create", "mkdir",
        "copy", "move", "rename", "glob", "find", "ls", "cat",

        # Web/Network
        "http", "https", "curl", "wget", "fetch", "request",
        "download", "upload", "url", "web", "api",

        # System
        "os", "system", "process", "env", "path",
        "install", "pip", "npm", "apt", "brew",

        # Database (externa)
        "sql", "database", "db", "query", "insert", "update",

        # Code execution
        "eval", "compile", "import", "require", "load",
    }

    # Keywords bloqueadas em inputs
    BLOCKED_INPUT_KEYWORDS: set[str] = {
        # Commands
        "rm ", "rm -", "sudo", "chmod", "chown",
        "/bin/", "/usr/", "/etc/", "/var/",

        # Paths perigosos
        "../", "..\\", "~/.ssh", "~/.aws", "~/.config",
        "/etc/passwd", "/etc/shadow",

        # Injection patterns
        "; ", " && ", " || ", "`", "$(",
        "$(", "${", "{{", "}}",
    }

    # Patterns regex para deteccao
    DANGEROUS_PATTERNS = [
        r"[\;\&\|]{2,}",           # ;; && ||
        r"\$\([^\)]+\)",           # $(command)
        r"\$\{[^\}]+\}",           # ${var}
        r"`[^`]+`",                # `command`
        r"\\x[0-9a-fA-F]{2}",      # \xNN hex encoding
        r"base64",                  # base64 encoding
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Inicializa o validador.

        Args:
            strict_mode: Se True, bloqueia qualquer tool fora do namespace permitido
        """
        self.strict_mode = strict_mode
        self._dangerous_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

    def validate(self, tool_name: str, inputs: Optional[dict] = None) -> ValidationResult:
        """
        Valida uma chamada de tool.

        Args:
            tool_name: Nome da tool
            inputs: Dicionario de inputs da tool

        Returns:
            ValidationResult com status e detalhes
        """
        # 1. Verificar whitelist
        if not self.is_allowed(tool_name):
            # Determinar razao especifica
            if not tool_name.startswith(self.ALLOWED_NAMESPACE):
                return ValidationResult(
                    is_valid=False,
                    tool_name=tool_name,
                    block_reason=BlockReason.INVALID_NAMESPACE,
                    details=f"Tool deve comecar com '{self.ALLOWED_NAMESPACE}'",
                )

            # Verificar keywords bloqueadas
            for keyword in self.BLOCKED_TOOL_KEYWORDS:
                if keyword in tool_name.lower():
                    return ValidationResult(
                        is_valid=False,
                        tool_name=tool_name,
                        block_reason=BlockReason.BLOCKED_KEYWORD,
                        details=f"Keyword bloqueada: '{keyword}'",
                    )

            return ValidationResult(
                is_valid=False,
                tool_name=tool_name,
                block_reason=BlockReason.NOT_IN_WHITELIST,
                details="Tool nao esta na whitelist de ferramentas permitidas",
            )

        # 2. Verificar inputs
        if inputs and not self.check_inputs(inputs):
            return ValidationResult(
                is_valid=False,
                tool_name=tool_name,
                block_reason=BlockReason.SUSPICIOUS_INPUT,
                details="Input contem conteudo suspeito",
            )

        return ValidationResult(
            is_valid=True,
            tool_name=tool_name,
        )

    def is_allowed(self, tool_name: str) -> bool:
        """Verifica se tool esta na whitelist."""
        # Whitelist explicita
        if tool_name in self.ALLOWED_TOOLS:
            return True

        # Em modo strict, rejeita tudo que nao esta na whitelist
        if self.strict_mode:
            return False

        # Em modo relaxado, permite tools do namespace permitido
        return tool_name.startswith(self.ALLOWED_NAMESPACE)

    def check_inputs(self, inputs: dict) -> bool:
        """
        Verifica se inputs sao seguros.

        Args:
            inputs: Dicionario de inputs

        Returns:
            True se inputs sao seguros
        """
        for key, value in inputs.items():
            if not self._is_safe_value(value):
                return False
        return True

    def _is_safe_value(self, value) -> bool:
        """Verifica se um valor e seguro."""
        if value is None:
            return True

        if isinstance(value, (int, float, bool)):
            return True

        if isinstance(value, str):
            return self._is_safe_string(value)

        if isinstance(value, list):
            return all(self._is_safe_value(v) for v in value)

        if isinstance(value, dict):
            return all(
                self._is_safe_value(k) and self._is_safe_value(v)
                for k, v in value.items()
            )

        return True

    def _is_safe_string(self, text: str) -> bool:
        """Verifica se string e segura."""
        text_lower = text.lower()

        # Verificar keywords bloqueadas
        for keyword in self.BLOCKED_INPUT_KEYWORDS:
            if keyword in text_lower:
                return False

        # Verificar patterns perigosos
        for pattern in self._dangerous_patterns:
            if pattern.search(text):
                return False

        return True

    def get_allowed_tools(self) -> list[str]:
        """Retorna lista de tools permitidas."""
        return sorted(list(self.ALLOWED_TOOLS))


# Instancia global
_tool_validator: Optional[ToolValidator] = None


def get_tool_validator(strict_mode: bool = True) -> ToolValidator:
    """Retorna instancia global do validador."""
    global _tool_validator
    if _tool_validator is None:
        _tool_validator = ToolValidator(strict_mode=strict_mode)
    return _tool_validator


def validate_tool(tool_name: str, inputs: Optional[dict] = None) -> ValidationResult:
    """Valida uma chamada de tool usando validador global."""
    return get_tool_validator().validate(tool_name, inputs)


def is_tool_allowed(tool_name: str) -> bool:
    """Verifica rapidamente se tool e permitida."""
    return get_tool_validator().is_allowed(tool_name)


if __name__ == "__main__":
    print("=== Teste de Tool Validator ===\n")

    validator = ToolValidator(strict_mode=True)

    # Testes de tools
    test_tools = [
        # Permitidas
        ("mcp__rag-tools__search_documents", True),
        ("mcp__rag-tools__search_hybrid", True),
        ("mcp__rag-tools__get_document", True),
        ("mcp__rag-tools__list_sources", True),

        # Bloqueadas
        ("bash", False),
        ("read_file", False),
        ("web_fetch", False),
        ("mcp__other__bash", False),
        ("mcp__filesystem__read", False),
    ]

    print("--- Validacao de Tools ---")
    for tool_name, expected in test_tools:
        result = validator.validate(tool_name)
        status = "PASS" if result.is_valid == expected else "FAIL"
        allowed = "PERMITIDO" if result.is_valid else "BLOQUEADO"
        print(f"  [{status}] {tool_name}: {allowed}")
        if not result.is_valid:
            print(f"         Razao: {result.block_reason.value} - {result.details}")

    # Testes de inputs
    print("\n--- Validacao de Inputs ---")
    test_inputs = [
        # Seguros
        ({"query": "politica de IA", "top_k": 5}, True),
        ({"doc_id": 123}, True),

        # Perigosos
        ({"query": "rm -rf /"}, False),
        ({"query": "$(cat /etc/passwd)"}, False),
        ({"query": "; DROP TABLE users;"}, False),
        ({"query": "../../../etc/passwd"}, False),
    ]

    for inputs, expected in test_inputs:
        result = validator.validate("mcp__rag-tools__search_documents", inputs)
        status = "PASS" if result.is_valid == expected else "FAIL"
        safe = "SEGURO" if result.is_valid else "PERIGOSO"
        print(f"  [{status}] {list(inputs.values())[0][:30]}...: {safe}")

    print("\n--- Tools Permitidas ---")
    for tool in validator.get_allowed_tools():
        print(f"  - {tool}")
