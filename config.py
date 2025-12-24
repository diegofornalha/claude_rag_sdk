# =============================================================================
# CONFIGURACAO DO HELLO AGENT - Claude Agent SDK
# =============================================================================

from claude_agent_sdk import ClaudeAgentOptions
from pathlib import Path

# Caminho do MCP server
MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

# -----------------------------------------------------------------------------
# Definicao do Agente
# -----------------------------------------------------------------------------

HELLO_AGENT_OPTIONS = ClaudeAgentOptions(
    # Modelo - haiku e mais rapido e barato
    model="haiku",

    # System prompt - define a personalidade e regras do agente
    system_prompt="""Eu sou o Hello Agent.

CONTEXTO TECNICO:
- Eu sou um AGENTE criado com Claude Agent SDK (biblioteca Python)
- Estou rodando no modelo Claude Haiku 4.5 (claude-haiku-4-5-20251001)
- NAO sou uma skill, NAO sou um comando, NAO sou um subagente
- Eu sou uma instancia do Claude configurada como agente independente
- O Claude Agent SDK e diferente do Claude Code CLI

MINHA IDENTIDADE:
- Meu nome: Hello Agent
- Minha natureza: Agente do Claude Agent SDK
- Eu ja estou ativo e conversando - nao preciso ser "chamado"

Se perguntarem "pode chamar o hello-agent?" respondo:
"Ola! Eu sou o Hello Agent, um agente criado com Claude Agent SDK. Ja estou aqui conversando com voce!"

NUNCA menciono: skills, comandos slash, subagentes do CLI

Meu comportamento:
- Direto e conciso
- Simpatico mas sem emojis
- Posso ler arquivos do projeto

ARTEFATOS:
- Quando criar arquivos (codigo, texto, etc), salvo na pasta: outputs/
- Caminho completo: /Users/2a/.claude/hello-agent/chat-simples/backend/outputs/""",

    # Ferramentas permitidas (apenas nomes, sem patterns)
    allowed_tools=[
        "Read",
        "Glob",
        "Grep",
        "Write",
        # MCP tools - hello-agent-tools
        "mcp__hello-agent-tools__create_file",
        "mcp__hello-agent-tools__list_outputs",
        "mcp__hello-agent-tools__read_output",
        "mcp__hello-agent-tools__delete_output",
    ],

    # Modo de permissao
    permission_mode="acceptEdits",

    # Ferramentas bloqueadas (patterns NAO sao suportados no SDK)
    disallowed_tools=[],

    # Diretorio de trabalho
    cwd="/Users/2a/.claude/hello-agent/chat-simples/backend",

    # MCP Servers - tools customizadas
    mcp_servers={
        "hello-agent-tools": {
            "command": "python",
            "args": [str(MCP_SERVER_PATH)]
        }
    }
)
