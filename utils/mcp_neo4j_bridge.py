"""
Bridge direto para servidores MCP Neo4j via subprocess.

Como o Claude Agent SDK não expõe corretamente os MCP servers (bugs #207, #3426),
este wrapper permite chamar os servidores MCP Neo4j diretamente via subprocess
e disponibilizar as ferramentas ao backend.
"""

import asyncio
import json
from typing import Any


class MCPNeo4jBridge:
    """Bridge para servidores MCP Neo4j via subprocess."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

    async def call_mcp_tool(self, server: str, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Chama uma ferramenta MCP via subprocess com inicialização correta.

        Args:
            server: Nome do servidor MCP (ex: 'mcp-neo4j-memory')
            tool: Nome da tool (ex: 'create_entities')
            params: Parâmetros da tool

        Returns:
            Resultado da execução da tool
        """
        env = {
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_USERNAME": self.neo4j_username,
            "NEO4J_PASSWORD": self.neo4j_password,
            "NEO4J_DATABASE": "neo4j",
        }

        try:
            process = await asyncio.create_subprocess_exec(
                server,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # 1. INICIALIZAR o servidor MCP primeiro
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "chat-simples-backend", "version": "1.0.0"},
                },
            }

            # Enviar initialize e aguardar resposta
            init_json = json.dumps(init_request) + "\n"
            process.stdin.write(init_json.encode())
            await process.stdin.drain()

            # Ler resposta de initialize (aguardar linha)
            init_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)

            if not init_response_line:
                return {"error": "No initialize response from server"}

            # 2. Agora chamar a tool
            tool_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool, "arguments": params},
            }

            tool_json = json.dumps(tool_request) + "\n"
            process.stdin.write(tool_json.encode())
            await process.stdin.drain()

            # Ler resposta da tool
            tool_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=30.0)

            # Fechar processo
            process.stdin.close()
            await process.wait()

            # Parsear resposta
            if tool_response_line:
                response = json.loads(tool_response_line.decode())
                if "result" in response:
                    return response["result"]
                elif "error" in response:
                    return {"error": response["error"]}

            return {"error": "No tool response from server"}

        except asyncio.TimeoutError:
            return {"error": "MCP server timeout"}
        except Exception as e:
            return {"error": f"Exception: {str(e)}"}

    # =========================================================================
    # NEO4J-CYPHER TOOLS
    # =========================================================================

    async def translate_to_cypher(self, natural_language: str) -> dict[str, Any]:
        """Traduz linguagem natural para query Cypher."""
        return await self.call_mcp_tool(
            "mcp-neo4j-cypher", "translate_to_cypher", {"query": natural_language}
        )

    async def execute_cypher(self, query: str) -> dict[str, Any]:
        """Executa query Cypher no Neo4j."""
        return await self.call_mcp_tool("mcp-neo4j-cypher", "execute_cypher", {"query": query})

    # =========================================================================
    # NEO4J-MEMORY TOOLS
    # =========================================================================

    async def create_entities(self, entities: list[dict]) -> dict[str, Any]:
        """Cria entidades no knowledge graph."""
        return await self.call_mcp_tool(
            "mcp-neo4j-memory", "create_entities", {"entities": entities}
        )

    async def create_relations(self, relations: list[dict]) -> dict[str, Any]:
        """Cria relacionamentos no knowledge graph."""
        return await self.call_mcp_tool(
            "mcp-neo4j-memory", "create_relations", {"relations": relations}
        )

    async def search_nodes(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Busca nós por texto."""
        return await self.call_mcp_tool(
            "mcp-neo4j-memory", "search_nodes", {"query": query, "limit": limit}
        )

    async def add_observations(self, observations: list[dict]) -> dict[str, Any]:
        """Adiciona observações ao knowledge graph."""
        return await self.call_mcp_tool(
            "mcp-neo4j-memory", "add_observations", {"observations": observations}
        )

    # =========================================================================
    # NEO4J-DATA-MODELING TOOLS
    # =========================================================================

    async def suggest_schema(self, description: str) -> dict[str, Any]:
        """Sugere um schema de grafo baseado em descrição."""
        return await self.call_mcp_tool(
            "mcp-neo4j-data-modeling", "suggest_schema", {"description": description}
        )

    # =========================================================================
    # NEO4J-AURA-MANAGER TOOLS
    # =========================================================================

    async def list_instances(self) -> dict[str, Any]:
        """Lista instâncias Neo4j Aura."""
        return await self.call_mcp_tool("mcp-neo4j-aura-manager", "list_instances", {})


# Singleton
_bridge: MCPNeo4jBridge | None = None


def get_neo4j_bridge() -> MCPNeo4jBridge:
    """Retorna instância singleton do bridge."""
    global _bridge
    if _bridge is None:
        _bridge = MCPNeo4jBridge()
    return _bridge
