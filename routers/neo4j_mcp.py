"""
Endpoints REST para MCP Neo4j via subprocess bridge.

Workaround para bug do Claude Agent SDK que não expõe MCP servers corretamente.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.mcp_neo4j_bridge import get_neo4j_bridge

router = APIRouter(prefix="/neo4j-mcp", tags=["Neo4j MCP"])


class CypherTranslateRequest(BaseModel):
    natural_language: str


class CypherExecuteRequest(BaseModel):
    query: str


class CreateEntitiesRequest(BaseModel):
    entities: list[dict]


class SearchNodesRequest(BaseModel):
    query: str
    limit: int = 10


@router.post("/cypher/translate")
async def translate_to_cypher(request: CypherTranslateRequest):
    """Traduz linguagem natural para Cypher."""
    bridge = get_neo4j_bridge()
    result = await bridge.translate_to_cypher(request.natural_language)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/cypher/execute")
async def execute_cypher(request: CypherExecuteRequest):
    """Executa query Cypher."""
    bridge = get_neo4j_bridge()
    result = await bridge.execute_cypher(request.query)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/memory/entities")
async def create_entities(request: CreateEntitiesRequest):
    """Cria entidades no knowledge graph."""
    bridge = get_neo4j_bridge()
    result = await bridge.create_entities(request.entities)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/memory/search")
async def search_nodes(request: SearchNodesRequest):
    """Busca nós no knowledge graph."""
    bridge = get_neo4j_bridge()
    result = await bridge.search_nodes(request.query, request.limit)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.get("/status")
async def mcp_status():
    """Verifica status dos servidores MCP Neo4j."""
    bridge = get_neo4j_bridge()

    # Testar conectividade com um comando simples
    cypher_test = await bridge.execute_cypher("RETURN 1 as test")
    memory_test = await bridge.search_nodes("test", 1)

    return {
        "status": "ok",
        "servers": {
            "neo4j-cypher": "ok" if "error" not in cypher_test else "error",
            "neo4j-memory": "ok" if "error" not in memory_test else "error",
        },
        "workaround": "subprocess-bridge",
        "reason": "Claude Agent SDK bugs #207, #3426",
    }
