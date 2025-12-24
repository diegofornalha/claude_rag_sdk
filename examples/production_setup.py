#!/usr/bin/env python3
"""Production setup example for Claude RAG SDK."""

import asyncio
import os
from claude_rag_sdk import (
    ClaudeRAG,
    ClaudeRAGOptions,
    EmbeddingModel,
    ChunkingStrategy,
    AgentModel,
)


async def main():
    """Demonstrate production-ready configuration."""
    print("=" * 60)
    print("Claude RAG SDK - Production Setup Example")
    print("=" * 60)

    # Production configuration
    options = ClaudeRAGOptions(
        # Identity
        id='production-agent',

        # Use better embedding model for production
        embedding_model=EmbeddingModel.BGE_BASE,  # 768 dims, good balance

        # Chunking optimized for your content
        chunk_size=500,
        chunk_overlap=50,
        chunking_strategy=ChunkingStrategy.SENTENCE,  # Better for documents

        # Agent model
        agent_model=AgentModel.SONNET,  # Better reasoning

        # Enable all safety features
        enable_reranking=True,
        enable_adaptive_topk=True,
        enable_prompt_guard=True,
        enable_hybrid_search=True,

        # Search configuration
        default_top_k=5,
        vector_weight=0.7,

        # Cache configuration
        cache_ttl=3600,  # 1 hour
        cache_max_size=10000,

        # Resilience
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0,
    )

    print("\n1. Configuration:")
    for key, value in options.to_dict().items():
        print(f"   {key}: {value}")

    async with await ClaudeRAG.open(options) as rag:
        print("\n2. RAG instance opened successfully")

        # AgentFS features
        print("\n3. Using AgentFS features:")

        # Store configuration
        await rag.kv.set('app:config', {
            'version': '1.0.0',
            'environment': 'production',
        })
        print("   - Stored app config in KV store")

        # Create output directory
        await rag.fs.write_file('/logs/startup.log', 'RAG system initialized\n')
        print("   - Created startup log")

        # Track custom operation
        call_id = await rag.tools.start('startup', {'version': '1.0.0'})
        await rag.tools.success(call_id, {'status': 'ready'})
        print("   - Logged startup to tool tracking")

        # Show stats
        print("\n4. System stats:")
        stats = await rag.stats()

        print(f"   Documents: {stats['documents']['total_documents']}")
        print(f"   Embeddings: {stats['documents']['total_embeddings']}")
        print(f"   Cache enabled: {stats['cache']['enabled']}")
        print(f"   Cache size: {stats['cache']['size']}")

    print("\n" + "=" * 60)
    print("Production setup complete!")
    print("\nNext steps:")
    print("1. Add your documents with rag.ingest.add_document()")
    print("2. Set up monitoring with rag.stats()")
    print("3. Configure your MCP server for full agent capabilities")
    print("=" * 60)


# Alternative: Load from environment
async def from_environment():
    """Load configuration from environment variables."""
    # Set environment variables
    os.environ['CLAUDE_RAG_ID'] = 'env-agent'
    os.environ['CLAUDE_RAG_EMBEDDING_MODEL'] = 'BAAI/bge-base-en-v1.5'
    os.environ['CLAUDE_RAG_ENABLE_RERANKING'] = 'true'
    os.environ['CLAUDE_RAG_DEFAULT_TOP_K'] = '10'

    options = ClaudeRAGOptions.from_env()
    print(f"Loaded config for agent: {options.id}")

    async with await ClaudeRAG.open(options) as rag:
        print("RAG ready from environment config")


if __name__ == "__main__":
    asyncio.run(main())
