#!/usr/bin/env python3
"""Basic search example for Claude RAG SDK."""

import asyncio
from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions


async def main():
    """Demonstrate basic search functionality."""
    print("=" * 60)
    print("Claude RAG SDK - Basic Search Example")
    print("=" * 60)

    # Open RAG instance
    async with await ClaudeRAG.open(ClaudeRAGOptions(id='search-example')) as rag:
        # Add some sample documents
        print("\n1. Adding sample documents...")

        await rag.add_text(
            content="""
            Retrieval-Augmented Generation (RAG) is a technique that combines
            the power of large language models with external knowledge retrieval.
            Instead of relying solely on the model's training data, RAG systems
            retrieve relevant documents from a knowledge base and use them to
            generate more accurate and up-to-date responses.
            """,
            source="rag-overview.txt",
        )

        await rag.add_text(
            content="""
            Vector databases store data as high-dimensional vectors, enabling
            semantic search capabilities. Unlike traditional keyword search,
            vector search understands the meaning behind queries, finding
            relevant results even when exact words don't match.
            """,
            source="vector-db.txt",
        )

        await rag.add_text(
            content="""
            Embedding models convert text into numerical vectors that capture
            semantic meaning. Popular models include BERT, sentence-transformers,
            and FastEmbed. These embeddings enable similarity comparisons
            between documents and queries.
            """,
            source="embeddings.txt",
        )

        # Check stats
        stats = await rag.stats()
        print(f"   Documents indexed: {stats['documents']['total_documents']}")

        # Perform search
        print("\n2. Searching for 'What is RAG?'...")
        results = await rag.search("What is RAG?", top_k=3)

        print(f"\n   Found {len(results)} results:\n")
        for r in results:
            print(f"   [{r.rank}] {r.source}")
            print(f"       Similarity: {r.similarity:.2%}")
            print(f"       Content: {r.content[:100].strip()}...")
            print()

        # Hybrid search
        print("3. Hybrid search for 'semantic vector search'...")
        results = await rag.search_hybrid(
            "semantic vector search",
            top_k=3,
            vector_weight=0.7,
        )

        print(f"\n   Found {len(results)} results:\n")
        for r in results:
            print(f"   [{r.rank}] {r.source}")
            print(f"       Vector: {r.vector_score:.2%}, BM25: {r.bm25_score:.2f}")
            print(f"       Hybrid: {r.hybrid_score:.2%}")
            print()

        # List all sources
        print("4. All indexed sources:")
        sources = await rag.list_sources()
        for s in sources:
            print(f"   - {s['nome']} ({s['tipo']}, {s['tamanho']} bytes)")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
