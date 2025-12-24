#!/usr/bin/env python3
"""Q&A with AI agent example for Claude RAG SDK."""

import asyncio
from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions, AgentModel


async def main():
    """Demonstrate AI-powered Q&A with citations."""
    print("=" * 60)
    print("Claude RAG SDK - AI Q&A Example")
    print("=" * 60)

    # Open RAG instance with agent configuration
    options = ClaudeRAGOptions(
        id='qa-example',
        agent_model=AgentModel.HAIKU,  # Fast and cost-effective
        enable_reranking=True,
        default_top_k=5,
    )

    async with await ClaudeRAG.open(options) as rag:
        # Add knowledge base
        print("\n1. Building knowledge base...")

        docs = [
            {
                "content": """
                # Security Best Practices

                ## Authentication
                - Always use strong passwords with at least 12 characters
                - Implement multi-factor authentication (MFA)
                - Use secure password hashing (bcrypt, Argon2)

                ## Authorization
                - Follow the principle of least privilege
                - Implement role-based access control (RBAC)
                - Regular access reviews and audits
                """,
                "source": "security-guide.md",
            },
            {
                "content": """
                # API Design Guidelines

                ## REST Principles
                - Use proper HTTP methods (GET, POST, PUT, DELETE)
                - Return appropriate status codes
                - Version your APIs

                ## Authentication
                - Use API keys or OAuth 2.0
                - Always use HTTPS
                - Rate limiting to prevent abuse
                """,
                "source": "api-guide.md",
            },
            {
                "content": """
                # Database Performance

                ## Indexing
                - Create indexes on frequently queried columns
                - Use composite indexes for multi-column queries
                - Monitor and remove unused indexes

                ## Query Optimization
                - Use EXPLAIN to analyze queries
                - Avoid SELECT * in production
                - Use connection pooling
                """,
                "source": "database-guide.md",
            },
        ]

        for doc in docs:
            result = await rag.add_text(doc["content"], doc["source"])
            print(f"   Added: {doc['source']} ({result.chunks} chunks)")

        # Ask questions
        print("\n2. Asking questions...\n")

        questions = [
            "What are the best practices for API authentication?",
            "How should I implement authorization in my application?",
            "What database indexing strategies should I use?",
        ]

        for q in questions:
            print(f"Q: {q}")
            print("-" * 40)

            response = await rag.query(q)

            print(f"A: {response.answer[:300]}...")
            print(f"\nConfidence: {response.confidence:.0%}")

            if response.citations:
                print("Sources:")
                for c in response.citations[:2]:
                    quote = c.get('quote', '')[:50]
                    print(f"  - {c['source']}: \"{quote}...\"")

            print("\n")

        # Show stats
        print("3. Session statistics:")
        stats = await rag.stats()
        print(f"   Documents: {stats['documents']['total_documents']}")
        print(f"   Tool calls: {len(stats['tools'])}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
