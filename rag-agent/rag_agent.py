#!/usr/bin/env python3
# =============================================================================
# RAG AGENT - Agente para Desafio Atlantyx
# =============================================================================
# Agente especializado em responder perguntas sobre IA em grandes empresas
# usando busca semantica nos documentos da base de conhecimento.
# =============================================================================

import asyncio
from claude_agent_sdk import query, AssistantMessage, TextBlock, ToolUseBlock, ResultMessage
from config import RAG_AGENT_OPTIONS


async def ask_question(question: str) -> str:
    """
    Envia uma pergunta para o agente RAG.

    Args:
        question: Pergunta sobre IA em grandes empresas

    Returns:
        Resposta do agente com citacoes
    """
    response_text = ""

    async for message in query(prompt=question, options=RAG_AGENT_OPTIONS):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text

    return response_text


async def interactive_mode():
    """Modo interativo para testar o agente."""
    print("=" * 60)
    print("RAG Agent - Desafio Atlantyx")
    print("=" * 60)
    print("Digite suas perguntas sobre IA em grandes empresas.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            question = input("\nVoce: ").strip()

            if not question:
                continue

            if question.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando...")
                break

            print("\nAgente: ", end="", flush=True)

            async for message in query(prompt=question, options=RAG_AGENT_OPTIONS):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="", flush=True)
                        elif isinstance(block, ToolUseBlock):
                            print(f"\n[Usando tool: {block.name}]", flush=True)

            print()  # Nova linha no final

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuario.")
            break
        except Exception as e:
            print(f"\nErro: {e}")


async def streaming_question(question: str):
    """Faz pergunta com streaming de resposta."""
    print(f"\nPergunta: {question}")
    print("-" * 40)
    print("Resposta: ", end="", flush=True)

    async for message in query(prompt=question, options=RAG_AGENT_OPTIONS):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Tool: {block.name}({block.input})]", flush=True)

    print("\n" + "-" * 40)


def test_questions():
    """Testa com as perguntas do desafio Atlantyx."""
    questions = [
        "Quais sao os principios obrigatorios da Politica de Uso de IA?",
        "Na arquitetura RAG enterprise, quais componentes sao obrigatorios?",
        "Cite 3 metricas minimas para operar um assistente de IA em producao.",
    ]

    print("=" * 60)
    print("TESTE - Perguntas do Desafio Atlantyx")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"\n--- Pergunta {i} ---")
        asyncio.run(streaming_question(q))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_questions()
        else:
            # Pergunta via linha de comando
            question = " ".join(sys.argv[1:])
            asyncio.run(streaming_question(question))
    else:
        # Modo interativo
        asyncio.run(interactive_mode())
