"""Title Generator - Geração inteligente de títulos usando Claude Agent SDK.

Usa a função `query` do Claude Agent SDK para gerar títulos curtos e
descritivos para conversas, baseado no contexto das mensagens.
"""

from claude_rag_sdk.core.logger import get_logger

logger = get_logger("title_generator")


async def generate_conversation_title(
    messages: list[dict],
    max_messages: int = 5,
    max_words: int = 5,
) -> str:
    """Gera um título curto para uma conversa usando Claude.

    Args:
        messages: Lista de mensagens da conversa [{"role": "user/assistant", "content": "..."}]
        max_messages: Número máximo de mensagens a considerar (default: 5)
        max_words: Número máximo de palavras no título (default: 5)

    Returns:
        Título gerado (4-5 palavras)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Como funciona o React hooks?"},
        ...     {"role": "assistant", "content": "React hooks são funções..."}
        ... ]
        >>> titulo = await generate_conversation_title(messages)
        >>> print(titulo)  # "Funcionamento do React Hooks"
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import query as sdk_query
    except ImportError as e:
        logger.warning(f"Claude Agent SDK não disponível: {e}")
        # Fallback para as 3 primeiras palavras
        if messages:
            words = messages[0].get("content", "").split()[:3]
            return " ".join(words)
        return "Nova conversa"

    # Limitar mensagens para não sobrecarregar
    limited_messages = messages[:max_messages]

    # Formatar contexto da conversa
    contexto = "\n".join([
        f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
        for msg in limited_messages
    ])

    system_prompt = f"""Você é um gerador de títulos. Sua única função é criar títulos curtos
de {max_words} palavras que capturem o tema principal de uma conversa.
Responda APENAS com o título, sem explicações, sem aspas, sem pontuação final."""

    prompt = f"""Conversa:
{contexto}

Título:"""

    titulo = ""

    try:
        options = ClaudeAgentOptions(
            model="claude-sonnet-4-20250514",  # Modelo rápido e barato
            system_prompt=system_prompt,
        )

        async for message in sdk_query(prompt=prompt, options=options):
            # Extrai o texto da resposta
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        titulo += block.text

        # Limpar título
        titulo = titulo.strip()

        # Remover aspas se houver
        titulo = titulo.strip('"\'')

        # Limitar palavras
        words = titulo.split()
        if len(words) > max_words:
            titulo = " ".join(words[:max_words])

        # Limitar caracteres
        if len(titulo) > 50:
            titulo = titulo[:50].rsplit(" ", 1)[0]

        logger.info(f"Título gerado: {titulo}")
        return titulo

    except Exception as e:
        logger.warning(f"Erro ao gerar título com Claude: {e}")
        # Fallback para as 3 primeiras palavras
        if messages:
            words = messages[0].get("content", "").split()[:3]
            return " ".join(words)
        return "Nova conversa"


async def should_generate_title(
    messages: list[dict],
    min_messages: int = 2,
) -> bool:
    """Verifica se deve gerar um título para a conversa.

    Args:
        messages: Lista de mensagens
        min_messages: Número mínimo de mensagens para gerar título

    Returns:
        True se deve gerar título
    """
    # Precisa de pelo menos N mensagens para ter contexto
    if len(messages) < min_messages:
        return False

    # Verifica se tem pelo menos uma mensagem do usuário com conteúdo substancial
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return False

    first_user_msg = user_messages[0].get("content", "")
    # Se a primeira mensagem for muito curta, talvez não valha gerar título
    if len(first_user_msg.split()) < 3:
        return False

    return True


# Função simplificada para uso direto
async def get_smart_title(
    user_message: str,
    assistant_response: str | None = None,
) -> str:
    """Gera título inteligente a partir de uma troca de mensagens.

    Args:
        user_message: Mensagem do usuário
        assistant_response: Resposta do assistente (opcional)

    Returns:
        Título gerado
    """
    messages = [{"role": "user", "content": user_message}]

    if assistant_response:
        messages.append({"role": "assistant", "content": assistant_response})

    return await generate_conversation_title(messages)
