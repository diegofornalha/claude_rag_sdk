"""Quiz endpoints - Sistema inteligente de avaliação com RAG."""

import asyncio
import json
import uuid
from enum import Enum
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import app_state
from claude_rag_sdk.core.auth import verify_api_key
from claude_rag_sdk.core.logger import get_logger

router = APIRouter(prefix="/quiz", tags=["Quiz"])
logger = get_logger("quiz")

# Store para quizzes em background (em memória por simplicidade)
# Em produção, usar Redis ou AgentFS KV
_quiz_store: dict[str, dict] = {}


# =============================================================================
# ENUMS & MODELS
# =============================================================================


class QuizDifficulty(str, Enum):
    """Níveis de dificuldade das questões."""

    EASY = "easy"  # 30% - Conceitos básicos
    MEDIUM = "medium"  # 50% - Regras e validações
    HARD = "hard"  # 20% - Nuances e detalhes complexos


class QuizRank(str, Enum):
    """Rankings baseados na trilha de benefícios Renda Extra Ton."""

    EMBAIXADOR = "embaixador"  # 100% aproveitamento
    ESPECIALISTA_III = "especialista_iii"  # 90-99%
    ESPECIALISTA_II = "especialista_ii"  # 80-89%
    ESPECIALISTA_I = "especialista_i"  # 60-79%
    INICIANTE = "iniciante"  # <60%


class QuizOption(BaseModel):
    """Alternativa de múltipla escolha."""

    label: str = Field(..., description="Letra da alternativa (A, B, C, D)")
    text: str = Field(..., description="Texto da alternativa")


class QuizQuestion(BaseModel):
    """Questão do quiz com metadata educacional."""

    id: int = Field(..., description="ID da questão (1-N)")
    question: str = Field(..., description="Enunciado da questão")
    options: list[QuizOption] = Field(..., description="4 alternativas")
    correct_index: int = Field(..., ge=0, le=3, description="Índice da resposta correta (0-3)")
    difficulty: QuizDifficulty = Field(..., description="Nível de dificuldade")
    points: int = Field(..., description="Pontos atribuídos (1=fácil, 2=médio, 3=difícil)")
    explanation: str = Field(..., description="Explicação detalhada da resposta correta")
    wrong_feedback: dict[int, str] = Field(
        ...,
        description="Feedback específico para cada alternativa incorreta (index -> feedback)",
    )
    learning_tip: str = Field(..., description="Dica de memorização ou conceito-chave")
    source_reference: str = Field(
        default="", description="Referência ao trecho do documento (página/seção)"
    )


class GenerateQuizRequest(BaseModel):
    """Request para geração de quiz."""

    num_questions: int = Field(default=10, ge=5, le=20, description="Número de questões (5-20)")
    focus_topics: list[str] = Field(
        default=[],
        description="Tópicos específicos para focar (vazio = todos os tópicos do documento)",
    )
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        description="Distribuição de dificuldade (deve somar 1.0)",
    )


class GenerateQuizResponse(BaseModel):
    """Response com quiz gerado."""

    quiz_id: str = Field(..., description="ID único do quiz gerado")
    title: str = Field(..., description="Título do quiz")
    description: str = Field(..., description="Descrição do conteúdo")
    total_questions: int = Field(..., description="Total de questões")
    max_score: int = Field(..., description="Pontuação máxima possível")
    questions: list[QuizQuestion] = Field(..., description="Lista de questões")
    difficulty_breakdown: dict[str, int] = Field(
        ..., description="Contagem por dificuldade (easy/medium/hard)"
    )


class QuizAnswerRequest(BaseModel):
    """Request para avaliar uma resposta."""

    quiz_id: str = Field(..., description="ID do quiz")
    question_id: int = Field(..., description="ID da questão")
    selected_index: int = Field(..., ge=0, le=3, description="Índice selecionado (0-3)")


class QuizAnswerResponse(BaseModel):
    """Response da avaliação de resposta."""

    is_correct: bool = Field(..., description="Se a resposta está correta")
    points_earned: int = Field(..., description="Pontos ganhos (0 se errado)")
    correct_index: int = Field(..., description="Índice da resposta correta")
    feedback: str = Field(..., description="Feedback educativo detalhado")
    explanation: str = Field(..., description="Explicação da resposta correta")
    learning_tip: str = Field(..., description="Dica de aprendizado")


class QuizResultsRequest(BaseModel):
    """Request para calcular resultado final."""

    quiz_id: str = Field(..., description="ID do quiz")
    answers: list[int] = Field(..., description="Lista de índices selecionados para cada questão")


class QuizResultsResponse(BaseModel):
    """Response com resultado final e ranking."""

    total_questions: int = Field(..., description="Total de questões")
    correct_answers: int = Field(..., description="Respostas corretas")
    score: int = Field(..., description="Pontuação obtida")
    max_score: int = Field(..., description="Pontuação máxima")
    percentage: float = Field(..., description="Percentual de aproveitamento")
    rank: QuizRank = Field(..., description="Ranking alcançado")
    rank_title: str = Field(..., description="Título do ranking")
    rank_message: str = Field(..., description="Mensagem personalizada de feedback")
    breakdown: dict[str, dict[str, int]] = Field(
        ..., description="Análise por dificuldade (corretas/total)"
    )


# =============================================================================
# LAZY GENERATION MODELS
# =============================================================================


class StartQuizResponse(BaseModel):
    """Response ao iniciar quiz com lazy generation."""

    quiz_id: str = Field(..., description="ID único do quiz")
    total_questions: int = Field(default=10, description="Total de questões")
    first_question: QuizQuestion = Field(..., description="Primeira pergunta (fixa)")


class QuestionStatusResponse(BaseModel):
    """Status de uma pergunta específica."""

    quiz_id: str
    index: int
    ready: bool = Field(..., description="Se a pergunta está pronta")
    question: QuizQuestion | None = Field(None, description="Pergunta se pronta")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QUIZ_GENERATION_PROMPT = """Você é um especialista em criar questões educativas de múltipla escolha.

Gere {num_questions} questões sobre o programa Renda Extra Ton, baseadas EXCLUSIVAMENTE no contexto fornecido.

CONTEXTO:
{context}

REQUISITOS:
1. Distribuição de dificuldade:
   - {easy_count} questões FÁCEIS (conceitos básicos, definições)
   - {medium_count} questões MÉDIAS (regras, validações, prazos)
   - {hard_count} questões DIFÍCEIS (nuances, cálculos, casos especiais)

2. Para cada questão, forneça:
   - Enunciado claro e objetivo
   - 4 alternativas (sendo 1 correta e 3 plausíveis mas incorretas)
   - Explicação detalhada da resposta correta
   - Feedback específico para cada alternativa incorreta (explicar por que está errada)
   - Dica de memorização ou conceito-chave
   - Referência ao documento (página/seção se possível)

3. Critérios de qualidade:
   - Alternativas incorretas devem ser plausíveis (não óbvias)
   - Feedback deve ser educativo (identificar o erro conceitual)
   - Questões difíceis devem envolver cálculos ou regras complexas
   - Use linguagem clara e profissional

FORMATO DE SAÍDA (JSON):
```json
{{
  "title": "Quiz: Renda Extra Ton",
  "description": "Avalie seu conhecimento sobre o programa",
  "questions": [
    {{
      "question": "Qual é...",
      "options": [
        {{"label": "A", "text": "..."}},
        {{"label": "B", "text": "..."}},
        {{"label": "C", "text": "..."}},
        {{"label": "D", "text": "..."}}
      ],
      "correct_index": 1,
      "difficulty": "medium",
      "explanation": "A resposta correta é B porque...",
      "wrong_feedback": {{
        "0": "Esta alternativa está incorreta porque...",
        "2": "Este conceito está errado pois...",
        "3": "Essa opção confunde..."
      }},
      "learning_tip": "Lembre-se que...",
      "source_reference": "Seção 2.3 do regulamento"
    }}
  ]
}}
```

Gere o JSON completo agora:"""


SINGLE_QUESTION_PROMPT = """Gere UMA questão de múltipla escolha BASEADA EXCLUSIVAMENTE no documento abaixo.

DOCUMENTO DE REFERÊNCIA (use APENAS estas informações):
{context}

REQUISITOS OBRIGATÓRIOS:
- Dificuldade: {difficulty}
- Número da pergunta: {question_number} de 10

🚫🚫🚫 ATENÇÃO MÁXIMA - TÓPICOS PROIBIDOS 🚫🚫🚫
Os tópicos abaixo JÁ FORAM USADOS. É ABSOLUTAMENTE PROIBIDO fazer perguntas sobre eles:
{previous_topics}

⚠️ QUALQUER pergunta que mencione palavras-chave desses tópicos será REJEITADA!

REGRAS CRÍTICAS:
1. A pergunta DEVE ser sobre informações PRESENTES no documento acima
2. A resposta correta DEVE estar explícita ou claramente inferível do documento
3. NÃO invente informações que não estão no documento
4. As alternativas erradas devem ser plausíveis mas claramente incorretas segundo o documento
5. A explicação deve CITAR qual parte do documento comprova a resposta
6. 🚫 SE JÁ FALAMOS SOBRE "prazo de pagamento" - NÃO PERGUNTE SOBRE QUANDO/PRAZO DE PAGAMENTO!
7. 🚫 SE JÁ FALAMOS SOBRE "indicações" - NÃO PERGUNTE SOBRE NÚMERO/QUANTIDADE DE INDICAÇÕES!
8. 🚫 SE JÁ FALAMOS SOBRE "níveis" - NÃO PERGUNTE SOBRE ATUALIZAÇÃO/PROGRESSÃO DE NÍVEIS!

TÓPICOS DISPONÍVEIS PARA ESTA PERGUNTA (escolha um que NÃO está na lista proibida):
1. Definição do programa e objetivo
2. Critérios de elegibilidade - número de indicações
3. Níveis e como subir de nível
4. Frequência de atualização dos níveis (dia do mês)
5. Taxa percentual do TPV
6. Regime de comodato dos equipamentos
7. Requisitos para Ponto Físico (nível mínimo)
8. Prazo de pagamento das recompensas (dia 10)
9. Regras de desligamento do programa
10. Validade das indicações
11. Condições para perda de benefícios
12. Carteira ativa de indicados
13. Permanência mínima na carteira (12 meses)
14. Recompensa fixa por indicação (R$50)
15. Programa Ton na Mão
16. Programa TapTon e link de indicação
17. Requisitos para elegibilidade inicial
18. Suspensão temporária do usuário
19. Cancelamento definitivo do programa
20. Plataforma Ton e seus recursos

Retorne APENAS um JSON válido no formato:
{{
  "question": "Pergunta clara e objetiva baseada no documento...",
  "options": [
    {{"label": "A", "text": "Alternativa A"}},
    {{"label": "B", "text": "Alternativa B"}},
    {{"label": "C", "text": "Alternativa C"}},
    {{"label": "D", "text": "Alternativa D"}}
  ],
  "correct_index": 1,
  "explanation": "A resposta correta é B porque, segundo o documento: '[citar trecho]'. Isso mostra que...",
  "wrong_feedback": {{
    "0": "A alternativa A está incorreta porque o documento diz que...",
    "2": "A alternativa C está incorreta porque o documento especifica que...",
    "3": "A alternativa D está incorreta porque contradiz o trecho que diz..."
  }},
  "learning_tip": "Lembre-se: [conceito-chave do documento]",
  "source_reference": "Conforme [seção/cláusula do documento]"
}}

Gere o JSON agora:"""


FIRST_QUESTION_PROMPT = """Gere a PRIMEIRA questão de um quiz sobre o programa Renda Extra Ton.

DOCUMENTO DE REFERÊNCIA (use APENAS estas informações):
{context}

REQUISITOS:
- Esta é a pergunta 1 de 10 (deve ser de nível FÁCIL - conceito introdutório)
- A pergunta deve ser sobre um conceito FUNDAMENTAL do programa

⚠️ IMPORTANTE - ESCOLHA UM TEMA DIFERENTE A CADA VEZ:
Escolha ALEATORIAMENTE UM dos temas abaixo (seed: {seed}):
1. O que é o programa Renda Extra?
2. O que é o programa Renda Ton?
3. Quem pode participar do programa?
4. Qual é o objetivo principal do programa?
5. O que são indicações válidas?
6. Como funciona a trilha de benefícios?
7. O que é a Plataforma Ton?
8. Qual a relação entre Renda Extra e Renda Ton?

Use o número seed ({seed}) para escolher: some os dígitos e use módulo 8 para selecionar o tema.

REGRAS CRÍTICAS:
1. A pergunta DEVE ser sobre informações PRESENTES no documento
2. A resposta correta DEVE estar explícita no documento
3. NÃO invente informações
4. As alternativas erradas devem ser plausíveis mas claramente incorretas
5. A explicação deve CITAR o documento
6. VARIE a formulação - não use sempre "O que é..."

Retorne APENAS um JSON válido:
{{
  "question": "Pergunta introdutória sobre o programa...",
  "options": [
    {{"label": "A", "text": "Alternativa A"}},
    {{"label": "B", "text": "Alternativa B"}},
    {{"label": "C", "text": "Alternativa C"}},
    {{"label": "D", "text": "Alternativa D"}}
  ],
  "correct_index": 1,
  "explanation": "A resposta correta é [X] porque o documento diz: '[citação]'...",
  "wrong_feedback": {{
    "0": "A alternativa A está incorreta porque...",
    "2": "A alternativa C está incorreta porque...",
    "3": "A alternativa D está incorreta porque..."
  }},
  "learning_tip": "Conceito-chave: [resumo do documento]",
  "source_reference": "Conforme [seção do documento]"
}}

Gere o JSON:"""


# FIRST_QUESTION removido - agora P1 é gerada dinamicamente via generate_first_question()
# Mantemos apenas como fallback em caso de erro na geração
FIRST_QUESTION_FALLBACK = QuizQuestion(
    id=1,
    question="O que é o programa Renda Extra oferecido pelo Ton?",
    options=[
        QuizOption(label="A", text="Um programa de cashback para clientes"),
        QuizOption(label="B", text="Um programa de indicação com recompensas financeiras"),
        QuizOption(label="C", text="Um programa de fidelidade com pontos"),
        QuizOption(label="D", text="Um programa de descontos em taxas"),
    ],
    correct_index=1,
    difficulty=QuizDifficulty.EASY,
    points=1,
    explanation="O Renda Extra é um programa de indicação. Consulte o regulamento para mais detalhes.",
    wrong_feedback={},
    learning_tip="Consulte o regulamento oficial para informações precisas.",
    source_reference="",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_rank(percentage: float) -> tuple[QuizRank, str, str]:
    """Calcula o ranking baseado no percentual de aproveitamento.

    Faixas de ranking:
    - 96-100%: 🏆 Embaixador (Domínio total)
    - 86-95%: 🌟 Especialista III (Conhecimento profundo)
    - 71-85%: ⭐ Especialista II (Boa compreensão)
    - 51-70%: 📚 Especialista I (Base sólida)
    - <50%: 🌱 Iniciante (Precisa revisar)

    Returns:
        tuple: (rank, title, message)
    """
    if percentage >= 96:
        return (
            QuizRank.EMBAIXADOR,
            "🏆 Embaixador do Renda Extra Ton",
            "Domínio total! Você possui conhecimento excepcional das regras do programa e está pronto para ser "
            "um verdadeiro embaixador, ajudando outros parceiros a maximizarem seus ganhos!",
        )
    elif percentage >= 86:
        return (
            QuizRank.ESPECIALISTA_III,
            "🌟 Especialista Nível III",
            "Excelente! Você possui conhecimento profundo do programa. Com esse domínio, você está muito próximo "
            "de alcançar o nível de Embaixador. Continue aprimorando os detalhes!",
        )
    elif percentage >= 71:
        return (
            QuizRank.ESPECIALISTA_II,
            "⭐ Especialista Nível II",
            "Muito bem! Você compreende bem as regras do Renda Extra Ton. Continue estudando as nuances e "
            "casos especiais para alcançar o Nível III!",
        )
    elif percentage >= 51:
        return (
            QuizRank.ESPECIALISTA_I,
            "📚 Especialista Nível I",
            "Bom trabalho! Você tem uma base sólida sobre o programa. Aprofunde seu conhecimento sobre as regras "
            "específicas e validações para evoluir para Especialista II!",
        )
    else:
        return (
            QuizRank.INICIANTE,
            "🌱 Iniciante no Programa",
            "Você está começando sua jornada! O conhecimento vem com estudo dedicado. "
            "Revise o regulamento com atenção, focando nos conceitos fundamentais e regras principais antes de avançar.",
        )


async def generate_questions_with_rag(
    num_questions: int, difficulty_distribution: dict[str, float], focus_topics: list[str]
) -> dict:
    """Gera questões usando RAG + Claude."""
    from claude_rag_sdk import ClaudeRAGOptions
    from claude_rag_sdk.agent import AgentEngine

    try:
        # Use the global RAG instance to access ingested documents
        rag = await app_state.get_rag()

        # Calculate question distribution
        easy_count = max(1, int(num_questions * difficulty_distribution.get("easy", 0.3)))
        medium_count = max(1, int(num_questions * difficulty_distribution.get("medium", 0.5)))
        hard_count = max(1, num_questions - easy_count - medium_count)

        # Build search query
        if focus_topics:
            search_query = f"Tópicos: {', '.join(focus_topics)}. Regras, validações e detalhes do programa Renda Extra Ton"
        else:
            search_query = (
                "Regras, validações, benefícios, prazos e detalhes do programa Renda Extra Ton"
            )

        # Search for relevant context
        logger.info("Buscando contexto RAG", query=search_query)
        search_results = await rag.search(search_query, top_k=10)

        if not search_results:
            raise HTTPException(
                status_code=404,
                detail="Nenhum documento encontrado. Faça a ingestão do regulamento primeiro.",
            )

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:8], 1):
            context_parts.append(f"[Trecho {i}]\n{result.content}\n")
        context = "\n".join(context_parts)

        # Generate prompt
        prompt = QUIZ_GENERATION_PROMPT.format(
            num_questions=num_questions,
            context=context,
            easy_count=easy_count,
            medium_count=medium_count,
            hard_count=hard_count,
        )

        # Use AgentEngine to call Claude (handles authentication)
        logger.info("Gerando questões com Claude", questions=num_questions)
        from claude_rag_sdk import AgentModel
        quiz_system_prompt = """Você é um gerador de quizzes. Responda APENAS com JSON válido.
Não use o formato padrão de answer/citations. Gere DIRETAMENTE o JSON do quiz no formato solicitado."""
        options = ClaudeRAGOptions(id="quiz-generator", agent_model=AgentModel.OPUS, system_prompt=quiz_system_prompt)
        agent = AgentEngine(options=options)
        response = await agent.query(prompt)

        # Parse response - AgentResponse has 'answer' attribute
        content = response.answer
        logger.info("Resposta do Claude recebida", length=len(content), preview=content[:500] if content else "empty")

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Try to find JSON object in the content if it's not pure JSON
        if not content.strip().startswith("{"):
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)

        logger.info("Tentando parsear JSON", preview=content[:300] if content else "empty")
        quiz_data = json.loads(content)

        # Validate and enrich
        questions = []
        for idx, q in enumerate(quiz_data["questions"][:num_questions], 1):
            # Normalize difficulty (Claude sometimes uses "difficult" instead of "hard")
            raw_difficulty = q.get("difficulty", "medium").lower()
            difficulty_map = {
                "easy": "easy",
                "medium": "medium",
                "hard": "hard",
                "difficult": "hard",  # Claude variation
                "fácil": "easy",
                "médio": "medium",
                "difícil": "hard",
            }
            normalized_difficulty = difficulty_map.get(raw_difficulty, "medium")

            # Determine points based on difficulty
            diff = QuizDifficulty(normalized_difficulty)
            points = 1 if diff == QuizDifficulty.EASY else 2 if diff == QuizDifficulty.MEDIUM else 3

            questions.append(
                QuizQuestion(
                    id=idx,
                    question=q["question"],
                    options=[QuizOption(**opt) for opt in q["options"]],
                    correct_index=q["correct_index"],
                    difficulty=diff,
                    points=points,
                    explanation=q["explanation"],
                    wrong_feedback={int(k): v for k, v in q["wrong_feedback"].items()},
                    learning_tip=q["learning_tip"],
                    source_reference=q.get("source_reference", ""),
                )
            )

        return {
            "title": quiz_data.get("title", "Quiz: Renda Extra Ton"),
            "description": quiz_data.get(
                "description", "Avalie seu conhecimento sobre o programa"
            ),
            "questions": questions,
        }

    except json.JSONDecodeError as e:
        logger.error("Erro ao parsear JSON do Claude", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar resposta do Claude: {str(e)}"
        )
    except Exception as e:
        logger.error("Erro ao gerar quiz", error=str(e))
        raise HTTPException(status_code=500, detail=f"Erro ao gerar quiz: {str(e)}")


async def generate_first_question(quiz_id: str, context: str) -> QuizQuestion:
    """Gera a primeira pergunta dinamicamente baseada no documento RAG.

    Args:
        quiz_id: ID do quiz
        context: Contexto do documento RAG

    Returns:
        QuizQuestion gerada dinamicamente
    """
    from claude_rag_sdk import ClaudeRAGOptions
    from claude_rag_sdk.agent import AgentEngine
    from claude_rag_sdk.options import AgentModel

    logger.info(f"[Quiz {quiz_id}] Gerando P1 dinamicamente...")

    try:
        # Configurar AgentEngine
        options = ClaudeRAGOptions(
            id=f"quiz-p1-{quiz_id}",
            agent_model=AgentModel.HAIKU,  # Rápido para P1
            system_prompt="Você é um gerador de questões de quiz. Responda APENAS com JSON válido, sem texto adicional.",
        )
        agent = AgentEngine(options=options)

        # Usar quiz_id como seed para variar o tema da P1
        # Converter hex para int e usar como seed
        seed = int(quiz_id.replace("-", "")[:8], 16) % 10000
        prompt = FIRST_QUESTION_PROMPT.format(context=context, seed=seed)
        logger.info(f"[Quiz {quiz_id}] P1 seed: {seed}")
        response = await agent.query(prompt)
        answer_text = response.answer if hasattr(response, "answer") else str(response)

        # Extrair JSON da resposta
        json_match = answer_text
        if "```json" in answer_text:
            json_match = answer_text.split("```json")[1].split("```")[0]
        elif "```" in answer_text:
            json_match = answer_text.split("```")[1].split("```")[0]

        q_data = json.loads(json_match.strip())

        question = QuizQuestion(
            id=1,
            question=q_data["question"],
            options=[QuizOption(**opt) for opt in q_data["options"]],
            correct_index=q_data["correct_index"],
            difficulty=QuizDifficulty.EASY,
            points=1,
            explanation=q_data["explanation"],
            wrong_feedback={int(k): v for k, v in q_data.get("wrong_feedback", {}).items()},
            learning_tip=q_data.get("learning_tip", ""),
            source_reference=q_data.get("source_reference", ""),
        )

        logger.info(f"[Quiz {quiz_id}] P1 gerada com sucesso: {question.question[:50]}...")
        return question

    except Exception as e:
        logger.error(f"[Quiz {quiz_id}] Erro ao gerar P1: {e}, usando fallback")
        # Fallback para pergunta genérica em caso de erro
        return FIRST_QUESTION_FALLBACK


async def generate_remaining_questions(quiz_id: str) -> None:
    """Gera perguntas 2-10 em background usando Claude Agent SDK.

    Esta função é executada via asyncio.create_task() e salva as perguntas
    no _quiz_store para serem recuperadas pelo endpoint /quiz/question.
    """
    from claude_rag_sdk import ClaudeRAGOptions
    from claude_rag_sdk.agent import AgentEngine
    from claude_rag_sdk.options import AgentModel

    logger.info(f"[Quiz {quiz_id}] Iniciando geração em background...")

    try:
        # Usar contexto já salvo no store (buscado em /start)
        quiz_data = _quiz_store.get(quiz_id)
        if not quiz_data:
            logger.error(f"[Quiz {quiz_id}] Quiz não encontrado no store")
            return

        context = quiz_data.get("context")
        if not context:
            # Fallback: buscar contexto RAG se não estiver no store
            rag = await app_state.get_rag()
            search_results = await rag.search(
                "Regras, validações, benefícios, prazos, níveis, recompensas do programa Renda Extra Ton",
                top_k=10,
            )

            if not search_results:
                logger.error(f"[Quiz {quiz_id}] Nenhum documento encontrado para RAG")
                _quiz_store[quiz_id]["error"] = "Nenhum documento encontrado"
                return

            context_parts = []
            for i, result in enumerate(search_results[:8], 1):
                context_parts.append(f"[Trecho {i}]\n{result.content}\n")
            context = "\n".join(context_parts)

        # 2. Configurar AgentEngine para geração rápida
        options = ClaudeRAGOptions(
            id=f"quiz-gen-{quiz_id}",
            agent_model=AgentModel.HAIKU,  # Haiku é mais rápido
            system_prompt="Você é um gerador de questões de quiz. Responda APENAS com JSON válido, sem texto adicional.",
        )
        agent = AgentEngine(options=options)

        # 3. Distribuição de dificuldades para perguntas 2-10
        # Total: 1 easy (P1 fixa), 2 easy, 5 medium, 2 hard = 10 perguntas
        difficulties = ["easy", "medium", "medium", "medium", "hard", "medium", "medium", "hard", "easy"]

        # Tópicos já usados (para evitar repetição) - começar com P1
        previous_topics = ["definição do programa Renda Extra"]

        # Mapeamento de palavras-chave para tópicos (ordem importa - mais específico primeiro)
        topic_keywords = {
            # === ESPECÍFICOS (alta prioridade) ===
            # Ponto Físico
            "ponto físico": "requisitos para Ponto Físico",
            "ponto ton": "requisitos para Ponto Físico",
            "elegível ao uso do ponto": "requisitos para Ponto Físico",
            # Comodato/Equipamentos
            "comodato": "regime de comodato dos equipamentos",
            "equipamento": "regime de comodato dos equipamentos",
            "ton na mão": "regime de comodato dos equipamentos",
            "disponibilizados": "regime de comodato dos equipamentos",
            # TPV/TapTon
            "tpv": "taxa percentual do TPV",
            "0,2%": "taxa percentual do TPV",
            "tapton": "programa Indique TapTon",
            "link de indicação": "programa Indique TapTon",
            # Permanência
            "permanência": "permanência na carteira",
            "12 meses": "permanência na carteira",
            "doze meses": "permanência na carteira",

            # === INDICAÇÕES (P2/P9 - mesmo tópico) ===
            "número mínimo de indicações": "quantidade mínima de indicações para elegibilidade",
            "mínimo de indicações": "quantidade mínima de indicações para elegibilidade",
            "3 indicações": "quantidade mínima de indicações para elegibilidade",
            "três indicações": "quantidade mínima de indicações para elegibilidade",
            "3 (três) indicações": "quantidade mínima de indicações para elegibilidade",
            "indicações válidas": "quantidade mínima de indicações para elegibilidade",
            "critério principal": "quantidade mínima de indicações para elegibilidade",
            "elegível a participar": "quantidade mínima de indicações para elegibilidade",

            # === ATUALIZAÇÃO DE NÍVEL (P3/P10 - mesmo tópico) ===
            "periodicidade de atualização": "data de atualização mensal do nível",
            "atualização do nível": "data de atualização mensal do nível",
            "dia do mês": "data de atualização mensal do nível",
            "dia 1": "data de atualização mensal do nível",
            "todo dia 1": "data de atualização mensal do nível",
            "primeiro dia": "data de atualização mensal do nível",
            "1º do mês": "data de atualização mensal do nível",
            "atualizado": "data de atualização mensal do nível",

            # === PRAZO DE PAGAMENTO (detectar TODAS as variações) ===
            "prazo máximo para o pagamento": "prazo de pagamento das recompensas",
            "prazo máximo para pagamento": "prazo de pagamento das recompensas",
            "prazo para o pagamento": "prazo de pagamento das recompensas",
            "prazo para pagamento": "prazo de pagamento das recompensas",
            "prazo de pagamento": "prazo de pagamento das recompensas",
            "pagamento das recompensas": "prazo de pagamento das recompensas",
            "pagamento da recompensa": "prazo de pagamento das recompensas",
            "efetue o pagamento": "prazo de pagamento das recompensas",
            "valores serão pagos": "prazo de pagamento das recompensas",
            "dia 10": "prazo de pagamento das recompensas",
            "10º dia": "prazo de pagamento das recompensas",
            "décimo dia": "prazo de pagamento das recompensas",
            "mês subsequente": "prazo de pagamento das recompensas",

            # === RECOMPENSA FIXA ===
            "r$50": "recompensa fixa por indicação",
            "r$ 50": "recompensa fixa por indicação",
            "cinquenta reais": "recompensa fixa por indicação",
            "50 reais": "recompensa fixa por indicação",
            "recompensa fixa": "recompensa fixa por indicação",
            "valor fixo": "recompensa fixa por indicação",

            # === TON NA MÃO ===
            "ton na mão": "programa Ton na Mão",
            "entrega direta": "programa Ton na Mão",
            "disponibilização direta": "programa Ton na Mão",

            # === SUSPENSÃO/CANCELAMENTO ===
            "suspensão": "suspensão e cancelamento",
            "suspenso": "suspensão e cancelamento",
            "cancelamento": "suspensão e cancelamento",
            "cancelado": "suspensão e cancelamento",
            "desligamento": "suspensão e cancelamento",
            "desligado": "suspensão e cancelamento",
            "excluído": "suspensão e cancelamento",

            # === VALIDADE DAS INDICAÇÕES ===
            "validade": "validade das indicações",
            "válidas": "validade das indicações",
            "indicações válidas": "validade das indicações",

            # === GENÉRICOS (baixa prioridade) ===
            "nível mínimo": "requisitos de nível para benefícios",
            "especialista i": "requisitos de nível para benefícios",
            "nível do usuário": "sistema de níveis",
            "nível": "sistema de níveis",
            "especialista": "sistema de níveis",
            "indicação": "programa de indicação",
            "elegibilidade": "critérios gerais de elegibilidade",
            "recompensa": "cálculo de recompensas",
            "pagamento": "prazo de pagamento das recompensas",
            "renda extra": "definição do programa Renda Extra",
            "renda ton": "definição do programa Renda Ton",
            "função": "objetivo do programa",
            "objetivo": "objetivo do programa",
            "principal": "objetivo do programa",
        }

        def extract_topic(question_text: str) -> str:
            """Extrai o tópico principal de uma pergunta."""
            q_lower = question_text.lower()
            for keyword, topic in topic_keywords.items():
                if keyword in q_lower:
                    return topic
            # Fallback: usar primeiros 60 chars
            return question_text[:60]

        def is_duplicate_topic(question_text: str, used_topics: list[str]) -> bool:
            """Verifica se a pergunta é sobre um tópico já usado."""
            topic = extract_topic(question_text)
            return topic in used_topics

        # 4. Gerar perguntas 2-10 com retry para duplicatas
        MAX_RETRIES = 5  # Aumentado para dar mais chances de encontrar tópico único

        for i, difficulty in enumerate(difficulties, start=2):
            retry_count = 0
            question_generated = False

            while not question_generated and retry_count < MAX_RETRIES:
                try:
                    # Formatar tópicos anteriores de forma clara
                    topics_str = "\n".join([f"  🚫 {t}" for t in previous_topics])

                    prompt = SINGLE_QUESTION_PROMPT.format(
                        context=context,
                        difficulty=difficulty,
                        question_number=i,
                        previous_topics=topics_str,
                    )

                    response = await agent.query(prompt)
                    answer_text = response.answer if hasattr(response, "answer") else str(response)

                    # Extrair JSON da resposta
                    json_match = answer_text
                    if "```json" in answer_text:
                        json_match = answer_text.split("```json")[1].split("```")[0]
                    elif "```" in answer_text:
                        json_match = answer_text.split("```")[1].split("```")[0]

                    q_data = json.loads(json_match.strip())

                    # VALIDAÇÃO DE DUPLICATA - Verificar ANTES de criar a pergunta
                    question_text = q_data["question"]
                    if is_duplicate_topic(question_text, previous_topics):
                        detected_topic = extract_topic(question_text)
                        logger.warning(
                            f"[Quiz {quiz_id}] P{i} DUPLICATA DETECTADA! "
                            f"Tópico '{detected_topic}' já usado. Retry {retry_count + 1}/{MAX_RETRIES}"
                        )
                        retry_count += 1
                        continue  # Tentar novamente

                    # Normalizar dificuldade
                    raw_difficulty = q_data.get("difficulty", difficulty).lower()
                    difficulty_map = {"easy": "easy", "medium": "medium", "hard": "hard", "difficult": "hard"}
                    normalized_diff = difficulty_map.get(raw_difficulty, difficulty)

                    diff_enum = QuizDifficulty(normalized_diff)
                    points = 1 if diff_enum == QuizDifficulty.EASY else 2 if diff_enum == QuizDifficulty.MEDIUM else 3

                    question = QuizQuestion(
                        id=i,
                        question=q_data["question"],
                        options=[QuizOption(**opt) for opt in q_data["options"]],
                        correct_index=q_data["correct_index"],
                        difficulty=diff_enum,
                        points=points,
                        explanation=q_data["explanation"],
                        wrong_feedback={int(k): v for k, v in q_data.get("wrong_feedback", {}).items()},
                        learning_tip=q_data.get("learning_tip", ""),
                        source_reference=q_data.get("source_reference", ""),
                    )

                    # Salvar no store
                    _quiz_store[quiz_id]["questions"][i] = question
                    _quiz_store[quiz_id]["generated_count"] = i

                    # Adicionar tópico para evitar repetição
                    topic = extract_topic(q_data["question"])
                    if topic not in previous_topics:
                        previous_topics.append(topic)
                    logger.info(f"[Quiz {quiz_id}] P{i} OK - Tópico: {topic}")

                    question_generated = True
                    logger.info(f"[Quiz {quiz_id}] Pergunta {i} gerada com sucesso")

                except Exception as e:
                    logger.error(f"[Quiz {quiz_id}] Erro ao gerar pergunta {i} (retry {retry_count}): {e}")
                    retry_count += 1

            # Se esgotou retries sem sucesso, criar fallback
            if not question_generated:
                logger.error(f"[Quiz {quiz_id}] Pergunta {i}: máximo de retries atingido, usando fallback")
                _quiz_store[quiz_id]["questions"][i] = QuizQuestion(
                    id=i,
                    question=f"Pergunta {i} sobre o programa Renda Extra Ton",
                    options=[
                        QuizOption(label="A", text="Opção A"),
                        QuizOption(label="B", text="Opção B"),
                        QuizOption(label="C", text="Opção C"),
                        QuizOption(label="D", text="Opção D"),
                    ],
                    correct_index=0,
                    difficulty=QuizDifficulty.MEDIUM,
                    points=2,
                    explanation="Erro ao gerar pergunta. Consulte o regulamento.",
                    wrong_feedback={},
                    learning_tip="",
                    source_reference="",
                )
                _quiz_store[quiz_id]["generated_count"] = i

        # Marcar como completo
        _quiz_store[quiz_id]["complete"] = True
        _quiz_store[quiz_id]["max_score"] = sum(
            q.points for q in _quiz_store[quiz_id]["questions"].values()
        )
        logger.info(f"[Quiz {quiz_id}] Geração completa! {len(_quiz_store[quiz_id]['questions'])} perguntas")

    except Exception as e:
        logger.error(f"[Quiz {quiz_id}] Erro fatal na geração: {e}")
        _quiz_store[quiz_id]["error"] = str(e)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/generate", response_model=GenerateQuizResponse)
async def generate_quiz(
    request: GenerateQuizRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    """Gera um quiz dinâmico usando RAG + Claude.

    - Busca contexto relevante no documento ingerido
    - Gera questões com distribuição de dificuldade (30/50/20)
    - Cada questão tem feedback educativo detalhado
    - Pontuação ponderada por dificuldade
    """
    import uuid

    logger.info("Gerando quiz", num_questions=request.num_questions)

    # Generate quiz
    result = await generate_questions_with_rag(
        request.num_questions, request.difficulty_distribution, request.focus_topics
    )

    # Calculate metadata
    quiz_id = str(uuid.uuid4())[:8]
    questions: list[QuizQuestion] = result["questions"]

    difficulty_breakdown = {
        "easy": sum(1 for q in questions if q.difficulty == QuizDifficulty.EASY),
        "medium": sum(1 for q in questions if q.difficulty == QuizDifficulty.MEDIUM),
        "hard": sum(1 for q in questions if q.difficulty == QuizDifficulty.HARD),
    }

    max_score = sum(q.points for q in questions)

    return GenerateQuizResponse(
        quiz_id=quiz_id,
        title=result["title"],
        description=result["description"],
        total_questions=len(questions),
        max_score=max_score,
        questions=questions,
        difficulty_breakdown=difficulty_breakdown,
    )


@router.post("/answer", response_model=QuizAnswerResponse)
async def evaluate_answer(
    request: QuizAnswerRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    """Avalia uma resposta individual.

    - Retorna se está correta
    - Fornece feedback educativo específico
    - Explica a resposta correta
    - Oferece dica de aprendizado

    Note: Este endpoint é stateless. O controle de estado do quiz
    deve ser feito no frontend.
    """
    # Note: Em produção, você armazenaria as questões do quiz em cache/db
    # Por ora, assumimos que o frontend mantém o estado completo
    raise HTTPException(
        status_code=501,
        detail="Use o endpoint /generate para obter questões e avalie no frontend",
    )


@router.post("/results", response_model=QuizResultsResponse)
async def calculate_results(
    request: QuizResultsRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    """Calcula resultado final e ranking.

    - Analisa desempenho por dificuldade
    - Calcula percentual de aproveitamento
    - Atribui ranking na trilha de carreira
    - Fornece feedback personalizado

    Note: Este endpoint é stateless. Passe as respostas e as questões
    serão buscadas do quiz_id (se implementar cache).
    """
    # Note: Similar ao /answer, precisaria de cache para funcionar completamente
    raise HTTPException(
        status_code=501, detail="Implemente cache de quiz para usar este endpoint"
    )


# =============================================================================
# LAZY GENERATION ENDPOINTS
# =============================================================================


@router.post("/start", response_model=StartQuizResponse)
async def start_quiz(
    _api_key: str | None = Depends(verify_api_key),
):
    """Inicia um quiz com lazy generation.

    - Valida que existem documentos no RAG antes de iniciar
    - Gera a primeira pergunta dinamicamente baseada no documento
    - Inicia geração das perguntas 2-10 em background
    - Frontend pode buscar perguntas via /question/{quiz_id}/{index}

    Esta arquitetura permite UX fluida enquanto as demais
    perguntas são geradas em paralelo.
    """
    # VALIDAÇÃO CRÍTICA: Verificar se RAG tem documentos
    rag = await app_state.get_rag()
    search_results = await rag.search(
        "programa Renda Extra Ton regras benefícios",
        top_k=5,
    )

    if not search_results:
        logger.error("Quiz não pode iniciar: RAG vazio")
        raise HTTPException(
            status_code=400,
            detail="Nenhum documento encontrado no RAG. Faça a ingestão do regulamento primeiro em /html/config.html",
        )

    # Construir contexto para P1
    context_parts = []
    for i, result in enumerate(search_results[:5], 1):
        context_parts.append(f"[Trecho {i}]\n{result.content}\n")
    context = "\n".join(context_parts)

    quiz_id = str(uuid.uuid4())[:8]

    # Gerar P1 dinamicamente baseada no documento
    first_question = await generate_first_question(quiz_id, context)

    # Inicializar store para este quiz
    _quiz_store[quiz_id] = {
        "questions": {1: first_question},
        "generated_count": 1,
        "complete": False,
        "error": None,
        "max_score": first_question.points,
        "context": context,  # Salvar contexto para P2-P10
    }

    # Iniciar geração em background (não bloqueia)
    asyncio.create_task(generate_remaining_questions(quiz_id))

    logger.info(f"[Quiz {quiz_id}] Iniciado com lazy generation (RAG: {len(search_results)} docs)")

    return StartQuizResponse(
        quiz_id=quiz_id,
        total_questions=10,
        first_question=first_question,
    )


@router.get("/question/{quiz_id}/{index}", response_model=QuizQuestion)
async def get_question(
    quiz_id: str,
    index: int,
    _api_key: str | None = Depends(verify_api_key),
):
    """Busca uma pergunta específica do quiz.

    - Se a pergunta já foi gerada, retorna imediatamente
    - Se ainda está sendo gerada, aguarda com polling (max 30s)
    - Se houver erro ou timeout, retorna HTTP 408/404

    Args:
        quiz_id: ID do quiz retornado por /start
        index: Número da pergunta (1-10)
    """
    if quiz_id not in _quiz_store:
        raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

    if index < 1 or index > 10:
        raise HTTPException(status_code=400, detail="Index deve ser entre 1 e 10")

    quiz_data = _quiz_store[quiz_id]

    # Verificar se houve erro na geração
    if quiz_data.get("error"):
        raise HTTPException(status_code=500, detail=f"Erro na geração: {quiz_data['error']}")

    # Se pergunta já está pronta, retornar imediatamente
    if index in quiz_data["questions"]:
        return quiz_data["questions"][index]

    # Polling: aguardar a pergunta ser gerada (max 30 tentativas x 1s = 30s)
    for attempt in range(30):
        await asyncio.sleep(1)

        # Verificar novamente
        if index in quiz_data["questions"]:
            logger.info(f"[Quiz {quiz_id}] Pergunta {index} pronta após {attempt + 1}s")
            return quiz_data["questions"][index]

        # Verificar erro
        if quiz_data.get("error"):
            raise HTTPException(status_code=500, detail=f"Erro na geração: {quiz_data['error']}")

    # Timeout
    raise HTTPException(
        status_code=408,
        detail=f"Timeout aguardando pergunta {index}. Geradas até agora: {quiz_data.get('generated_count', 0)}",
    )


@router.get("/status/{quiz_id}")
async def get_quiz_status(
    quiz_id: str,
    _api_key: str | None = Depends(verify_api_key),
):
    """Retorna status do quiz (para debug/monitoramento).

    Útil para verificar quantas perguntas já foram geradas.
    """
    if quiz_id not in _quiz_store:
        raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

    quiz_data = _quiz_store[quiz_id]

    return {
        "quiz_id": quiz_id,
        "generated_count": quiz_data.get("generated_count", 0),
        "total_questions": 10,
        "complete": quiz_data.get("complete", False),
        "error": quiz_data.get("error"),
        "max_score": quiz_data.get("max_score", 0),
        "questions_ready": list(quiz_data["questions"].keys()),
    }


@router.get("/all/{quiz_id}")
async def get_all_questions(
    quiz_id: str,
    _api_key: str | None = Depends(verify_api_key),
):
    """Retorna todas as perguntas do quiz (quando geração completa).

    Útil para o frontend obter max_score e calcular resultado final.
    """
    if quiz_id not in _quiz_store:
        raise HTTPException(status_code=404, detail=f"Quiz {quiz_id} não encontrado")

    quiz_data = _quiz_store[quiz_id]

    # Verificar se está completo
    if not quiz_data.get("complete"):
        raise HTTPException(
            status_code=202,
            detail=f"Quiz ainda em geração. Perguntas prontas: {quiz_data.get('generated_count', 0)}/10",
        )

    # Retornar todas as perguntas ordenadas
    questions = [quiz_data["questions"][i] for i in range(1, 11) if i in quiz_data["questions"]]

    return {
        "quiz_id": quiz_id,
        "total_questions": len(questions),
        "max_score": quiz_data.get("max_score", 0),
        "questions": questions,
    }
