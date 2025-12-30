"""Quiz Templates - Prompts e constantes para geracao de questoes."""

from ..models.enums import QuizDifficulty
from ..models.schemas import QuizOption, QuizQuestion

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

QUIZ_SYSTEM_PROMPT = """Voce e um gerador de questoes de quiz. Responda APENAS com JSON valido, sem texto adicional."""

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QUIZ_GENERATION_PROMPT = """Voce e um especialista em criar questoes educativas de multipla escolha.

Gere {num_questions} questoes sobre o programa Renda Extra Ton, baseadas EXCLUSIVAMENTE no contexto fornecido.

CONTEXTO:
{context}

REQUISITOS:
1. Distribuicao de dificuldade:
   - {easy_count} questoes FACEIS (conceitos basicos, definicoes)
   - {medium_count} questoes MEDIAS (regras, validacoes, prazos)
   - {hard_count} questoes DIFICEIS (nuances, calculos, casos especiais)

2. Para cada questao, forneca:
   - Enunciado claro e objetivo
   - 4 alternativas (sendo 1 correta e 3 plausiveis mas incorretas)
   - Explicacao detalhada da resposta correta
   - Feedback especifico para cada alternativa incorreta (explicar por que esta errada)
   - Dica de memorizacao ou conceito-chave
   - Referencia ao documento (pagina/secao se possivel)

3. Criterios de qualidade:
   - Alternativas incorretas devem ser plausiveis (nao obvias)
   - Feedback deve ser educativo (identificar o erro conceitual)
   - Questoes dificeis devem envolver calculos ou regras complexas
   - Use linguagem clara e profissional

FORMATO DE SAIDA (JSON):
```json
{{
  "title": "Quiz: Renda Extra Ton",
  "description": "Avalie seu conhecimento sobre o programa",
  "questions": [
    {{
      "question": "Qual e...",
      "options": [
        {{"label": "A", "text": "..."}},
        {{"label": "B", "text": "..."}},
        {{"label": "C", "text": "..."}},
        {{"label": "D", "text": "..."}}
      ],
      "correct_index": 1,
      "difficulty": "medium",
      "explanation": "A resposta correta e B porque...",
      "wrong_feedback": {{
        "0": "Esta alternativa esta incorreta porque...",
        "2": "Este conceito esta errado pois...",
        "3": "Essa opcao confunde..."
      }},
      "learning_tip": "Lembre-se que...",
      "source_reference": "Secao 2.3 do regulamento"
    }}
  ]
}}
```

Gere o JSON completo agora:"""


SINGLE_QUESTION_PROMPT = """Gere UMA questao de multipla escolha BASEADA EXCLUSIVAMENTE no documento abaixo.

DOCUMENTO DE REFERENCIA (use APENAS estas informacoes):
{context}

REQUISITOS OBRIGATORIOS:
- Dificuldade: {difficulty}
- Numero da pergunta: {question_number} de 10

ATENCAO MAXIMA - TOPICOS PROIBIDOS
Os topicos abaixo JA FORAM USADOS. E ABSOLUTAMENTE PROIBIDO fazer perguntas sobre eles:
{previous_topics}

QUALQUER pergunta que mencione palavras-chave desses topicos sera REJEITADA!

REGRAS CRITICAS:
1. A pergunta DEVE ser sobre informacoes PRESENTES no documento acima
2. A resposta correta DEVE estar explicita ou claramente inferivel do documento
3. NAO invente informacoes que nao estao no documento
4. As alternativas erradas devem ser plausiveis mas claramente incorretas segundo o documento
5. A explicacao deve CITAR qual parte do documento comprova a resposta
6. SE JA FALAMOS SOBRE "prazo de pagamento" - NAO PERGUNTE SOBRE QUANDO/PRAZO DE PAGAMENTO!
7. SE JA FALAMOS SOBRE "indicacoes" - NAO PERGUNTE SOBRE NUMERO/QUANTIDADE DE INDICACOES!
8. SE JA FALAMOS SOBRE "niveis" - NAO PERGUNTE SOBRE ATUALIZACAO/PROGRESSAO DE NIVEIS!

TOPICOS DISPONIVEIS PARA ESTA PERGUNTA (escolha um que NAO esta na lista proibida):
1. Definicao do programa e objetivo
2. Criterios de elegibilidade - numero de indicacoes
3. Niveis e como subir de nivel
4. Frequencia de atualizacao dos niveis (dia do mes)
5. Taxa percentual do TPV
6. Regime de comodato dos equipamentos
7. Requisitos para Ponto Fisico (nivel minimo)
8. Prazo de pagamento das recompensas (dia 10)
9. Regras de desligamento do programa
10. Validade das indicacoes
11. Condicoes para perda de beneficios
12. Carteira ativa de indicados
13. Permanencia minima na carteira (12 meses)
14. Recompensa fixa por indicacao (R$50)
15. Programa Ton na Mao
16. Programa TapTon e link de indicacao
17. Requisitos para elegibilidade inicial
18. Suspensao temporaria do usuario
19. Cancelamento definitivo do programa
20. Plataforma Ton e seus recursos

Retorne APENAS um JSON valido no formato:
{{
  "question": "Pergunta clara e objetiva baseada no documento...",
  "options": [
    {{"label": "A", "text": "Alternativa A"}},
    {{"label": "B", "text": "Alternativa B"}},
    {{"label": "C", "text": "Alternativa C"}},
    {{"label": "D", "text": "Alternativa D"}}
  ],
  "correct_index": 1,
  "explanation": "A resposta correta e B porque, segundo o documento: '[citar trecho]'. Isso mostra que...",
  "wrong_feedback": {{
    "0": "A alternativa A esta incorreta porque o documento diz que...",
    "2": "A alternativa C esta incorreta porque o documento especifica que...",
    "3": "A alternativa D esta incorreta porque contradiz o trecho que diz..."
  }},
  "learning_tip": "Lembre-se: [conceito-chave do documento]",
  "source_reference": "Conforme [secao/clausula do documento]"
}}

Gere o JSON agora:"""


FIRST_QUESTION_PROMPT = """Gere a PRIMEIRA questao de um quiz sobre o programa Renda Extra Ton.

DOCUMENTO DE REFERENCIA (use APENAS estas informacoes):
{context}

REQUISITOS:
- Esta e a pergunta 1 de 10 (deve ser de nivel FACIL - conceito introdutorio)
- A pergunta deve ser sobre um conceito FUNDAMENTAL do programa

IMPORTANTE - ESCOLHA UM TEMA DIFERENTE A CADA VEZ:
Escolha ALEATORIAMENTE UM dos temas abaixo (seed: {seed}):
1. O que e o programa Renda Extra?
2. O que e o programa Renda Ton?
3. Quem pode participar do programa?
4. Qual e o objetivo principal do programa?
5. O que sao indicacoes validas?
6. Como funciona a trilha de beneficios?
7. O que e a Plataforma Ton?
8. Qual a relacao entre Renda Extra e Renda Ton?

Use o numero seed ({seed}) para escolher: some os digitos e use modulo 8 para selecionar o tema.

REGRAS CRITICAS:
1. A pergunta DEVE ser sobre informacoes PRESENTES no documento
2. A resposta correta DEVE estar explicita no documento
3. NAO invente informacoes
4. As alternativas erradas devem ser plausiveis mas claramente incorretas
5. A explicacao deve CITAR o documento
6. VARIE a formulacao - nao use sempre "O que e..."

Retorne APENAS um JSON valido:
{{
  "question": "Pergunta introdutoria sobre o programa...",
  "options": [
    {{"label": "A", "text": "Alternativa A"}},
    {{"label": "B", "text": "Alternativa B"}},
    {{"label": "C", "text": "Alternativa C"}},
    {{"label": "D", "text": "Alternativa D"}}
  ],
  "correct_index": 1,
  "explanation": "A resposta correta e [X] porque o documento diz: '[citacao]'...",
  "wrong_feedback": {{
    "0": "A alternativa A esta incorreta porque...",
    "2": "A alternativa C esta incorreta porque...",
    "3": "A alternativa D esta incorreta porque..."
  }},
  "learning_tip": "Conceito-chave: [resumo do documento]",
  "source_reference": "Conforme [secao do documento]"
}}

Gere o JSON:"""


# =============================================================================
# FALLBACK QUESTION
# =============================================================================

FIRST_QUESTION_FALLBACK = QuizQuestion(
    id=1,
    question="O que e o programa Renda Extra oferecido pelo Ton?",
    options=[
        QuizOption(label="A", text="Um programa de cashback para clientes"),
        QuizOption(label="B", text="Um programa de indicacao com recompensas financeiras"),
        QuizOption(label="C", text="Um programa de fidelidade com pontos"),
        QuizOption(label="D", text="Um programa de descontos em taxas"),
    ],
    correct_index=1,
    difficulty=QuizDifficulty.EASY,
    points=1,
    explanation="O Renda Extra e um programa de indicacao. Consulte o regulamento para mais detalhes.",
    wrong_feedback={},
    learning_tip="Consulte o regulamento oficial para informacoes precisas.",
    source_reference="",
)


# =============================================================================
# TOPIC KEYWORDS - Mapeamento para deduplicacao
# =============================================================================

TOPIC_KEYWORDS: dict[str, str] = {
    # === ESPECIFICOS (alta prioridade) ===
    # Ponto Fisico
    "ponto fisico": "requisitos para Ponto Fisico",
    "ponto ton": "requisitos para Ponto Fisico",
    "elegivel ao uso do ponto": "requisitos para Ponto Fisico",
    # Comodato/Equipamentos
    "comodato": "regime de comodato dos equipamentos",
    "equipamento": "regime de comodato dos equipamentos",
    "ton na mao": "regime de comodato dos equipamentos",
    "disponibilizados": "regime de comodato dos equipamentos",
    # TPV/TapTon
    "tpv": "taxa percentual do TPV",
    "0,2%": "taxa percentual do TPV",
    "tapton": "programa Indique TapTon",
    "link de indicacao": "programa Indique TapTon",
    # Permanencia
    "permanencia": "permanencia na carteira",
    "12 meses": "permanencia na carteira",
    "doze meses": "permanencia na carteira",
    # === INDICACOES ===
    "numero minimo de indicacoes": "quantidade minima de indicacoes para elegibilidade",
    "minimo de indicacoes": "quantidade minima de indicacoes para elegibilidade",
    "3 indicacoes": "quantidade minima de indicacoes para elegibilidade",
    "tres indicacoes": "quantidade minima de indicacoes para elegibilidade",
    "3 (tres) indicacoes": "quantidade minima de indicacoes para elegibilidade",
    "indicacoes validas": "quantidade minima de indicacoes para elegibilidade",
    "criterio principal": "quantidade minima de indicacoes para elegibilidade",
    "elegivel a participar": "quantidade minima de indicacoes para elegibilidade",
    # === ATUALIZACAO DE NIVEL ===
    "periodicidade de atualizacao": "data de atualizacao mensal do nivel",
    "atualizacao do nivel": "data de atualizacao mensal do nivel",
    "dia do mes": "data de atualizacao mensal do nivel",
    "dia 1": "data de atualizacao mensal do nivel",
    "todo dia 1": "data de atualizacao mensal do nivel",
    "primeiro dia": "data de atualizacao mensal do nivel",
    "1o do mes": "data de atualizacao mensal do nivel",
    "atualizado": "data de atualizacao mensal do nivel",
    # === PRAZO DE PAGAMENTO ===
    "prazo maximo para o pagamento": "prazo de pagamento das recompensas",
    "prazo maximo para pagamento": "prazo de pagamento das recompensas",
    "prazo para o pagamento": "prazo de pagamento das recompensas",
    "prazo para pagamento": "prazo de pagamento das recompensas",
    "prazo de pagamento": "prazo de pagamento das recompensas",
    "pagamento das recompensas": "prazo de pagamento das recompensas",
    "pagamento da recompensa": "prazo de pagamento das recompensas",
    "efetue o pagamento": "prazo de pagamento das recompensas",
    "valores serao pagos": "prazo de pagamento das recompensas",
    "dia 10": "prazo de pagamento das recompensas",
    "10o dia": "prazo de pagamento das recompensas",
    "decimo dia": "prazo de pagamento das recompensas",
    "mes subsequente": "prazo de pagamento das recompensas",
    # === RECOMPENSA FIXA ===
    "r$50": "recompensa fixa por indicacao",
    "r$ 50": "recompensa fixa por indicacao",
    "cinquenta reais": "recompensa fixa por indicacao",
    "50 reais": "recompensa fixa por indicacao",
    "recompensa fixa": "recompensa fixa por indicacao",
    "valor fixo": "recompensa fixa por indicacao",
    # === TON NA MAO ===
    # NOTA: "ton na mao" já definido em Comodato/Equipamentos (linha 234)
    "entrega direta": "programa Ton na Mao",
    "disponibilizacao direta": "programa Ton na Mao",
    # === SUSPENSAO/CANCELAMENTO ===
    "suspensao": "suspensao e cancelamento",
    "suspenso": "suspensao e cancelamento",
    "cancelamento": "suspensao e cancelamento",
    "cancelado": "suspensao e cancelamento",
    "desligamento": "suspensao e cancelamento",
    "desligado": "suspensao e cancelamento",
    "excluido": "suspensao e cancelamento",
    # === VALIDADE DAS INDICACOES ===
    "validade": "validade das indicacoes",
    "validas": "validade das indicacoes",
    # NOTA: "indicacoes validas" já definido em INDICACOES (linha 251)
    # === GENERICOS (baixa prioridade) ===
    "nivel minimo": "requisitos de nivel para beneficios",
    "especialista i": "requisitos de nivel para beneficios",
    "nivel do usuario": "sistema de niveis",
    "nivel": "sistema de niveis",
    "especialista": "sistema de niveis",
    "indicacao": "programa de indicacao",
    "elegibilidade": "criterios gerais de elegibilidade",
    "recompensa": "calculo de recompensas",
    "pagamento": "prazo de pagamento das recompensas",
    "renda extra": "definicao do programa Renda Extra",
    "renda ton": "definicao do programa Renda Ton",
    "funcao": "objetivo do programa",
    "objetivo": "objetivo do programa",
    "principal": "objetivo do programa",
}


# =============================================================================
# DIFFICULTY DISTRIBUTION
# =============================================================================

# Distribuicao padrao de dificuldades para perguntas 2-10
# Total: 1 easy (P1), 2 easy, 5 medium, 2 hard = 10 perguntas
DEFAULT_DIFFICULTY_DISTRIBUTION = [
    "easy",    # P2
    "medium",  # P3
    "medium",  # P4
    "medium",  # P5
    "hard",    # P6
    "medium",  # P7
    "medium",  # P8
    "hard",    # P9
    "easy",    # P10
]
