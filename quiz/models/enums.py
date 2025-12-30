"""Quiz Enums - Dificuldade e Rankings."""

from enum import Enum


class QuizDifficulty(str, Enum):
    """Niveis de dificuldade das questoes."""

    EASY = "easy"  # 30% - Conceitos basicos
    MEDIUM = "medium"  # 50% - Regras e validacoes
    HARD = "hard"  # 20% - Nuances e detalhes complexos


class QuizRank(str, Enum):
    """Rankings baseados na trilha de beneficios Renda Extra Ton."""

    EMBAIXADOR = "embaixador"  # 100% aproveitamento
    ESPECIALISTA_III = "especialista_iii"  # 90-99%
    ESPECIALISTA_II = "especialista_ii"  # 80-89%
    ESPECIALISTA_I = "especialista_i"  # 60-79%
    INICIANTE = "iniciante"  # <60%
