"""Quiz Scoring Engine - Motor de pontuacao e ranking."""

from ..models.enums import QuizDifficulty, QuizRank
from ..models.schemas import QuizQuestion


class QuizScoringEngine:
    """Motor de pontuação e ranking para quizzes.

    Calcula pontuação baseada em dificuldade e determina ranking
    baseado na trilha de benefícios do Renda Extra Ton.

    Pontuação por dificuldade:
        - EASY: 1 ponto
        - MEDIUM: 2 pontos
        - HARD: 3 pontos

    Faixas de ranking:
        - 96-100%: Embaixador (Domínio total)
        - 86-95%: Especialista III (Conhecimento profundo)
        - 71-85%: Especialista II (Boa compreensão)
        - 51-70%: Especialista I (Base sólida)
        - <51%: Iniciante (Precisa revisar)

    Example:
        >>> engine = QuizScoringEngine()
        >>> rank, title, message = engine.calculate_rank(85.0)
        >>> print(title)  # "⭐ Especialista Nível II"
    """

    # Pontos por dificuldade
    POINTS = {
        QuizDifficulty.EASY: 1,
        QuizDifficulty.MEDIUM: 2,
        QuizDifficulty.HARD: 3,
    }

    # Faixas de ranking (threshold, rank, title, message)
    RANK_THRESHOLDS = [
        (
            96,
            QuizRank.EMBAIXADOR,
            "🏆 Embaixador do Renda Extra Ton",
            "Domínio total! Você possui conhecimento excepcional das regras do programa "
            "e está pronto para ser um verdadeiro embaixador, ajudando outros parceiros "
            "a maximizarem seus ganhos!",
        ),
        (
            86,
            QuizRank.ESPECIALISTA_III,
            "🌟 Especialista Nível III",
            "Excelente! Você possui conhecimento profundo do programa. Com esse domínio, "
            "você está muito próximo de alcançar o nível de Embaixador. Continue "
            "aprimorando os detalhes!",
        ),
        (
            71,
            QuizRank.ESPECIALISTA_II,
            "⭐ Especialista Nível II",
            "Muito bem! Você compreende bem as regras do Renda Extra Ton. Continue "
            "estudando as nuances e casos especiais para alcançar o Nível III!",
        ),
        (
            51,
            QuizRank.ESPECIALISTA_I,
            "📚 Especialista Nível I",
            "Bom trabalho! Você tem uma base sólida sobre o programa. Aprofunde seu "
            "conhecimento sobre as regras específicas e validações para evoluir para "
            "Especialista II!",
        ),
        (
            0,
            QuizRank.INICIANTE,
            "🌱 Iniciante no Programa",
            "Você está começando sua jornada! O conhecimento vem com estudo dedicado. "
            "Revise o regulamento com atenção, focando nos conceitos fundamentais e "
            "regras principais antes de avançar.",
        ),
    ]

    def get_points_for_difficulty(self, difficulty: QuizDifficulty) -> int:
        """Retorna pontos para uma dificuldade.

        Args:
            difficulty: Nível de dificuldade

        Returns:
            Pontos correspondentes (1, 2 ou 3)
        """
        return self.POINTS.get(difficulty, 2)  # Default: medium

    def calculate_rank(self, percentage: float) -> tuple[QuizRank, str, str]:
        """Calcula o ranking baseado no percentual de aproveitamento.

        Args:
            percentage: Percentual de acerto (0-100)

        Returns:
            Tuple de (rank, title, message)
        """
        for threshold, rank, title, message in self.RANK_THRESHOLDS:
            if percentage >= threshold:
                return rank, title, message

        # Fallback (nunca deve chegar aqui)
        return self.RANK_THRESHOLDS[-1][1:4]

    def calculate_score(
        self, questions: list[QuizQuestion], answers: list[int]
    ) -> dict:
        """Calcula pontuação completa do quiz.

        Args:
            questions: Lista de perguntas do quiz
            answers: Lista de índices selecionados pelo usuário

        Returns:
            Dict com score, percentage, rank e breakdown por dificuldade
        """
        if len(questions) != len(answers):
            raise ValueError(
                f"Número de respostas ({len(answers)}) diferente do número de perguntas ({len(questions)})"
            )

        total_score = 0
        max_score = 0
        correct_count = 0
        breakdown = {
            "easy": {"correct": 0, "total": 0},
            "medium": {"correct": 0, "total": 0},
            "hard": {"correct": 0, "total": 0},
        }

        for question, answer in zip(questions, answers, strict=False):
            diff_key = question.difficulty.value
            breakdown[diff_key]["total"] += 1
            max_score += question.points

            if answer == question.correct_index:
                total_score += question.points
                correct_count += 1
                breakdown[diff_key]["correct"] += 1

        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        rank, rank_title, rank_message = self.calculate_rank(percentage)

        return {
            "total_questions": len(questions),
            "correct_answers": correct_count,
            "score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1),
            "rank": rank,
            "rank_title": rank_title,
            "rank_message": rank_message,
            "breakdown": breakdown,
        }

    def evaluate_answer(
        self, question: QuizQuestion, selected_index: int
    ) -> dict:
        """Avalia uma resposta individual.

        Args:
            question: Pergunta respondida
            selected_index: Índice selecionado (0-3)

        Returns:
            Dict com is_correct, points_earned, feedback
        """
        is_correct = selected_index == question.correct_index

        if is_correct:
            feedback = question.explanation
            points_earned = question.points
        else:
            # Buscar feedback específico para alternativa errada
            feedback = question.wrong_feedback.get(
                selected_index,
                f"A resposta correta era a alternativa {question.options[question.correct_index].label}.",
            )
            points_earned = 0

        return {
            "is_correct": is_correct,
            "points_earned": points_earned,
            "correct_index": question.correct_index,
            "feedback": feedback,
            "explanation": question.explanation,
            "learning_tip": question.learning_tip,
        }
