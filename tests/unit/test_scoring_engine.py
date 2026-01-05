# =============================================================================
# TESTES - Quiz Scoring Engine
# =============================================================================
# Testes unitarios para motor de pontuacao e ranking
# =============================================================================

import pytest


class TestQuizScoringEnginePoints:
    """Testes para calculo de pontos."""

    def test_points_by_difficulty(self):
        """Verifica pontos por dificuldade."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizDifficulty

        engine = QuizScoringEngine()

        assert engine.get_points_for_difficulty(QuizDifficulty.EASY) == 1
        assert engine.get_points_for_difficulty(QuizDifficulty.MEDIUM) == 2
        assert engine.get_points_for_difficulty(QuizDifficulty.HARD) == 3

    def test_points_default_fallback(self):
        """Verifica fallback para dificuldade desconhecida."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # Passa valor invalido, deve retornar default (2)
        result = engine.get_points_for_difficulty("unknown")
        assert result == 2


class TestQuizScoringEngineRank:
    """Testes para calculo de ranking."""

    def test_rank_embaixador(self):
        """Verifica ranking Embaixador (96-100%)."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, title, message = engine.calculate_rank(100.0)
        assert rank == QuizRank.EMBAIXADOR
        assert "Embaixador" in title

        rank, title, message = engine.calculate_rank(96.0)
        assert rank == QuizRank.EMBAIXADOR

    def test_rank_especialista_iii(self):
        """Verifica ranking Especialista III (86-95%)."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, title, message = engine.calculate_rank(95.0)
        assert rank == QuizRank.ESPECIALISTA_III
        assert "Nível III" in title

        rank, title, message = engine.calculate_rank(86.0)
        assert rank == QuizRank.ESPECIALISTA_III

    def test_rank_especialista_ii(self):
        """Verifica ranking Especialista II (71-85%)."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, title, message = engine.calculate_rank(85.0)
        assert rank == QuizRank.ESPECIALISTA_II
        assert "Nível II" in title

        rank, title, message = engine.calculate_rank(71.0)
        assert rank == QuizRank.ESPECIALISTA_II

    def test_rank_especialista_i(self):
        """Verifica ranking Especialista I (51-70%)."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, title, message = engine.calculate_rank(70.0)
        assert rank == QuizRank.ESPECIALISTA_I
        assert "Nível I" in title

        rank, title, message = engine.calculate_rank(51.0)
        assert rank == QuizRank.ESPECIALISTA_I

    def test_rank_iniciante(self):
        """Verifica ranking Iniciante (<51%)."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, title, message = engine.calculate_rank(50.0)
        assert rank == QuizRank.INICIANTE
        assert "Iniciante" in title

        rank, title, message = engine.calculate_rank(0.0)
        assert rank == QuizRank.INICIANTE


class TestQuizScoringEngineCalculateScore:
    """Testes para calculo de pontuacao completa."""

    def test_calculate_score_all_correct(self, sample_quiz_questions):
        """Verifica pontuacao com todas corretas."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # Respostas corretas para cada pergunta (baseado na fixture)
        # A fixture cria perguntas onde i % 4 determina a correta
        answers = [0, 1, 2, 3, 0]  # indices corretos

        result = engine.calculate_score(sample_quiz_questions, answers)

        assert result["correct_answers"] == 5
        assert result["percentage"] == 100.0
        assert result["score"] == result["max_score"]

    def test_calculate_score_all_wrong(self, sample_quiz_questions):
        """Verifica pontuacao com todas erradas."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # Respostas erradas (escolhe sempre 3 que nao e o padrao)
        answers = [3, 3, 3, 0, 3]

        result = engine.calculate_score(sample_quiz_questions, answers)

        assert result["correct_answers"] == 0
        assert result["score"] == 0

    def test_calculate_score_partial(self, sample_quiz_questions):
        """Verifica pontuacao parcial."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # Metade certas
        answers = [0, 1, 0, 0, 0]  # 3 certas (0, 1, 4)

        result = engine.calculate_score(sample_quiz_questions, answers)

        assert result["correct_answers"] == 3
        assert 0 < result["percentage"] < 100

    def test_calculate_score_breakdown(self, sample_quiz_questions):
        """Verifica breakdown por dificuldade."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()
        answers = [0, 1, 2, 3, 0]

        result = engine.calculate_score(sample_quiz_questions, answers)

        assert "breakdown" in result
        assert "facil" in result["breakdown"]
        assert "media" in result["breakdown"]
        assert "dificil" in result["breakdown"]

    def test_calculate_score_mismatched_lengths(self, sample_quiz_questions):
        """Verifica erro com numero diferente de respostas."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        with pytest.raises(ValueError) as exc_info:
            engine.calculate_score(sample_quiz_questions, [0, 1, 2])  # 3 respostas, 5 perguntas

        assert "diferente do número de perguntas" in str(exc_info.value)


class TestQuizScoringEngineEvaluateAnswer:
    """Testes para avaliacao de resposta individual."""

    def test_evaluate_correct_answer(self, sample_pergunta_quiz):
        """Verifica avaliacao de resposta correta."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # A alternativa B e a correta
        result = engine.evaluate_answer(sample_pergunta_quiz, 1)  # index 1 = B

        assert result["is_correct"] is True
        assert result["points_earned"] == sample_pergunta_quiz.points
        assert result["correct_index"] == 1

    def test_evaluate_wrong_answer(self, sample_pergunta_quiz):
        """Verifica avaliacao de resposta errada."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        # A alternativa A e incorreta
        result = engine.evaluate_answer(sample_pergunta_quiz, 0)  # index 0 = A

        assert result["is_correct"] is False
        assert result["points_earned"] == 0
        assert result["correct_index"] == 1  # B e a correta

    def test_evaluate_answer_includes_feedback(self, sample_pergunta_quiz):
        """Verifica que feedback esta incluso."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        result = engine.evaluate_answer(sample_pergunta_quiz, 0)

        assert "feedback" in result
        assert "explanation" in result
        assert len(result["explanation"]) > 0


class TestQuizScoringEngineEdgeCases:
    """Testes de casos extremos."""

    def test_empty_questions_list(self):
        """Verifica com lista vazia de perguntas."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        result = engine.calculate_score([], [])

        assert result["total_questions"] == 0
        assert result["percentage"] == 0

    def test_single_question(self, sample_pergunta_quiz):
        """Verifica com uma unica pergunta."""
        from quiz.engine.scoring_engine import QuizScoringEngine

        engine = QuizScoringEngine()

        result = engine.calculate_score([sample_pergunta_quiz], [1])  # B e correta

        assert result["total_questions"] == 1
        assert result["correct_answers"] == 1
        assert result["percentage"] == 100.0

    def test_rank_boundary_96(self):
        """Verifica limite exato de 96%."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, _, _ = engine.calculate_rank(96.0)
        assert rank == QuizRank.EMBAIXADOR

        rank, _, _ = engine.calculate_rank(95.9)
        assert rank == QuizRank.ESPECIALISTA_III

    def test_rank_boundary_51(self):
        """Verifica limite exato de 51%."""
        from quiz.engine.scoring_engine import QuizScoringEngine
        from quiz.models.enums import QuizRank

        engine = QuizScoringEngine()

        rank, _, _ = engine.calculate_rank(51.0)
        assert rank == QuizRank.ESPECIALISTA_I

        rank, _, _ = engine.calculate_rank(50.9)
        assert rank == QuizRank.INICIANTE
