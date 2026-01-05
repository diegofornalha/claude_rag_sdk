# =============================================================================
# TESTES - Quiz Schemas Module
# =============================================================================
# Testes unitarios para schemas Pydantic do quiz
# =============================================================================

import pytest
from pydantic import ValidationError


class TestAlternativa:
    """Testes para Alternativa dataclass."""

    def test_create_alternativa_valid(self):
        """Verifica criacao de alternativa valida."""
        from quiz.models.schemas_supabase import Alternativa

        alt = Alternativa(
            texto="Texto da alternativa",
            correta=True,
            explicacao="Explicacao detalhada",
        )

        assert alt.texto == "Texto da alternativa"
        assert alt.correta is True
        assert alt.explicacao == "Explicacao detalhada"

    def test_alternativa_incorrect(self):
        """Verifica alternativa incorreta."""
        from quiz.models.schemas_supabase import Alternativa

        alt = Alternativa(
            texto="Opcao errada",
            correta=False,
            explicacao="Incorreto porque...",
        )

        assert alt.correta is False

    def test_alternativa_missing_field(self):
        """Verifica erro com campo faltando."""
        from quiz.models.schemas_supabase import Alternativa

        with pytest.raises(ValidationError):
            Alternativa(texto="Texto", correta=True)  # falta explicacao


class TestPerguntaQuiz:
    """Testes para PerguntaQuiz dataclass."""

    def test_create_pergunta_valid(self, sample_pergunta_quiz):
        """Verifica criacao de pergunta valida."""
        q = sample_pergunta_quiz

        assert q.numero == 1
        assert "idade mínima" in q.texto.lower()
        assert len(q.alternativas) == 4
        assert q.dificuldade == "facil"
        assert q.topico == "Elegibilidade"

    def test_alternativa_correta_property(self, sample_pergunta_quiz):
        """Verifica property alternativa_correta."""
        q = sample_pergunta_quiz

        assert q.alternativa_correta == "B"

    def test_points_property(self):
        """Verifica property points por dificuldade."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        base_alts = {
            "A": Alternativa(texto="A", correta=True, explicacao="Correto"),
            "B": Alternativa(texto="B", correta=False, explicacao="Errado"),
            "C": Alternativa(texto="C", correta=False, explicacao="Errado"),
            "D": Alternativa(texto="D", correta=False, explicacao="Errado"),
        }

        # Facil = 1 ponto
        q_facil = PerguntaQuiz(
            numero=1,
            texto="Pergunta facil sobre o programa",
            alternativas=base_alts,
            dificuldade="facil",
            topico="Geral",
            regulamento_ref="Ref",
        )
        assert q_facil.points == 1

        # Media = 2 pontos
        q_media = PerguntaQuiz(
            numero=2,
            texto="Pergunta media sobre o programa",
            alternativas=base_alts,
            dificuldade="media",
            topico="Geral",
            regulamento_ref="Ref",
        )
        assert q_media.points == 2

        # Dificil = 3 pontos
        q_dificil = PerguntaQuiz(
            numero=3,
            texto="Pergunta dificil sobre o programa",
            alternativas=base_alts,
            dificuldade="dificil",
            topico="Geral",
            regulamento_ref="Ref",
        )
        assert q_dificil.points == 3

    def test_validar_alternativas_wrong_letters(self):
        """Verifica erro com letras incorretas."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        with pytest.raises(ValidationError) as exc_info:
            PerguntaQuiz(
                numero=1,
                texto="Pergunta sobre o programa Renda Extra",
                alternativas={
                    "A": Alternativa(texto="A", correta=True, explicacao="X"),
                    "B": Alternativa(texto="B", correta=False, explicacao="X"),
                    "X": Alternativa(texto="X", correta=False, explicacao="X"),  # Errado
                    "D": Alternativa(texto="D", correta=False, explicacao="X"),
                },
                dificuldade="facil",
                topico="Geral",
                regulamento_ref="Ref",
            )

        assert "Deve ter alternativas A, B, C, D" in str(exc_info.value)

    def test_validar_alternativas_multiple_correct(self):
        """Verifica erro com mais de uma correta."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        with pytest.raises(ValidationError) as exc_info:
            PerguntaQuiz(
                numero=1,
                texto="Pergunta sobre o programa Renda Extra",
                alternativas={
                    "A": Alternativa(texto="A", correta=True, explicacao="X"),
                    "B": Alternativa(texto="B", correta=True, explicacao="X"),  # 2 corretas
                    "C": Alternativa(texto="C", correta=False, explicacao="X"),
                    "D": Alternativa(texto="D", correta=False, explicacao="X"),
                },
                dificuldade="facil",
                topico="Geral",
                regulamento_ref="Ref",
            )

        assert "exatamente 1 alternativa correta" in str(exc_info.value)

    def test_validar_alternativas_no_correct(self):
        """Verifica erro sem alternativa correta."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        with pytest.raises(ValidationError) as exc_info:
            PerguntaQuiz(
                numero=1,
                texto="Pergunta sobre o programa Renda Extra",
                alternativas={
                    "A": Alternativa(texto="A", correta=False, explicacao="X"),
                    "B": Alternativa(texto="B", correta=False, explicacao="X"),
                    "C": Alternativa(texto="C", correta=False, explicacao="X"),
                    "D": Alternativa(texto="D", correta=False, explicacao="X"),
                },
                dificuldade="facil",
                topico="Geral",
                regulamento_ref="Ref",
            )

        assert "exatamente 1 alternativa correta" in str(exc_info.value)

    def test_validar_dificuldade_invalid(self):
        """Verifica erro com dificuldade invalida."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        base_alts = {
            "A": Alternativa(texto="A", correta=True, explicacao="X"),
            "B": Alternativa(texto="B", correta=False, explicacao="X"),
            "C": Alternativa(texto="C", correta=False, explicacao="X"),
            "D": Alternativa(texto="D", correta=False, explicacao="X"),
        }

        with pytest.raises(ValidationError) as exc_info:
            PerguntaQuiz(
                numero=1,
                texto="Pergunta sobre o programa Renda Extra",
                alternativas=base_alts,
                dificuldade="extrema",  # Invalida
                topico="Geral",
                regulamento_ref="Ref",
            )

        assert "dificuldade deve ser" in str(exc_info.value)

    def test_texto_min_length(self):
        """Verifica tamanho minimo do texto."""
        from quiz.models.schemas_supabase import Alternativa, PerguntaQuiz

        base_alts = {
            "A": Alternativa(texto="A", correta=True, explicacao="X"),
            "B": Alternativa(texto="B", correta=False, explicacao="X"),
            "C": Alternativa(texto="C", correta=False, explicacao="X"),
            "D": Alternativa(texto="D", correta=False, explicacao="X"),
        }

        with pytest.raises(ValidationError):
            PerguntaQuiz(
                numero=1,
                texto="Curto",  # Menos de 10 chars
                alternativas=base_alts,
                dificuldade="facil",
                topico="Geral",
                regulamento_ref="Ref",
            )


class TestQuizQuestionLegacy:
    """Testes para QuizQuestionLegacy (compatibilidade)."""

    def test_to_supabase_conversion(self):
        """Verifica conversao para formato Supabase."""
        from quiz.models.schemas_supabase import QuizQuestionLegacy

        legacy = QuizQuestionLegacy(
            id=1,
            question="Qual a idade minima?",
            options=[
                {"text": "16 anos"},
                {"text": "18 anos"},
                {"text": "21 anos"},
                {"text": "25 anos"},
            ],
            correct_index=1,
            difficulty="easy",
            points=1,
            explanation="A idade minima e 18 anos",
            wrong_feedback={0: "Errado", 2: "Errado", 3: "Errado"},
            learning_tip="Memorize: 18 anos",
            source_reference="Item 2.1",
        )

        supabase = legacy.to_supabase()

        assert supabase.numero == 1
        assert "idade" in supabase.texto.lower()
        assert supabase.alternativa_correta == "B"
        assert supabase.dificuldade == "facil"
        assert supabase.alternativas["B"].correta is True

    def test_extrair_topico(self):
        """Verifica extracao automatica de topico."""
        from quiz.models.schemas_supabase import QuizQuestionLegacy

        # Elegibilidade
        q1 = QuizQuestionLegacy(
            id=1,
            question="Qual idade minima para elegibilidade?",
            options=[{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}],
            correct_index=0,
            difficulty="easy",
            points=1,
            explanation="X",
            wrong_feedback={},
        )
        assert q1._extrair_topico() == "Elegibilidade"

        # Pagamentos
        q2 = QuizQuestionLegacy(
            id=2,
            question="Qual o prazo de pagamento?",
            options=[{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}],
            correct_index=0,
            difficulty="easy",
            points=1,
            explanation="X",
            wrong_feedback={},
        )
        assert q2._extrair_topico() == "Pagamentos"

        # Geral (fallback)
        q3 = QuizQuestionLegacy(
            id=3,
            question="Pergunta generica sobre o programa",
            options=[{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}],
            correct_index=0,
            difficulty="easy",
            points=1,
            explanation="X",
            wrong_feedback={},
        )
        assert q3._extrair_topico() == "Geral"


class TestQuizDifficulty:
    """Testes para QuizDifficulty enum."""

    def test_difficulty_values(self):
        """Verifica valores do enum."""
        from quiz.models.enums import QuizDifficulty

        assert QuizDifficulty.EASY.value == "easy"
        assert QuizDifficulty.MEDIUM.value == "medium"
        assert QuizDifficulty.HARD.value == "hard"


class TestQuizRank:
    """Testes para QuizRank enum."""

    def test_rank_values(self):
        """Verifica valores do enum."""
        from quiz.models.enums import QuizRank

        assert QuizRank.EMBAIXADOR.value == "embaixador"
        assert QuizRank.ESPECIALISTA_III.value == "especialista_iii"
        assert QuizRank.ESPECIALISTA_II.value == "especialista_ii"
        assert QuizRank.ESPECIALISTA_I.value == "especialista_i"
        assert QuizRank.INICIANTE.value == "iniciante"


class TestQuizState:
    """Testes para QuizState dataclass."""

    def test_create_state(self):
        """Verifica criacao de estado."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="abc-123")

        assert state.quiz_id == "abc-123"
        assert state.questions == {}
        assert state.generated_count == 0
        assert state.complete is False
        assert state.error is None

    def test_add_question(self, sample_pergunta_quiz):
        """Verifica adicao de pergunta."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="test-123")
        state.add_question(1, sample_pergunta_quiz)

        assert 1 in state.questions
        assert state.generated_count == 1
        assert state.max_score == sample_pergunta_quiz.points

    def test_is_ready(self, sample_pergunta_quiz):
        """Verifica check de pergunta pronta."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="test-123")

        assert state.is_ready(1) is False

        state.add_question(1, sample_pergunta_quiz)

        assert state.is_ready(1) is True
        assert state.is_ready(2) is False

    def test_mark_complete(self, sample_pergunta_quiz):
        """Verifica marcacao como completo."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="test-123")
        state.add_question(1, sample_pergunta_quiz)
        state.mark_complete()

        assert state.complete is True
        assert state.max_score == sample_pergunta_quiz.points

    def test_set_error(self):
        """Verifica definicao de erro."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="test-123")
        state.set_error("Erro de teste")

        assert state.error == "Erro de teste"

    def test_add_topic(self):
        """Verifica adicao de topico."""
        from quiz.models.state import QuizState

        state = QuizState(quiz_id="test-123")
        state.add_topic("Elegibilidade")
        state.add_topic("Pagamentos")
        state.add_topic("Elegibilidade")  # Duplicata

        assert len(state.previous_topics) == 2
        assert "Elegibilidade" in state.previous_topics
        assert "Pagamentos" in state.previous_topics

    def test_to_dict(self, sample_quiz_state):
        """Verifica conversao para dict."""
        data = sample_quiz_state.to_dict()

        assert data["quiz_id"] == "test-123"
        assert "questions" in data
        assert "generated_count" in data
        assert data["context"] == "Contexto RAG de teste"

    def test_from_dict(self, sample_quiz_state):
        """Verifica criacao a partir de dict."""
        from quiz.models.state import QuizState

        data = sample_quiz_state.to_dict()
        restored = QuizState.from_dict(data)

        assert restored.quiz_id == sample_quiz_state.quiz_id
        assert len(restored.questions) == len(sample_quiz_state.questions)
        assert restored.context == sample_quiz_state.context


class TestQuizOption:
    """Testes para QuizOption schema."""

    def test_create_option(self):
        """Verifica criacao de opcao."""
        from quiz.models.schemas import QuizOption

        opt = QuizOption(label="A", text="Texto da opcao")

        assert opt.label == "A"
        assert opt.text == "Texto da opcao"


class TestGenerateQuizRequest:
    """Testes para GenerateQuizRequest."""

    def test_default_values(self):
        """Verifica valores padrao."""
        from quiz.models.schemas import GenerateQuizRequest

        req = GenerateQuizRequest()

        assert req.num_questions == 10
        assert req.focus_topics == []
        assert req.difficulty_distribution == {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    def test_custom_values(self):
        """Verifica valores customizados."""
        from quiz.models.schemas import GenerateQuizRequest

        req = GenerateQuizRequest(
            num_questions=15,
            focus_topics=["Elegibilidade", "Pagamentos"],
            difficulty_distribution={"easy": 0.5, "medium": 0.3, "hard": 0.2},
        )

        assert req.num_questions == 15
        assert len(req.focus_topics) == 2

    def test_num_questions_validation(self):
        """Verifica validacao de num_questions."""
        from quiz.models.schemas import GenerateQuizRequest

        # Muito baixo
        with pytest.raises(ValidationError):
            GenerateQuizRequest(num_questions=3)

        # Muito alto
        with pytest.raises(ValidationError):
            GenerateQuizRequest(num_questions=25)


class TestStartQuizResponse:
    """Testes para StartQuizResponse."""

    def test_create_response(self, sample_pergunta_quiz):
        """Verifica criacao de response."""
        from quiz.models.schemas import StartQuizResponse

        resp = StartQuizResponse(
            quiz_id="abc-123",
            total_questions=10,
            first_question=sample_pergunta_quiz,
        )

        assert resp.quiz_id == "abc-123"
        assert resp.total_questions == 10
        assert resp.first_question.numero == 1
