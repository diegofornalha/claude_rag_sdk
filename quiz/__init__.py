"""Quiz Module - Sistema inteligente de avaliação com RAG.

Arquitetura:
- models/: Enums, Schemas Pydantic, QuizState
- engine/: QuizEngine, DedupEngine, ScoringEngine
- llm/: LLMClientFactory
- storage/: QuizStore (AgentFS integration)
- prompts/: Templates de prompts
- router.py: FastAPI endpoints
"""

from .engine import QuizEngine, QuizScoringEngine, TopicDeduplicationEngine
from .llm import LLMClientFactory
from .models import QuizDifficulty, QuizOption, QuizQuestion, QuizRank, QuizState
from .storage import QuizStore

__all__ = [
    # Models
    "QuizDifficulty",
    "QuizRank",
    "QuizOption",
    "QuizQuestion",
    "QuizState",
    # Engines
    "QuizEngine",
    "TopicDeduplicationEngine",
    "QuizScoringEngine",
    # LLM
    "LLMClientFactory",
    # Storage
    "QuizStore",
]
