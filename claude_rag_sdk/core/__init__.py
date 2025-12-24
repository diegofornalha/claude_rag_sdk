# Core modules for RAG Agent
from .logger import logger, set_conversation_id, set_request_id, get_conversation_id, get_request_id
from .rbac import User, Role, RBACFilter, set_current_user, get_current_user, get_rbac_filter
from .circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerError, circuit_breaker, get_or_create_circuit_breaker
from .cache import LRUCache, EmbeddingCache, ResponseCache, get_embedding_cache, get_response_cache
from .hybrid_search import HybridSearch, BM25, SearchResult
from .reranker import CrossEncoderReranker, LightweightReranker, create_reranker, RerankResult
from .security import CORSMiddleware, CORSConfig, SecurityHeaders, get_security_headers
from .rate_limiter import SlidingWindowRateLimiter, TokenBucketRateLimiter, RateLimitResult, get_rate_limiter, check_rate_limit
from .prompt_guard import PromptGuard, ThreatLevel, ScanResult, get_prompt_guard, scan_prompt, is_safe_prompt
from .auth import APIKeyManager, APIKey, AuthScope, AuthResult, get_key_manager, authenticate, extract_api_key

__all__ = [
    # Logger
    "logger",
    "set_conversation_id",
    "set_request_id",
    "get_conversation_id",
    "get_request_id",
    # RBAC
    "User",
    "Role",
    "RBACFilter",
    "set_current_user",
    "get_current_user",
    "get_rbac_filter",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    "circuit_breaker",
    "get_or_create_circuit_breaker",
    # Cache
    "LRUCache",
    "EmbeddingCache",
    "ResponseCache",
    "get_embedding_cache",
    "get_response_cache",
    # Hybrid Search
    "HybridSearch",
    "BM25",
    "SearchResult",
    # Reranker
    "CrossEncoderReranker",
    "LightweightReranker",
    "create_reranker",
    "RerankResult",
    # Security
    "CORSMiddleware",
    "CORSConfig",
    "SecurityHeaders",
    "get_security_headers",
    # Rate Limiter
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "RateLimitResult",
    "get_rate_limiter",
    "check_rate_limit",
    # Prompt Guard
    "PromptGuard",
    "ThreatLevel",
    "ScanResult",
    "get_prompt_guard",
    "scan_prompt",
    "is_safe_prompt",
    # Auth
    "APIKeyManager",
    "APIKey",
    "AuthScope",
    "AuthResult",
    "get_key_manager",
    "authenticate",
    "extract_api_key",
]
