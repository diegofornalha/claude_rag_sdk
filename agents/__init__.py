"""Agents module - Abstrações de alto nível usando Claude Agent SDK."""

from agents.chat_agent import ChatAgent, ChatRequest, StreamChunk, create_chat_agent
from agents.metrics import (
    MetricsManager,
    RequestMetrics,
    SessionMetrics,
    get_metrics_manager,
    estimate_tokens,
    calculate_cost,
    PRICING,
)
from agents.session_cache import (
    SessionCache,
    AgentFSPool,
    LRUCache,
    CacheEntry,
    CacheStats,
    get_session_cache,
    get_agentfs_pool,
)
from agents.title_generator import (
    generate_conversation_title,
    should_generate_title,
    get_smart_title,
)

__all__ = [
    # Chat Agent
    "ChatAgent",
    "ChatRequest",
    "StreamChunk",
    "create_chat_agent",
    # Metrics
    "MetricsManager",
    "RequestMetrics",
    "SessionMetrics",
    "get_metrics_manager",
    "estimate_tokens",
    "calculate_cost",
    "PRICING",
    # Session Cache
    "SessionCache",
    "AgentFSPool",
    "LRUCache",
    "CacheEntry",
    "CacheStats",
    "get_session_cache",
    "get_agentfs_pool",
    # Title Generator
    "generate_conversation_title",
    "should_generate_title",
    "get_smart_title",
]
