"""Agents module - Abstrações de alto nível usando Claude Agent SDK."""

from agents.chat_agent import ChatAgent, ChatRequest, StreamChunk, create_chat_agent
from agents.metrics import (
    PRICING,
    MetricsManager,
    RequestMetrics,
    SessionMetrics,
    calculate_cost,
    estimate_tokens,
    get_metrics_manager,
)
from agents.session_cache import (
    AgentFSPool,
    CacheEntry,
    CacheStats,
    LRUCache,
    SessionCache,
    get_agentfs_pool,
    get_session_cache,
)
from agents.title_generator import (
    generate_conversation_title,
    get_smart_title,
    should_generate_title,
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
