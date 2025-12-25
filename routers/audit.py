"""Audit endpoints."""
from fastapi import APIRouter
from typing import Optional
import time

from app_state import AGENTFS_DIR, get_current_session_id
from utils.debug_parser import parse_debug_file

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("/tools")
async def get_audit_tools(limit: int = 100, session_id: Optional[str] = None):
    """Get tool call history."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session", "records": []}

    session_db = AGENTFS_DIR / f"{sid}.db"
    if not session_db.exists():
        return {"error": "No active session", "session_id": None, "records": [], "count": 0}

    session_afs = None
    try:
        session_afs = await AgentFS.open(AgentFSOptions(id=sid))
        stats = await session_afs.tools.get_stats()

        since_timestamp = int(time.time()) - 86400
        recent = await session_afs.tools.get_recent(since=since_timestamp, limit=limit)

        recent_dicts = []
        for call in recent[:limit]:
            recent_dicts.append({
                "id": call.id,
                "name": call.name,
                "started_at": call.started_at,
                "completed_at": call.completed_at,
                "duration_ms": call.duration_ms,
                "status": call.status,
                "parameters": call.parameters,
                "result": call.result,
                "error": call.error,
            })

        return {
            "session_id": sid,
            "stats": [{"name": s.name, "calls": s.total_calls, "avg_ms": s.avg_duration_ms} for s in stats],
            "recent": recent_dicts,
            "count": len(recent_dicts),
        }
    except Exception as e:
        print(f"[AUDIT] Error getting tools for {sid}: {e}")
        return {"session_id": sid, "records": [], "count": 0, "error": "Failed to get tool records"}
    finally:
        if session_afs:
            await session_afs.close()


@router.get("/stats")
async def get_audit_stats(session_id: Optional[str] = None):
    """Get audit statistics."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    sid = session_id or get_current_session_id()
    if not sid:
        return {"error": "No active session"}

    session_db = AGENTFS_DIR / f"{sid}.db"
    if not session_db.exists():
        return {"error": "No active session", "session_id": None}

    session_afs = None
    try:
        session_afs = await AgentFS.open(AgentFSOptions(id=sid))
        stats = await session_afs.tools.get_stats()

        return {
            "session_id": sid,
            "total_calls": sum(s.total_calls for s in stats),
            "by_tool": {s.name: s.total_calls for s in stats},
            "avg_duration_ms": sum(s.avg_duration_ms for s in stats) / len(stats) if stats else 0,
        }
    except Exception as e:
        print(f"[AUDIT] Error getting stats for {sid}: {e}")
        return {"session_id": sid, "total_calls": 0, "by_tool": {}, "error": "Failed to get statistics"}
    finally:
        if session_afs:
            await session_afs.close()


@router.get("/debug/{session_id}")
async def get_debug_log(session_id: str):
    """Retorna logs de debug do CLI para uma sessão."""
    entries = parse_debug_file(session_id)

    if not entries:
        return {
            "session_id": session_id,
            "found": False,
            "entries": [],
            "count": 0,
            "summary": {"total_events": 0, "tool_events": 0, "file_writes": 0, "streams": 0}
        }

    return {
        "session_id": session_id,
        "found": True,
        "entries": [
            {
                "timestamp": e.timestamp,
                "timestamp_ms": e.timestamp_ms,
                "level": e.level,
                "message": e.message,
                "tool_name": e.tool_name,
                "event_type": e.event_type
            }
            for e in entries
        ],
        "count": len(entries),
        "summary": {
            "total_events": len(entries),
            "tool_events": len([e for e in entries if e.tool_name]),
            "file_writes": len([e for e in entries if e.event_type == 'file_write']),
            "streams": len([e for e in entries if e.event_type == 'stream'])
        }
    }


@router.get("/tools/enriched")
async def get_enriched_tools(session_id: str, limit: int = 50):
    """Retorna tool calls enriquecidas com dados de debug do CLI."""
    from agentfs_sdk import AgentFS, AgentFSOptions

    tools_data = []
    session_afs = None
    try:
        session_afs = await AgentFS.open(AgentFSOptions(id=session_id))
        recent = await session_afs.tools.get_recent(limit=limit)

        tools_data = [
            {
                "id": call.id,
                "name": call.name,
                "started_at": call.started_at,
                "completed_at": call.completed_at,
                "duration_ms": call.duration_ms,
                "status": call.status,
                "parameters": call.parameters,
                "result": call.result
            }
            for call in recent
        ]
    except Exception as e:
        print(f"[WARN] Could not get AgentFS tools: {e}")
    finally:
        if session_afs:
            await session_afs.close()

    debug_entries = parse_debug_file(session_id)

    for tool in tools_data:
        tool_name = tool["name"]
        tool_start = (tool["started_at"] or 0) * 1000

        relevant_debug = []
        for entry in debug_entries:
            time_diff = abs(entry.timestamp_ms - tool_start)
            if time_diff < 2000:
                if entry.tool_name == tool_name or entry.event_type in ['pre_hook', 'post_hook', 'file_write']:
                    relevant_debug.append({
                        "timestamp": entry.timestamp,
                        "event_type": entry.event_type,
                        "message": entry.message[:200]
                    })

        tool["debug"] = relevant_debug
        tool["debug_count"] = len(relevant_debug)

    return {
        "session_id": session_id,
        "tools": tools_data,
        "debug_available": len(debug_entries) > 0,
        "debug_total": len(debug_entries),
        "enriched": True
    }
