"""Debug parser for Claude Code CLI logs."""
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Debug directory (Claude Code CLI logs)
DEBUG_DIR = Path.home() / ".claude" / "debug"


@dataclass
class DebugEntry:
    """Entrada de log de debug do CLI."""
    timestamp: str
    timestamp_ms: int
    level: str
    message: str
    tool_name: Optional[str] = None
    event_type: Optional[str] = None  # pre_hook, post_hook, file_write, stream, temp_file


def parse_debug_file(session_id: str) -> list[DebugEntry]:
    """Parseia arquivo de debug do CLI."""
    debug_file = DEBUG_DIR / f"{session_id}.txt"
    if not debug_file.exists():
        return []

    entries = []
    pattern = r'^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+\[(\w+)\]\s+(.+)$'

    try:
        with open(debug_file, 'r') as f:
            for line in f:
                match = re.match(pattern, line.strip())
                if match:
                    timestamp_str, level, message = match.groups()

                    # Converter timestamp para ms
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamp_ms = int(dt.timestamp() * 1000)

                    # Detectar tipo de evento
                    event_type = None
                    tool_name = None

                    if 'PreToolUse' in message or 'executePreToolHooks' in message:
                        event_type = 'pre_hook'
                        tool_match = re.search(r'tool:\s*(\w+)|for:\s*(\w+)', message)
                        if tool_match:
                            tool_name = tool_match.group(1) or tool_match.group(2)
                    elif 'PostToolUse' in message:
                        event_type = 'post_hook'
                        tool_match = re.search(r'for:\s*(\w+)', message)
                        if tool_match:
                            tool_name = tool_match.group(1)
                    elif 'written atomically' in message or ('File' in message and 'written' in message):
                        event_type = 'file_write'
                    elif 'Stream started' in message:
                        event_type = 'stream'
                    elif 'Temp file' in message:
                        event_type = 'temp_file'

                    entries.append(DebugEntry(
                        timestamp=timestamp_str,
                        timestamp_ms=timestamp_ms,
                        level=level,
                        message=message,
                        tool_name=tool_name,
                        event_type=event_type
                    ))
    except Exception as e:
        print(f"[WARN] Could not parse debug file {session_id}: {e}")

    return entries
