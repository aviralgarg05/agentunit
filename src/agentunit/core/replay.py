"""
Replay utilities leveraging stored traces.
"""

from __future__ import annotations

from pathlib import Path

from .trace import TraceLog


def load_traces(traces_dir: str | Path) -> list[TraceLog]:
    """
    Load stored traces from disk for deterministic replay or analysis.
    """

    path = Path(traces_dir)
    logs: list[TraceLog] = []
    for trace_file in sorted(path.glob("*.json")):
        logs.append(TraceLog.load(trace_file))
    return logs
