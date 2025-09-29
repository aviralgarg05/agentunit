"""Replay utilities leveraging stored traces."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .trace import TraceLog


def load_traces(traces_dir: str | Path) -> List[TraceLog]:
    """Load stored traces from disk for deterministic replay or analysis."""

    path = Path(traces_dir)
    logs: List[TraceLog] = []
    for trace_file in sorted(path.glob("*.json")):
        logs.append(TraceLog.load(trace_file))
    return logs
