"""
Tracing utilities shared between adapters and the runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    """
    Represents a single prompt, tool call, or response in an agent run.
    """

    type: str
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class TraceLog:
    """
    A collection of chronological events for a scenario iteration.
    """

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, event_type: str, **payload: Any) -> None:
        self.events.append(TraceEvent(type=event_type, payload=payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {
                    "type": event.type,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                for event in self.events
            ]
        }

    def last_response(self) -> str | None:
        for event in reversed(self.events):
            if event.type == "agent_response":
                return str(event.payload.get("content"))
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceLog:
        events = []
        for event in data.get("events", []):
            created = event.get("created_at")
            timestamp = datetime.fromisoformat(created) if created else datetime.now(timezone.utc)
            events.append(
                TraceEvent(
                    type=event.get("type", "event"),
                    payload=event.get("payload", {}),
                    created_at=timestamp,
                )
            )
        return cls(events=events)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    @classmethod
    def load(cls, path: str | Path) -> TraceLog:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
