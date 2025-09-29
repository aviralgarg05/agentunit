"""Abstract base adapter for bridging external agent frameworks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional
import abc

from ..core.trace import TraceLog
from ..datasets.base import DatasetCase


@dataclass(slots=True)
class AdapterOutcome:
    """Normalized response from executing a scenario iteration."""

    success: bool
    output: Any
    tool_calls: Iterable[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] | None = None
    error: Optional[str] = None


class BaseAdapter(abc.ABC):
    """Adapters wrap framework-specific execution details."""

    name: str = "adapter"

    @abc.abstractmethod
    def prepare(self) -> None:
        """Perform any lazy setup (loading graphs, flows, etc.)."""

    @abc.abstractmethod
    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        """Run the agent flow on a single dataset case."""

    def cleanup(self) -> None:  # pragma: no cover - default no-op
        """Hook for cleaning up resources such as temporary files or servers."""

    def supports_replay(self) -> bool:
        return True
