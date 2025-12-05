"""Abstract base adapter for bridging external agent frameworks."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable

    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase


@dataclass(slots=True)
class AdapterOutcome:
    """Normalized response from executing a scenario iteration."""

    success: bool
    output: Any
    tool_calls: Iterable[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] | None = None
    error: str | None = None


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
