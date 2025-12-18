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
        """
        Perform any lazy setup required before execution.

        This may include loading graphs, flows, or other resources.

        Returns:
            None
        """

    @abc.abstractmethod
    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        """
        Run the agent flow on a single dataset case.

        Args:
            case (DatasetCase): The dataset case to be processed.
            trace (TraceLog): Trace log used to record execution details.

        Returns:
            AdapterOutcome: The outcome produced by executing the adapter.
        """

    def cleanup(self) -> None:  # pragma: no cover - default no-op
        """
        Clean up resources after execution.

        This hook can be used to release resources such as temporary files
        or running servers.

        Returns:
            None
        """

    def supports_replay(self) -> bool:
        return True
