"""Adapter for CrewAI workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentunit.core.exceptions import AgentUnitError

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter


if TYPE_CHECKING:
    from collections.abc import Callable

    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from crewai import Crew
except Exception:  # pragma: no cover
    Crew = None  # type: ignore


class CrewAIAdapter(BaseAdapter):
    name = "crewai"

    def __init__(
        self, crew: Any, task_builder: Callable[[DatasetCase], Any] | None = None, **options: Any
    ) -> None:
        self._crew = crew
        self._task_builder = task_builder
        self._options = options

    @classmethod
    def from_crew(cls, crew: Any, **options: Any) -> CrewAIAdapter:
        return cls(crew=crew, **options)

    def prepare(self) -> None:
        if self._crew is None:
            msg = "CrewAI crew is not defined"
            raise AgentUnitError(msg)
        if Crew is None:
            logger.warning("CrewAI not installed; running in mock mode")

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        self.prepare()
        if Crew is not None and not isinstance(self._crew, Crew):
            msg = "CrewAIAdapter expects a Crew instance when CrewAI is installed"
            raise AgentUnitError(msg)

        trace.record(
            "agent_prompt",
            input={"query": case.query, "context": case.context, "tools": case.tools},
        )

        try:
            if self._task_builder is not None:
                task = self._task_builder(case)
            elif Crew is not None and hasattr(self._crew, "tasks"):
                task = self._crew.tasks[0] if self._crew.tasks else None
            else:
                task = None

            if task is not None and hasattr(task, "input"):
                task.input = case.query

            if hasattr(self._crew, "kickoff"):
                response = self._crew.kickoff(inputs={"query": case.query, "context": case.context})
            elif callable(self._crew):
                response = self._crew(case)
            else:
                msg = "Cannot execute CrewAI scenario - unsupported crew object"
                raise AgentUnitError(msg)

            trace.record("agent_response", content=response)
            return AdapterOutcome(success=True, output=response)
        except Exception as exc:  # pragma: no cover
            logger.exception("CrewAI execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))


register_adapter(CrewAIAdapter, aliases=("crew_ai",))
