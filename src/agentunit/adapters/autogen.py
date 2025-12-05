"""Adapter for AutoGen conversational orchestrations."""

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
    import autogen  # type: ignore
except Exception:  # pragma: no cover
    autogen = None


class AutoGenAdapter(BaseAdapter):
    name = "autogen"

    def __init__(
        self,
        orchestrator: Any,
        *,
        task_builder: Callable[[DatasetCase], str] | None = None,
        message_builder: Callable[[DatasetCase], list[dict[str, Any]]] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if orchestrator is None:
            msg = "AutoGenAdapter requires an orchestrator or conversation callable"
            raise AgentUnitError(msg)
        self._orchestrator = orchestrator
        self._task_builder = task_builder or (lambda case: case.query)
        self._message_builder = message_builder or self._default_message_builder
        self._extra_kwargs = extra_kwargs or {}
        self._runner: Callable[[dict[str, Any]], Any] | None = None

    def prepare(self) -> None:
        if self._runner is not None:
            return
        self._runner = self._resolve_runner(self._orchestrator)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._runner is None:
            self.prepare()
        assert self._runner is not None  # mypy guard

        payload = {
            "task": self._task_builder(case),
            "messages": self._message_builder(case),
            "metadata": case.metadata,
            "tools": case.tools,
        }
        trace.record("autogen_task", payload=payload)
        try:
            result = self._invoke_runner(self._runner, payload)
            final_output = self._extract_output(result)
            trace.record("agent_response", content=final_output)
            return AdapterOutcome(success=True, output=final_output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("AutoGen execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    # Helpers -----------------------------------------------------------------
    def _resolve_runner(self, candidate: Any) -> Callable[[dict[str, Any]], Any]:
        if callable(candidate):  # plain function or lambda
            return candidate
        for attr in ("run", "invoke", "chat", "__call__"):
            if hasattr(candidate, attr):
                method = getattr(candidate, attr)
                if callable(method):
                    return method
        msg = "Unsupported AutoGen orchestrator; expected callable or object with run/chat method"
        raise AgentUnitError(msg)

    def _invoke_runner(
        self, runner: Callable[[dict[str, Any]], Any], payload: dict[str, Any]
    ) -> Any:
        try:
            return runner(payload, **self._extra_kwargs)
        except TypeError:
            return runner(payload)

    def _default_message_builder(self, case: DatasetCase) -> list[dict[str, Any]]:
        messages = []
        system_prompt = case.metadata.get("system") if case.metadata else None
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        context_lines = case.context or []
        if context_lines:
            context_blob = "\n".join(context_lines)
            messages.append({"role": "user", "content": context_blob})
        messages.append({"role": "user", "content": case.query})
        return messages

    def _extract_output(self, result: Any) -> Any:
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ("response", "output", "final", "result"):
                if key in result:
                    return result[key]
        if autogen is not None and hasattr(result, "messages"):
            return result.messages  # pragma: no cover - depends on autogen objects
        if hasattr(result, "content"):
            return result.content
        return result


register_adapter(AutoGenAdapter, aliases=("autogen_chat", "autogen_orchestrator"))
