"""Adapter for OpenAI Swarm orchestrations."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter
from ..core.exceptions import AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None


class OpenAISwarmAdapter(BaseAdapter):
    """Executes multi-agent swarms built with OpenAI Swarm APIs."""

    name = "openai_swarm"

    def __init__(
        self,
        swarm: Any,
        *,
        message_builder: Optional[Callable[[DatasetCase], list[Dict[str, Any]]]] = None,
        metadata_builder: Optional[Callable[[DatasetCase], Dict[str, Any]]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if swarm is None:
            raise AgentUnitError("OpenAISwarmAdapter requires a swarm or callable")
        self._swarm = swarm
        self._message_builder = message_builder or self._default_message_builder
        self._metadata_builder = metadata_builder or (lambda case: case.metadata or {})
        self._run_kwargs = run_kwargs or {}
        self._runner: Optional[Callable[[list[Dict[str, Any]]], Any]] = None

    def prepare(self) -> None:
        if self._runner is not None:
            return
        self._runner = self._resolve_runner(self._swarm)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._runner is None:
            self.prepare()
        assert self._runner is not None

        messages = self._message_builder(case)
        trace.record("openai_swarm_messages", payload={"messages": messages})
        kwargs = {"metadata": self._metadata_builder(case)}
        kwargs.update(self._run_kwargs)
        try:
            result = self._invoke_runner(self._runner, messages, kwargs)
            output = self._extract_output(result)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("OpenAI Swarm execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, swarm: Any) -> Callable[[list[Dict[str, Any]]], Any]:
        if callable(swarm):
            return swarm
        for attr in ("run", "execute", "invoke", "__call__"):
            if hasattr(swarm, attr):
                candidate = getattr(swarm, attr)
                if callable(candidate):
                    return candidate
        raise AgentUnitError("Unsupported OpenAI Swarm object; expected callable or object with run/invoke")

    def _invoke_runner(
        self,
        runner: Callable[[list[Dict[str, Any]]], Any],
        messages: list[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Any:
        try:
            return runner(messages=messages, **kwargs)
        except TypeError:
            return runner(messages)

    def _default_message_builder(self, case: DatasetCase) -> list[Dict[str, Any]]:
        messages: list[Dict[str, Any]] = []
        if case.context:
            messages.append({"role": "system", "content": "\n".join(case.context)})
        messages.append({"role": "user", "content": case.query})
        return messages

    def _extract_output(self, result: Any) -> Any:
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ("output", "response", "result"):
                if key in result:
                    return result[key]
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "messages"):
            return result.messages
        return result


register_adapter(OpenAISwarmAdapter, aliases=("swarm", "openai-swarm"))
