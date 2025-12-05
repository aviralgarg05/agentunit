"""Adapter for Haystack pipelines."""

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


class HaystackAdapter(BaseAdapter):
    name = "haystack"

    def __init__(
        self,
        pipeline: Any,
        *,
        input_key: str = "query",
        params: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if pipeline is None:
            msg = "HaystackAdapter requires a pipeline or callable"
            raise AgentUnitError(msg)
        self._pipeline = pipeline
        self._input_key = input_key
        self._params = params or {}
        self._run_kwargs = run_kwargs or {}
        self._callable: Callable[[dict[str, Any]], Any] | None = None

    def prepare(self) -> None:
        if self._callable is not None:
            return
        self._callable = self._resolve_runner(self._pipeline)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None

        payload = {
            self._input_key: case.query,
            "metadata": case.metadata,
            "context": case.context,
            "tools": case.tools,
        }
        trace.record("haystack_input", payload=payload)
        try:
            response = self._invoke_runner(self._callable, payload)
            parsed = self._extract_output(response)
            trace.record("agent_response", content=parsed)
            return AdapterOutcome(success=True, output=parsed)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Haystack execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, pipeline: Any) -> Callable[[dict[str, Any]], Any]:
        if callable(pipeline):
            return pipeline
        for attr in ("run", "__call__", "invoke"):
            if hasattr(pipeline, attr):
                method = getattr(pipeline, attr)
                if callable(method):
                    return method
        msg = "Unsupported Haystack pipeline; expected callable or object with run method"
        raise AgentUnitError(msg)

    def _invoke_runner(
        self, runner: Callable[[dict[str, Any]], Any], payload: dict[str, Any]
    ) -> Any:
        try:
            return runner(payload, params=self._params, **self._run_kwargs)
        except TypeError:
            return runner(payload)

    def _extract_output(self, response: Any) -> Any:
        if response is None:
            return None
        if isinstance(response, dict):
            for key in ("answers", "results", "output"):
                if key in response:
                    return response[key]
        if hasattr(response, "answers"):
            return response.answers
        return response


register_adapter(HaystackAdapter, aliases=("haystack_pipeline",))
