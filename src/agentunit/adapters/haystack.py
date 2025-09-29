"""Adapter for Haystack pipelines."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter
from ..core.exceptions import AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)

class HaystackAdapter(BaseAdapter):
    name = "haystack"

    def __init__(
        self,
        pipeline: Any,
        *,
        input_key: str = "query",
        params: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pipeline is None:
            raise AgentUnitError("HaystackAdapter requires a pipeline or callable")
        self._pipeline = pipeline
        self._input_key = input_key
        self._params = params or {}
        self._run_kwargs = run_kwargs or {}
        self._callable: Optional[Callable[[Dict[str, Any]], Any]] = None

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

    def _resolve_runner(self, pipeline: Any) -> Callable[[Dict[str, Any]], Any]:
        if callable(pipeline):
            return pipeline
        for attr in ("run", "__call__", "invoke"):
            if hasattr(pipeline, attr):
                method = getattr(pipeline, attr)
                if callable(method):
                    return method
        raise AgentUnitError("Unsupported Haystack pipeline; expected callable or object with run method")

    def _invoke_runner(self, runner: Callable[[Dict[str, Any]], Any], payload: Dict[str, Any]) -> Any:
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
