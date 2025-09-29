"""Adapter for Microsoft PromptFlow orchestrations."""
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
    import promptflow  # type: ignore
except Exception:  # pragma: no cover
    promptflow = None


class PromptFlowAdapter(BaseAdapter):
    """Executes PromptFlow flows or activities."""

    name = "promptflow"

    def __init__(
        self,
        flow: Any,
        *,
        context_builder: Optional[Callable[[DatasetCase], Dict[str, Any]]] = None,
        output_key: str = "output",
        run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if flow is None:
            raise AgentUnitError("PromptFlowAdapter requires a flow or callable")
        self._flow = flow
        self._context_builder = context_builder or self._default_context_builder
        self._output_key = output_key
        self._run_kwargs = run_kwargs or {}
        self._runner: Optional[Callable[[Dict[str, Any]], Any]] = None

    def prepare(self) -> None:
        if self._runner is not None:
            return
        self._runner = self._resolve_runner(self._flow)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._runner is None:
            self.prepare()
        assert self._runner is not None

        context = self._context_builder(case)
        trace.record("promptflow_context", payload=context)
        try:
            response = self._invoke_runner(self._runner, context)
            output = self._extract_output(response)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("PromptFlow execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, flow: Any) -> Callable[[Dict[str, Any]], Any]:
        if callable(flow):
            return flow
        for attr in ("run", "invoke", "execute", "__call__"):
            if hasattr(flow, attr):
                candidate = getattr(flow, attr)
                if callable(candidate):
                    return candidate
        raise AgentUnitError("Unsupported PromptFlow flow; expected callable or object with run/invoke")

    def _invoke_runner(self, runner: Callable[[Dict[str, Any]], Any], context: Dict[str, Any]) -> Any:
        try:
            return runner(context, **self._run_kwargs)
        except TypeError:
            return runner(context)

    def _default_context_builder(self, case: DatasetCase) -> Dict[str, Any]:
        return {
            "inputs": {
                "query": case.query,
                "context": case.context,
                "metadata": case.metadata,
            }
        }

    def _extract_output(self, response: Any) -> Any:
        if response is None:
            return None
        if isinstance(response, dict):
            if self._output_key in response:
                return response[self._output_key]
            if "outputs" in response and isinstance(response["outputs"], dict):
                return response["outputs"].get(self._output_key)
        if hasattr(response, self._output_key):
            return getattr(response, self._output_key)
        return response


register_adapter(PromptFlowAdapter, aliases=("ms_promptflow", "prompt_flow"))
