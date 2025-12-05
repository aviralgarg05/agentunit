"""Adapter for Microsoft Semantic Kernel executors."""

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


class SemanticKernelAdapter(BaseAdapter):
    name = "semantic_kernel"

    def __init__(
        self,
        invoker: Any,
        *,
        input_variable: str = "input",
        variable_builder: Callable[[DatasetCase], dict[str, Any]] | None = None,
        extra_variables: dict[str, Any] | None = None,
    ) -> None:
        if invoker is None:
            msg = "SemanticKernelAdapter requires a kernel/function invoker"
            raise AgentUnitError(msg)
        self._invoker = invoker
        self._input_variable = input_variable
        self._variable_builder = variable_builder or self._default_variable_builder
        self._extra_variables = extra_variables or {}
        self._callable: Callable[[dict[str, Any]], Any] | None = None

    def prepare(self) -> None:
        if self._callable is not None:
            return
        self._callable = self._resolve_runner(self._invoker)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None

        variables = self._variable_builder(case)
        variables[self._input_variable] = case.query
        variables.update(self._extra_variables)
        trace.record("semantic_kernel_variables", payload=variables)
        try:
            response = self._invoke_runner(self._callable, variables)
            output = self._extract_output(response)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Semantic Kernel execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, invoker: Any) -> Callable[[dict[str, Any]], Any]:
        if callable(invoker):
            return invoker
        for attr in ("invoke", "run", "__call__"):
            if hasattr(invoker, attr):
                method = getattr(invoker, attr)
                if callable(method):
                    return method
        msg = "Unsupported Semantic Kernel invoker; expected callable or object with invoke method"
        raise AgentUnitError(msg)

    def _invoke_runner(
        self, runner: Callable[[dict[str, Any]], Any], variables: dict[str, Any]
    ) -> Any:
        try:
            return runner(variables)
        except TypeError:
            return runner(**variables)

    def _default_variable_builder(self, case: DatasetCase) -> dict[str, Any]:
        return {
            "metadata": case.metadata,
            "context": case.context,
            "tools": case.tools,
        }

    def _extract_output(self, response: Any) -> Any:
        if response is None:
            return None
        if isinstance(response, dict) and "result" in response:
            return response["result"]
        if hasattr(response, "result"):
            return response.result
        if hasattr(response, "value"):
            return response.value
        return response


register_adapter(SemanticKernelAdapter, aliases=("sk", "ms_semantic_kernel"))
