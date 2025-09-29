"""Adapter for LlamaIndex query engines."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter
from ..core.exceptions import AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)

class LlamaIndexAdapter(BaseAdapter):
    name = "llama_index"

    def __init__(
        self,
        engine: Any,
        *,
        prompt_builder: Optional[Callable[[DatasetCase], str]] = None,
        response_attribute: str = "response",
    ) -> None:
        if engine is None:
            raise AgentUnitError("LlamaIndexAdapter requires a query engine or callable")
        self._engine = engine
        self._prompt_builder = prompt_builder or (lambda case: case.query)
        self._response_attribute = response_attribute
        self._callable: Optional[Callable[[str], Any]] = None

    def prepare(self) -> None:
        if self._callable is not None:
            return
        self._callable = self._resolve_runner(self._engine)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None

        prompt = self._prompt_builder(case)
        trace.record("llama_index_prompt", payload={"prompt": prompt})
        try:
            result = self._callable(prompt)
            output = self._extract_output(result)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LlamaIndex execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, engine: Any) -> Callable[[str], Any]:
        if hasattr(engine, "as_query_engine"):
            engine = engine.as_query_engine()
        if callable(engine):
            return engine
        for attr in ("query", "run", "__call__"):
            if hasattr(engine, attr):
                method = getattr(engine, attr)
                if callable(method):
                    return method
        raise AgentUnitError("Unsupported LlamaIndex engine; expected callable or object with query method")

    def _extract_output(self, result: Any) -> Any:
        if result is None:
            return None
        if isinstance(result, dict) and self._response_attribute in result:
            return result[self._response_attribute]
        if hasattr(result, self._response_attribute):
            return getattr(result, self._response_attribute)
        if hasattr(result, "message"):
            return result.message
        return result


register_adapter(LlamaIndexAdapter, aliases=("llamaindex", "gpt_index"))
