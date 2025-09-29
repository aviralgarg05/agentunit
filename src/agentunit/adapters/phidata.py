"""Adapter for Phidata data-centric agents."""
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
    import phi  # type: ignore
except Exception:  # pragma: no cover - best effort guard
    phi = None


class PhidataAdapter(BaseAdapter):
    """Executes scenarios against Phidata agents or workspaces."""

    name = "phidata"

    def __init__(
        self,
        agent: Any,
        *,
        input_builder: Optional[Callable[[DatasetCase], Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if agent is None:
            raise AgentUnitError("PhidataAdapter requires an agent or callable")
        self._agent = agent
        self._input_builder = input_builder or self._default_input_builder
        self._extra = extra or {}
        self._callable: Optional[Callable[[Dict[str, Any]], Any]] = None

    def prepare(self) -> None:
        if self._callable is not None:
            return
        self._callable = self._resolve_runner(self._agent)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None

        payload = self._input_builder(case)
        payload.update(self._extra)
        trace.record("phidata_input", payload=payload)
        try:
            result = self._callable(payload)
            output = self._extract_output(result)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive handling
            logger.exception("Phidata execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _resolve_runner(self, agent: Any) -> Callable[[Dict[str, Any]], Any]:
        if callable(agent):
            return agent
        for attr in ("run", "execute", "__call__"):
            if hasattr(agent, attr):
                candidate = getattr(agent, attr)
                if callable(candidate):
                    return candidate
        raise AgentUnitError("Unsupported Phidata agent; expected callable or object with run/execute")

    def _default_input_builder(self, case: DatasetCase) -> Dict[str, Any]:
        return {
            "query": case.query,
            "context": case.context,
            "metadata": case.metadata,
            "expected_output": case.expected_output,
        }

    def _extract_output(self, result: Any) -> Any:
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ("response", "output", "result"):
                if key in result:
                    return result[key]
        if hasattr(result, "result"):
            return getattr(result, "result")
        if hasattr(result, "content"):
            return result.content
        return result


register_adapter(PhidataAdapter, aliases=("phi", "phidata_agent"))
