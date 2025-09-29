"""Adapter for the OpenAI Agents SDK (March 2025 release)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import logging

from .base import BaseAdapter, AdapterOutcome
from ..core.exceptions import AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai_agents_sdk import AgentsClient
except Exception:  # pragma: no cover
    AgentsClient = None  # type: ignore


class OpenAIAgentsAdapter(BaseAdapter):
    name = "openai-agents"

    def __init__(self, flow: Any, client: Optional[Any] = None, **options: Any) -> None:
        self._flow = flow
        self._client = client or self._default_client(options)
        self._options = options
        self._callable: Optional[Callable[[Dict[str, Any]], Any]] = None

    @classmethod
    def from_flow(cls, flow: Any, **options: Any) -> "OpenAIAgentsAdapter":
        return cls(flow=flow, **options)

    def prepare(self) -> None:
        if self._callable is not None:
            return
        if self._flow is None:
            raise AgentUnitError("OpenAI Agents flow is not defined")
        if hasattr(self._flow, "run"):
            self._callable = self._flow.run
            return
        if callable(self._flow):
            self._callable = self._flow
            return
        raise AgentUnitError("Unsupported OpenAI Agents flow; expected callable or .run method")

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None
        payload = {
            "query": case.query,
            "context": case.context,
            "tools": case.tools,
            "metadata": case.metadata,
        }
        trace.record("agent_prompt", input=payload)
        try:
            if self._client is not None and hasattr(self._callable, "__self__"):
                response = self._callable(payload, client=self._client)  # type: ignore[arg-type]
            else:
                response = self._callable(payload)
            normalized = self._normalize_response(response, trace)
            trace.record("agent_response", content=normalized.output)
            return normalized
        except Exception as exc:  # pragma: no cover
            logger.exception("OpenAI Agents execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _normalize_response(self, response: Any, trace: TraceLog) -> AdapterOutcome:
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls") or []
            for tool in tool_calls:
                trace.record("tool_call", **tool)
            return AdapterOutcome(success=response.get("success", True), output=response.get("output"), tool_calls=tool_calls)
        return AdapterOutcome(success=True, output=response)

    def _default_client(self, options: Dict[str, Any]) -> Optional[Any]:
        if AgentsClient is None:
            logger.warning("OpenAI Agents SDK not installed; running in mock mode")
            return None
        api_key = options.get("api_key")
        if api_key is None:
            logger.debug("Instantiating AgentsClient using environment credentials")
            return AgentsClient()
        return AgentsClient(api_key=api_key)
