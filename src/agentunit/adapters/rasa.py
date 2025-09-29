"""Adapter for Rasa conversational agents."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import httpx

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter
from ..core.exceptions import AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)


class RasaAdapter(BaseAdapter):
    """Executes conversations against Rasa REST endpoints or callables."""

    name = "rasa"

    def __init__(
        self,
        target: str | Callable[[Dict[str, Any]], Any],
        *,
        sender_id: str = "agentunit",
        session_params: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
        response_key: str = "text",
    ) -> None:
        if not target:
            raise AgentUnitError("RasaAdapter requires an endpoint URL or callable handler")
        self._target = target
        self._sender_id = sender_id
        self._session_params = session_params or {}
        self._timeout = timeout
        self._headers = headers or {"Content-Type": "application/json"}
        self._response_key = response_key
        self._callable: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._client: Optional[httpx.Client] = None

    def prepare(self) -> None:
        if callable(self._target):
            self._callable = self._target  # type: ignore[assignment]
        else:
            if self._client is None:
                self._client = httpx.Client(timeout=self._timeout)

    def cleanup(self) -> None:  # pragma: no cover - cleanup path
        if self._client is not None:
            self._client.close()
            self._client = None

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None and self._client is None:
            self.prepare()

        payload = {
            "sender": self._sender_id,
            "message": case.query,
            "metadata": {**self._session_params, **(case.metadata or {})},
        }
        trace.record("rasa_request", payload=payload)

        try:
            if self._callable is not None:
                response = self._callable(payload)
            else:
                assert isinstance(self._target, str)
                assert self._client is not None
                result = self._client.post(self._target, json=payload, headers=self._headers)
                result.raise_for_status()
                response = result.json()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Rasa execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

        output = self._extract_output(response)
        trace.record("agent_response", content=output)
        return AdapterOutcome(success=True, output=output)

    def _extract_output(self, response: Any) -> Any:
        if response is None:
            return None
        if isinstance(response, list):
            # REST API returns list of responses
            if not response:
                return None
            if isinstance(response[0], dict) and self._response_key in response[0]:
                return " ".join(str(item.get(self._response_key, "")) for item in response)
            return response
        if isinstance(response, dict) and self._response_key in response:
            return response[self._response_key]
        return response


register_adapter(RasaAdapter, aliases=("rasa_rest", "rasa_endpoint"))
