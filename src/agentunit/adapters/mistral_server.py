"""Adapter for the Mistral open-source server deployment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from agentunit.core.exceptions import AgentUnitError

from .base import AdapterOutcome, BaseAdapter
from .registry import register_adapter


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase


logger = logging.getLogger(__name__)


class MistralServerAdapter(BaseAdapter):
    """Runs chat completions against a Mistral-compatible HTTP server."""

    name = "mistral_server"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str = "mistral-large-latest",
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra_headers: dict[str, str] | None = None,
        message_builder: Callable[[DatasetCase], Iterable[dict[str, Any]]] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            msg = "MistralServerAdapter requires a base_url"
            raise AgentUnitError(msg)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra_headers = extra_headers or {}
        self._message_builder = message_builder or self._default_message_builder
        self._client = http_client

    def prepare(self) -> None:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            headers.update(self._extra_headers)
            self._client = httpx.Client(base_url=self._base_url, headers=headers, timeout=30.0)

    def cleanup(self) -> None:  # pragma: no cover - cleanup path
        if self._client is not None:
            self._client.close()
            self._client = None

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._client is None:
            self.prepare()
        assert self._client is not None

        messages = self._message_builder(case)
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        trace.record("mistral_request", payload=payload)
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            output = self._extract_output(data)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Mistral server request failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    def _default_message_builder(self, case: DatasetCase) -> Iterable[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if case.context:
            messages.append({"role": "system", "content": "\n".join(case.context)})
        messages.append({"role": "user", "content": case.query})
        return messages

    def _extract_output(self, data: dict[str, Any]) -> Any:
        if not data:
            return None
        choices = data.get("choices")
        if not choices:
            return data
        message = choices[0].get("message", {})
        return message.get("content") or message


register_adapter(MistralServerAdapter, aliases=("mistral", "mistral-http"))
