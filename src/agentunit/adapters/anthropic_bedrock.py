"""Adapter for Anthropic Claude models running via Amazon Bedrock."""

from __future__ import annotations

import json
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


class AnthropicBedrockAdapter(BaseAdapter):
    """Executes prompts against Anthropic Claude models deployed on Amazon Bedrock."""

    name = "anthropic_bedrock"

    def __init__(
        self,
        *,
        client: Any,
        model_id: str,
        prompt_builder: Callable[[DatasetCase], dict[str, Any]] | None = None,
        invoke_kwargs: dict[str, Any] | None = None,
        response_key: str = "content",
    ) -> None:
        if client is None:
            msg = "AnthropicBedrockAdapter requires a Bedrock runtime client or callable"
            raise AgentUnitError(msg)
        if not model_id:
            msg = "AnthropicBedrockAdapter requires a model_id"
            raise AgentUnitError(msg)
        self._client = client
        self._model_id = model_id
        self._prompt_builder = prompt_builder or self._default_prompt_builder
        self._invoke_kwargs = invoke_kwargs or {}
        self._response_key = response_key
        self._callable: Callable[[dict[str, Any]], Any] | None = None

    def prepare(self) -> None:
        if self._callable is not None:
            return
        self._callable = self._resolve_invoker(self._client)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._callable is None:
            self.prepare()
        assert self._callable is not None

        body = self._prompt_builder(case)
        payload = {
            "modelId": self._model_id,
            "body": json.dumps(body),
        }
        payload.update(self._invoke_kwargs)
        trace.record("bedrock_request", payload=body)
        try:
            response = self._callable(payload)
            output = self._extract_output(response)
            trace.record("agent_response", content=output)
            return AdapterOutcome(success=True, output=output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Bedrock invocation failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    # Helpers -----------------------------------------------------------------
    def _resolve_invoker(self, client: Any) -> Callable[[dict[str, Any]], Any]:
        if callable(client):
            return client
        for attr in ("invoke_model", "invoke_model_with_response_stream", "__call__"):
            if hasattr(client, attr):
                candidate = getattr(client, attr)
                if callable(candidate):
                    return lambda request, _c=candidate: _c(**request)
        msg = "Unsupported Bedrock client; expected invoke_model callable"
        raise AgentUnitError(msg)

    def _default_prompt_builder(self, case: DatasetCase) -> dict[str, Any]:
        messages = []
        if case.context:
            messages.append({"role": "system", "content": "\n".join(case.context)})
        messages.append({"role": "user", "content": case.query})
        return {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
        }

    def _extract_output(self, response: Any) -> Any:
        if response is None:
            return None
        if isinstance(response, dict):
            body_payload = self._parse_body_if_present(response)
            if body_payload is not None:
                return self._extract_output(body_payload)
            if self._response_key in response:
                return response[self._response_key]
            if "completion" in response:
                return response["completion"]
        extracted = self._extract_from_mapping_like(response)
        return response if extracted is None else extracted

    def _parse_body_if_present(self, response: dict[str, Any]) -> Any:
        if "body" not in response:
            return None
        data = response["body"]
        try:
            if hasattr(data, "read"):
                data = data.read()
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            if isinstance(data, str):
                return json.loads(data)
        except Exception:  # pragma: no cover - resilience
            logger.debug("Failed to parse Bedrock body payload", exc_info=True)
        return data

    def _extract_from_mapping_like(self, response: Any) -> Any:
        if hasattr(response, "get"):
            try:
                return response.get(self._response_key)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to extract response via mapping", exc_info=True)
        return None


register_adapter(AnthropicBedrockAdapter, aliases=("claude_bedrock", "bedrock_claude"))
