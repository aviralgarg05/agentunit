from __future__ import annotations

from typing import Any, Dict, List

from agentunit.adapters.autogen import AutoGenAdapter
from agentunit.adapters.haystack import HaystackAdapter
from agentunit.adapters.llama_index import LlamaIndexAdapter
from agentunit.adapters.semantic_kernel import SemanticKernelAdapter
from agentunit.adapters.phidata import PhidataAdapter
from agentunit.adapters.promptflow import PromptFlowAdapter
from agentunit.adapters.openai_swarm import OpenAISwarmAdapter
from agentunit.adapters.anthropic_bedrock import AnthropicBedrockAdapter
from agentunit.adapters.mistral_server import MistralServerAdapter
from agentunit.adapters.rasa import RasaAdapter
from agentunit.adapters.registry import resolve_adapter
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase

def _dataset_case() -> DatasetCase:
    return DatasetCase(
        id="case-1",
        query="Explain quantum tunneling",
        expected_output="",
        context=["Physics"],
        tools=["search"],
        metadata={"system": "You are a helpful assistant."},
    )


def test_autogen_adapter_with_run_method() -> None:
    calls: List[Dict[str, Any]] = []

    class DummyOrchestrator:
        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            calls.append(payload)
            return {"response": "Autogen reply"}

    adapter = AutoGenAdapter(orchestrator=DummyOrchestrator())
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Autogen reply"
    assert calls and calls[0]["task"] == "Explain quantum tunneling"


def test_haystack_adapter_handles_answers_dict() -> None:
    recorded: List[Dict[str, Any]] = []

    class DummyPipeline:
        def run(self, payload: Dict[str, Any], **_: Any) -> Dict[str, Any]:
            recorded.append(payload)
            return {"answers": ["Haystack answer"]}

    adapter = HaystackAdapter(pipeline=DummyPipeline())
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == ["Haystack answer"]
    assert recorded and recorded[0]["query"] == "Explain quantum tunneling"


def test_llama_index_adapter_allows_callable_engine() -> None:
    adapter = LlamaIndexAdapter(engine=lambda prompt: {"response": prompt.upper()})
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "EXPLAIN QUANTUM TUNNELING"


def test_semantic_kernel_adapter_invokes_callable() -> None:
    captured: List[Dict[str, Any]] = []

    class DummyInvoker:
        def __call__(self, variables: Dict[str, Any]) -> Dict[str, Any]:
            captured.append(variables)
            return {"result": "Semantic Kernel reply"}

    adapter = SemanticKernelAdapter(invoker=DummyInvoker())
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Semantic Kernel reply"
    assert captured and captured[0]["input"] == "Explain quantum tunneling"


def test_phidata_adapter_runs_callable_agent() -> None:
    calls: List[Dict[str, Any]] = []

    def agent(payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(payload)
        return {"response": "Phi reply"}

    adapter = PhidataAdapter(agent=agent)
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Phi reply"
    assert calls and calls[0]["query"] == "Explain quantum tunneling"


def test_promptflow_adapter_invokes_flow_callable() -> None:
    calls: List[Dict[str, Any]] = []

    def flow(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"output": "PromptFlow reply"}

    adapter = PromptFlowAdapter(flow=flow)
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "PromptFlow reply"
    assert calls and calls[0]["inputs"]["query"] == "Explain quantum tunneling"


def test_openai_swarm_adapter_runs_swarm_callable() -> None:
    calls: List[List[Dict[str, Any]]] = []

    def swarm(*, messages: List[Dict[str, Any]], **_: Any) -> Dict[str, Any]:
        calls.append(messages)
        return {"output": "Swarm reply"}

    adapter = OpenAISwarmAdapter(swarm=swarm)
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Swarm reply"
    assert calls and calls[0][0]["role"] == "system"


def test_anthropic_bedrock_adapter_uses_callable_client() -> None:
    recorded: List[Dict[str, Any]] = []

    def client(request: Dict[str, Any]) -> Dict[str, Any]:
        recorded.append(request)
        body = {"completion": "Claude reply"}
        return body

    adapter = AnthropicBedrockAdapter(client=client, model_id="anthropic.claude-v2")
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Claude reply"
    assert recorded and recorded[0]["modelId"] == "anthropic.claude-v2"


def test_mistral_server_adapter_parses_response() -> None:
    class DummyResponse:
        def __init__(self, payload: Dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return self._payload

    class DummyClient:
        def __init__(self) -> None:
            self.captured: List[Dict[str, Any]] = []

        def post(self, _: str, json: Dict[str, Any]) -> DummyResponse:
            self.captured.append(json)
            return DummyResponse(
                {
                    "choices": [
                        {"message": {"content": "Mistral reply"}}
                    ]
                }
            )

    client = DummyClient()
    adapter = MistralServerAdapter(base_url="http://localhost:8000", http_client=client)  # type: ignore[arg-type]
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Mistral reply"
    assert client.captured and client.captured[0]["model"] == "mistral-large-latest"


def test_rasa_adapter_supports_callable_target() -> None:
    def rasa_handler(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"text": "Rasa reply"}]

    adapter = RasaAdapter(target=rasa_handler)
    outcome = adapter.execute(_dataset_case(), TraceLog())

    assert outcome.success is True
    assert outcome.output == "Rasa reply"


def test_registry_resolves_new_adapters() -> None:
    assert resolve_adapter("autogen") is AutoGenAdapter
    assert resolve_adapter("haystack") is HaystackAdapter
    assert resolve_adapter("llama_index") is LlamaIndexAdapter
    assert resolve_adapter("semantic_kernel") is SemanticKernelAdapter
    assert resolve_adapter("phidata") is PhidataAdapter
    assert resolve_adapter("promptflow") is PromptFlowAdapter
    assert resolve_adapter("openai_swarm") is OpenAISwarmAdapter
    assert resolve_adapter("anthropic_bedrock") is AnthropicBedrockAdapter
    assert resolve_adapter("mistral_server") is MistralServerAdapter
    assert resolve_adapter("rasa") is RasaAdapter
