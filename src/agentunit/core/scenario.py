"""
Scenario definition API exposed to end users.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentunit.datasets.registry import resolve_dataset


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from agentunit.adapters.base import BaseAdapter
    from agentunit.datasets.base import DatasetCase, DatasetSource


@dataclass(slots=True)
class Scenario:
    """
    Defines a reproducible agent evaluation scenario.
    """

    name: str
    adapter: BaseAdapter
    dataset: DatasetSource
    retries: int = 1
    max_turns: int = 20
    timeout: float = 60.0
    tags: list[str] = field(default_factory=list)
    seed: int | None = None
    metadata: dict = field(default_factory=dict)

    def iter_cases(self) -> Iterable[DatasetCase]:
        if self.seed is not None:
            random.seed(self.seed)
        yield from self.dataset.iter_cases()

    # Factories ----------------------------------------------------------------
    @classmethod
    def load_langgraph(
        cls,
        path: str | Path | object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **config: object,
    ) -> Scenario:
        from agentunit.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter.from_source(path, **config)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(path, fallback="langgraph-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_openai_agents(
        cls,
        flow: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: object,
    ) -> Scenario:
        from agentunit.adapters.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter.from_flow(flow, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or getattr(flow, "__name__", None) or "openai-agents-scenario"
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_crewai(
        cls,
        crew: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: object,
    ) -> Scenario:
        """
        Create scenario from CrewAI crew.
        """
        from agentunit.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter.from_crew(crew, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(crew, fallback="crewai-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_autogen(
        cls,
        orchestrator: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        """
        Create scenario from AutoGen orchestrator.
        """
        from agentunit.adapters.autogen import AutoGenAdapter

        adapter = AutoGenAdapter(orchestrator=orchestrator, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(orchestrator, fallback="autogen-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_haystack(
        cls,
        pipeline: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.haystack import HaystackAdapter

        adapter = HaystackAdapter(pipeline=pipeline, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(pipeline, fallback="haystack-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_llama_index(
        cls,
        engine: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.llama_index import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(engine=engine, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(engine, fallback="llama-index-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_semantic_kernel(
        cls,
        invoker: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.semantic_kernel import SemanticKernelAdapter

        adapter = SemanticKernelAdapter(invoker=invoker, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(invoker, fallback="semantic-kernel-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_phidata(
        cls,
        agent: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.phidata import PhidataAdapter

        adapter = PhidataAdapter(agent=agent, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(agent, fallback="phidata-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_promptflow(
        cls,
        flow: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.promptflow import PromptFlowAdapter

        adapter = PromptFlowAdapter(flow=flow, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(flow, fallback="promptflow-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_openai_swarm(
        cls,
        swarm: object,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.openai_swarm import OpenAISwarmAdapter

        adapter = OpenAISwarmAdapter(swarm=swarm, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(swarm, fallback="openai-swarm-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_anthropic_bedrock(
        cls,
        client: object,
        model_id: str,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.anthropic_bedrock import AnthropicBedrockAdapter

        adapter = AnthropicBedrockAdapter(client=client, model_id=model_id, **options)
        ds = resolve_dataset(dataset)
        base_name = name or f"{model_id}-bedrock"
        return cls(name=base_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_mistral_server(
        cls,
        base_url: str,
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.mistral_server import MistralServerAdapter

        adapter = MistralServerAdapter(base_url=base_url, **options)
        ds = resolve_dataset(dataset)
        if name is not None:
            scenario_name = name
        elif isinstance(base_url, str) and "://" in base_url:
            scenario_name = "mistral-server-scenario"
        else:
            scenario_name = _infer_name(base_url, fallback="mistral-server-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_rasa_endpoint(
        cls,
        target: str | Callable[[dict], object],
        dataset: str | DatasetSource | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Scenario:
        from agentunit.adapters.rasa import RasaAdapter

        adapter = RasaAdapter(target=target, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name if name is not None else _infer_name(target, fallback="rasa-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    # Convenience ----------------------------------------------------------------
    def with_dataset(self, dataset: str | DatasetSource) -> Scenario:
        return Scenario(
            name=self.name,
            adapter=self.adapter,
            dataset=resolve_dataset(dataset),
            retries=self.retries,
            max_turns=self.max_turns,
            timeout=self.timeout,
            tags=list(self.tags),
            seed=self.seed,
            metadata=dict(self.metadata),
        )

    def clone(self, **overrides: object) -> Scenario:
        data = {
            "name": self.name,
            "adapter": self.adapter,
            "dataset": self.dataset,
            "retries": self.retries,
            "max_turns": self.max_turns,
            "timeout": self.timeout,
            "tags": list(self.tags),
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }
        data.update(overrides)
        return Scenario(**data)  # type: ignore[arg-type]


def _infer_name(source: object, fallback: str) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).stem
    if hasattr(source, "__name__"):
        return source.__name__
    if hasattr(source, "__class__") and hasattr(source.__class__, "__name__"):
        return source.__class__.__name__
    return fallback
