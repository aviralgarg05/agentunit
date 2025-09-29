"""Scenario definition API exposed to end users."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional
from pathlib import Path
import random

from ..adapters.base import BaseAdapter
from ..adapters.langgraph import LangGraphAdapter
from ..adapters.openai_agents import OpenAIAgentsAdapter
from ..adapters.crewai import CrewAIAdapter
from ..adapters.autogen import AutoGenAdapter
from ..adapters.haystack import HaystackAdapter
from ..adapters.llama_index import LlamaIndexAdapter
from ..adapters.semantic_kernel import SemanticKernelAdapter
from ..adapters.phidata import PhidataAdapter
from ..adapters.promptflow import PromptFlowAdapter
from ..adapters.openai_swarm import OpenAISwarmAdapter
from ..adapters.anthropic_bedrock import AnthropicBedrockAdapter
from ..adapters.mistral_server import MistralServerAdapter
from ..adapters.rasa import RasaAdapter
from ..datasets.base import DatasetSource, DatasetCase
from ..datasets.registry import resolve_dataset


@dataclass(slots=True)
class Scenario:
    """Defines a reproducible agent evaluation scenario."""

    name: str
    adapter: BaseAdapter
    dataset: DatasetSource
    retries: int = 1
    max_turns: int = 20
    timeout: float = 60.0
    tags: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def iter_cases(self) -> Iterable[DatasetCase]:
        if self.seed is not None:
            random.seed(self.seed)
        for case in self.dataset.iter_cases():
            yield case

    # Factories ----------------------------------------------------------------
    @classmethod
    def load_langgraph(
        cls,
        path: str | Path | object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **config: object,
    ) -> "Scenario":
        adapter = LangGraphAdapter.from_source(path, **config)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(path, fallback="langgraph-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_openai_agents(
        cls,
        flow: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = OpenAIAgentsAdapter.from_flow(flow, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or getattr(flow, "__name__", "openai-agents-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_crewai(
        cls,
        crew: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = CrewAIAdapter.from_crew(crew, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or getattr(crew, "name", "crewai-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_autogen(
        cls,
        orchestrator: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = AutoGenAdapter(orchestrator=orchestrator, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(orchestrator, fallback="autogen-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_haystack(
        cls,
        pipeline: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = HaystackAdapter(pipeline=pipeline, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(pipeline, fallback="haystack-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_llama_index(
        cls,
        engine: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = LlamaIndexAdapter(engine=engine, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(engine, fallback="llama-index-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_semantic_kernel(
        cls,
        invoker: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = SemanticKernelAdapter(invoker=invoker, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(invoker, fallback="semantic-kernel-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_phidata(
        cls,
        agent: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = PhidataAdapter(agent=agent, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(agent, fallback="phidata-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_promptflow(
        cls,
        flow: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = PromptFlowAdapter(flow=flow, **options)
        ds = resolve_dataset(dataset)
        scenario_name = name or _infer_name(flow, fallback="promptflow-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_openai_swarm(
        cls,
        swarm: object,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
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
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = AnthropicBedrockAdapter(client=client, model_id=model_id, **options)
        ds = resolve_dataset(dataset)
        base_name = name or f"{model_id}-bedrock"
        return cls(name=base_name, adapter=adapter, dataset=ds)

    @classmethod
    def from_mistral_server(
        cls,
        base_url: str,
        dataset: str | DatasetSource | None = None,
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
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
        name: Optional[str] = None,
        **options: object,
    ) -> "Scenario":
        adapter = RasaAdapter(target=target, **options)
        ds = resolve_dataset(dataset)
        if name is not None:
            scenario_name = name
        else:
            scenario_name = _infer_name(target, fallback="rasa-scenario")
        return cls(name=scenario_name, adapter=adapter, dataset=ds)

    # Convenience ----------------------------------------------------------------
    def with_dataset(self, dataset: str | DatasetSource) -> "Scenario":
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

    def clone(self, **overrides: object) -> "Scenario":
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
        return getattr(source, "__name__")
    if hasattr(source, "__class__") and hasattr(source.__class__, "__name__"):
        return source.__class__.__name__
    return fallback
