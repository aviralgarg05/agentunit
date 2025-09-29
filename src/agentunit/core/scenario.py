"""Scenario definition API exposed to end users."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional
from pathlib import Path
import random

from ..adapters.base import BaseAdapter
from ..adapters.langgraph import LangGraphAdapter
from ..adapters.openai_agents import OpenAIAgentsAdapter
from ..adapters.crewai import CrewAIAdapter
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
    return getattr(source, "__name__", fallback)
