"""Metric interface for AgentUnit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol

from ..core.trace import TraceLog
from ..datasets.base import DatasetCase


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float | None
    detail: Dict[str, Any]


class Metric(Protocol):
    name: str

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        ...


class CompositeMetric(Metric):
    """Allows computing multiple metrics in one pass."""

    name = "composite"

    def __init__(self, metrics: Iterable[Metric]) -> None:
        self._metrics = list(metrics)

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        results = [metric.evaluate(case, trace, outcome) for metric in self._metrics]
        aggregate = sum(filter(None, (r.value for r in results if r.value is not None)), 0.0)
        count = len([r for r in results if r.value is not None]) or 1
        return MetricResult(name="composite", value=aggregate / count, detail={r.name: r.detail for r in results})
