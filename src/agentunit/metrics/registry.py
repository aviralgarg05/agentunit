"""Metric registry mapping string names to implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .builtin import (
    AnswerCorrectnessMetric,
    FaithfulnessMetric,
    HallucinationRateMetric,
    RetrievalQualityMetric,
    ToolSuccessMetric,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import Metric


DEFAULT_METRICS: dict[str, Metric] = {
    "faithfulness": FaithfulnessMetric(),
    "tool_success": ToolSuccessMetric(),
    "answer_correctness": AnswerCorrectnessMetric(),
    "hallucination_rate": HallucinationRateMetric(),
    "retrieval_quality": RetrievalQualityMetric(),
}


def resolve_metrics(names: Sequence[str] | None) -> list[Metric]:
    if not names:
        return list(DEFAULT_METRICS.values())
    resolved = []
    for name in names:
        metric = DEFAULT_METRICS.get(name)
        if metric is None:
            msg = f"Unknown metric '{name}'"
            raise KeyError(msg)
        resolved.append(metric)
    return resolved


def register_metric(name: str, metric: Metric) -> None:
    if name in DEFAULT_METRICS:
        msg = f"Metric '{name}' is already registered"
        raise ValueError(msg)
    DEFAULT_METRICS[name] = metric


def get_metric(name: str) -> Metric:
    metric = DEFAULT_METRICS.get(name)
    if metric is None:
        msg = f"Metric '{name}' not found"
        raise KeyError(msg)
    return metric
