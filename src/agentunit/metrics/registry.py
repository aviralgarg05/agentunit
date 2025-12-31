"""Metric registry mapping string names to implementations."""

from __future__ import annotations

import threading
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

_registry_lock = threading.Lock()


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
    with _registry_lock:
        if name in DEFAULT_METRICS:
            msg = f"Metric '{name}' is already registered"
            raise ValueError(msg)
        DEFAULT_METRICS[name] = metric


def get_metric(name: str) -> Metric:
    # Deduplicates logic by using resolve_metrics,
    # which ensures the same error message is used.
    return resolve_metrics([name])[0]
