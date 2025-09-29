"""Metric registry mapping string names to implementations."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .base import Metric
from .builtin import (
    FaithfulnessMetric,
    ToolSuccessMetric,
    AnswerCorrectnessMetric,
    HallucinationRateMetric,
    RetrievalQualityMetric,
)


DEFAULT_METRICS: Dict[str, Metric] = {
    "faithfulness": FaithfulnessMetric(),
    "tool_success": ToolSuccessMetric(),
    "answer_correctness": AnswerCorrectnessMetric(),
    "hallucination_rate": HallucinationRateMetric(),
    "retrieval_quality": RetrievalQualityMetric(),
}


def resolve_metrics(names: Sequence[str] | None) -> List[Metric]:
    if not names:
        return list(DEFAULT_METRICS.values())
    resolved = []
    for name in names:
        metric = DEFAULT_METRICS.get(name)
        if metric is None:
            raise KeyError(f"Unknown metric '{name}'")
        resolved.append(metric)
    return resolved
