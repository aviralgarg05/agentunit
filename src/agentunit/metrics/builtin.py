"""Built-in metrics leveraging RAGAS when available."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import Metric, MetricResult


if TYPE_CHECKING:
    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase


logger = logging.getLogger(__name__)

try:  # pragma: no cover - heavy optional dependency
    from ragas.metrics import answer_correctness as ragas_answer_correctness
    from ragas.metrics import context_precision as ragas_context_precision
    from ragas.metrics import faithfulness as ragas_faithfulness
    from ragas.metrics import hallucination as ragas_hallucination

    RAGAS_AVAILABLE = True
except Exception:  # pragma: no cover
    RAGAS_AVAILABLE = False


def _coerce_to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return "\n".join(map(str, obj))
    return str(obj)


class FaithfulnessMetric(Metric):
    name = "faithfulness"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        answer = _coerce_to_text(getattr(outcome, "output", outcome))
        references = case.context or []
        if RAGAS_AVAILABLE and references:
            try:
                value = float(ragas_faithfulness(answer=answer, contexts=references))
            except Exception as exc:  # pragma: no cover
                logger.warning("RAGAS faithfulness failed: %s", exc)
                value = None
        elif references:
            # Simple heuristic: check if at least one context string appears in the answer
            matches = sum(1 for ref in references if ref.lower() in answer.lower())
            value = matches / len(references)
        else:
            value = None
        return MetricResult(
            name=self.name, value=value, detail={"answer": answer, "references": references}
        )


class ToolSuccessMetric(Metric):
    name = "tool_success"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        tool_calls = [event.payload for event in trace.events if event.type == "tool_call"]
        if not tool_calls:
            value = 1.0 if getattr(outcome, "success", False) else 0.0
        else:
            successes = sum(1 for tool in tool_calls if tool.get("status", "success") == "success")
            value = successes / len(tool_calls)
        return MetricResult(name=self.name, value=value, detail={"tool_calls": tool_calls})


class AnswerCorrectnessMetric(Metric):
    name = "answer_correctness"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        expected = _coerce_to_text(case.expected_output)
        answer = _coerce_to_text(getattr(outcome, "output", outcome))
        if not expected:
            value = None
        elif answer.strip() == expected.strip():
            value = 1.0
        elif RAGAS_AVAILABLE:
            try:
                value = float(ragas_answer_correctness(answer=answer, reference=expected))
            except Exception as exc:  # pragma: no cover
                logger.warning("RAGAS answer_correctness failed: %s", exc)
                value = 0.0
        else:
            value = 0.0
        return MetricResult(
            name=self.name, value=value, detail={"expected": expected, "answer": answer}
        )


class HallucinationRateMetric(Metric):
    name = "hallucination_rate"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        answer = _coerce_to_text(getattr(outcome, "output", outcome))
        references = case.context or []
        value = None
        if RAGAS_AVAILABLE and references:
            try:
                score = float(ragas_hallucination(answer=answer, contexts=references))
                value = 1.0 - score
            except Exception as exc:  # pragma: no cover
                logger.warning("RAGAS hallucination failed: %s", exc)
                value = None
        elif references:
            matches = sum(1 for ref in references if ref.lower() in answer.lower())
            value = 0.0 if matches == len(references) else 1.0
        return MetricResult(name=self.name, value=value, detail={"answer": answer})


class RetrievalQualityMetric(Metric):
    name = "retrieval_quality"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        references = case.context or []
        answer = _coerce_to_text(getattr(outcome, "output", outcome))
        if not references:
            value = None
        elif RAGAS_AVAILABLE:
            try:
                value = float(ragas_context_precision(answer=answer, contexts=references))
            except Exception as exc:  # pragma: no cover
                logger.warning("RAGAS context_precision failed: %s", exc)
                value = None
        else:
            mentions = sum(1 for ref in references if ref.lower() in answer.lower())
            value = mentions / len(references)
        return MetricResult(
            name=self.name, value=value, detail={"references": references, "answer": answer}
        )


class CostMetric(Metric):
    name = "cost"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        # Try to extract cost from trace metadata or outcome
        cost = 0.0

        # Check trace metadata
        if trace.metadata and "cost" in trace.metadata:
            cost = float(trace.metadata["cost"])

        # Check outcome
        elif hasattr(outcome, "cost"):
            cost = float(outcome.cost)

        # Sum up cost from tool calls if available
        tool_calls = [event.payload for event in trace.events if event.type == "tool_call"]
        for tool in tool_calls:
            if "cost" in tool:
                cost += float(tool["cost"])

        return MetricResult(name=self.name, value=cost, detail={"cost": cost})


class TokenUsageMetric(Metric):
    name = "token_usage"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Check trace metadata
        if trace.metadata and "usage" in trace.metadata:
            usage = trace.metadata["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

        # Check outcome
        elif hasattr(outcome, "usage"):
            usage = outcome.usage
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            else:
                # Assume object with attributes
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

        return MetricResult(
            name=self.name,
            value=float(total_tokens),
            detail={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
