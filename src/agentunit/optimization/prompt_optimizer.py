"""Prompt optimization for AI agents.

Standard feature that DeepEval and LangSmith provide.
A/B test prompts, automatic improvements, metric-driven optimization.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant for testing."""

    id: str
    template: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTestResult:
    """Result of testing a prompt variant."""

    variant_id: str
    num_samples: int
    avg_score: float
    scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    passed_rate: float = 0.0


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    improvement_score: float
    suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""


class PromptOptimizer:
    """Optimize prompts for better agent performance.

    Standard feature in DeepEval (auto-optimize) and LangSmith.

    Features:
    - A/B test prompt variants
    - Automatic improvement suggestions
    - Metric-driven optimization
    - Best variant selection

    Example:
        ```python
        optimizer = PromptOptimizer()

        # A/B test variants
        variants = [
            PromptVariant("v1", "Answer the question: {query}"),
            PromptVariant("v2", "You are an expert. Answer: {query}"),
        ]
        results = optimizer.ab_test(variants, test_cases)
        print(f"Best: {results['winner']}")

        # Auto-optimize
        improved = optimizer.optimize("Answer the question: {query}")
        ```
    """

    OPTIMIZATION_PROMPT = """Analyze this prompt and suggest improvements:

Original prompt:
```
{prompt}
```

Performance context:
- Current avg score: {avg_score}
- Issues observed: {issues}

Suggest improvements to:
1. Clarity and specificity
2. Output format guidance
3. Edge case handling
4. Safety and guardrails

Output as JSON:
```json
{{
    "optimized_prompt": "<improved prompt>",
    "changes": [
        {{"aspect": "<what changed>", "reason": "<why>"}},
        ...
    ],
    "expected_improvement": "<explanation>"
}}
```
"""

    VARIATION_PROMPT = """Generate {num_variants} variations of this prompt.

Original:
```
{prompt}
```

Each variation should:
- Keep the core intent
- Try different approaches (concise, detailed, structured, etc.)
- Include different instruction styles

Output as JSON:
```json
[
    {{
        "id": "v1",
        "template": "<variation>",
        "style": "<description of approach>"
    }},
    ...
]
```
"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        api_key: str | None = None,
    ):
        """Initialize optimizer.

        Args:
            model: LLM model for optimization
            provider: LLM provider
            api_key: Optional API key
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self._client = None

    def _get_client(self) -> Any:
        """Get LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            import openai

            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=api_key)
        elif self.provider == "groq":
            from groq import Groq

            api_key = self.api_key or os.environ.get("GROQ_API_KEY")
            self._client = Groq(api_key=api_key)

        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call LLM."""
        client = self._get_client()

        if self.provider in ["openai", "groq"]:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        return ""

    def generate_variants(
        self,
        prompt: str,
        num_variants: int = 3,
    ) -> list[PromptVariant]:
        """Generate prompt variations.

        Args:
            prompt: Original prompt
            num_variants: Number of variants to generate

        Returns:
            List of PromptVariant objects
        """
        gen_prompt = self.VARIATION_PROMPT.format(
            prompt=prompt,
            num_variants=num_variants,
        )

        raw = self._call_llm(gen_prompt)

        # Parse JSON
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group(1))
                return [
                    PromptVariant(
                        id=item.get("id", f"v{i}"),
                        template=item.get("template", ""),
                        description=item.get("style", ""),
                    )
                    for i, item in enumerate(items)
                ]
            except json.JSONDecodeError:
                pass

        return []

    def ab_test(
        self,
        variants: list[PromptVariant],
        test_cases: list[dict[str, Any]],
        evaluator: Callable[[str, str, str | None], float],
    ) -> dict[str, Any]:
        """A/B test prompt variants.

        Args:
            variants: List of prompt variants to test
            test_cases: List of test cases with 'query', 'expected' keys
            evaluator: Function(query, response, expected) -> score

        Returns:
            Test results with winner
        """
        results = {}

        for variant in variants:
            scores = []
            total_latency = 0.0

            for case in test_cases:
                import time

                # Format prompt with query
                prompt = variant.template.format(query=case.get("query", ""))

                start = time.time()
                response = self._call_llm(prompt)
                latency = (time.time() - start) * 1000
                total_latency += latency

                # Evaluate
                score = evaluator(case.get("query", ""), response, case.get("expected"))
                scores.append(score)

            results[variant.id] = PromptTestResult(
                variant_id=variant.id,
                num_samples=len(test_cases),
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                scores=scores,
                latency_ms=total_latency / len(test_cases) if test_cases else 0.0,
                passed_rate=sum(1 for s in scores if s >= 0.7) / len(scores) if scores else 0.0,
            )

        # Find winner
        winner = max(results.values(), key=lambda r: r.avg_score)

        return {
            "winner": winner.variant_id,
            "results": {
                k: {
                    "avg_score": v.avg_score,
                    "passed_rate": v.passed_rate,
                    "latency_ms": v.latency_ms,
                }
                for k, v in results.items()
            },
            "improvement": winner.avg_score - min(r.avg_score for r in results.values()),
        }

    def optimize(
        self,
        prompt: str,
        current_score: float = 0.5,
        issues: list[str] | None = None,
    ) -> OptimizationResult:
        """Optimize a prompt.

        Args:
            prompt: Original prompt
            current_score: Current performance score
            issues: Known issues with current prompt

        Returns:
            OptimizationResult with improved prompt
        """
        opt_prompt = self.OPTIMIZATION_PROMPT.format(
            prompt=prompt,
            avg_score=current_score,
            issues=", ".join(issues) if issues else "None specified",
        )

        raw = self._call_llm(opt_prompt)

        # Parse response
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=data.get("optimized_prompt", prompt),
                    improvement_score=0.0,  # Unknown until tested
                    suggestions=[c.get("aspect", "") for c in data.get("changes", [])],
                    reasoning=data.get("expected_improvement", ""),
                )
            except json.JSONDecodeError:
                pass

        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=prompt,
            improvement_score=0.0,
        )

    def iterative_optimize(
        self,
        prompt: str,
        test_cases: list[dict[str, Any]],
        evaluator: Callable[[str, str, str | None], float],
        max_iterations: int = 3,
        target_score: float = 0.9,
    ) -> dict[str, Any]:
        """Iteratively optimize a prompt.

        Args:
            prompt: Starting prompt
            test_cases: Test cases for evaluation
            evaluator: Scoring function
            max_iterations: Max optimization iterations
            target_score: Target score to stop at

        Returns:
            Optimization history and final prompt
        """
        history = []
        current_prompt = prompt
        current_score = 0.0

        for i in range(max_iterations):
            # Test current prompt
            scores = []
            for case in test_cases:
                formatted = current_prompt.format(query=case.get("query", ""))
                response = self._call_llm(formatted)
                score = evaluator(case.get("query", ""), response, case.get("expected"))
                scores.append(score)

            current_score = sum(scores) / len(scores) if scores else 0.0

            history.append(
                {
                    "iteration": i + 1,
                    "prompt": current_prompt,
                    "score": current_score,
                }
            )

            if current_score >= target_score:
                break

            # Optimize
            result = self.optimize(current_prompt, current_score)
            current_prompt = result.optimized_prompt

        return {
            "final_prompt": current_prompt,
            "final_score": current_score,
            "iterations": len(history),
            "history": history,
            "target_reached": current_score >= target_score,
        }


__all__ = [
    "OptimizationResult",
    "PromptOptimizer",
    "PromptTestResult",
    "PromptVariant",
]
