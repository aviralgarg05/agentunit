"""Auto-optimizer for applying recommendations."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .analyzer import AnalysisResult, RunAnalyzer
from .recommender import Recommendation, RecommendationType, Recommender


class OptimizationStrategy(Enum):
    """Optimization strategies."""

    CONSERVATIVE = "conservative"  # Apply only high-confidence changes
    BALANCED = "balanced"  # Apply most recommendations
    AGGRESSIVE = "aggressive"  # Apply all recommendations


@dataclass
class OptimizationResult:
    """Result of applying optimizations.

    Attributes:
        applied_recommendations: Recommendations that were applied
        skipped_recommendations: Recommendations that were skipped
        changes: Dictionary of changes made
        estimated_improvement: Estimated improvement metrics
    """

    applied_recommendations: list[Recommendation]
    skipped_recommendations: list[Recommendation]
    changes: dict[str, Any]
    estimated_improvement: dict[str, float]


class AutoOptimizer:
    """Automatically applies optimization recommendations.

    This optimizer can:
    - Analyze test runs
    - Generate recommendations
    - Apply safe optimizations automatically
    - Report changes and estimated improvements
    """

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        auto_apply: bool = False,
        approval_callback: Callable[[Recommendation], bool] | None = None,
    ):
        """Initialize auto-optimizer.

        Args:
            strategy: Optimization strategy to use
            auto_apply: Whether to automatically apply recommendations
            approval_callback: Callback for manual approval of changes
        """
        self.strategy = strategy
        self.auto_apply = auto_apply
        self.approval_callback = approval_callback

        self.analyzer = RunAnalyzer()
        self.recommender = Recommender()

    def optimize(
        self, run_data: dict[str, Any], config: dict[str, Any] | None = None
    ) -> OptimizationResult:
        """Analyze run and apply optimizations.

        Args:
            run_data: Test run data
            config: Current configuration

        Returns:
            OptimizationResult with applied changes
        """
        # Analyze run
        analysis = self.analyzer.analyze_run(run_data)

        # Generate recommendations
        recommendations = self.recommender.generate_recommendations(
            analysis, context={"config": config}
        )

        # Filter by strategy
        filtered = self._filter_by_strategy(recommendations)

        # Apply recommendations
        applied = []
        skipped = []
        changes = {}

        for rec in filtered:
            if self._should_apply(rec):
                change = self._apply_recommendation(rec, config)
                if change:
                    applied.append(rec)
                    changes.update(change)
                else:
                    skipped.append(rec)
            else:
                skipped.append(rec)

        # Estimate improvement
        estimated_improvement = self._estimate_improvement(applied, analysis)

        return OptimizationResult(
            applied_recommendations=applied,
            skipped_recommendations=skipped,
            changes=changes,
            estimated_improvement=estimated_improvement,
        )

    def _filter_by_strategy(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """Filter recommendations based on strategy.

        Args:
            recommendations: All recommendations

        Returns:
            Filtered recommendations
        """
        if self.strategy == OptimizationStrategy.CONSERVATIVE:
            # Only high priority (>= 8)
            return [r for r in recommendations if r.priority >= 8]

        if self.strategy == OptimizationStrategy.BALANCED:
            # Medium to high priority (>= 6)
            return [r for r in recommendations if r.priority >= 6]

        # AGGRESSIVE
        # All recommendations
        return recommendations

    def _should_apply(self, recommendation: Recommendation) -> bool:
        """Check if recommendation should be applied.

        Args:
            recommendation: Recommendation to check

        Returns:
            True if should apply
        """
        if not self.auto_apply:
            # Manual approval required
            if self.approval_callback:
                return self.approval_callback(recommendation)
            return False

        # Auto-apply based on type
        safe_types = {
            RecommendationType.PARAMETER,
            RecommendationType.PERFORMANCE,
            RecommendationType.COST,
        }

        if recommendation.type in safe_types:
            return True

        # For other types, require higher priority
        return recommendation.priority >= 9

    def _apply_recommendation(
        self, recommendation: Recommendation, config: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Apply a recommendation.

        Args:
            recommendation: Recommendation to apply
            config: Current configuration

        Returns:
            Dictionary of changes made, or None if not applicable
        """
        changes = {}

        if recommendation.type == RecommendationType.PARAMETER:
            # Apply parameter changes
            changes.update(self._apply_parameter_changes(recommendation, config))

        elif recommendation.type == RecommendationType.MODEL:
            # Suggest model change
            changes["model"] = self._suggest_model(recommendation)

        elif recommendation.type == RecommendationType.PROMPT:
            # Provide prompt suggestions
            changes["prompt_suggestions"] = self._suggest_prompt_improvements(recommendation)

        elif recommendation.type == RecommendationType.PERFORMANCE:
            # Apply performance optimizations
            changes.update(self._apply_performance_optimizations(recommendation, config))

        elif recommendation.type == RecommendationType.COST:
            # Apply cost optimizations
            changes.update(self._apply_cost_optimizations(recommendation, config))

        return changes if changes else None

    def _apply_parameter_changes(
        self, recommendation: Recommendation, config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply parameter changes.

        Args:
            recommendation: Parameter recommendation
            config: Current config

        Returns:
            Parameter changes
        """
        changes = {}

        # Example: Adjust timeout
        if "timeout" in recommendation.metadata:
            changes["timeout"] = recommendation.metadata["timeout"] * 1.5

        return changes

    def _suggest_model(self, recommendation: Recommendation) -> str:
        """Suggest a better model.

        Args:
            recommendation: Model recommendation

        Returns:
            Suggested model name
        """
        # Simple heuristic: suggest based on current success rate
        metadata = recommendation.metadata
        success_rate = metadata.get("current_success_rate", 0.5)

        if success_rate < 0.5:
            return "gpt-4-turbo"  # Most capable
        if success_rate < 0.7:
            return "gpt-4o"  # Balanced
        return "gpt-4o-mini"  # Cost-effective

    def _suggest_prompt_improvements(self, recommendation: Recommendation) -> list[str]:
        """Suggest prompt improvements.

        Args:
            recommendation: Prompt recommendation

        Returns:
            List of suggestions
        """
        suggestions = []

        if "clarity" in recommendation.description.lower():
            suggestions.append("Add specific output format requirements")
            suggestions.append("Include 2-3 examples of desired responses")

        if "examples" in recommendation.action.lower():
            suggestions.append("Use few-shot prompting with diverse examples")

        if "step-by-step" in recommendation.action.lower():
            suggestions.append("Add 'Let's think step by step' instruction")
            suggestions.append("Request numbered reasoning steps")

        return suggestions

    def _apply_performance_optimizations(
        self, recommendation: Recommendation, config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply performance optimizations.

        Args:
            recommendation: Performance recommendation
            config: Current config

        Returns:
            Performance changes
        """
        changes = {}

        if "caching" in recommendation.action.lower():
            changes["enable_cache"] = True
            changes["cache_ttl"] = 3600  # 1 hour

        if "parallel" in recommendation.action.lower():
            changes["parallel_tools"] = True
            changes["max_parallel"] = 5

        return changes

    def _apply_cost_optimizations(
        self, recommendation: Recommendation, config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply cost optimizations.

        Args:
            recommendation: Cost recommendation
            config: Current config

        Returns:
            Cost optimization changes
        """
        changes = {}

        metadata = recommendation.metadata

        # Suggest smaller model
        if "total_tokens" in metadata and metadata["total_tokens"] > 100000:
            changes["suggested_model"] = "gpt-4o-mini"

        # Reduce context window
        if "avg_cost" in metadata and metadata["avg_cost"] > 0.1:
            changes["max_context_length"] = 4000
            changes["compress_history"] = True

        return changes

    def _estimate_improvement(
        self, applied: list[Recommendation], analysis: AnalysisResult
    ) -> dict[str, float]:
        """Estimate improvement from applied recommendations.

        Args:
            applied: Applied recommendations
            analysis: Original analysis

        Returns:
            Estimated improvements
        """
        improvements = {"success_rate": 0.0, "latency_reduction": 0.0, "cost_reduction": 0.0}

        for rec in applied:
            # Parse impact statements for estimates
            if "success rate" in rec.impact.lower():
                # Extract percentage improvement
                if ">" in rec.impact:
                    target = float(rec.impact.split(">")[1].split("%")[0]) / 100
                    improvements["success_rate"] = max(
                        improvements["success_rate"], target - analysis.success_rate
                    )

            if "latency" in rec.impact.lower() and "30-50%" in rec.impact:
                improvements["latency_reduction"] = 0.4  # midpoint

            if "cost" in rec.impact.lower() or "50-70%" in rec.impact:
                improvements["cost_reduction"] = 0.6  # midpoint

        return improvements
