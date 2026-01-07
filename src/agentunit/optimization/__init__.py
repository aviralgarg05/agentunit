"""Auto-optimization recommendations for AgentUnit.

This module provides meta-evaluation capabilities to analyze test runs
and generate actionable recommendations for improving agent performance.
"""

from .analyzer import AnalysisResult, RunAnalyzer
from .optimizer import AutoOptimizer, OptimizationStrategy
from .recommender import Recommendation, RecommendationType, Recommender


__all__ = [
    "AnalysisResult",
    "AutoOptimizer",
    "OptimizationStrategy",
    "Recommendation",
    "RecommendationType",
    "Recommender",
    "RunAnalyzer",
]


def __getattr__(name: str):
    """Lazy loading of optimization components."""
    if name == "RunAnalyzer":
        from .analyzer import RunAnalyzer

        return RunAnalyzer
    if name == "AnalysisResult":
        from .analyzer import AnalysisResult

        return AnalysisResult
    if name == "Recommender":
        from .recommender import Recommender

        return Recommender
    if name == "Recommendation":
        from .recommender import Recommendation

        return Recommendation
    if name == "RecommendationType":
        from .recommender import RecommendationType

        return RecommendationType
    if name == "AutoOptimizer":
        from .optimizer import AutoOptimizer

        return AutoOptimizer
    if name == "OptimizationStrategy":
        from .optimizer import OptimizationStrategy

        return OptimizationStrategy
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
