"""Auto-optimization recommendations for AgentUnit.

This module provides meta-evaluation capabilities to analyze test runs
and generate actionable recommendations for improving agent performance.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import RunAnalyzer, AnalysisResult
    from .recommender import Recommender, Recommendation, RecommendationType
    from .optimizer import AutoOptimizer, OptimizationStrategy

__all__ = [
    "RunAnalyzer",
    "AnalysisResult",
    "Recommender",
    "Recommendation",
    "RecommendationType",
    "AutoOptimizer",
    "OptimizationStrategy",
]


def __getattr__(name: str):
    """Lazy loading of optimization components."""
    if name == "RunAnalyzer":
        from .analyzer import RunAnalyzer
        return RunAnalyzer
    elif name == "AnalysisResult":
        from .analyzer import AnalysisResult
        return AnalysisResult
    elif name == "Recommender":
        from .recommender import Recommender
        return Recommender
    elif name == "Recommendation":
        from .recommender import Recommendation
        return Recommendation
    elif name == "RecommendationType":
        from .recommender import RecommendationType
        return RecommendationType
    elif name == "AutoOptimizer":
        from .optimizer import AutoOptimizer
        return AutoOptimizer
    elif name == "OptimizationStrategy":
        from .optimizer import OptimizationStrategy
        return OptimizationStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
