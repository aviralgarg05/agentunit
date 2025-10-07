"""Advanced A/B testing and regression detection for agent evaluation.

This module provides tools for comparing agent versions, detecting performance
regressions, and conducting statistical analysis with confidence intervals.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .comparator import (
        VersionComparator,
        ConfigurationComparator,
        ABTestRunner,
        RegressionDetector,
    )
    from .statistics import (
        BootstrapCI,
        StatisticalTest,
        MetricAggregator,
        SignificanceAnalyzer,
    )
    from .reports import ComparisonReport, RegressionReport

__all__ = [
    "VersionComparator",
    "ConfigurationComparator",
    "ABTestRunner",
    "RegressionDetector",
    "BootstrapCI",
    "StatisticalTest",
    "MetricAggregator",
    "SignificanceAnalyzer",
    "ComparisonReport",
    "RegressionReport",
]

_COMPARISON_IMPORTS = {
    "VersionComparator": "agentunit.comparison.comparator",
    "ConfigurationComparator": "agentunit.comparison.comparator",
    "ABTestRunner": "agentunit.comparison.comparator",
    "RegressionDetector": "agentunit.comparison.comparator",
    "BootstrapCI": "agentunit.comparison.statistics",
    "StatisticalTest": "agentunit.comparison.statistics",
    "MetricAggregator": "agentunit.comparison.statistics",
    "SignificanceAnalyzer": "agentunit.comparison.statistics",
    "ComparisonReport": "agentunit.comparison.reports",
    "RegressionReport": "agentunit.comparison.reports",
}


def __getattr__(name: str):
    """Lazy loading for comparison components."""
    if name in _COMPARISON_IMPORTS:
        import importlib
        module_path = _COMPARISON_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Support for dir() and autocomplete."""
    return sorted(__all__)
