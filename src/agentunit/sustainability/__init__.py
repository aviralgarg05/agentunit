"""Resource and sustainability tracking for AgentUnit.

This module provides tracking for:
- Carbon footprint (via CodeCarbon)
- GPU/TPU utilization
- Memory usage
- Energy consumption
- Cost tracking
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import ResourceTracker, ResourceMetrics
    from .carbon import CarbonTracker, CarbonReport
    from .metrics import EnergyMetric, CarbonMetric, ResourceUtilizationMetric

__all__ = [
    "ResourceTracker",
    "ResourceMetrics",
    "CarbonTracker",
    "CarbonReport",
    "EnergyMetric",
    "CarbonMetric",
    "ResourceUtilizationMetric",
]


def __getattr__(name: str):
    """Lazy loading of sustainability components."""
    if name == "ResourceTracker":
        from .tracker import ResourceTracker
        return ResourceTracker
    elif name == "ResourceMetrics":
        from .tracker import ResourceMetrics
        return ResourceMetrics
    elif name == "CarbonTracker":
        from .carbon import CarbonTracker
        return CarbonTracker
    elif name == "CarbonReport":
        from .carbon import CarbonReport
        return CarbonReport
    elif name == "EnergyMetric":
        from .metrics import EnergyMetric
        return EnergyMetric
    elif name == "CarbonMetric":
        from .metrics import CarbonMetric
        return CarbonMetric
    elif name == "ResourceUtilizationMetric":
        from .metrics import ResourceUtilizationMetric
        return ResourceUtilizationMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
