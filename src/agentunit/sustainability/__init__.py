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
    from .carbon import CarbonReport, CarbonTracker
    from .metrics import CarbonMetric, EnergyMetric, ResourceUtilizationMetric
    from .tracker import ResourceMetrics, ResourceTracker

__all__ = [
    "CarbonMetric",
    "CarbonReport",
    "CarbonTracker",
    "EnergyMetric",
    "ResourceMetrics",
    "ResourceTracker",
    "ResourceUtilizationMetric",
]


def __getattr__(name: str):
    """Lazy loading of sustainability components."""
    if name == "ResourceTracker":
        from .tracker import ResourceTracker

        return ResourceTracker
    if name == "ResourceMetrics":
        from .tracker import ResourceMetrics

        return ResourceMetrics
    if name == "CarbonTracker":
        from .carbon import CarbonTracker

        return CarbonTracker
    if name == "CarbonReport":
        from .carbon import CarbonReport

        return CarbonReport
    if name == "EnergyMetric":
        from .metrics import EnergyMetric

        return EnergyMetric
    if name == "CarbonMetric":
        from .metrics import CarbonMetric

        return CarbonMetric
    if name == "ResourceUtilizationMetric":
        from .metrics import ResourceUtilizationMetric

        return ResourceUtilizationMetric
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
