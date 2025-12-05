# Production Integration & Monitoring Framework
"""
Production monitoring and observability integration for AgentUnit.

This module provides comprehensive production monitoring capabilities including:
- Real-time metrics collection and analysis
- Drift detection and alerting
- Integration with popular monitoring platforms
- Baseline establishment and comparison
- Performance regression detection
"""

# Core monitoring components
# Platform integrations
from .integrations import MonitoringPlatform, ProductionIntegration
from .monitoring import (
    AlertID,
    AlertManager,
    AlertSeverity,
    BaselineID,
    BaselineMetrics,
    DriftDetection,
    DriftDetector,
    DriftType,
    EvaluationID,
    EvaluationTrigger,
    MetricsCollector,
    ProductionMetrics,
)


# Version and metadata
__version__ = "0.4.0"
__author__ = "AgentUnit Team"
__description__ = "Production integration and monitoring framework"

__all__ = [
    "AlertID",
    "AlertManager",
    "AlertSeverity",
    "BaselineID",
    "BaselineMetrics",
    "DriftDetection",
    "DriftDetector",
    "DriftType",
    # Type aliases
    "EvaluationID",
    # Enums
    "EvaluationTrigger",
    # Protocols
    "MetricsCollector",
    "MonitoringPlatform",
    # Base classes
    "ProductionIntegration",
    # Core metrics and data structures
    "ProductionMetrics",
    "__author__",
    "__description__",
    # Metadata
    "__version__",
]
