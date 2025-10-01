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
from .monitoring import (
    ProductionMetrics,
    BaselineMetrics,
    DriftDetection,
    EvaluationTrigger,
    DriftType,
    AlertSeverity,
    EvaluationID,
    BaselineID,
    AlertID,
    MetricsCollector,
    DriftDetector,
    AlertManager
)

# Platform integrations
from .integrations import (
    MonitoringPlatform,
    ProductionIntegration
)

# Version and metadata
__version__ = "0.4.0"
__author__ = "AgentUnit Team"
__description__ = "Production integration and monitoring framework"

__all__ = [
    # Core metrics and data structures
    "ProductionMetrics",
    "BaselineMetrics", 
    "DriftDetection",
    
    # Enums
    "EvaluationTrigger",
    "DriftType",
    "AlertSeverity",
    "MonitoringPlatform",
    
    # Type aliases
    "EvaluationID",
    "BaselineID", 
    "AlertID",
    
    # Protocols
    "MetricsCollector",
    "DriftDetector",
    "AlertManager",
    
    # Base classes
    "ProductionIntegration",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]