# Production Monitoring Core Components
"""
Core monitoring components for production integration.
Includes metrics collection, drift detection, and alerting.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
EvaluationID = str
BaselineID = str
AlertID = str


@dataclass
class ProductionMetrics:
    """Production metrics for an evaluation run."""

    evaluation_id: EvaluationID
    timestamp: datetime
    scenario_name: str
    performance: Dict[str, float] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)
    reliability: Dict[str, float] = field(default_factory=dict)
    efficiency: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get a metric value from any category."""
        for category in [self.performance, self.quality, self.reliability, self.efficiency]:
            if metric_name in category:
                return category[metric_name]

        if metric_name in self.custom_metrics:
            value = self.custom_metrics[metric_name]
            return float(value) if isinstance(value, (int, float)) else None

        return None


@dataclass
class BaselineMetrics:
    """Baseline metrics for comparison."""

    id: BaselineID
    scenario_name: str
    created_at: datetime
    run_count: int
    performance_baseline: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_baseline: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reliability_baseline: Dict[str, Dict[str, float]] = field(default_factory=dict)
    efficiency_baseline: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_baseline_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a metric."""
        for baseline in [
            self.performance_baseline,
            self.quality_baseline,
            self.reliability_baseline,
            self.efficiency_baseline
        ]:
            if metric_name in baseline:
                return baseline[metric_name]
        return None


class EvaluationTrigger(Enum):
    """Triggers for evaluation execution."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    DEPLOYMENT = "deployment"
    DRIFT_DETECTED = "drift_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    ROLLBACK = "rollback"


class DriftType(Enum):
    """Types of drift that can be detected."""

    PERFORMANCE = "performance"
    QUALITY = "quality"
    DATA = "data"
    CONCEPT = "concept"
    COVARIATE = "covariate"
    PRIOR = "prior"
    BEHAVIORAL = "behavioral"
    INFRASTRUCTURE = "infrastructure"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DriftDetection:
    """Drift detection result."""

    id: str
    detection_time: datetime
    drift_type: DriftType
    severity: AlertSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float
    threshold: float
    confidence: float
    description: str
    affected_scenarios: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def collect_metrics(
        self,
        scenario: Any,
        result: Any,
        **kwargs
    ) -> Optional[ProductionMetrics]:
        """Collect metrics from a scenario run."""
        raise NotImplementedError("Subclasses must implement collect_metrics")


class DriftDetector(Protocol):
    """Protocol for drift detection."""

    def detect_drift(
        self,
        current_metrics: ProductionMetrics,
        baseline: BaselineMetrics,
        thresholds: Dict[str, float]
    ) -> List[DriftDetection]:
        """Detect drift between current metrics and baseline."""
        raise NotImplementedError("Subclasses must implement detect_drift")


class AlertManager(Protocol):
    """Protocol for alert management."""

    def send_alert(
        self,
        alert_id: AlertID,
        severity: AlertSeverity,
        message: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send an alert."""
        raise NotImplementedError("Subclasses must implement send_alert")

    def check_alert_rules(
        self,
        metrics: ProductionMetrics
    ) -> List[Dict[str, Any]]:
        """Check if any alert rules are triggered."""
        raise NotImplementedError("Subclasses must implement check_alert_rules")