# Production Monitoring Core Components
"""
Core monitoring components for production integration.
Includes metrics collection, drift detection, and alerting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


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
    performance: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, float] = field(default_factory=dict)
    efficiency: dict[str, float] = field(default_factory=dict)
    custom_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric_value(self, metric_name: str) -> float | None:
        """Get a metric value from any category."""
        for category in [self.performance, self.quality, self.reliability, self.efficiency]:
            if metric_name in category:
                return category[metric_name]

        if metric_name in self.custom_metrics:
            value = self.custom_metrics[metric_name]
            return float(value) if isinstance(value, int | float) else None

        return None


@dataclass
class BaselineMetrics:
    """Baseline metrics for comparison."""

    id: BaselineID
    scenario_name: str
    created_at: datetime
    run_count: int
    performance_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    quality_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    reliability_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    efficiency_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_baseline_stats(self, metric_name: str) -> dict[str, float] | None:
        """Get baseline statistics for a metric."""
        for baseline in [
            self.performance_baseline,
            self.quality_baseline,
            self.reliability_baseline,
            self.efficiency_baseline,
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
    affected_scenarios: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector(Protocol):
    """Protocol for metrics collection."""

    def collect_metrics(self, scenario: Any, result: Any, **kwargs) -> ProductionMetrics | None:
        """Collect metrics from a scenario run."""
        msg = "Subclasses must implement collect_metrics"
        raise NotImplementedError(msg)


class DriftDetector(Protocol):
    """Protocol for drift detection."""

    def detect_drift(
        self,
        current_metrics: ProductionMetrics,
        baseline: BaselineMetrics,
        thresholds: dict[str, float],
    ) -> list[DriftDetection]:
        """Detect drift between current metrics and baseline."""
        msg = "Subclasses must implement detect_drift"
        raise NotImplementedError(msg)


class AlertManager(Protocol):
    """Protocol for alert management."""

    def send_alert(
        self, alert_id: AlertID, severity: AlertSeverity, message: str, metadata: dict[str, Any]
    ) -> bool:
        """Send an alert."""
        msg = "Subclasses must implement send_alert"
        raise NotImplementedError(msg)

    def check_alert_rules(self, metrics: ProductionMetrics) -> list[dict[str, Any]]:
        """Check if any alert rules are triggered."""
        msg = "Subclasses must implement check_alert_rules"
        raise NotImplementedError(msg)
