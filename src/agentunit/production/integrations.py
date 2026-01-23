# Production Platform Integrations
"""
Platform-specific integrations for production monitoring.
Includes LangSmith, AgentOps, Wandb, Langfuse, and Phoenix integrations.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from .monitoring import (
    AlertID,
    AlertSeverity,
    BaselineID,
    BaselineMetrics,
    DriftDetection,
    EvaluationID,
    ProductionMetrics,
)


# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentunit.core import Scenario


class MonitoringPlatform(Enum):
    """Supported monitoring platforms."""

    LANGSMITH = "langsmith"
    AGENTOPS = "agentops"
    WANDB = "wandb"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    CUSTOM = "custom"


class ProductionIntegration(ABC):
    """Abstract base class for production monitoring integrations."""

    def __init__(self, platform: MonitoringPlatform, config: dict[str, Any] | None = None):
        self.platform = platform
        self.config = config or {}
        self.baselines: dict[BaselineID, BaselineMetrics] = {}
        self.alert_rules: dict[str, dict[str, Any]] = {}
        self.evaluation_history: list[ProductionMetrics] = []

        self._setup_platform()

    @abstractmethod
    def _setup_platform(self) -> None:
        """Setup platform-specific configuration."""
        ...

    @abstractmethod
    def collect_metrics(
        self, scenario: "Scenario", result: Any, **kwargs
    ) -> ProductionMetrics | None:
        """Collect metrics from a scenario run."""
        ...

    def establish_baseline(
        self, historical_data: list[dict[str, Any]], metrics: list[str], **kwargs
    ) -> BaselineMetrics:
        """Establish baseline metrics."""
        baseline_id = f"baseline_{uuid.uuid4().hex[:8]}"
        logger.info(f"Establishing baseline {baseline_id}")

        try:
            # Calculate baseline statistics
            baseline_stats = self._calculate_baseline_stats(historical_data, metrics)

            baseline = BaselineMetrics(
                id=baseline_id,
                scenario_name=kwargs.get("scenario_name", "unknown"),
                created_at=datetime.now(),
                run_count=len(historical_data),
                **baseline_stats,
            )

            self.baselines[baseline_id] = baseline
            logger.info(f"Baseline {baseline_id} established successfully")

            return baseline

        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
            raise

    def _calculate_baseline_stats(
        self, historical_data: list[dict[str, Any]], metrics: list[str]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate baseline statistics from historical data."""
        baseline_stats: dict[str, dict[str, dict[str, float]]] = {
            "performance_baseline": {},
            "quality_baseline": {},
            "reliability_baseline": {},
            "efficiency_baseline": {},
        }

        for metric in metrics:
            values = []
            for data_point in historical_data:
                if metric in data_point:
                    values.append(float(data_point[metric]))

            if values:
                import numpy as np

                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                }

                # Categorize metric
                category = self._categorize_metric(metric)
                baseline_stats[f"{category}_baseline"][metric] = stats

        return baseline_stats

    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize a metric into performance, quality, reliability, or efficiency."""
        metric_lower = metric_name.lower()

        if any(
            term in metric_lower for term in ["latency", "response_time", "throughput", "speed"]
        ):
            return "performance"
        if any(
            term in metric_lower for term in ["accuracy", "precision", "recall", "f1", "quality"]
        ):
            return "quality"
        if any(
            term in metric_lower for term in ["error_rate", "failure", "uptime", "availability"]
        ):
            return "reliability"
        if any(
            term in metric_lower for term in ["cost", "resource", "memory", "cpu", "efficiency"]
        ):
            return "efficiency"
        return "performance"  # Default category

    def detect_drift(
        self,
        current_metrics: ProductionMetrics,
        baseline_id: BaselineID,
        thresholds: dict[str, float] | None = None,
    ) -> list[DriftDetection]:
        """Detect drift between current metrics and baseline."""
        if baseline_id not in self.baselines:
            logger.warning(f"Baseline {baseline_id} not found")
            return []

        baseline = self.baselines[baseline_id]
        thresholds = thresholds or {"default": 2.0}  # Default 2 sigma threshold

        drift_detections = []

        # Check all metric categories
        metric_categories = [
            ("performance", current_metrics.performance, baseline.performance_baseline),
            ("quality", current_metrics.quality, baseline.quality_baseline),
            ("reliability", current_metrics.reliability, baseline.reliability_baseline),
            ("efficiency", current_metrics.efficiency, baseline.efficiency_baseline),
        ]

        for _, current_vals, baseline_vals in metric_categories:
            for metric_name, current_value in current_vals.items():
                if metric_name in baseline_vals:
                    drift = self._check_metric_drift(
                        metric_name, current_value, baseline_vals[metric_name], thresholds
                    )
                    if drift:
                        drift_detections.append(drift)

        return drift_detections

    def _check_metric_drift(
        self,
        metric_name: str,
        current_value: float,
        baseline_stats: dict[str, float],
        thresholds: dict[str, float],
    ) -> DriftDetection | None:
        """Check if a specific metric has drifted."""
        baseline_mean = baseline_stats["mean"]
        baseline_std = baseline_stats["std"]

        # Use metric-specific threshold or default
        threshold = thresholds.get(metric_name, thresholds.get("default", 2.0))

        # Calculate z-score
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std

            if z_score > threshold:
                severity = self._determine_severity(z_score, threshold)

                return DriftDetection(
                    id=f"drift_{uuid.uuid4().hex[:8]}",
                    detection_time=datetime.now(),
                    drift_type=self._determine_drift_type(metric_name),
                    severity=severity,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_mean,
                    deviation=z_score,
                    threshold=threshold,
                    confidence=min(z_score / threshold, 1.0),
                    description=f"Metric {metric_name} drifted by {z_score:.2f} sigma",
                )

        return None

    def _determine_severity(self, z_score: float, threshold: float) -> AlertSeverity:
        """Determine alert severity based on z-score."""
        if z_score > threshold * 3:
            return AlertSeverity.CRITICAL
        if z_score > threshold * 2:
            return AlertSeverity.ERROR
        if z_score > threshold * 1.5:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def _determine_drift_type(self, metric_name: str):
        """Determine drift type based on metric name."""
        from .monitoring import DriftType

        metric_lower = metric_name.lower()
        if any(term in metric_lower for term in ["performance", "latency", "speed"]):
            return DriftType.PERFORMANCE
        if any(term in metric_lower for term in ["quality", "accuracy", "precision"]):
            return DriftType.QUALITY
        return DriftType.BEHAVIORAL

    def add_alert_rule(
        self,
        rule_name: str,
        condition: Callable[[ProductionMetrics], bool],
        severity: AlertSeverity,
        message_template: str,
    ) -> None:
        """Add an alert rule."""
        self.alert_rules[rule_name] = {
            "condition": condition,
            "severity": severity,
            "message_template": message_template,
        }
        logger.info(f"Added alert rule: {rule_name}")

    def check_alert_rules(self, metrics: ProductionMetrics) -> list[dict[str, Any]]:
        """Check if any alert rules are triggered."""
        triggered_alerts = []

        for rule_name, rule in self.alert_rules.items():
            try:
                if rule["condition"](metrics):
                    alert = {
                        "id": f"alert_{uuid.uuid4().hex[:8]}",
                        "rule_name": rule_name,
                        "severity": rule["severity"],
                        "message": rule["message_template"].format(
                            scenario=metrics.scenario_name, timestamp=metrics.timestamp
                        ),
                        "metrics": metrics,
                        "triggered_at": datetime.now(),
                    }
                    triggered_alerts.append(alert)
                    logger.warning(f"Alert triggered: {rule_name}")

            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")

        return triggered_alerts

    @abstractmethod
    def send_alert(
        self, alert_id: AlertID, severity: AlertSeverity, message: str, metadata: dict[str, Any]
    ) -> bool:
        """Send an alert through the platform."""
        ...

    @abstractmethod
    def log_evaluation(
        self, evaluation_id: EvaluationID, metrics: ProductionMetrics, metadata: dict[str, Any]
    ) -> bool:
        """Log evaluation results to the platform."""
        ...
