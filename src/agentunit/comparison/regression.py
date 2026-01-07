"""Regression detection for AI agent evaluation.

Standard feature that LangSmith and DeepEval provide.
Automatic baseline comparison, alert on degradation, statistical testing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Alert for detected regression."""

    metric_name: str
    baseline_value: float
    current_value: float
    degradation_pct: float
    severity: str  # low, medium, high, critical
    statistical_significance: float
    message: str


@dataclass
class RegressionReport:
    """Report from regression analysis."""

    timestamp: str
    baseline_run: str
    current_run: str
    has_regression: bool
    alerts: list[RegressionAlert] = field(default_factory=list)
    metrics_compared: dict[str, dict[str, float]] = field(default_factory=dict)
    summary: str = ""


class RegressionDetector:
    """Detect performance regressions in AI agents.

    Standard feature in LangSmith and DeepEval.

    Features:
    - Automatic baseline comparison
    - Statistical significance testing
    - Configurable thresholds
    - Alert generation

    Example:
        ```python
        detector = RegressionDetector()

        # Set baseline
        detector.set_baseline({
            "accuracy": 0.95,
            "latency_ms": 250,
            "cost_usd": 0.01,
        })

        # Check for regressions
        report = detector.check({
            "accuracy": 0.88,
            "latency_ms": 400,
            "cost_usd": 0.015,
        })

        if report.has_regression:
            for alert in report.alerts:
                print(f"ALERT: {alert.message}")
        ```
    """

    # Default thresholds (percentage degradation to trigger alert)
    DEFAULT_THRESHOLDS = {
        "accuracy": 0.05,  # 5% drop
        "precision": 0.05,
        "recall": 0.05,
        "f1_score": 0.05,
        "latency_ms": 0.20,  # 20% increase
        "cost_usd": 0.30,  # 30% increase
        "tokens_total": 0.25,
        "passed_rate": 0.05,
    }

    # Metrics where higher is worse
    INVERSE_METRICS = {"latency_ms", "cost_usd", "tokens_total", "error_rate"}

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        baseline_path: str | Path | None = None,
    ):
        """Initialize detector.

        Args:
            thresholds: Custom thresholds per metric
            baseline_path: Path to store baseline
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

        self.baseline: dict[str, float] = {}
        self.baseline_metadata: dict[str, Any] = {}
        self.baseline_path = Path(baseline_path) if baseline_path else None

        if self.baseline_path and self.baseline_path.exists():
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline from file."""
        if self.baseline_path and self.baseline_path.exists():
            try:
                with open(self.baseline_path) as f:
                    data = json.load(f)
                self.baseline = data.get("metrics", {})
                self.baseline_metadata = data.get("metadata", {})
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")

    def _save_baseline(self) -> None:
        """Save baseline to file."""
        if self.baseline_path:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "metrics": self.baseline,
                "metadata": self.baseline_metadata,
            }
            with open(self.baseline_path, "w") as f:
                json.dump(data, f, indent=2)

    def set_baseline(
        self,
        metrics: dict[str, float],
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set baseline metrics.

        Args:
            metrics: Baseline metric values
            run_id: Optional run ID
            metadata: Optional metadata
        """
        self.baseline = dict(metrics)
        self.baseline_metadata = {
            "run_id": run_id,
            "set_at": datetime.now().isoformat(),
            **(metadata or {}),
        }
        self._save_baseline()
        logger.info(f"Baseline set with {len(metrics)} metrics")

    def check(
        self,
        current_metrics: dict[str, float],
        run_id: str | None = None,
    ) -> RegressionReport:
        """Check for regressions against baseline.

        Args:
            current_metrics: Current metric values
            run_id: Optional current run ID

        Returns:
            RegressionReport with any detected regressions
        """
        alerts = []
        metrics_compared = {}

        for metric, current in current_metrics.items():
            if metric not in self.baseline:
                continue

            baseline = self.baseline[metric]
            threshold = self.thresholds.get(metric, 0.10)

            # Calculate degradation
            is_inverse = metric in self.INVERSE_METRICS
            if is_inverse:
                # Higher is worse
                degradation = (current - baseline) / baseline if baseline != 0 else 0
            else:
                # Lower is worse
                degradation = (baseline - current) / baseline if baseline != 0 else 0

            degradation_pct = degradation * 100

            metrics_compared[metric] = {
                "baseline": baseline,
                "current": current,
                "degradation_pct": degradation_pct,
                "threshold_pct": threshold * 100,
            }

            # Check if regression
            if degradation > threshold:
                severity = self._determine_severity(degradation, threshold)
                significance = self._calculate_significance(baseline, current)

                alerts.append(
                    RegressionAlert(
                        metric_name=metric,
                        baseline_value=baseline,
                        current_value=current,
                        degradation_pct=degradation_pct,
                        severity=severity,
                        statistical_significance=significance,
                        message=self._format_alert_message(
                            metric, baseline, current, degradation_pct, severity
                        ),
                    )
                )

        has_regression = len(alerts) > 0
        summary = self._generate_summary(alerts, metrics_compared)

        return RegressionReport(
            timestamp=datetime.now().isoformat(),
            baseline_run=self.baseline_metadata.get("run_id", "unknown"),
            current_run=run_id or "current",
            has_regression=has_regression,
            alerts=alerts,
            metrics_compared=metrics_compared,
            summary=summary,
        )

    def _determine_severity(self, degradation: float, threshold: float) -> str:
        """Determine alert severity."""
        ratio = degradation / threshold

        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

    def _calculate_significance(self, baseline: float, current: float) -> float:
        """Calculate statistical significance (simplified)."""
        # Simple approach - real implementation would use t-test
        if baseline == 0:
            return 0.0

        effect = abs(current - baseline) / baseline

        # Higher effect = higher significance
        if effect > 0.3:
            return 0.99
        elif effect > 0.2:
            return 0.95
        elif effect > 0.1:
            return 0.90
        elif effect > 0.05:
            return 0.80
        else:
            return 0.50

    def _format_alert_message(
        self,
        metric: str,
        baseline: float,
        current: float,
        degradation_pct: float,
        severity: str,
    ) -> str:
        """Format alert message."""
        direction = "increased" if metric in self.INVERSE_METRICS else "decreased"
        return (
            f"[{severity.upper()}] {metric} {direction} by {abs(degradation_pct):.1f}% "
            f"(baseline: {baseline:.4f}, current: {current:.4f})"
        )

    def _generate_summary(
        self,
        alerts: list[RegressionAlert],
        metrics_compared: dict[str, dict],
    ) -> str:
        """Generate summary text."""
        if not alerts:
            return f"No regressions detected across {len(metrics_compared)} metrics."

        critical = sum(1 for a in alerts if a.severity == "critical")
        high = sum(1 for a in alerts if a.severity == "high")
        medium = sum(1 for a in alerts if a.severity == "medium")
        low = sum(1 for a in alerts if a.severity == "low")

        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if high:
            parts.append(f"{high} high")
        if medium:
            parts.append(f"{medium} medium")
        if low:
            parts.append(f"{low} low")

        return f"Detected {len(alerts)} regressions ({', '.join(parts)}) across {len(metrics_compared)} metrics."

    def update_baseline_if_better(
        self,
        current_metrics: dict[str, float],
        primary_metric: str = "accuracy",
        run_id: str | None = None,
    ) -> bool:
        """Update baseline if current is better.

        Args:
            current_metrics: Current metrics
            primary_metric: Metric to use for comparison
            run_id: Optional run ID

        Returns:
            True if baseline was updated
        """
        if not self.baseline:
            self.set_baseline(current_metrics, run_id)
            return True

        current = current_metrics.get(primary_metric, 0)
        baseline = self.baseline.get(primary_metric, 0)

        is_inverse = primary_metric in self.INVERSE_METRICS
        is_better = current < baseline if is_inverse else current > baseline

        if is_better:
            self.set_baseline(current_metrics, run_id)
            logger.info(f"Baseline updated: {primary_metric} improved from {baseline} to {current}")
            return True

        return False


class CIRegressionChecker:
    """CI/CD integration for regression checking.

    For use in CI pipelines to fail builds on regressions.
    """

    def __init__(
        self,
        detector: RegressionDetector,
        fail_on_severity: str = "medium",
    ):
        """Initialize checker.

        Args:
            detector: RegressionDetector instance
            fail_on_severity: Minimum severity to fail CI
        """
        self.detector = detector
        self.fail_on_severity = fail_on_severity
        self.severity_order = ["low", "medium", "high", "critical"]

    def check_and_report(
        self,
        current_metrics: dict[str, float],
        run_id: str | None = None,
    ) -> tuple[bool, str]:
        """Check regressions and report for CI.

        Args:
            current_metrics: Current metrics
            run_id: Optional run ID

        Returns:
            Tuple of (should_fail, report_text)
        """
        report = self.detector.check(current_metrics, run_id)

        fail_idx = self.severity_order.index(self.fail_on_severity)
        should_fail = any(self.severity_order.index(a.severity) >= fail_idx for a in report.alerts)

        lines = [
            "# Regression Check Report",
            f"Baseline: {report.baseline_run}",
            f"Current: {report.current_run}",
            "",
            f"**Status: {'FAIL' if should_fail else 'PASS'}**",
            "",
            report.summary,
            "",
        ]

        if report.alerts:
            lines.append("## Alerts")
            for alert in sorted(
                report.alerts,
                key=lambda a: self.severity_order.index(a.severity),
                reverse=True,
            ):
                lines.append(f"- {alert.message}")

        return should_fail, "\n".join(lines)


__all__ = [
    "CIRegressionChecker",
    "RegressionAlert",
    "RegressionDetector",
    "RegressionReport",
]
