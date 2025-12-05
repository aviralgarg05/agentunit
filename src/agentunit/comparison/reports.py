"""Reports for comparison and regression analysis."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import datetime


def _format_numeric(value: Any, fmt: str) -> str:
    """Format numeric values safely, returning fallback strings unchanged."""
    if isinstance(value, Number):
        try:
            return fmt.format(value)
        except Exception:  # pragma: no cover - defensive
            return str(value)

    # Strings that already look like numbers should be formatted if possible
    if isinstance(value, str):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        else:
            try:
                return fmt.format(numeric_value)
            except Exception:  # pragma: no cover - defensive
                return value

    if value is None:
        return "N/A"

    return str(value)


@dataclass
class ComparisonReport:
    """Report for version/configuration comparison."""

    title: str
    baseline_id: str
    treatment_id: str
    metric_comparisons: dict[str, dict[str, Any]]
    overall_assessment: str
    timestamp: datetime
    metadata: dict[str, Any]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"**Generated:** {self.timestamp.isoformat()}",
            "",
            "## Comparison Overview",
            f"- **Baseline:** {self.baseline_id}",
            f"- **Treatment:** {self.treatment_id}",
            f"- **Overall Assessment:** {self.overall_assessment}",
            "",
            "## Metric Comparisons",
            "",
        ]

        for metric_name, comparison in self.metric_comparisons.items():
            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append(
                f"- Baseline Mean: {_format_numeric(comparison.get('baseline_mean', 'N/A'), '{:.4f}')}"
            )
            lines.append(
                f"- Treatment Mean: {_format_numeric(comparison.get('treatment_mean', 'N/A'), '{:.4f}')}"
            )
            lines.append(
                f"- Difference: {_format_numeric(comparison.get('difference', 'N/A'), '{:.4f}')}"
            )
            lines.append(
                f"- Percent Change: {_format_numeric(comparison.get('percent_change', 'N/A'), '{:.2f}')}%"
            )
            lines.append(
                f"- P-value: {_format_numeric(comparison.get('p_value', 'N/A'), '{:.4f}')}"
            )
            lines.append(
                f"- Effect Size: {_format_numeric(comparison.get('effect_size', 'N/A'), '{:.4f}')}"
            )
            lines.append(
                f"- Statistically Significant: {comparison.get('statistically_significant', False)}"
            )
            lines.append(
                f"- Practically Significant: {comparison.get('practically_significant', False)}"
            )
            lines.append(f"- **Recommendation:** {comparison.get('recommendation', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "baseline_id": self.baseline_id,
            "treatment_id": self.treatment_id,
            "metric_comparisons": self.metric_comparisons,
            "overall_assessment": self.overall_assessment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RegressionReport:
    """Report for regression detection."""

    title: str
    baseline_version: str
    new_version: str
    regressions: list[dict[str, Any]]
    passed: bool
    timestamp: datetime
    metadata: dict[str, Any]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        status = "PASS" if self.passed else "FAIL"

        lines = [
            f"# {self.title}",
            "",
            f"**Status:** {status}",
            f"**Generated:** {self.timestamp.isoformat()}",
            "",
            "## Version Information",
            f"- **Baseline:** {self.baseline_version}",
            f"- **New Version:** {self.new_version}",
            "",
            "## Regression Analysis",
            "",
        ]

        if not self.regressions:
            lines.append("No regressions detected.")
        else:
            for regression in self.regressions:
                metric_name = regression.get("metric_name", "Unknown")
                severity = regression.get("severity", "unknown")

                lines.append(f"### {metric_name} - Severity: {severity.upper()}")
                lines.append("")
                lines.append(
                    f"- Baseline Mean: {_format_numeric(regression.get('baseline_mean', 'N/A'), '{:.4f}')}"
                )
                lines.append(
                    f"- New Mean: {_format_numeric(regression.get('new_mean', 'N/A'), '{:.4f}')}"
                )
                lines.append(
                    f"- Difference: {_format_numeric(regression.get('difference', 'N/A'), '{:.4f}')}"
                )
                lines.append(
                    f"- Percent Change: {_format_numeric(regression.get('percent_change', 'N/A'), '{:.2f}')}%"
                )
                lines.append(
                    f"- P-value: {_format_numeric(regression.get('p_value', 'N/A'), '{:.4f}')}"
                )
                lines.append(
                    f"- Effect Size: {_format_numeric(regression.get('effect_size', 'N/A'), '{:.4f}')}"
                )
                lines.append(f"- **Recommendation:** {regression.get('recommendation', 'N/A')}")
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "baseline_version": self.baseline_version,
            "new_version": self.new_version,
            "regressions": self.regressions,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def get_critical_regressions(self) -> list[dict[str, Any]]:
        """Get only critical regressions."""
        return [r for r in self.regressions if r.get("severity") == "critical"]

    def get_summary(self) -> str:
        """Get summary of regression report."""
        if self.passed:
            return f"PASS: No regressions detected in {self.new_version}"

        critical = len(self.get_critical_regressions())
        total = len(self.regressions)

        return f"FAIL: {total} regression(s) detected ({critical} critical) in {self.new_version}"


__all__ = [
    "ComparisonReport",
    "RegressionReport",
]
