"""Tests for A/B testing and regression detection."""

from datetime import datetime, timezone

import numpy as np
import pytest

from agentunit import comparison
from agentunit.comparison.comparator import ABTestRunner, RegressionDetector
from agentunit.comparison.reports import ComparisonReport, RegressionReport
from agentunit.comparison.statistics import (
    BootstrapCI,
    MetricAggregator,
    SignificanceAnalyzer,
    StatisticalTest,
)


def test_comparison_imports():
    """Test that comparison module can be imported."""
    assert hasattr(comparison, "__all__")
    assert "VersionComparator" in comparison.__all__
    assert "RegressionDetector" in comparison.__all__


def test_bootstrap_ci():
    """Test Bootstrap confidence interval estimation."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    bootstrap = BootstrapCI(n_iterations=1000, confidence_level=0.95, random_seed=42)

    ci = bootstrap.estimate(data, statistic=np.mean)

    assert ci.point_estimate == pytest.approx(3.0)
    assert ci.lower_bound < ci.point_estimate < ci.upper_bound
    assert ci.confidence_level == pytest.approx(0.95)
    assert ci.method == "bootstrap_percentile"


def test_statistical_test():
    """Test statistical hypothesis testing."""
    test = StatisticalTest(alpha=0.05)

    sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample2 = [2.0, 3.0, 4.0, 5.0, 6.0]

    result = test.t_test(sample1, sample2)

    assert result.test_name == "t_test_independent"
    assert result.p_value >= 0
    assert result.p_value <= 1
    assert result.alpha == pytest.approx(0.05)


def test_metric_aggregator():
    """Test metric aggregation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    agg_results = MetricAggregator.aggregate(values)

    assert agg_results["mean"] == pytest.approx(3.0)
    assert agg_results["median"] == pytest.approx(3.0)
    assert agg_results["min"] == pytest.approx(1.0)
    assert agg_results["max"] == pytest.approx(5.0)


def test_significance_analyzer():
    """Test significance analysis."""
    analyzer = SignificanceAnalyzer(alpha=0.05, min_effect_size=0.2)

    baseline = [0.8, 0.82, 0.79, 0.81, 0.80] * 10  # 50 samples
    treatment = [0.85, 0.87, 0.84, 0.86, 0.85] * 10  # 50 samples with improvement

    analysis = analyzer.analyze_difference(baseline, treatment, "accuracy")

    assert "baseline_mean" in analysis
    assert "treatment_mean" in analysis
    assert "difference" in analysis
    assert "effect_size" in analysis
    assert "recommendation" in analysis


def test_regression_detector():
    """Test regression detection."""
    detector = RegressionDetector(regression_threshold=0.05, min_effect_size=0.2)

    # No regression case
    baseline = [0.9, 0.91, 0.89, 0.90, 0.91] * 10
    new_version = [0.89, 0.90, 0.88, 0.89, 0.90] * 10  # Slight decrease, not significant

    result = detector.detect_regression(baseline, new_version, "accuracy")

    assert "is_regression" in result
    assert "severity" in result
    assert "recommendation" in result

    # Clear regression case
    baseline_good = [0.9] * 50
    new_version_bad = [0.7] * 50  # 22% degradation

    result_bad = detector.detect_regression(baseline_good, new_version_bad, "accuracy")

    assert result_bad["is_regression"] is True
    assert result_bad["severity"] in ["minor", "major", "critical"]


def test_ab_test_runner():
    """Test A/B testing runner."""
    runner = ABTestRunner(min_sample_size=30, alpha=0.05)

    # Would test actual A/B test run, but requires async
    assert runner.min_sample_size == 30
    assert runner.alpha == pytest.approx(0.05)


def test_comparison_report():
    """Test comparison report generation."""
    report = ComparisonReport(
        title="Test Comparison",
        baseline_id="v1.0",
        treatment_id="v2.0",
        metric_comparisons={
            "accuracy": {
                "baseline_mean": 0.8,
                "treatment_mean": 0.85,
                "difference": 0.05,
                "percent_change": 6.25,
                "p_value": 0.02,
                "statistically_significant": True,
                "recommendation": "Significant improvement detected",
            }
        },
        overall_assessment="IMPROVEMENT_DETECTED",
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )

    markdown = report.to_markdown()
    assert "Test Comparison" in markdown
    assert "v1.0" in markdown
    assert "v2.0" in markdown

    data = report.to_dict()
    assert data["title"] == "Test Comparison"


def test_regression_report():
    """Test regression report generation."""
    report = RegressionReport(
        title="Regression Test",
        baseline_version="v1.0",
        new_version="v2.0",
        regressions=[
            {
                "metric_name": "accuracy",
                "severity": "critical",
                "baseline_mean": 0.9,
                "new_mean": 0.7,
                "difference": -0.2,
                "percent_change": -22.2,
                "recommendation": "BLOCK: critical regression detected",
            }
        ],
        passed=False,
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )

    markdown = report.to_markdown()
    assert "FAIL" in markdown
    assert "critical" in markdown.lower()

    critical = report.get_critical_regressions()
    assert len(critical) == 1

    summary = report.get_summary()
    assert "FAIL" in summary
    assert "1 regression" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
