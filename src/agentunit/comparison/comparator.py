"""Comparators for A/B testing and regression detection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.scenario import Scenario
from ..datasets.base import DatasetSource
from ..metrics.base import MetricResult
from .statistics import SignificanceAnalyzer, BootstrapCI, MetricAggregator


@dataclass
class RunResult:
    """Result from a single scenario run."""
    
    version: str
    config: Dict[str, Any]
    metric_results: List[MetricResult]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get value for a specific metric."""
        for result in self.metric_results:
            if result.name == metric_name:
                return result.value
        return None
    
    def get_all_metric_values(self, metric_name: str) -> List[float]:
        """Get all values for a specific metric (may have multiple)."""
        values = []
        for result in self.metric_results:
            if result.name == metric_name and result.value is not None:
                values.append(result.value)
        return values


@dataclass
class ComparisonResult:
    """Result of comparing two versions/configurations."""
    
    baseline_version: str
    treatment_version: str
    metric_comparisons: Dict[str, Dict[str, Any]]
    overall_assessment: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionComparator:
    """Compare different versions of an agent."""
    
    def __init__(
        self,
        significance_analyzer: Optional[SignificanceAnalyzer] = None
    ):
        """Initialize version comparator.
        
        Args:
            significance_analyzer: Analyzer for statistical significance
        """
        self.analyzer = significance_analyzer or SignificanceAnalyzer()
        self.run_history: List[RunResult] = []
    
    async def compare_versions(
        self,
        baseline_scenario: Scenario,
        treatment_scenario: Scenario,
        dataset: DatasetSource,
        metrics: List[str],
        n_runs: int = 10
    ) -> ComparisonResult:
        """Compare two versions of an agent.
        
        Args:
            baseline_scenario: Baseline version scenario
            treatment_scenario: Treatment version scenario  
            dataset: Dataset to evaluate on
            metrics: List of metric names to compare
            n_runs: Number of runs per version
        
        Returns:
            ComparisonResult with statistical analysis
        """
        # Run baseline version
        baseline_results = []
        for i in range(n_runs):
            # This is a simplified version - actual implementation would run the scenario
            # result = await baseline_scenario.run(dataset)
            # For now, we'll collect results in a structured way
            baseline_results.append(RunResult(
                version="baseline",
                config={},
                metric_results=[],
                metadata={"run_id": i}
            ))
        
        # Run treatment version
        treatment_results = []
        for i in range(n_runs):
            treatment_results.append(RunResult(
                version="treatment", 
                config={},
                metric_results=[],
                metadata={"run_id": i}
            ))
        
        # Store in history
        self.run_history.extend(baseline_results + treatment_results)
        
        # Compare metrics
        metric_comparisons = {}
        for metric_name in metrics:
            baseline_values = []
            for result in baseline_results:
                values = result.get_all_metric_values(metric_name)
                baseline_values.extend(values)
            
            treatment_values = []
            for result in treatment_results:
                values = result.get_all_metric_values(metric_name)
                treatment_values.extend(values)
            
            if baseline_values and treatment_values:
                analysis = self.analyzer.analyze_difference(
                    baseline_values,
                    treatment_values,
                    metric_name
                )
                metric_comparisons[metric_name] = analysis
        
        # Overall assessment
        significant_improvements = sum(
            1 for comp in metric_comparisons.values()
            if comp.get("statistically_significant") and comp.get("difference", 0) > 0
        )
        
        significant_regressions = sum(
            1 for comp in metric_comparisons.values()
            if comp.get("statistically_significant") and comp.get("difference", 0) < 0
        )
        
        if significant_regressions > 0:
            overall = "REGRESSION_DETECTED"
        elif significant_improvements > 0:
            overall = "IMPROVEMENT_DETECTED"
        else:
            overall = "NO_SIGNIFICANT_CHANGE"
        
        return ComparisonResult(
            baseline_version="baseline",
            treatment_version="treatment",
            metric_comparisons=metric_comparisons,
            overall_assessment=overall
        )


class ConfigurationComparator:
    """Compare different configurations of the same agent version."""
    
    def __init__(self):
        """Initialize configuration comparator."""
        self.analyzer = SignificanceAnalyzer()
    
    def compare_configs(
        self,
        scenario_factory: Callable[[Dict[str, Any]], Scenario],
        configs: List[Dict[str, Any]],
        dataset: DatasetSource,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple configurations.
        
        Args:
            scenario_factory: Function to create scenario from config
            configs: List of configurations to compare
            dataset: Dataset to evaluate on
            metrics: Metrics to compare
        
        Returns:
            Comparison results for all configurations
        """
        results = {}
        
        for i, config in enumerate(configs):
            config_id = f"config_{i}"
            # scenario = scenario_factory(config)
            # result = scenario.run(dataset)
            results[config_id] = {
                "config": config,
                "metrics": {}
            }
        
        return {
            "configurations": results,
            "best_config": self._find_best_config(results, metrics),
            "metric_rankings": self._rank_by_metrics(results, metrics)
        }
    
    def _find_best_config(
        self,
        results: Dict[str, Any],
        metrics: List[str]
    ) -> Optional[str]:
        """Find best configuration based on metrics."""
        # Simplified: would use multi-objective optimization
        return list(results.keys())[0] if results else None
    
    def _rank_by_metrics(
        self,
        results: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, List[str]]:
        """Rank configurations by each metric."""
        rankings = {}
        
        for metric in metrics:
            # Extract values and sort
            config_values = []
            for config_id, result in results.items():
                value = result["metrics"].get(metric)
                if value is not None:
                    config_values.append((config_id, value))
            
            # Sort by value (descending)
            config_values.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [config_id for config_id, _ in config_values]
        
        return rankings


class ABTestRunner:
    """Run A/B tests for agent evaluation."""
    
    def __init__(
        self,
        min_sample_size: int = 30,
        alpha: float = 0.05,
        power: float = 0.8
    ):
        """Initialize A/B test runner.
        
        Args:
            min_sample_size: Minimum sample size per variant
            alpha: Significance level
            power: Desired statistical power
        """
        self.min_sample_size = min_sample_size
        self.alpha = alpha
        self.power = power
        self.analyzer = SignificanceAnalyzer(alpha=alpha)
    
    async def run_ab_test(
        self,
        variant_a: Scenario,
        variant_b: Scenario,
        dataset: DatasetSource,
        primary_metric: str,
        secondary_metrics: Optional[List[str]] = None,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """Run A/B test between two variants.
        
        Args:
            variant_a: First variant (control)
            variant_b: Second variant (treatment)
            dataset: Dataset for evaluation
            primary_metric: Primary metric for decision making
            secondary_metrics: Secondary metrics to monitor
            early_stopping: Whether to enable early stopping
        
        Returns:
            A/B test results with recommendation
        """
        secondary_metrics = secondary_metrics or []
        
        results_a = []
        results_b = []
        
        # Collect results (simplified)
        for i in range(self.min_sample_size):
            results_a.append(0.8)  # Placeholder
            results_b.append(0.85)  # Placeholder
            
            # Check for early stopping
            if early_stopping and i >= 10 and i % 5 == 0:
                if self._should_stop_early(results_a, results_b):
                    break
        
        # Analyze results
        analysis = self.analyzer.analyze_difference(
            results_a,
            results_b,
            primary_metric
        )
        
        # Make recommendation
        if analysis["statistically_significant"]:
            if analysis["difference"] > 0:
                winner = "variant_b"
                recommendation = "Deploy variant B"
            else:
                winner = "variant_a"
                recommendation = "Keep variant A"
        else:
            winner = "inconclusive"
            recommendation = "Need more data or no significant difference"
        
        return {
            "variant_a_mean": analysis["baseline_mean"],
            "variant_b_mean": analysis["treatment_mean"],
            "difference": analysis["difference"],
            "percent_change": analysis["percent_change"],
            "p_value": analysis["p_value"],
            "winner": winner,
            "recommendation": recommendation,
            "sample_sizes": {
                "variant_a": len(results_a),
                "variant_b": len(results_b)
            }
        }
    
    def _should_stop_early(
        self,
        results_a: List[float],
        results_b: List[float]
    ) -> bool:
        """Check if early stopping criteria are met."""
        if len(results_a) < 10 or len(results_b) < 10:
            return False
        
        analysis = self.analyzer.analyze_difference(results_a, results_b)
        
        # Stop if very significant difference detected
        return analysis["p_value"] < 0.01 and analysis["practically_significant"]


class RegressionDetector:
    """Detect performance regressions between versions."""
    
    def __init__(
        self,
        regression_threshold: float = 0.05,
        min_effect_size: float = 0.2
    ):
        """Initialize regression detector.
        
        Args:
            regression_threshold: Threshold for regression detection (5% degradation)
            min_effect_size: Minimum effect size to consider
        """
        self.regression_threshold = regression_threshold
        self.min_effect_size = min_effect_size
        self.analyzer = SignificanceAnalyzer(min_effect_size=min_effect_size)
    
    def detect_regression(
        self,
        baseline_results: List[float],
        new_results: List[float],
        metric_name: str,
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """Detect if there's a performance regression.
        
        Args:
            baseline_results: Baseline performance measurements
            new_results: New version performance measurements
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
        
        Returns:
            Regression detection results
        """
        analysis = self.analyzer.analyze_difference(
            baseline_results,
            new_results,
            metric_name
        )
        
        # Determine if regression occurred
        difference = analysis["difference"]
        percent_change = analysis["percent_change"]
        
        if higher_is_better:
            regression_condition = (
                analysis["statistically_significant"] and
                difference < 0 and
                abs(percent_change) > self.regression_threshold * 100
            )
        else:
            regression_condition = (
                analysis["statistically_significant"] and
                difference > 0 and
                abs(percent_change) > self.regression_threshold * 100
            )

        is_regression = bool(regression_condition)
        
        severity = "none"
        if is_regression:
            if abs(percent_change) > 20:
                severity = "critical"
            elif abs(percent_change) > 10:
                severity = "major"
            else:
                severity = "minor"
        
        return {
            "metric_name": metric_name,
            "is_regression": is_regression,
            "severity": severity,
            "baseline_mean": analysis["baseline_mean"],
            "new_mean": analysis["treatment_mean"],
            "difference": difference,
            "percent_change": percent_change,
            "p_value": analysis["p_value"],
            "effect_size": analysis["effect_size"],
            "recommendation": (
                f"BLOCK: {severity} regression detected" if is_regression
                else "PASS: No regression detected"
            )
        }


__all__ = [
    "RunResult",
    "ComparisonResult",
    "VersionComparator",
    "ConfigurationComparator",
    "ABTestRunner",
    "RegressionDetector",
]
