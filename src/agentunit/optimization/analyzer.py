"""Run analysis for optimization recommendations."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """Result of analyzing a test run.

    Attributes:
        total_cases: Total number of test cases
        passed_cases: Number of passed test cases
        failed_cases: Number of failed test cases
        avg_latency: Average latency across all cases (seconds)
        total_tokens: Total tokens used (if available)
        total_cost: Total cost (if available)
        failure_patterns: Common patterns in failures
        performance_bottlenecks: Identified performance issues
        metric_summary: Summary of all metrics
    """

    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    performance_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    metric_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def avg_cost_per_case(self) -> float:
        """Calculate average cost per case."""
        if self.total_cases == 0:
            return 0.0
        return self.total_cost / self.total_cases


class RunAnalyzer:
    """Analyzes test run results to identify optimization opportunities.

    This analyzer examines:
    - Success/failure rates
    - Latency patterns
    - Token usage
    - Cost efficiency
    - Common failure modes
    - Performance bottlenecks
    """

    def __init__(
        self,
        latency_threshold: float = 5.0,
        cost_threshold: float = 0.1,
    ):
        """Initialize run analyzer.

        Args:
            latency_threshold: Threshold for considering latency a bottleneck (seconds)
            cost_threshold: Threshold for considering cost high (dollars per case)
        """
        self.latency_threshold = latency_threshold
        self.cost_threshold = cost_threshold

    def analyze_run(self, run_data: dict[str, Any]) -> AnalysisResult:
        """Analyze a test run.

        Args:
            run_data: Dictionary containing run results with keys:
                - cases: List of test case results
                - metrics: Metric results
                - metadata: Run metadata

        Returns:
            AnalysisResult with analysis findings
        """
        result = AnalysisResult()

        cases = run_data.get("cases", [])
        result.total_cases = len(cases)

        # Analyze individual cases
        latencies = []
        costs = []
        tokens = []
        failures = []

        for case in cases:
            # Success/failure
            if case.get("passed", False):
                result.passed_cases += 1
            else:
                result.failed_cases += 1
                failures.append(case)

            # Latency
            if "latency" in case:
                latencies.append(case["latency"])

            # Tokens
            if "tokens" in case:
                tokens.append(case["tokens"])

            # Cost
            if "cost" in case:
                costs.append(case["cost"])

        # Calculate averages
        if latencies:
            result.avg_latency = sum(latencies) / len(latencies)
        if tokens:
            result.total_tokens = sum(tokens)
        if costs:
            result.total_cost = sum(costs)

        # Identify failure patterns
        result.failure_patterns = self._analyze_failures(failures)

        # Identify performance bottlenecks
        result.performance_bottlenecks = self._analyze_performance(
            latencies, costs, result.avg_latency, result.avg_cost_per_case
        )

        # Summarize metrics
        result.metric_summary = self._summarize_metrics(run_data.get("metrics", {}))

        return result

    def _analyze_failures(self, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze failure patterns.

        Args:
            failures: List of failed test cases

        Returns:
            List of identified patterns
        """
        if not failures:
            return []

        patterns = []

        # Group by error type
        error_types = defaultdict(list)
        for failure in failures:
            error = failure.get("error", "Unknown")
            error_types[error].append(failure)

        # Identify common patterns
        for error, cases in error_types.items():
            if len(cases) >= 2:  # At least 2 occurrences
                patterns.append(
                    {
                        "type": "recurring_error",
                        "error": error,
                        "count": len(cases),
                        "percentage": len(cases) / len(failures) * 100,
                        "cases": [c.get("id") for c in cases[:5]],  # Sample
                    }
                )

        # Check for timeout patterns
        timeouts = [f for f in failures if "timeout" in str(f.get("error", "")).lower()]
        if timeouts:
            patterns.append(
                {
                    "type": "timeout",
                    "count": len(timeouts),
                    "percentage": len(timeouts) / len(failures) * 100,
                }
            )

        return patterns

    def _analyze_performance(
        self, latencies: list[float], costs: list[float], avg_latency: float, avg_cost: float
    ) -> list[dict[str, Any]]:
        """Analyze performance bottlenecks.

        Args:
            latencies: List of latencies
            costs: List of costs
            avg_latency: Average latency
            avg_cost: Average cost per case

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        # High latency
        if avg_latency > self.latency_threshold:
            bottlenecks.append(
                {
                    "type": "high_latency",
                    "avg_latency": avg_latency,
                    "threshold": self.latency_threshold,
                    "severity": "high" if avg_latency > self.latency_threshold * 2 else "medium",
                }
            )

        # High cost
        if avg_cost > self.cost_threshold:
            bottlenecks.append(
                {
                    "type": "high_cost",
                    "avg_cost": avg_cost,
                    "threshold": self.cost_threshold,
                    "severity": "high" if avg_cost > self.cost_threshold * 2 else "medium",
                }
            )

        # Variable latency (high std dev)
        if latencies and len(latencies) > 1:
            import statistics

            std_dev = statistics.stdev(latencies)
            if std_dev > avg_latency * 0.5:  # > 50% of average
                bottlenecks.append(
                    {
                        "type": "variable_latency",
                        "std_dev": std_dev,
                        "avg": avg_latency,
                        "severity": "medium",
                    }
                )

        return bottlenecks

    def _summarize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Summarize metric results.

        Args:
            metrics: Metric results from run

        Returns:
            Summary dictionary
        """
        summary = {}

        for metric_name, values in metrics.items():
            if not values:
                continue

            # Calculate statistics
            numeric_values = [v for v in values if isinstance(v, int | float)]
            if numeric_values:
                summary[metric_name] = {
                    "avg": sum(numeric_values) / len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "count": len(numeric_values),
                }

        return summary
