"""Statistical utilities for rigorous evaluation analysis.

This module provides statistical significance testing, confidence intervals,
and other analysis tools for agent evaluation - addressing the lack of
rigorous statistical validation in current AI agent benchmarks.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results.

    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value of the test
        significant: Whether result is significant at alpha level
        alpha: Significance level used
        effect_size: Effect size measure (if applicable)
        confidence_interval: Tuple of (lower, upper) bounds
        interpretation: Human-readable interpretation
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    interpretation: str = ""


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis.

    Attributes:
        estimate: Point estimate
        std_error: Standard error of estimate
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        confidence_level: Confidence level (e.g., 0.95)
        bootstrap_samples: Number of bootstrap samples used
    """

    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


@dataclass
class ComparisonResult:
    """Result of comparing two systems/methods.

    Attributes:
        system_a: Name of first system
        system_b: Name of second system
        metric: Metric being compared
        mean_a: Mean for system A
        mean_b: Mean for system B
        difference: Mean difference (A - B)
        statistical_test: StatisticalResult from comparison
        winner: Which system is significantly better (or 'tie')
        practical_significance: Whether difference is practically meaningful
    """

    system_a: str
    system_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    statistical_test: StatisticalResult
    winner: str = "tie"
    practical_significance: bool = False


class StatisticalAnalyzer:
    """Perform statistical analysis on evaluation results.

    Addresses GAP: Lack of rigorous statistical validation in agent benchmarks.
    Most papers report only mean accuracy without confidence intervals or
    significance testing.
    """

    def __init__(self, alpha: float = 0.05, random_seed: int | None = None):
        """Initialize statistical analyzer.

        Args:
            alpha: Significance level for hypothesis tests
            random_seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random = random.Random(random_seed)

    def mean(self, values: list[float]) -> float:
        """Calculate mean of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def variance(self, values: list[float], ddof: int = 1) -> float:
        """Calculate variance with degrees of freedom adjustment."""
        if len(values) <= ddof:
            return 0.0
        mean_val = self.mean(values)
        return sum((x - mean_val) ** 2 for x in values) / (len(values) - ddof)

    def std_dev(self, values: list[float], ddof: int = 1) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.variance(values, ddof))

    def std_error(self, values: list[float]) -> float:
        """Calculate standard error of the mean."""
        if len(values) <= 1:
            return 0.0
        return self.std_dev(values) / math.sqrt(len(values))

    def confidence_interval(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for the mean.

        Uses t-distribution approximation for small samples.

        Args:
            values: List of values
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            mean_val = self.mean(values)
            return (mean_val, mean_val)

        mean_val = self.mean(values)
        se = self.std_error(values)
        n = len(values)

        # T-distribution critical value approximation
        # For n > 30, use z-score; otherwise estimate t
        if n > 30:
            z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        else:
            # Approximate t-critical value for common confidence levels
            t_values = {
                (0.95, 5): 2.571,
                (0.95, 10): 2.228,
                (0.95, 20): 2.086,
                (0.95, 30): 2.042,
                (0.99, 5): 4.032,
                (0.99, 10): 3.169,
                (0.99, 20): 2.845,
                (0.99, 30): 2.750,
            }
            # Find closest match
            conf_key = 0.95 if confidence < 0.97 else 0.99
            n_keys = sorted([k[1] for k in t_values if k[0] == conf_key])
            closest_n = min(n_keys, key=lambda x: abs(x - n))
            z = t_values.get((conf_key, closest_n), 2.0)

        margin = z * se
        return (mean_val - margin, mean_val + margin)

    def bootstrap_confidence_interval(
        self,
        values: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        statistic: Callable[[list[float]], float] | None = None,
    ) -> BootstrapResult:
        """Calculate bootstrap confidence interval.

        Non-parametric method that doesn't assume normality.

        Args:
            values: List of values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            statistic: Function to compute statistic (default: mean)

        Returns:
            BootstrapResult with confidence interval
        """
        if statistic is None:
            statistic = self.mean

        if not values:
            return BootstrapResult(
                estimate=0.0,
                std_error=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                confidence_level=confidence,
                bootstrap_samples=n_bootstrap,
            )

        # Original estimate
        original_estimate = statistic(values)

        # Bootstrap resampling
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            sample = [self.random.choice(values) for _ in range(len(values))]
            bootstrap_estimates.append(statistic(sample))

        # Calculate percentile confidence interval
        bootstrap_estimates.sort()
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

        return BootstrapResult(
            estimate=original_estimate,
            std_error=self.std_dev(bootstrap_estimates),
            ci_lower=bootstrap_estimates[max(0, lower_idx)],
            ci_upper=bootstrap_estimates[min(len(bootstrap_estimates) - 1, upper_idx)],
            confidence_level=confidence,
            bootstrap_samples=n_bootstrap,
        )

    def paired_t_test(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> StatisticalResult:
        """Perform paired t-test for matched samples.

        Tests whether the mean difference between paired observations is zero.

        Args:
            values_a: Values from system A
            values_b: Values from system B (same length, paired)

        Returns:
            StatisticalResult with test results
        """
        if len(values_a) != len(values_b):
            raise ValueError("Paired samples must have equal length")

        n = len(values_a)
        if n < 2:
            return StatisticalResult(
                test_name="paired_t_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for paired t-test",
            )

        # Calculate differences
        differences = [a - b for a, b in zip(values_a, values_b, strict=False)]

        mean_diff = self.mean(differences)
        std_diff = self.std_dev(differences)
        se_diff = std_diff / math.sqrt(n)

        if se_diff == 0:
            t_stat = 0.0
        else:
            t_stat = mean_diff / se_diff

        # Approximate p-value using normal distribution for large n
        # For small n, this is an approximation
        p_value = self._approximate_t_pvalue(abs(t_stat), n - 1)

        # Effect size (Cohen's d for paired samples)
        effect_size = mean_diff / std_diff if std_diff > 0 else 0.0

        significant = p_value < self.alpha

        interpretation = self._interpret_effect_size(effect_size)
        if significant:
            direction = "higher" if mean_diff > 0 else "lower"
            interpretation = (
                f"Significant difference (p={p_value:.4f}). A is {direction}. {interpretation}"
            )
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"

        return StatisticalResult(
            test_name="paired_t_test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=self.confidence_interval(differences),
            interpretation=interpretation,
        )

    def independent_t_test(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> StatisticalResult:
        """Perform independent samples t-test.

        Tests whether means of two independent groups are equal.
        Uses Welch's t-test (doesn't assume equal variances).

        Args:
            values_a: Values from group A
            values_b: Values from group B

        Returns:
            StatisticalResult with test results
        """
        n_a, n_b = len(values_a), len(values_b)

        if n_a < 2 or n_b < 2:
            return StatisticalResult(
                test_name="independent_t_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for t-test",
            )

        mean_a = self.mean(values_a)
        mean_b = self.mean(values_b)
        var_a = self.variance(values_a)
        var_b = self.variance(values_b)

        # Welch's t-test
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean_a - mean_b) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else n_a + n_b - 2

        p_value = self._approximate_t_pvalue(abs(t_stat), df)

        # Pooled effect size (Cohen's d)
        pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        significant = p_value < self.alpha

        interpretation = self._interpret_effect_size(effect_size)
        if significant:
            direction = "higher" if mean_a > mean_b else "lower"
            interpretation = (
                f"Significant difference (p={p_value:.4f}). A is {direction}. {interpretation}"
            )
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"

        return StatisticalResult(
            test_name="independent_t_test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=interpretation,
        )

    def wilcoxon_signed_rank(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> StatisticalResult:
        """Perform Wilcoxon signed-rank test (non-parametric paired test).

        Non-parametric alternative to paired t-test.

        Args:
            values_a: Values from system A
            values_b: Values from system B (paired)

        Returns:
            StatisticalResult with test results
        """
        if len(values_a) != len(values_b):
            raise ValueError("Paired samples must have equal length")

        n = len(values_a)
        if n < 5:
            return StatisticalResult(
                test_name="wilcoxon_signed_rank",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for Wilcoxon test",
            )

        # Calculate differences and their absolute values
        differences = [a - b for a, b in zip(values_a, values_b, strict=False)]

        # Remove zero differences
        nonzero = [(d, abs(d)) for d in differences if d != 0]
        if not nonzero:
            return StatisticalResult(
                test_name="wilcoxon_signed_rank",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="No differences between samples",
            )

        # Rank by absolute value
        ranked = sorted(enumerate(nonzero), key=lambda x: x[1][1])
        ranks = [0] * len(nonzero)
        for i, (orig_idx, _) in enumerate(ranked):
            ranks[orig_idx] = i + 1

        # Sum of positive and negative ranks
        w_plus = sum(r for r, (d, _) in zip(ranks, nonzero, strict=False) if d > 0)
        w_minus = sum(r for r, (d, _) in zip(ranks, nonzero, strict=False) if d < 0)

        w_stat = min(w_plus, w_minus)
        n_eff = len(nonzero)

        # Normal approximation for large n
        mean_w = n_eff * (n_eff + 1) / 4
        std_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)

        if std_w == 0:
            z_stat = 0.0
        else:
            z_stat = (w_stat - mean_w) / std_w

        # Two-tailed p-value from normal approximation
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))

        significant = p_value < self.alpha

        # Effect size (matched-pairs rank biserial correlation)
        r_effect = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0

        interpretation = f"{'Significant' if significant else 'No significant'} difference "
        interpretation += f"(W={w_stat:.1f}, p={p_value:.4f})"

        return StatisticalResult(
            test_name="wilcoxon_signed_rank",
            statistic=w_stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            effect_size=r_effect,
            interpretation=interpretation,
        )

    def mann_whitney_u(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric independent test).

        Non-parametric alternative to independent t-test.

        Args:
            values_a: Values from group A
            values_b: Values from group B

        Returns:
            StatisticalResult with test results
        """
        n_a, n_b = len(values_a), len(values_b)

        if n_a < 2 or n_b < 2:
            return StatisticalResult(
                test_name="mann_whitney_u",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data for Mann-Whitney test",
            )

        # Combine and rank
        combined = [(v, "a") for v in values_a] + [(v, "b") for v in values_b]
        combined.sort(key=lambda x: x[0])

        # Assign ranks (handling ties with average rank)
        ranks = {}
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[id(combined[k])] = avg_rank
            i = j

        # Sum of ranks for group A
        r_a = sum(ranks[id((v, "a"))] for v in values_a if id((v, "a")) in ranks)

        # Approximate: try matching by value
        combined_lookup = combined.copy()
        r_a = 0
        for v in values_a:
            for idx, (cv, label) in enumerate(combined_lookup):
                if cv == v and label == "a":
                    r_a += idx + 1
                    combined_lookup[idx] = (cv, "used")
                    break

        # U statistic
        u_a = r_a - n_a * (n_a + 1) / 2
        u_b = n_a * n_b - u_a
        u_stat = min(u_a, u_b)

        # Normal approximation
        mean_u = n_a * n_b / 2
        std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

        if std_u == 0:
            z_stat = 0.0
        else:
            z_stat = (u_stat - mean_u) / std_u

        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))

        significant = p_value < self.alpha

        # Effect size (rank-biserial correlation)
        r_effect = 1 - (2 * u_stat) / (n_a * n_b)

        interpretation = f"{'Significant' if significant else 'No significant'} difference "
        interpretation += f"(U={u_stat:.1f}, p={p_value:.4f})"

        return StatisticalResult(
            test_name="mann_whitney_u",
            statistic=u_stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            effect_size=r_effect,
            interpretation=interpretation,
        )

    def mcnemar_test(
        self,
        successes_a: list[bool],
        successes_b: list[bool],
    ) -> StatisticalResult:
        """Perform McNemar's test for paired binary outcomes.

        Commonly used for comparing classifiers on the same test set.

        Args:
            successes_a: Binary outcomes for system A
            successes_b: Binary outcomes for system B (paired)

        Returns:
            StatisticalResult with test results
        """
        if len(successes_a) != len(successes_b):
            raise ValueError("Paired samples must have equal length")

        # Count discordant pairs
        b = sum(
            1 for a, b in zip(successes_a, successes_b, strict=False) if a and not b
        )  # A correct, B wrong
        c = sum(
            1 for a, b in zip(successes_a, successes_b, strict=False) if not a and b
        )  # A wrong, B correct

        if b + c == 0:
            return StatisticalResult(
                test_name="mcnemar_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
                interpretation="No discordant pairs",
            )

        # Chi-square statistic with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0

        # P-value from chi-square distribution (1 df)
        p_value = 1 - self._chi2_cdf(chi2, 1)

        significant = p_value < self.alpha

        interpretation = f"{'Significant' if significant else 'No significant'} difference "
        interpretation += f"(chi2={chi2:.2f}, p={p_value:.4f}). "
        interpretation += f"A better on {b} cases, B better on {c} cases."

        return StatisticalResult(
            test_name="mcnemar_test",
            statistic=chi2,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interpretation,
        )

    def compare_systems(
        self,
        results_a: list[float | bool],
        results_b: list[float | bool],
        system_a_name: str = "System A",
        system_b_name: str = "System B",
        metric_name: str = "score",
        paired: bool = True,
    ) -> ComparisonResult:
        """Compare two systems with appropriate statistical tests.

        Automatically selects appropriate test based on data type.

        Args:
            results_a: Results from system A
            results_b: Results from system B
            system_a_name: Name of system A
            system_b_name: Name of system B
            metric_name: Name of the metric
            paired: Whether samples are paired

        Returns:
            ComparisonResult with detailed comparison
        """
        # Convert to floats if boolean
        if results_a and isinstance(results_a[0], bool):
            values_a = [1.0 if x else 0.0 for x in results_a]
            values_b = [1.0 if x else 0.0 for x in results_b]
            is_binary = True
        else:
            values_a = [float(x) for x in results_a]
            values_b = [float(x) for x in results_b]
            is_binary = False

        mean_a = self.mean(values_a)
        mean_b = self.mean(values_b)
        difference = mean_a - mean_b

        # Select appropriate test
        if is_binary and paired:
            stat_result = self.mcnemar_test(
                [bool(x) for x in results_a],
                [bool(x) for x in results_b],
            )
        elif paired:
            stat_result = self.paired_t_test(values_a, values_b)
        else:
            stat_result = self.independent_t_test(values_a, values_b)

        # Determine winner
        if stat_result.significant:
            winner = system_a_name if difference > 0 else system_b_name
        else:
            winner = "tie"

        # Practical significance (e.g., > 2% difference for accuracy)
        practical_threshold = 0.02 if is_binary else 0.1 * max(abs(mean_a), abs(mean_b), 0.01)
        practical_significance = abs(difference) > practical_threshold

        return ComparisonResult(
            system_a=system_a_name,
            system_b=system_b_name,
            metric=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            difference=difference,
            statistical_test=stat_result,
            winner=winner,
            practical_significance=practical_significance,
        )

    def _approximate_t_pvalue(self, t: float, df: float) -> float:
        """Approximate p-value for t-distribution using normal approximation."""
        # For large df, t approaches normal
        if df > 100:
            return 2 * (1 - self._normal_cdf(abs(t)))

        # Rough approximation for smaller df
        adjustment = 1 + (1 / (4 * df))
        z_approx = t / adjustment
        return 2 * (1 - self._normal_cdf(abs(z_approx)))

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return (1 + math.erf(x / math.sqrt(2))) / 2

    def _chi2_cdf(self, x: float, df: int) -> float:
        """Approximate chi-square CDF."""
        if x <= 0:
            return 0.0

        # For df=1, use normal approximation
        if df == 1:
            return 2 * self._normal_cdf(math.sqrt(x)) - 1

        # General approximation using Wilson-Hilferty transformation
        z = (((x / df) ** (1 / 3)) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return self._normal_cdf(z)

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect size."
        elif abs_d < 0.5:
            return "Small effect size."
        elif abs_d < 0.8:
            return "Medium effect size."
        else:
            return "Large effect size."


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results across systems.

    Addresses GAP: Need for reproducible, rigorous benchmark analysis.
    """

    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """Initialize benchmark analyzer.

        Args:
            alpha: Significance level
            random_seed: Random seed for reproducibility
        """
        self.stats = StatisticalAnalyzer(alpha=alpha, random_seed=random_seed)
        self.results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def add_result(
        self,
        system_name: str,
        benchmark: str,
        task_id: str,
        score: float,
        passed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a benchmark result.

        Args:
            system_name: Name of the system
            benchmark: Benchmark name (e.g., 'gaia', 'arena')
            task_id: Task identifier
            score: Numeric score
            passed: Whether task passed
            metadata: Additional metadata
        """
        self.results[system_name][benchmark].append(
            {
                "task_id": task_id,
                "score": score,
                "passed": passed,
                "metadata": metadata or {},
            }
        )

    def get_summary(self, system_name: str, benchmark: str) -> dict[str, Any]:
        """Get summary statistics for a system on a benchmark.

        Args:
            system_name: Name of the system
            benchmark: Benchmark name

        Returns:
            Dictionary with summary statistics
        """
        results = self.results.get(system_name, {}).get(benchmark, [])

        if not results:
            return {
                "system": system_name,
                "benchmark": benchmark,
                "n_tasks": 0,
                "n_passed": 0,
                "pass_rate": 0.0,
                "mean_score": 0.0,
                "ci_95": (0.0, 0.0),
            }

        scores = [r["score"] for r in results]
        passed = [r["passed"] for r in results]

        n_tasks = len(results)
        n_passed = sum(passed)
        pass_rate = n_passed / n_tasks if n_tasks > 0 else 0.0

        mean_score = self.stats.mean(scores)
        ci_95 = self.stats.confidence_interval(scores, 0.95)

        bootstrap = self.stats.bootstrap_confidence_interval(
            [1.0 if p else 0.0 for p in passed],
            n_bootstrap=1000,
            confidence=0.95,
        )

        return {
            "system": system_name,
            "benchmark": benchmark,
            "n_tasks": n_tasks,
            "n_passed": n_passed,
            "pass_rate": pass_rate,
            "pass_rate_ci": (bootstrap.ci_lower, bootstrap.ci_upper),
            "mean_score": mean_score,
            "score_ci_95": ci_95,
        }

    def compare_systems(
        self,
        system_a: str,
        system_b: str,
        benchmark: str,
    ) -> ComparisonResult:
        """Compare two systems on a benchmark.

        Args:
            system_a: First system name
            system_b: Second system name
            benchmark: Benchmark name

        Returns:
            ComparisonResult with statistical analysis
        """
        results_a = self.results.get(system_a, {}).get(benchmark, [])
        results_b = self.results.get(system_b, {}).get(benchmark, [])

        if not results_a or not results_b:
            raise ValueError(f"Missing results for comparison on {benchmark}")

        # Match by task_id for paired comparison
        task_ids_a = {r["task_id"]: r for r in results_a}
        task_ids_b = {r["task_id"]: r for r in results_b}

        common_tasks = set(task_ids_a.keys()) & set(task_ids_b.keys())

        if common_tasks:
            # Paired comparison
            scores_a = [task_ids_a[t]["score"] for t in sorted(common_tasks)]
            scores_b = [task_ids_b[t]["score"] for t in sorted(common_tasks)]
            paired = True
        else:
            # Unpaired comparison
            scores_a = [r["score"] for r in results_a]
            scores_b = [r["score"] for r in results_b]
            paired = False

        return self.stats.compare_systems(
            scores_a,
            scores_b,
            system_a,
            system_b,
            f"{benchmark}_score",
            paired=paired,
        )

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive analysis report.

        Returns:
            Dictionary with full benchmark analysis
        """
        report = {
            "systems": [],
            "benchmarks": set(),
            "summaries": [],
            "comparisons": [],
        }

        # Get all systems and benchmarks
        for system_name, benchmarks in self.results.items():
            report["systems"].append(system_name)
            for benchmark in benchmarks:
                report["benchmarks"].add(benchmark)

        report["benchmarks"] = list(report["benchmarks"])

        # Generate summaries
        for system in report["systems"]:
            for benchmark in report["benchmarks"]:
                summary = self.get_summary(system, benchmark)
                if summary["n_tasks"] > 0:
                    report["summaries"].append(summary)

        # Generate pairwise comparisons
        systems = report["systems"]
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                for benchmark in report["benchmarks"]:
                    try:
                        comparison = self.compare_systems(systems[i], systems[j], benchmark)
                        report["comparisons"].append(
                            {
                                "system_a": comparison.system_a,
                                "system_b": comparison.system_b,
                                "benchmark": benchmark,
                                "mean_a": comparison.mean_a,
                                "mean_b": comparison.mean_b,
                                "difference": comparison.difference,
                                "p_value": comparison.statistical_test.p_value,
                                "significant": comparison.statistical_test.significant,
                                "winner": comparison.winner,
                                "effect_size": comparison.statistical_test.effect_size,
                            }
                        )
                    except ValueError:
                        continue

        return report


__all__ = [
    "BenchmarkAnalyzer",
    "BootstrapResult",
    "ComparisonResult",
    "StatisticalAnalyzer",
    "StatisticalResult",
]
