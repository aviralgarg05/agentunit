"""Statistical analysis tools for A/B testing and regression detection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback implementations
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n == 0:
                return 0
            mid = n // 2
            if n % 2 == 0:
                return (sorted_data[mid - 1] + sorted_data[mid]) / 2
            return sorted_data[mid]
        
        @staticmethod
        def std(data, ddof=0):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - ddof if ddof else len(data))
            return variance ** 0.5
        
        @staticmethod
        def min(data):
            return min(data) if data else 0
        
        @staticmethod
        def max(data):
            return max(data) if data else 0
        
        @staticmethod
        def percentile(data, q):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n == 0:
                return 0
            k = (n - 1) * q / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            d0 = sorted_data[int(f)] * (c - k)
            d1 = sorted_data[int(c)] * (k - f)
            return d0 + d1
        
        class random:
            @staticmethod
            def choice(data, size, replace=True):
                return [random.choice(data) for _ in range(size)]

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Fallback for t-test using normal approximation
    class scipy_stats:
        @staticmethod
        def ttest_ind(a, b, equal_var=False):
            # Simple t-test implementation
            mean_a = np.mean(a)
            mean_b = np.mean(b)
            std_a = np.std(a, ddof=1)
            std_b = np.std(b, ddof=1)
            n_a = len(a)
            n_b = len(b)
            
            if equal_var:
                pooled_std = math.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
                se = pooled_std * math.sqrt(1/n_a + 1/n_b)
                df = n_a + n_b - 2
            else:
                se = math.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b)
                # Welch-Satterthwaite df
                df = (std_a ** 2 / n_a + std_b ** 2 / n_b) ** 2 / (
                    (std_a ** 2 / n_a) ** 2 / (n_a - 1) + (std_b ** 2 / n_b) ** 2 / (n_b - 1)
                )
            
            t_stat = (mean_a - mean_b) / se if se > 0 else 0
            # Approximate p-value using normal distribution
            p_value = 2 * (1 - (1 + math.erf(abs(t_stat) / math.sqrt(2))) / 2)
            
            class Result:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            
            return Result(t_stat, p_value)
        
        @staticmethod
        def mannwhitneyu(a, b, alternative='two-sided'):
            # Simple rank-based test
            combined = [(x, 0) for x in a] + [(x, 1) for x in b]
            combined.sort()
            
            rank_sum = sum(i + 1 for i, (_, group) in enumerate(combined) if group == 0)
            n_a = len(a)
            n_b = len(b)
            
            u_a = rank_sum - n_a * (n_a + 1) / 2
            mean_u = n_a * n_b / 2
            std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
            
            z = (u_a - mean_u) / std_u if std_u > 0 else 0
            p_value = 2 * (1 - (1 + math.erf(abs(z) / math.sqrt(2))) / 2)
            
            class Result:
                def __init__(self, statistic, pvalue):
                    self.statistic = statistic
                    self.pvalue = pvalue
            
            return Result(u_a, p_value)


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval for a metric."""
    
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    
    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower_bound <= value <= self.upper_bound
    
    def width(self) -> float:
        """Get the width of the confidence interval."""
        return self.upper_bound - self.lower_bound


class BootstrapCI:
    """Bootstrap confidence interval estimation."""
    
    def __init__(
        self, 
        n_iterations: int = 10000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None
    ):
        """Initialize bootstrap CI estimator.
        
        Args:
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def estimate(
        self, 
        data: List[float], 
        statistic: Callable[[List[float]], float] = np.mean
    ) -> ConfidenceInterval:
        """Estimate confidence interval using bootstrap.
        
        Args:
            data: Sample data
            statistic: Function to compute statistic (default: mean)
        
        Returns:
            ConfidenceInterval object
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        n = len(data)
        bootstrap_estimates = []
        
        # Bootstrap resampling
        for _ in range(self.n_iterations):
            sample = np.random.choice(data, size=n, replace=True)
            estimate = statistic(sample)
            bootstrap_estimates.append(estimate)
        
        # Compute percentile confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
        upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
        point_estimate = statistic(data)
        
        return ConfidenceInterval(
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=self.confidence_level,
            method="bootstrap_percentile"
        )


@dataclass
class TestResult:
    """Result of a statistical test."""
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    details: Dict[str, Any]


class StatisticalTest:
    """Statistical hypothesis testing."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical test.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def t_test(
        self, 
        sample1: List[float], 
        sample2: List[float],
        paired: bool = False
    ) -> TestResult:
        """Perform t-test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            paired: Whether samples are paired
        
        Returns:
            TestResult object
        """
        n1 = len(sample1)
        n2 = len(sample2)
        
        if n1 == 0 or n2 == 0:
            raise ValueError("Samples cannot be empty")
        
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        
        if paired:
            if n1 != n2:
                raise ValueError("Paired samples must have same length")
            
            differences = [a - b for a, b in zip(sample1, sample2)]
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            
            t_statistic = mean_diff / (std_diff / math.sqrt(n1))
            df = n1 - 1
        else:
            std1 = np.std(sample1, ddof=1)
            std2 = np.std(sample2, ddof=1)
            
            # Welch's t-test (unequal variances)
            pooled_std = math.sqrt(std1**2 / n1 + std2**2 / n2)
            t_statistic = (mean1 - mean2) / pooled_std
            
            # Welch-Satterthwaite degrees of freedom
            df = ((std1**2 / n1 + std2**2 / n2)**2 /
                  ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1)))
        
        # Approximate p-value using normal distribution for large samples
        # For small samples, this is an approximation
        from scipy.stats import t as t_dist
        p_value = 2 * (1 - t_dist.cdf(abs(t_statistic), df))
        
        return TestResult(
            test_name="t_test_paired" if paired else "t_test_independent",
            statistic=t_statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            details={
                "mean1": mean1,
                "mean2": mean2,
                "df": df,
                "effect_size": abs(mean1 - mean2) / np.std(sample1 + sample2, ddof=1)
            }
        )
    
    def mann_whitney_u(
        self, 
        sample1: List[float], 
        sample2: List[float]
    ) -> TestResult:
        """Perform Mann-Whitney U test (non-parametric).
        
        Args:
            sample1: First sample
            sample2: Second sample
        
        Returns:
            TestResult object
        """
        from scipy.stats import mannwhitneyu
        
        statistic, p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        return TestResult(
            test_name="mann_whitney_u",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            details={
                "median1": np.median(sample1),
                "median2": np.median(sample2)
            }
        )


class MetricAggregator:
    """Aggregate metrics across multiple runs."""
    
    @staticmethod
    def aggregate(
        values: List[float], 
        methods: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Aggregate values using multiple methods.
        
        Args:
            values: List of metric values
            methods: Aggregation methods to use
        
        Returns:
            Dictionary of aggregated values
        """
        if not values:
            return {}
        
        methods = methods or ["mean", "median", "std", "min", "max", "p25", "p75"]
        
        results = {}
        
        if "mean" in methods:
            results["mean"] = np.mean(values)
        if "median" in methods:
            results["median"] = np.median(values)
        if "std" in methods:
            results["std"] = np.std(values, ddof=1)
        if "min" in methods:
            results["min"] = np.min(values)
        if "max" in methods:
            results["max"] = np.max(values)
        if "p25" in methods:
            results["p25"] = np.percentile(values, 25)
        if "p75" in methods:
            results["p75"] = np.percentile(values, 75)
        if "count" in methods:
            results["count"] = len(values)
        
        return results


class SignificanceAnalyzer:
    """Analyze statistical significance of differences."""
    
    def __init__(
        self, 
        alpha: float = 0.05,
        min_effect_size: float = 0.2
    ):
        """Initialize significance analyzer.
        
        Args:
            alpha: Significance level
            min_effect_size: Minimum effect size to consider meaningful
        """
        self.alpha = alpha
        self.min_effect_size = min_effect_size
    
    def analyze_difference(
        self,
        baseline: List[float],
        treatment: List[float],
        metric_name: str = "metric"
    ) -> Dict[str, Any]:
        """Analyze difference between baseline and treatment.
        
        Args:
            baseline: Baseline measurements
            treatment: Treatment measurements
            metric_name: Name of the metric being analyzed
        
        Returns:
            Analysis results dictionary
        """
        # Compute basic statistics
        baseline_mean = np.mean(baseline)
        treatment_mean = np.mean(treatment)
        difference = treatment_mean - baseline_mean
        percent_change = (difference / baseline_mean * 100) if baseline_mean != 0 else 0
        
        # Effect size (Cohen's d)
        pooled_std = np.std(baseline + treatment, ddof=1)
        effect_size = abs(difference) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test
        stat_test = StatisticalTest(alpha=self.alpha)
        test_result = stat_test.t_test(baseline, treatment)
        
        # Bootstrap confidence interval for difference
        bootstrap = BootstrapCI(n_iterations=5000, confidence_level=0.95)
        
        # Combined data for difference estimation
        def mean_diff(indices):
            baseline_sample = [baseline[i % len(baseline)] for i in indices[:len(baseline)]]
            treatment_sample = [treatment[i % len(treatment)] for i in indices[len(baseline):]]
            return np.mean(treatment_sample) - np.mean(baseline_sample)
        
        # Simplified: CI on treatment mean
        treatment_ci = bootstrap.estimate(treatment)
        
        # Determine significance
        statistically_significant = test_result.significant
        practically_significant = effect_size >= self.min_effect_size
        
        return {
            "metric_name": metric_name,
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "difference": difference,
            "percent_change": percent_change,
            "effect_size": effect_size,
            "statistically_significant": statistically_significant,
            "practically_significant": practically_significant,
            "p_value": test_result.p_value,
            "confidence_interval": {
                "lower": treatment_ci.lower_bound,
                "upper": treatment_ci.upper_bound,
                "confidence_level": treatment_ci.confidence_level
            },
            "recommendation": self._generate_recommendation(
                statistically_significant,
                practically_significant,
                difference
            )
        }
    
    def _generate_recommendation(
        self,
        stat_sig: bool,
        pract_sig: bool,
        difference: float
    ) -> str:
        """Generate recommendation based on significance."""
        if stat_sig and pract_sig:
            direction = "improvement" if difference > 0 else "degradation"
            return f"Significant {direction} detected. Action recommended."
        elif stat_sig and not pract_sig:
            return "Statistically significant but effect size is small. Monitor."
        elif not stat_sig and pract_sig:
            return "Large effect size but not statistically significant. Need more data."
        else:
            return "No significant difference detected."


__all__ = [
    "ConfidenceInterval",
    "BootstrapCI",
    "TestResult",
    "StatisticalTest",
    "MetricAggregator",
    "SignificanceAnalyzer",
]
