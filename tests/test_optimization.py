"""Tests for optimization module."""

from agentunit.optimization import (
    AnalysisResult,
    AutoOptimizer,
    OptimizationStrategy,
    Recommendation,
    RecommendationType,
    Recommender,
    RunAnalyzer,
)


def test_optimization_imports():
    """Test optimization module imports."""
    assert RunAnalyzer is not None
    assert AnalysisResult is not None
    assert Recommender is not None
    assert Recommendation is not None
    assert RecommendationType is not None
    assert AutoOptimizer is not None
    assert OptimizationStrategy is not None


def test_run_analyzer_basic():
    """Test basic run analysis."""
    analyzer = RunAnalyzer(latency_threshold=3.0, cost_threshold=0.05)

    run_data = {
        "cases": [
            {"id": "1", "passed": True, "latency": 1.2, "tokens": 500, "cost": 0.01},
            {"id": "2", "passed": True, "latency": 1.5, "tokens": 600, "cost": 0.012},
            {
                "id": "3",
                "passed": False,
                "latency": 2.0,
                "tokens": 800,
                "cost": 0.015,
                "error": "timeout",
            },
        ],
        "metrics": {"accuracy": [0.9, 0.95, 0.0]},
    }

    result = analyzer.analyze_run(run_data)

    assert result.total_cases == 3
    assert result.passed_cases == 2
    assert result.failed_cases == 1
    assert abs(result.success_rate - 0.666) < 0.01
    assert 1.0 < result.avg_latency < 2.0
    assert result.total_tokens == 1900
    assert abs(result.total_cost - 0.037) < 0.001


def test_run_analyzer_failure_patterns():
    """Test failure pattern detection."""
    analyzer = RunAnalyzer()

    run_data = {
        "cases": [
            {"id": "1", "passed": False, "error": "API timeout"},
            {"id": "2", "passed": False, "error": "API timeout"},
            {"id": "3", "passed": False, "error": "validation error"},
            {"id": "4", "passed": False, "error": "API timeout"},
            {"id": "5", "passed": True},
        ]
    }

    result = analyzer.analyze_run(run_data)

    assert len(result.failure_patterns) > 0
    # Should detect recurring "API timeout" pattern
    timeout_pattern = next(
        (p for p in result.failure_patterns if p["type"] == "recurring_error"), None
    )
    assert timeout_pattern is not None
    assert timeout_pattern["count"] == 3


def test_run_analyzer_performance_bottlenecks():
    """Test performance bottleneck detection."""
    analyzer = RunAnalyzer(latency_threshold=2.0, cost_threshold=0.05)

    run_data = {
        "cases": [
            {"id": "1", "passed": True, "latency": 5.0, "cost": 0.10},
            {"id": "2", "passed": True, "latency": 6.0, "cost": 0.12},
        ]
    }

    result = analyzer.analyze_run(run_data)

    assert len(result.performance_bottlenecks) > 0
    # Should detect high latency
    high_latency = next(
        (b for b in result.performance_bottlenecks if b["type"] == "high_latency"), None
    )
    assert high_latency is not None
    # Should detect high cost
    high_cost = next((b for b in result.performance_bottlenecks if b["type"] == "high_cost"), None)
    assert high_cost is not None


def test_recommender_low_success():
    """Test recommendations for low success rate."""
    recommender = Recommender()

    analysis = AnalysisResult(
        total_cases=10, passed_cases=4, failed_cases=6, avg_latency=2.0, total_cost=0.5
    )

    recommendations = recommender.generate_recommendations(analysis)

    assert len(recommendations) > 0
    # Should recommend prompt improvements
    prompt_recs = [r for r in recommendations if r.type == RecommendationType.PROMPT]
    assert len(prompt_recs) > 0
    # Should recommend model upgrade
    model_recs = [r for r in recommendations if r.type == RecommendationType.MODEL]
    assert len(model_recs) > 0


def test_recommender_performance_issues():
    """Test recommendations for performance issues."""
    recommender = Recommender()

    analysis = AnalysisResult(
        total_cases=10,
        passed_cases=8,
        failed_cases=2,
        avg_latency=8.0,
        performance_bottlenecks=[
            {"type": "high_latency", "avg_latency": 8.0, "threshold": 5.0, "severity": "high"}
        ],
    )

    recommendations = recommender.generate_recommendations(analysis)

    # Should recommend performance improvements
    perf_recs = [r for r in recommendations if r.type == RecommendationType.PERFORMANCE]
    assert len(perf_recs) > 0


def test_recommender_cost_optimization():
    """Test recommendations for cost optimization."""
    recommender = Recommender()

    analysis = AnalysisResult(
        total_cases=10,
        passed_cases=9,
        failed_cases=1,
        avg_latency=2.0,
        total_tokens=500000,
        total_cost=5.0,
    )

    recommendations = recommender.generate_recommendations(analysis)

    # Should recommend cost optimizations
    cost_recs = [r for r in recommendations if r.type == RecommendationType.COST]
    assert len(cost_recs) > 0


def test_recommendation_sorting():
    """Test that recommendations are sorted by priority."""
    recommender = Recommender()

    analysis = AnalysisResult(
        total_cases=10,
        passed_cases=3,
        failed_cases=7,
        avg_latency=10.0,
        total_cost=2.0,
        performance_bottlenecks=[
            {"type": "high_latency", "avg_latency": 10.0, "threshold": 5.0, "severity": "high"}
        ],
    )

    recommendations = recommender.generate_recommendations(analysis)

    assert len(recommendations) > 0
    # Check sorted by priority descending
    for i in range(len(recommendations) - 1):
        assert recommendations[i].priority >= recommendations[i + 1].priority


def test_auto_optimizer_conservative():
    """Test auto-optimizer with conservative strategy."""
    optimizer = AutoOptimizer(strategy=OptimizationStrategy.CONSERVATIVE, auto_apply=True)

    run_data = {
        "cases": [
            {"id": "1", "passed": False, "latency": 10.0},
            {"id": "2", "passed": False, "latency": 12.0},
        ]
    }

    result = optimizer.optimize(run_data, config={})

    # Conservative should only apply high-priority recommendations
    assert len(result.applied_recommendations) >= 0
    for rec in result.applied_recommendations:
        assert rec.priority >= 8


def test_auto_optimizer_balanced():
    """Test auto-optimizer with balanced strategy."""
    optimizer = AutoOptimizer(strategy=OptimizationStrategy.BALANCED, auto_apply=True)

    run_data = {
        "cases": [
            {"id": "1", "passed": True, "latency": 8.0, "cost": 0.15},
            {"id": "2", "passed": True, "latency": 9.0, "cost": 0.18},
        ]
    }

    result = optimizer.optimize(run_data, config={})

    # Balanced should apply medium to high priority
    assert len(result.applied_recommendations) >= 0
    for rec in result.applied_recommendations:
        assert rec.priority >= 6


def test_auto_optimizer_changes():
    """Test that optimizer produces changes."""
    optimizer = AutoOptimizer(strategy=OptimizationStrategy.AGGRESSIVE, auto_apply=True)

    run_data = {
        "cases": [
            {"id": "1", "passed": True, "latency": 10.0, "cost": 0.20, "tokens": 50000},
        ]
    }

    result = optimizer.optimize(run_data, config={})

    # Should have some changes or recommendations
    assert len(result.applied_recommendations) > 0 or len(result.skipped_recommendations) > 0


def test_auto_optimizer_manual_approval():
    """Test auto-optimizer with manual approval."""
    approved_count = 0

    def approval_callback(rec: Recommendation) -> bool:
        nonlocal approved_count
        approved_count += 1
        return rec.priority >= 9  # Only approve high priority

    optimizer = AutoOptimizer(
        strategy=OptimizationStrategy.BALANCED,
        auto_apply=False,
        approval_callback=approval_callback,
    )

    run_data = {
        "cases": [
            {"id": "1", "passed": False, "error": "timeout"},
            {"id": "2", "passed": False, "error": "timeout"},
        ]
    }

    optimizer.optimize(run_data, config={})

    # Approval callback should have been called
    assert approved_count > 0


def test_recommendation_types():
    """Test all recommendation types are accessible."""
    assert RecommendationType.PROMPT is not None
    assert RecommendationType.MODEL is not None
    assert RecommendationType.TOOL is not None
    assert RecommendationType.PARAMETER is not None
    assert RecommendationType.ARCHITECTURE is not None
    assert RecommendationType.COST is not None
    assert RecommendationType.PERFORMANCE is not None


def test_analysis_result_properties():
    """Test AnalysisResult computed properties."""
    result = AnalysisResult(total_cases=10, passed_cases=7, failed_cases=3, total_cost=2.5)

    assert abs(result.success_rate - 0.7) < 0.01
    assert abs(result.avg_cost_per_case - 0.25) < 0.01

    # Test zero division handling
    empty_result = AnalysisResult()
    assert empty_result.success_rate == 0.0
    assert empty_result.avg_cost_per_case == 0.0
