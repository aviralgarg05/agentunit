"""Tests for benchmarks module."""

from pathlib import Path

from agentunit.benchmarks import (
    AgentArenaBenchmark,
    ArenaTask,
    ArenaTaskType,
    BenchmarkResult,
    BenchmarkRunner,
    GAIABenchmark,
    GAIALevel,
    LeaderboardConfig,
    LeaderboardSubmitter,
)


def test_benchmarks_imports():
    """Test benchmarks module imports."""
    assert GAIABenchmark is not None
    assert GAIALevel is not None
    assert AgentArenaBenchmark is not None
    assert ArenaTask is not None
    assert LeaderboardSubmitter is not None
    assert LeaderboardConfig is not None
    assert BenchmarkRunner is not None
    assert BenchmarkResult is not None


def test_gaia_levels():
    """Test GAIA difficulty levels."""
    assert GAIALevel.LEVEL_1.value == 1
    assert GAIALevel.LEVEL_2.value == 2
    assert GAIALevel.LEVEL_3.value == 3


def test_gaia_sample_tasks():
    """Test GAIA sample tasks loading."""
    gaia = GAIABenchmark()
    tasks = gaia.load_dataset("validation")

    assert len(tasks) == 3
    assert all(hasattr(task, "task_id") for task in tasks)
    assert all(hasattr(task, "question") for task in tasks)
    assert all(hasattr(task, "level") for task in tasks)
    assert all(hasattr(task, "final_answer") for task in tasks)


def test_gaia_to_agentunit_dataset():
    """Test GAIA to AgentUnit dataset conversion."""
    gaia = GAIABenchmark()
    tasks = gaia.load_dataset("validation")
    dataset = gaia.to_agentunit_dataset()

    assert len(dataset) == len(tasks)
    for case in dataset:
        assert hasattr(case, "id")
        assert hasattr(case, "query")
        assert hasattr(case, "expected_output")
        assert case.metadata.get("benchmark") == "gaia"


def test_gaia_score_calculation():
    """Test GAIA score calculation."""
    gaia = GAIABenchmark()

    results = [
        {"task_id": "1", "passed": True, "level": 1},
        {"task_id": "2", "passed": True, "level": 1},
        {"task_id": "3", "passed": False, "level": 2},
        {"task_id": "4", "passed": True, "level": 3},
    ]

    scores = gaia.calculate_score(results)

    assert "overall" in scores
    assert "level_1" in scores
    assert "level_2" in scores
    assert "level_3" in scores
    assert abs(scores["overall"] - 75.0) < 0.1  # 3 out of 4 passed


def test_gaia_submission_format():
    """Test GAIA submission formatting."""
    gaia = GAIABenchmark()

    results = [
        {"task_id": "1", "output": "Paris", "passed": True},
        {"task_id": "2", "output": "77", "passed": True},
    ]

    submission = gaia.format_submission(results, model_name="test_model")

    assert submission["model_name"] == "test_model"
    assert len(submission["results"]) == 2
    assert all("task_id" in r for r in submission["results"])


def test_arena_sample_tasks():
    """Test AgentArena sample tasks."""
    arena = AgentArenaBenchmark()
    tasks = arena.load_dataset("test")

    assert len(tasks) == 3
    assert all(isinstance(task, ArenaTask) for task in tasks)


def test_arena_to_agentunit_dataset():
    """Test AgentArena to AgentUnit dataset conversion."""
    arena = AgentArenaBenchmark()
    tasks = arena.load_dataset("test")
    dataset = arena.to_agentunit_dataset()

    assert len(dataset) == len(tasks)
    for case in dataset:
        assert case.metadata.get("benchmark") == "agent_arena"
        assert "success_criteria" in case.metadata


def test_arena_success_evaluation():
    """Test AgentArena success evaluation."""
    arena = AgentArenaBenchmark()

    # Test "equals" criteria
    task_equals = ArenaTask(
        task_id="test1",
        task_type=ArenaTaskType.CODE_EXECUTION,
        instruction="Calculate 10!",
        success_criteria={"type": "equals", "value": 3628800}
    )
    assert arena.evaluate_success(task_equals, 3628800, []) is True
    assert arena.evaluate_success(task_equals, 123, []) is False

    # Test "contains" criteria
    task_contains = ArenaTask(
        task_id="test2",
        task_type=ArenaTaskType.WEB_BROWSING,
        instruction="Find BTC price",
        success_criteria={"type": "contains", "value": "BTC"}
    )
    assert arena.evaluate_success(task_contains, "BTC: $50000", []) is True
    assert arena.evaluate_success(task_contains, "ETH: $3000", []) is False


def test_arena_score_calculation():
    """Test AgentArena score calculation."""
    arena = AgentArenaBenchmark()

    results = [
        {"task_id": "1", "passed": True, "task_type": "web_browsing", "steps": 3},
        {"task_id": "2", "passed": True, "task_type": "code_execution", "steps": 2},
        {"task_id": "3", "passed": False, "task_type": "web_browsing", "steps": 5},
    ]

    scores = arena.calculate_score(results)

    assert "overall" in scores
    assert "avg_steps" in scores
    assert "task_type" in scores
    assert abs(scores["overall"] - 66.67) < 0.1  # 2 out of 3 passed


def test_leaderboard_config():
    """Test leaderboard configuration."""
    config = LeaderboardConfig(
        leaderboard_name="gaia",
        api_url="https://api.gaia-benchmark.com",
        api_key="test_key",
        model_name="test_model",
        organization="test_org"
    )

    assert config.leaderboard_name == "gaia"
    assert config.api_url == "https://api.gaia-benchmark.com"
    assert config.model_name == "test_model"


def test_leaderboard_submission(tmp_path):
    """Test leaderboard submission."""
    config = LeaderboardConfig(
        leaderboard_name="test",
        api_url="https://example.com",
        model_name="test_model"
    )

    submitter = LeaderboardSubmitter(config, output_dir=tmp_path)

    results = [
        {"id": "1", "output": "answer1", "passed": True},
        {"id": "2", "output": "answer2", "passed": False},
    ]

    response = submitter.submit(results, benchmark_name="test", dry_run=True)

    assert response["status"] == "saved"
    assert "file" in response
    assert Path(response["file"]).exists()


def test_leaderboard_comparison():
    """Test leaderboard comparison with baseline."""
    config = LeaderboardConfig(
        leaderboard_name="test",
        api_url="https://example.com",
        model_name="custom_model"
    )

    submitter = LeaderboardSubmitter(config)

    results = [
        {"passed": True},
        {"passed": True},
        {"passed": True},
        {"passed": False},
    ]

    comparison = submitter.compare_with_baseline(results, baseline_model="gpt-4o-mini")

    assert comparison["current_model"] == "custom_model"
    assert comparison["baseline_model"] == "gpt-4o-mini"
    assert abs(comparison["current_score"] - 75.0) < 0.1  # 3 out of 4 passed


def test_benchmark_runner_gaia():
    """Test benchmark runner with GAIA."""
    config = LeaderboardConfig(
        leaderboard_name="gaia",
        api_url="https://example.com",
        model_name="test_model"
    )

    runner = BenchmarkRunner(leaderboard_config=config)
    result = runner.run_gaia(level=GAIALevel.LEVEL_1, submit=False)

    assert result.benchmark_type.value == "gaia"
    assert result.total_tasks > 0
    assert "overall" in result.scores


def test_benchmark_runner_arena():
    """Test benchmark runner with AgentArena."""
    runner = BenchmarkRunner()
    result = runner.run_arena(submit=False)

    assert result.benchmark_type.value == "agent_arena"
    assert result.total_tasks > 0
    assert "overall" in result.scores


def test_benchmark_comparison():
    """Test cross-benchmark comparison."""
    runner = BenchmarkRunner()

    gaia_result = runner.run_gaia()
    arena_result = runner.run_arena()

    comparison = runner.compare_benchmarks([gaia_result, arena_result])

    assert len(comparison["benchmarks"]) == 2
    assert "avg_success_rate" in comparison
    assert "total_tasks" in comparison
    assert comparison["total_tasks"] == gaia_result.total_tasks + arena_result.total_tasks
