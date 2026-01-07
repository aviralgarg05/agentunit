"""Tests for benchmarks module."""

import json

import pytest

from agentunit.benchmarks import BenchmarkRunner, BenchmarkScenario
from agentunit.benchmarks.definitions import BenchmarkTask


@pytest.fixture
def mock_agent():
    """Create a mock agent function."""
    return lambda x: "Simulated response"


@pytest.fixture
def sample_scenario():
    """Create a sample benchmark scenario."""
    return BenchmarkScenario(
        name="TestBenchmark",
        description="A test benchmark scenario",
        tasks=[
            BenchmarkTask(
                id="task_1", prompt="Question 1", expected_answer="answer1", difficulty="easy"
            ),
            BenchmarkTask(
                id="task_2", prompt="Question 2", expected_answer="answer2", difficulty="hard"
            ),
        ],
    )


def test_benchmark_runner_init(tmp_path):
    """Test BenchmarkRunner initialization."""
    runner = BenchmarkRunner(submissions_dir=tmp_path)
    assert runner.submissions_dir == tmp_path
    assert runner.submissions_dir.exists()


def test_run_scenario_success(tmp_path, sample_scenario, mock_agent):
    """Test successful scenario execution."""
    runner = BenchmarkRunner(submissions_dir=tmp_path)

    # Run scenario
    output_path = runner.run_scenario(
        scenario=sample_scenario, agent_func=mock_agent, model_name="test-model"
    )

    # Verify output file
    assert output_path.exists()
    assert output_path.suffix == ".json"

    with open(output_path) as f:
        data = json.load(f)

    assert data["benchmark"] == "TestBenchmark"
    assert data["model_name"] == "test-model"
    assert len(data["results"]) == 2

    # Check individual results
    assert data["results"][0]["task_id"] == "task_1"
    assert "latency" in data["results"][0]


def test_run_scenario_correctness(tmp_path):
    """Test correctness checking logic."""
    scenario = BenchmarkScenario(
        name="MathTest",
        description="Math",
        tasks=[BenchmarkTask(id="1", prompt="2+2", expected_answer="4")],
    )

    runner = BenchmarkRunner(submissions_dir=tmp_path)

    # Correct agent
    runner.run_scenario(
        scenario=scenario, agent_func=lambda x: "The answer is 4", model_name="smart-agent"
    )

    # Incorrect agent
    last_run = runner.run_scenario(
        scenario=scenario, agent_func=lambda x: "The answer is 5", model_name="dumb-agent"
    )

    with open(last_run) as f:
        data = json.load(f)

    assert data["results"][0]["correct"] is False
