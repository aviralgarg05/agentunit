"""Integration tests for LangGraph with AgentUnit."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentunit import Scenario, run_suite
from agentunit.datasets.base import DatasetCase, DatasetSource


# Skip all tests in this module if LangGraph is not available
langgraph = pytest.importorskip("langgraph", reason="LangGraph not installed")


class SimpleDataset(DatasetSource):
    """Simple dataset for testing."""

    def __init__(self):
        """Initialize the simple dataset."""
        super().__init__(name="simple-test-dataset", loader=self._generate_cases)

    def _generate_cases(self):
        """Generate test cases."""
        return [
            DatasetCase(
                id="quantum-query",
                query="What is quantum tunneling?",
                expected_output="Quantum tunneling is a quantum mechanical phenomenon",
                context=["Physics", "Quantum Mechanics"],
                tools=["search"],
                metadata={"difficulty": "intermediate"}
            ),
            DatasetCase(
                id="python-query",
                query="Tell me about Python programming",
                expected_output="Python is a high-level programming language",
                context=["Programming", "Software Development"],
                tools=["documentation"],
                metadata={"difficulty": "beginner"}
            ),
            DatasetCase(
                id="weather-query",
                query="What's the weather like today?",
                expected_output="I would need access to weather APIs",
                context=["Weather", "APIs"],
                tools=["weather_api"],
                metadata={"difficulty": "advanced"}
            )
        ]


@pytest.mark.langgraph
@pytest.mark.integration
class TestLangGraphIntegration:
    """Integration tests for LangGraph adapter."""

    def test_langgraph_scenario_from_callable(self):
        """Test creating a scenario from a callable LangGraph agent."""
        from .simple_langgraph_agent import invoke_agent

        # Create scenario using the callable
        scenario = Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name="test-langgraph-callable"
        )

        assert scenario.name == "test-langgraph-callable"
        assert scenario.adapter.name == "langgraph"

        # Test that the scenario can be prepared
        scenario.adapter.prepare()

        # Test execution with a single case
        test_case = DatasetCase(
            id="test-case",
            query="What is quantum tunneling?",
            expected_output="",
            context=["Physics"],
            tools=["search"],
            metadata={}
        )

        from agentunit.core.trace import TraceLog
        trace = TraceLog()
        outcome = scenario.adapter.execute(test_case, trace)

        assert outcome.success is True
        assert outcome.output is not None
        assert "quantum" in outcome.output.lower()

    def test_langgraph_scenario_from_python_file(self):
        """Test creating a scenario from a Python file."""
        # Create a temporary Python file with the agent
        agent_file = Path("tests/integration/simple_langgraph_agent.py")

        scenario = Scenario.load_langgraph(
            path=agent_file,
            dataset=SimpleDataset(),
            name="test-langgraph-file",
            callable="invoke_agent"  # Specify the callable name
        )

        assert scenario.name == "test-langgraph-file"
        assert scenario.adapter.name == "langgraph"

    def test_full_evaluation_cycle(self):
        """Test running a full evaluation cycle with LangGraph."""
        from .simple_langgraph_agent import invoke_agent

        # Create scenario
        scenario = Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name="full-cycle-test"
        )

        # Run the evaluation
        result = run_suite([scenario])

        # Verify results
        assert len(result.scenarios) == 1
        scenario_result = result.scenarios[0]

        assert scenario_result.name == "full-cycle-test"
        assert len(scenario_result.runs) == 3  # Three test cases in SimpleDataset

        # Check that all runs completed
        for run in scenario_result.runs:
            assert run.success is True
            assert run.duration_ms > 0
            assert run.error is None

    def test_scenario_with_metrics(self):
        """Test running scenario with metrics evaluation."""
        from .simple_langgraph_agent import invoke_agent

        scenario = Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name="metrics-test"
        )

        # Run with basic metrics (if available)
        result = run_suite([scenario], metrics=["response_length"])

        scenario_result = result.scenarios[0]

        # Check that metrics were calculated (or gracefully handled if not available)
        for run in scenario_result.runs:
            assert isinstance(run.metrics, dict)

    def test_scenario_error_handling(self):
        """Test that scenarios handle errors gracefully."""
        def failing_agent(payload):
            """An agent that always fails."""
            raise ValueError("Simulated agent failure")

        scenario = Scenario.load_langgraph(
            path=failing_agent,
            dataset=SimpleDataset(),
            name="error-test"
        )

        result = run_suite([scenario])
        scenario_result = result.scenarios[0]

        # All runs should fail but be recorded
        for run in scenario_result.runs:
            assert run.success is False
            assert run.error is not None
            assert "Simulated agent failure" in run.error

    def test_scenario_with_retries(self):
        """Test scenario retry functionality."""
        from .simple_langgraph_agent import invoke_agent

        scenario = Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name="retry-test"
        )

        # Set retries
        scenario.retries = 2

        result = run_suite([scenario])
        scenario_result = result.scenarios[0]

        # Should still succeed (our agent doesn't fail)
        for run in scenario_result.runs:
            assert run.success is True

    def test_scenario_clone_and_modify(self):
        """Test cloning and modifying scenarios."""
        from .simple_langgraph_agent import invoke_agent

        original = Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name="original"
        )

        # Clone with modifications
        cloned = original.clone(
            name="cloned",
            retries=3,
            timeout=120.0
        )

        assert cloned.name == "cloned"
        assert cloned.retries == 3
        assert cloned.timeout == 120.0
        assert original.name == "original"  # Original unchanged
        assert original.retries == 1  # Default value


@pytest.mark.langgraph
@pytest.mark.integration
def test_langgraph_adapter_registry():
    """Test that LangGraph adapter is properly registered."""
    from agentunit.adapters.langgraph import LangGraphAdapter
    from agentunit.adapters.registry import resolve_adapter

    adapter_class = resolve_adapter("langgraph")
    assert adapter_class is LangGraphAdapter

    # Test alias
    adapter_class = resolve_adapter("langgraph_graph")
    assert adapter_class is LangGraphAdapter


@pytest.mark.langgraph
@pytest.mark.integration
def test_multiple_scenarios():
    """Test running multiple LangGraph scenarios together."""
    from .simple_langgraph_agent import invoke_agent

    # Create multiple scenarios
    scenarios = [
        Scenario.load_langgraph(
            path=invoke_agent,
            dataset=SimpleDataset(),
            name=f"scenario-{i}"
        )
        for i in range(3)
    ]

    # Run all scenarios
    result = run_suite(scenarios)

    assert len(result.scenarios) == 3
    for i, scenario_result in enumerate(result.scenarios):
        assert scenario_result.name == f"scenario-{i}"
        assert len(scenario_result.runs) == 3  # Each has 3 test cases
