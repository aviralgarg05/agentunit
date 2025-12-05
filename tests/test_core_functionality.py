"""
Simple test scenario to validate AgentUnit core functionality.
"""

from agentunit.core import Scenario
from agentunit.reporting.results import ScenarioResult, ScenarioRun


def test_basic_scenario():
    """Test that basic scenario creation and execution works."""
    # Create a simple scenario
    scenario = Scenario(
        name="test_basic_scenario",
        adapter=None,  # We'll skip the adapter for this basic test
        dataset=None,  # We'll skip the dataset for this basic test
    )

    # Verify scenario creation
    assert scenario.name == "test_basic_scenario"


def test_imports():
    """Test script for AgentUnit core functionality"""
    # Test core imports

    # Test result creation
    result = ScenarioResult(name="test_result")
    run = ScenarioRun(
        scenario_name="test_scenario",
        case_id="test_case",
        success=True,
        metrics={"test_metric": 0.95},
        duration_ms=1000.0,
        trace=None,
    )
    result.add_run(run)

    assert result.name == "test_result"
    assert len(result.runs) == 1
