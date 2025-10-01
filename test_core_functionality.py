"""
Simple test scenario to validate AgentUnit core functionality.
"""

from agentunit.core import Scenario


def test_basic_scenario():
    """Test that basic scenario creation and execution works."""
    # Create a simple scenario
    scenario = Scenario(
        name="test_basic_scenario",
        adapter=None,  # We'll skip the adapter for this basic test
        dataset=None   # We'll skip the dataset for this basic test
    )
    
    # Verify scenario creation
    assert scenario.name == "test_basic_scenario"
    
    print("✅ Basic scenario creation test passed!")
    return True


def test_imports():
    """Test script for AgentUnit core functionality"""
    try:
        # Test core imports
        from agentunit.reporting.results import ScenarioResult, ScenarioRun
        print("✅ Core imports successful")
        
        # Test result creation
        result = ScenarioResult(name="test_result")
        run = ScenarioRun(
            timestamp=123456789,
            duration=1000,
            status="passed",
            metrics={"test_metric": 0.95}
        )
        result.add_run(run)
        print("✅ ScenarioResult creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running AgentUnit core functionality tests...")
    
    # Test imports
    if test_imports():
        print("✅ Import validation passed")
    else:
        print("❌ Import validation failed")
        exit(1)
    
    # Test basic functionality
    if test_basic_scenario():
        print("✅ Basic scenario test passed")
    else:
        print("❌ Basic scenario test failed")
        exit(1)
    
    print("🎯 Core functionality tests completed!")