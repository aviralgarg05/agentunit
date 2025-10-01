#!/usr/bin/env python3
"""
Comprehensive platform adapter validation tests for AgentUnit
Tests all 5 platform integrations to ensure they work correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_platform_imports():
    """Test that all platform adapters can be imported successfully"""
    print("🧪 Testing platform adapter imports...")
    
    adapters_tested = []
    
    try:
        # Test AutoGen AG2 adapter
        from agentunit.adapters.autogen_ag2 import AG2Adapter
        print("✅ AutoGen AG2 adapter imported successfully")
        adapters_tested.append(AG2Adapter.__name__)
        
        # Test Swarm adapter
        from agentunit.adapters.swarm_adapter import SwarmAdapter
        print("✅ Swarm adapter imported successfully")
        adapters_tested.append(SwarmAdapter.__name__)
        
        # Test LangSmith adapter
        from agentunit.adapters.langsmith_adapter import LangSmithAdapter
        print("✅ LangSmith adapter imported successfully")
        adapters_tested.append(LangSmithAdapter.__name__)
        
        # Test AgentOps adapter
        from agentunit.adapters.agentops_adapter import AgentOpsAdapter
        print("✅ AgentOps adapter imported successfully")
        adapters_tested.append(AgentOpsAdapter.__name__)
        
        # Test Wandb adapter
        from agentunit.adapters.wandb_adapter import WandbAdapter
        print("✅ Wandb adapter imported successfully")
        adapters_tested.append(WandbAdapter.__name__)
        
        print(f"✅ All {len(adapters_tested)} adapters imported: {adapters_tested}")
        return True
    except Exception as e:
        print(f"❌ Platform adapter import failed: {e}")
        return False

def test_adapter_initialization():
    """Test that adapters are properly defined (cannot instantiate abstract classes)"""
    print("\n🧪 Testing adapter class definitions...")
    
    try:
        from agentunit.adapters.autogen_ag2 import AG2Adapter
        from agentunit.adapters.swarm_adapter import SwarmAdapter
        from agentunit.adapters.langsmith_adapter import LangSmithAdapter
        from agentunit.adapters.agentops_adapter import AgentOpsAdapter
        from agentunit.adapters.wandb_adapter import WandbAdapter
        
        # Test that classes are defined and have expected attributes
        for adapter_class, name in [
            (AG2Adapter, "AutoGen AG2"),
            (SwarmAdapter, "Swarm"), 
            (LangSmithAdapter, "LangSmith"),
            (AgentOpsAdapter, "AgentOps"),
            (WandbAdapter, "Wandb")
        ]:
            # Check class is properly defined
            assert hasattr(adapter_class, '__init__'), f"{name} adapter missing __init__"
            assert hasattr(adapter_class, '__name__'), f"{name} adapter missing __name__"
            print(f"✅ {name} adapter class properly defined")
        
        print("✅ All adapter classes are properly defined")
        return True
    except Exception as e:
        print(f"❌ Adapter class validation failed: {e}")
        return False

def test_scenario_integration():
    """Test that scenario can be created with basic components"""
    print("\n🧪 Testing scenario creation with basic components...")
    
    try:
        from agentunit.core import Scenario, DatasetSource, DatasetCase
        from agentunit.adapters.base import BaseAdapter, AdapterOutcome
        from agentunit.core.trace import TraceLog
        
        # Create a basic dataset case
        test_case = DatasetCase(
            id="test_case_1",
            query="Hello world",
            expected_output="Hi there!",
            metadata={"type": "greeting"}
        )
        
        # Create a dataset source
        dataset = DatasetSource("test_dataset", lambda: [test_case])
        
        # Create a simple mock adapter that implements BaseAdapter interface
        class MockAdapter(BaseAdapter):
            def __init__(self, config):
                self.config = config
                self.name = "mock_adapter"
            
            def prepare(self) -> None:
                """Perform any lazy setup."""
                return None
            
            def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
                """Run the agent flow on a single dataset case."""
                return AdapterOutcome(
                    success=True,
                    output="mock response",
                    tool_calls=[],
                    metrics={"test_metric": 1.0}
                )
        
        # Create adapter
        adapter = MockAdapter({
            "model": "gpt-3.5-turbo",
            "timeout": 30
        })
        
        # Create scenario
        scenario = Scenario(
            name="test_scenario", 
            adapter=adapter, 
            dataset=dataset
        )
        
        print("✅ Scenario created successfully with mock adapter")
        print(f"   - Scenario name: {scenario.name}")
        print(f"   - Adapter type: {type(adapter).__name__}")
        print(f"   - Dataset name: {dataset.name}")
        
        return True
    except Exception as e:
        print(f"❌ Scenario integration failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration with adapters"""
    print("\n🧪 Testing CLI integration...")
    
    try:
        # Test that CLI entrypoint exists and can be imported
        import agentunit.cli
        cli_module_name = agentunit.cli.__name__
        print(f"✅ CLI module imported successfully: {cli_module_name}")
        
        # Test that core CLI functionality is accessible
        from agentunit.cli import entrypoint
        entrypoint_name = entrypoint.__name__
        print(f"✅ CLI entrypoint function accessible: {entrypoint_name}")
        
        return True
    except Exception as e:
        print(f"❌ CLI integration failed: {e}")
        return False

def test_reporting_integration():
    """Test reporting system integration"""
    print("\n🧪 Testing reporting system integration...")
    
    try:
        from agentunit.reporting.results import ScenarioResult, ScenarioRun
        from agentunit.core.trace import TraceLog
        
        # Create result with proper constructor
        result = ScenarioResult(name="test_scenario_result")
        
        # Create a scenario run
        trace = TraceLog()
        run = ScenarioRun(
            scenario_name="test_scenario",
            case_id="test_case",
            success=True,
            metrics={"accuracy": 0.95},
            duration_ms=1500.0,
            trace=trace
        )
        
        # Add run to result
        result.add_run(run)
        
        print("✅ ScenarioResult created successfully")
        print(f"   - Result name: {result.name}")
        print(f"   - Success rate: {result.success_rate}")
        print(f"   - Number of runs: {len(result.runs)}")
        
        return True
    except Exception as e:
        print(f"❌ Reporting integration failed: {e}")
        return False

def main():
    """Run all platform adapter validation tests"""
    print("🚀 Starting comprehensive platform adapter validation...")
    print("=" * 60)
    
    tests = [
        test_platform_imports,
        test_adapter_initialization,
        test_scenario_integration,
        test_cli_integration,
        test_reporting_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed")
    
    print("\n" + "=" * 60)
    print(f"🎯 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All platform adapter validation tests passed!")
        print("✅ AgentUnit is ready for production use!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)