"""Tests for the AgentUnit pytest plugin."""

from pathlib import Path
from textwrap import dedent

import pytest

from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.pytest.plugin import AgentUnitFile, AgentUnitItem, _is_eval_directory


class SimpleTestAdapter(BaseAdapter):
    """Simple adapter for testing."""

    name = "test"

    def __init__(self, agent_func):
        self.agent_func = agent_func

    def prepare(self):
        pass

    def execute(self, case, trace):
        try:
            result = self.agent_func({"query": case.query})
            output = result.get("result", "")
            success = output == case.expected_output
            return AdapterOutcome(success=success, output=output)
        except Exception as e:
            return AdapterOutcome(success=False, output=None, error=str(e))


class MockConfig:
    """Mock pytest config."""

    def __init__(self, rootpath=None):
        self.rootpath = rootpath or Path()


class MockSession:
    """Mock pytest session."""

    def __init__(self, rootpath=None):
        self.config = MockConfig(rootpath)


class MockParent:
    """Mock pytest parent node."""

    def __init__(self, path=None, rootpath=None):
        self.path = path or Path("test.py")
        self.config = MockConfig(rootpath)
        self.session = MockSession(rootpath)
        self.nodeid = str(self.path)
        self.own_markers = []
        self.parent = None  # Root node has no parent


class TestPytestPlugin:
    """Test the AgentUnit pytest plugin functionality."""

    def test_is_eval_directory(self):
        """Test the eval directory detection."""
        # Should detect files in tests/eval/
        assert _is_eval_directory(Path("tests/eval/scenarios.py"))
        assert _is_eval_directory(Path("project/tests/eval/test.yaml"))

        # Should not detect files elsewhere
        assert not _is_eval_directory(Path("tests/test_something.py"))
        assert not _is_eval_directory(Path("src/agentunit/core.py"))
        assert not _is_eval_directory(Path("eval/scenarios.py"))

    def test_scenario_discovery_from_python_file(self, tmp_path):
        """Test discovering scenarios from Python files."""
        # Create a temporary Python file with scenarios
        scenario_file = tmp_path / "tests" / "eval" / "test_scenarios.py"
        scenario_file.parent.mkdir(parents=True)

        scenario_content = dedent("""
            from agentunit import Scenario
            from agentunit.datasets.base import DatasetCase, DatasetSource
            from agentunit.adapters.base import BaseAdapter, AdapterOutcome

            class TestAdapter(BaseAdapter):
                name = "test"
                def __init__(self, agent_func):
                    self.agent_func = agent_func
                def prepare(self):
                    pass
                def execute(self, case, trace):
                    result = self.agent_func({"query": case.query})
                    return AdapterOutcome(success=True, output=result.get("result"))

            class TestDataset(DatasetSource):
                def __init__(self):
                    super().__init__(name="test", loader=lambda: [
                        DatasetCase(id="test1", query="hello", expected_output="hi")
                    ])

            def test_agent(payload):
                return {"result": "hi"}

            # This should be discovered
            test_scenario = Scenario(
                name="test-scenario",
                adapter=TestAdapter(test_agent),
                dataset=TestDataset(),
            )

            def scenario_factory():
                return Scenario(
                    name="factory-scenario",
                    adapter=TestAdapter(test_agent),
                    dataset=TestDataset(),
                )
        """)

        scenario_file.write_text(scenario_content)

        # Create a mock parent collector
        parent = MockParent(path=tmp_path, rootpath=tmp_path)

        # Test file collection
        agentunit_file = AgentUnitFile.from_parent(parent, path=scenario_file)
        scenarios = agentunit_file._discover_scenarios()

        # Should find at least one scenario
        assert len(scenarios) >= 1
        scenario_names = [s.name for s in scenarios]
        assert "test-scenario" in scenario_names

    def test_agentunit_item_success(self):
        """Test AgentUnit item with successful scenario."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource

        class SuccessDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="success",
                    loader=lambda: [
                        DatasetCase(id="success1", query="test", expected_output="test")
                    ],
                )

        def success_agent(payload):
            return {"result": "test"}

        scenario = Scenario(
            name="success-test",
            adapter=SimpleTestAdapter(success_agent),
            dataset=SuccessDataset(),
        )

        # Create mock parent
        parent = MockParent()
        item = AgentUnitItem.from_parent(parent, name="test", scenario=scenario)

        # Should not raise any exception
        item.runtest()

    def test_agentunit_item_failure(self):
        """Test AgentUnit item with failing scenario."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource

        class FailDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="fail",
                    loader=lambda: [
                        DatasetCase(id="fail1", query="test", expected_output="expected")
                    ],
                )

        def fail_agent(payload):
            return {"result": "wrong"}

        scenario = Scenario(
            name="fail-test",
            adapter=SimpleTestAdapter(fail_agent),
            dataset=FailDataset(),
        )

        # Create mock parent
        parent = MockParent()
        item = AgentUnitItem.from_parent(parent, name="test", scenario=scenario)

        # Should raise AssertionError for failed scenario
        with pytest.raises(AssertionError, match="Scenario 'fail-test' failed"):
            item.runtest()

    def test_agentunit_item_load_error(self):
        """Test AgentUnit item with load error."""
        # Create mock parent
        parent = MockParent()
        item = AgentUnitItem.from_parent(parent, name="test", load_error="Failed to load")

        # Should raise AgentUnitError for load error
        from agentunit.core.exceptions import AgentUnitError

        with pytest.raises(AgentUnitError, match="Failed to load scenario"):
            item.runtest()

    def test_pytest_markers(self):
        """Test that pytest markers are properly added."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="test",
                    loader=lambda: [DatasetCase(id="test1", query="test", expected_output="test")],
                )

        def test_agent(payload):
            return {"result": "test"}

        scenario = Scenario(
            name="marker-test",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        # Create mock parent
        parent = MockParent()
        item = AgentUnitItem.from_parent(parent, name="test", scenario=scenario)

        # Check markers
        marker_names = [marker.name for marker in item.iter_markers()]
        assert "agentunit" in marker_names
        assert "scenario" in marker_names
