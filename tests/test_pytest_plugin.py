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


class TestScenarioCache:
    """Test the AgentUnit scenario caching functionality."""

    def test_cache_creation_and_hit(self, tmp_path):
        """Test that cache stores and retrieves results correctly."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="cache-test",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        def test_agent(payload):
            return {"result": "hi"}

        scenario = Scenario(
            name="cache-test-scenario",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        cache = ScenarioCache(tmp_path, enabled=True)
        assert cache.get(scenario) is None

        cache_key = cache.set(scenario, success=True, failures=[])
        assert cache_key

        cached = cache.get(scenario)
        assert cached is not None
        assert cached.success is True
        assert cached.failures == []

    def test_cache_disabled(self, tmp_path):
        """Test that cache operations are no-op when disabled."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="disabled-test",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        def test_agent(payload):
            return {"result": "hi"}

        scenario = Scenario(
            name="disabled-cache-scenario",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        cache = ScenarioCache(tmp_path, enabled=False)
        cache_key = cache.set(scenario, success=True, failures=[])
        assert cache_key == ""
        assert cache.get(scenario) is None

    def test_cache_invalidation_on_source_change(self, tmp_path):
        """Test that cache is invalidated when source file changes."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="invalidation-test",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        def test_agent(payload):
            return {"result": "hi"}

        scenario = Scenario(
            name="invalidation-scenario",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        source_file = tmp_path / "source.py"
        source_file.write_text("# original content")

        cache = ScenarioCache(tmp_path, enabled=True)
        cache.set(scenario, success=True, failures=[], source_path=source_file)

        cached = cache.get(scenario, source_path=source_file)
        assert cached is not None
        assert cached.success is True

        source_file.write_text("# modified content")
        cached = cache.get(scenario, source_path=source_file)
        assert cached is None

    def test_cache_clear(self, tmp_path):
        """Test clearing the cache."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="clear-test",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        def test_agent(payload):
            return {"result": "hi"}

        scenario = Scenario(
            name="clear-cache-scenario",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        cache = ScenarioCache(tmp_path, enabled=True)
        cache.set(scenario, success=True, failures=[])
        assert cache.get(scenario) is not None

        count = cache.clear()
        assert count >= 1
        assert cache.get(scenario) is None

    def test_cache_stores_failures(self, tmp_path):
        """Test that cache stores failure information."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class TestDataset(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="failure-test",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        def test_agent(payload):
            return {"result": "wrong"}

        scenario = Scenario(
            name="failure-cache-scenario",
            adapter=SimpleTestAdapter(test_agent),
            dataset=TestDataset(),
        )

        cache = ScenarioCache(tmp_path, enabled=True)
        failures = ["Case test1: Expected 'hi' but got 'wrong'"]
        cache.set(scenario, success=False, failures=failures)

        cached = cache.get(scenario)
        assert cached is not None
        assert cached.success is False
        assert cached.failures == failures

    def test_different_scenarios_have_different_keys(self, tmp_path):
        """Test that different scenarios produce different cache keys."""
        from agentunit import Scenario
        from agentunit.datasets.base import DatasetCase, DatasetSource
        from agentunit.pytest.cache import ScenarioCache

        class Dataset1(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="dataset1",
                    loader=lambda: [DatasetCase(id="test1", query="hello", expected_output="hi")],
                )

        class Dataset2(DatasetSource):
            def __init__(self):
                super().__init__(
                    name="dataset2",
                    loader=lambda: [
                        DatasetCase(id="test2", query="bye", expected_output="goodbye")
                    ],
                )

        def test_agent(payload):
            return {"result": "hi"}

        scenario1 = Scenario(
            name="scenario1",
            adapter=SimpleTestAdapter(test_agent),
            dataset=Dataset1(),
        )

        scenario2 = Scenario(
            name="scenario2",
            adapter=SimpleTestAdapter(test_agent),
            dataset=Dataset2(),
        )

        cache = ScenarioCache(tmp_path, enabled=True)
        key1 = cache.set(scenario1, success=True, failures=[])
        key2 = cache.set(scenario2, success=False, failures=["error"])

        assert key1 != key2

        cached1 = cache.get(scenario1)
        cached2 = cache.get(scenario2)

        assert cached1 is not None
        assert cached1.success is True

        assert cached2 is not None
        assert cached2.success is False
