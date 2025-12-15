"""Pytest plugin for AgentUnit scenario discovery and execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agentunit import Scenario, run_suite
from agentunit.core.exceptions import AgentUnitError


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from _pytest.config import Config
    from _pytest.nodes import Collector
    from _pytest.python import Module


def pytest_configure(config: Config) -> None:
    """Configure pytest with AgentUnit markers."""
    config.addinivalue_line("markers", "agentunit: mark test as an AgentUnit scenario evaluation")
    config.addinivalue_line("markers", "scenario(name): mark test with specific scenario name")


def pytest_collect_file(file_path: Path, parent: Collector) -> Module | None:
    """Collect AgentUnit scenario files as pytest tests."""
    # Only collect files in tests/eval/ directory
    if not _is_eval_directory(file_path):
        return None

    # Look for scenario files (Python files or YAML/JSON configs)
    if file_path.suffix in {".py", ".yaml", ".yml", ".json"}:
        return AgentUnitFile.from_parent(parent, path=file_path)

    return None


def _is_eval_directory(file_path: Path) -> bool:
    """Check if file is in tests/eval/ directory."""
    parts = file_path.parts
    return "tests" in parts and "eval" in parts


class AgentUnitFile(pytest.File):
    """Pytest file collector for AgentUnit scenarios."""

    def collect(self) -> Generator[AgentUnitItem, None, None]:
        """Collect scenario items from the file."""
        try:
            scenarios = self._discover_scenarios()
            for scenario in scenarios:
                yield AgentUnitItem.from_parent(self, name=scenario.name, scenario=scenario)
        except Exception as e:
            # If we can't load scenarios, create a single failing test
            yield AgentUnitItem.from_parent(
                self, name=f"load_error_{self.path.stem}", scenario=None, load_error=str(e)
            )

    def _discover_scenarios(self) -> list[Scenario]:
        """Discover scenarios from the file."""
        scenarios = []

        if self.path.suffix == ".py":
            scenarios.extend(self._discover_python_scenarios())
        elif self.path.suffix in {".yaml", ".yml", ".json"}:
            scenarios.extend(self._discover_config_scenarios())

        return scenarios

    def _discover_python_scenarios(self) -> list[Scenario]:
        """Discover scenarios from Python files."""
        scenarios = []

        # Import the module and look for scenario objects or functions
        spec = self._import_module()
        if spec is None:
            return scenarios

        module = spec

        # Look for Scenario objects
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, Scenario):
                scenarios.append(obj)
            elif callable(obj) and name.startswith("scenario_"):
                # Try to call functions that look like scenario factories
                try:
                    result = obj()
                    if isinstance(result, Scenario):
                        scenarios.append(result)
                except Exception:
                    # Skip functions that can't be called or don't return scenarios
                    continue

        return scenarios

    def _discover_config_scenarios(self) -> list[Scenario]:
        """Discover scenarios from config files."""
        # This would integrate with the nocode module to load scenarios
        # from YAML/JSON configuration files
        try:
            from agentunit.nocode import ScenarioBuilder

            builder = ScenarioBuilder.from_file(self.path)
            scenario = builder.to_scenario()
            return [scenario]
        except ImportError:
            # nocode module not available
            return []
        except Exception:
            # Failed to load config
            return []

    def _import_module(self) -> Any:
        """Import Python module from file path."""
        try:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(self.path.stem, self.path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[self.path.stem] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None


class AgentUnitItem(pytest.Item):
    """Pytest test item for AgentUnit scenarios."""

    def __init__(
        self,
        name: str,
        parent: AgentUnitFile,
        scenario: Scenario | None = None,
        load_error: str | None = None,
    ) -> None:
        super().__init__(name, parent)
        self.scenario = scenario
        self.load_error = load_error

        # Add agentunit marker
        self.add_marker(pytest.mark.agentunit)

        # Add scenario name marker if available
        if scenario:
            self.add_marker(pytest.mark.scenario(name=scenario.name))

    def runtest(self) -> None:
        """Run the AgentUnit scenario as a pytest test."""
        if self.load_error:
            raise AgentUnitError(f"Failed to load scenario: {self.load_error}")

        if self.scenario is None:
            raise AgentUnitError("No scenario to run")

        # Run the scenario using AgentUnit
        result = run_suite([self.scenario])

        # Check if the scenario passed
        scenario_result = result.scenarios[0]

        # Collect failures
        failures = []
        for run in scenario_result.runs:
            if not run.success:
                error_msg = run.error or "Unknown error"
                failures.append(f"Case {run.case_id}: {error_msg}")

        if failures:
            failure_summary = "\n".join(failures)
            raise AssertionError(f"Scenario '{self.scenario.name}' failed:\n{failure_summary}")

    def repr_failure(self, excinfo: Any) -> str:
        """Represent test failure."""
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self) -> tuple[str, int | None, str]:
        """Report test location info."""
        return str(self.path), None, f"agentunit::{self.name}"
