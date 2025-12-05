"""Main scenario builder interface for no-code scenario creation."""

from __future__ import annotations

import inspect
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from agentunit.adapters.base import BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.datasets.base import DatasetCase, DatasetSource

from .generator import CodeGenerator
from .validator import SchemaValidator


class ScenarioBuilder:
    """Build AgentUnit scenarios from YAML/JSON configuration files.

    This is the main entry point for the no-code scenario builder,
    combining validation, parsing, and scenario instantiation.

    Examples:
        >>> builder = ScenarioBuilder()
        >>>
        >>> # Load and validate scenario from YAML
        >>> scenario = builder.from_yaml("my_scenario.yaml")
        >>>
        >>> # Generate Python code
        >>> code = builder.to_python("my_scenario.yaml")
        >>> print(code)
        >>>
        >>> # Load multiple scenarios from a directory
        >>> scenarios = builder.from_directory("scenarios/")
    """

    def __init__(self, validate: bool = True):
        """Initialize scenario builder.

        Args:
            validate: Whether to validate configs before building (default True)
        """
        self.validate = validate
        self.validator = SchemaValidator() if validate else None
        self.generator = CodeGenerator()

    def from_dict(
        self,
        config: dict[str, Any],
        adapter: BaseAdapter | None = None,
    ) -> Scenario:
        """Build scenario from configuration dictionary.

        Args:
            config: Scenario configuration
            adapter: Optional adapter instance (if not in config)

        Returns:
            Configured Scenario object
        """
        # Validate if enabled
        if self.validate:
            result = self.validator.validate(config)
            if not result.valid:
                errors_str = "\n".join(f"  {e.path}: {e.message}" for e in result.errors)
                msg = f"Invalid scenario configuration:\n{errors_str}"
                raise ValueError(msg)

        # Extract fields
        name = config["name"]

        # Build dataset
        dataset = self._build_dataset(config["dataset"])

        # Use provided adapter or build from config
        if adapter is None:
            if "adapter" not in config:
                msg = "Adapter must be provided or specified in config"
                raise ValueError(msg)
            adapter = self._build_adapter(config["adapter"])

        # Build metrics
        metrics = self._build_metrics(config.get("metrics", []))
        scenario_kwargs = {
            "name": name,
            "adapter": adapter,
            "dataset": dataset,
        }
        if "retries" in config:
            scenario_kwargs["retries"] = config["retries"]
        if "timeout" in config:
            scenario_kwargs["timeout"] = config["timeout"]
        scenario = self._instantiate_scenario(scenario_kwargs)
        self._apply_metrics(scenario, metrics)
        return scenario

    def from_yaml(
        self,
        filepath: str | Path,
        adapter: BaseAdapter | None = None,
    ) -> Scenario:
        """Build scenario from YAML file.

        Args:
            filepath: Path to YAML configuration file
            adapter: Optional adapter instance

        Returns:
            Configured Scenario object
        """
        with open(filepath) as f:
            config = yaml.safe_load(f)

        return self.from_dict(config, adapter)

    def from_json(
        self,
        filepath: str | Path,
        adapter: BaseAdapter | None = None,
    ) -> Scenario:
        """Build scenario from JSON file.

        Args:
            filepath: Path to JSON configuration file
            adapter: Optional adapter instance

        Returns:
            Configured Scenario object
        """
        with open(filepath) as f:
            config = json.load(f)

        return self.from_dict(config, adapter)

    def from_directory(
        self,
        dirpath: str | Path,
        pattern: str = "*.yaml",
        adapter: BaseAdapter | None = None,
    ) -> list[Scenario]:
        """Load all scenarios from a directory.

        Args:
            dirpath: Path to directory containing scenario files
            pattern: Glob pattern for scenario files (default: *.yaml)
            adapter: Optional adapter to use for all scenarios

        Returns:
            List of Scenario objects
        """
        dirpath = Path(dirpath)
        scenarios = []

        for filepath in sorted(dirpath.glob(pattern)):
            try:
                if filepath.suffix in [".yaml", ".yml"]:
                    scenario = self.from_yaml(filepath, adapter)
                elif filepath.suffix == ".json":
                    scenario = self.from_json(filepath, adapter)
                else:
                    continue

                scenarios.append(scenario)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

        return scenarios

    def to_python(self, filepath: str | Path) -> str:
        """Generate Python code from configuration file.

        Args:
            filepath: Path to configuration file

        Returns:
            Generated Python code as string
        """
        result = self.generator.from_file(filepath)
        return result.code

    def to_yaml(self, scenario: Scenario, filepath: str | Path | None = None) -> str:
        """Convert Scenario object to YAML configuration.

        Args:
            scenario: Scenario to convert
            filepath: Optional path to write YAML to

        Returns:
            YAML configuration string
        """
        config = self._scenario_to_dict(scenario)
        yaml_str = yaml.dump(config, sort_keys=False, default_flow_style=False)

        if filepath:
            Path(filepath).write_text(yaml_str)

        return yaml_str

    def to_json(self, scenario: Scenario, filepath: str | Path | None = None) -> str:
        """Convert Scenario object to JSON configuration.

        Args:
            scenario: Scenario to convert
            filepath: Optional path to write JSON to

        Returns:
            JSON configuration string
        """
        config = self._scenario_to_dict(scenario)
        json_str = json.dumps(config, indent=2)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    def _build_dataset(self, dataset_config: dict[str, Any]) -> DatasetSource:
        """Build dataset from configuration."""
        if "cases" in dataset_config:
            case_params = inspect.signature(DatasetCase).parameters
            cases = []
            for i, case_data in enumerate(dataset_config["cases"]):
                case_input = case_data.get("input", case_data.get("query"))
                if case_input is None:
                    msg = "Dataset case must define 'input' or 'query'"
                    raise ValueError(msg)
                case_expected = case_data.get("expected", case_data.get("expected_output"))
                case_kwargs: dict[str, Any] = {}
                case_id = case_data.get("id", f"case_{i}")
                if "id" in case_params:
                    case_kwargs["id"] = case_id
                if "query" in case_params:
                    case_kwargs["query"] = case_input
                elif "input" in case_params:
                    case_kwargs["input"] = case_input
                else:
                    msg = "DatasetCase must accept either 'query' or 'input'"
                    raise ValueError(msg)
                if case_expected is not None:
                    if "expected_output" in case_params:
                        case_kwargs["expected_output"] = case_expected
                    elif "expected" in case_params:
                        case_kwargs["expected"] = case_expected
                if "context" in case_params and "context" in case_data:
                    case_kwargs["context"] = case_data["context"]
                if "metadata" in case_params:
                    case_kwargs["metadata"] = case_data.get("metadata", {})
                cases.append(DatasetCase(**case_kwargs))
            return DatasetSource.from_list(cases)

        if dataset_config.get("source") == "file":
            return DatasetSource.from_file(dataset_config["path"])

        msg = f"Unsupported dataset configuration: {dataset_config}"
        raise ValueError(msg)

    def _build_adapter(self, adapter_config: dict[str, Any]) -> BaseAdapter:
        """Build adapter from configuration."""
        adapter_type = adapter_config.get("type")
        if not adapter_type:
            msg = "Adapter configuration must include a 'type' field pointing to the adapter class"
            raise ValueError(msg)

        module_path, _, class_name = adapter_type.rpartition(".")
        if not module_path or not class_name:
            msg = (
                "Adapter instantiation from shorthand type names is not implemented yet; "
                "provide a fully qualified class path"
            )
            raise NotImplementedError(msg)

        try:
            module = import_module(module_path)
        except ImportError as exc:
            msg = f"Could not import adapter module '{module_path}'"
            raise ImportError(msg) from exc

        try:
            adapter_cls = getattr(module, class_name)
        except AttributeError as exc:
            msg = f"Adapter class '{class_name}' not found in module '{module_path}'"
            raise ImportError(msg) from exc

        init_kwargs = adapter_config.get("config") or {}
        if not isinstance(init_kwargs, dict):
            msg = "Adapter 'config' must be a mapping of constructor arguments"
            raise ValueError(msg)

        adapter_instance = adapter_cls(**init_kwargs)

        if not isinstance(adapter_instance, BaseAdapter):
            msg = f"Adapter '{adapter_type}' must inherit from BaseAdapter"
            raise TypeError(msg)

        return adapter_instance

    def _build_metrics(self, metrics_config: list[Any]) -> list[Any]:
        """Build metrics from configuration."""
        # Simplified - would need to import and instantiate actual metrics
        metrics = []

        for metric in metrics_config:
            if isinstance(metric, (str, dict)):
                # Placeholder: metric instantiation not yet implemented
                continue

        return metrics

    def _scenario_to_dict(self, scenario: Scenario) -> dict[str, Any]:
        """Convert Scenario to configuration dictionary."""
        config = {
            "name": scenario.name,
            "adapter": {
                "type": "custom",  # Would need to determine actual type
                "config": {},
            },
            "dataset": {
                "cases": [],
            },
        }

        # Extract cases from dataset
        if hasattr(scenario.dataset, "cases"):
            for case in scenario.dataset.cases:
                input_value = getattr(case, "input", getattr(case, "query", None))
                expected_value = getattr(case, "expected", getattr(case, "expected_output", None))
                config["dataset"]["cases"].append(
                    {
                        "input": input_value,
                        "expected": expected_value,
                        "context": getattr(case, "context", None),
                        "metadata": getattr(case, "metadata", None),
                    }
                )

        # Add optional fields
        if hasattr(scenario, "metrics") and scenario.metrics:
            config["metrics"] = [m.__class__.__name__ for m in scenario.metrics]

        if hasattr(scenario, "retries"):
            config["retries"] = scenario.retries

        if hasattr(scenario, "timeout"):
            config["timeout"] = scenario.timeout

        return config

    def _instantiate_scenario(self, scenario_kwargs: dict[str, Any]) -> Scenario:
        init_signature = inspect.signature(Scenario.__init__)
        allows_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in init_signature.parameters.values()
        )
        valid_params = {
            name
            for name, param in init_signature.parameters.items()
            if name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered_kwargs = (
            scenario_kwargs.copy()
            if allows_var_kwargs
            else {key: value for key, value in scenario_kwargs.items() if key in valid_params}
        )
        missing_required = [
            name
            for name, param in init_signature.parameters.items()
            if name in valid_params
            and param.default is inspect._empty
            and name not in filtered_kwargs
        ]
        if missing_required:
            missing = ", ".join(sorted(missing_required))
            msg = f"Scenario constructor missing required arguments: {missing}"
            raise ValueError(msg)
        return Scenario(**filtered_kwargs)

    def _apply_metrics(self, scenario: Scenario, metrics: list[Any]) -> None:
        if not metrics:
            return
        setter = getattr(scenario, "set_metrics", None)
        if callable(setter):
            setter(metrics)
            return
        if hasattr(scenario, "metrics"):
            scenario.metrics = metrics
            return
        msg = "Scenario configuration includes metrics, but the Scenario implementation does not support them."
        raise ValueError(msg)
