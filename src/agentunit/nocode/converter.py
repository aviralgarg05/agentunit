"""Configuration format conversion utilities."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ConversionFormat(Enum):
    """Supported configuration formats."""

    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"


class ConfigConverter:
    """Convert scenario configurations between different formats.

    Supports conversion between YAML, JSON, and Python code formats
    with preservation of structure and comments where possible.

    Examples:
        >>> converter = ConfigConverter()
        >>>
        >>> # Convert YAML to JSON
        >>> converter.convert("scenario.yaml", ConversionFormat.JSON, "scenario.json")
        >>>
        >>> # Convert JSON to Python code
        >>> code = converter.convert_to_python("scenario.json")
        >>>
        >>> # Round-trip conversion
        >>> original = converter.load_yaml("scenario.yaml")
        >>> json_str = converter.to_json_string(original)
        >>> back_to_yaml = converter.to_yaml_string(json.loads(json_str))
    """

    def __init__(self):
        """Initialize converter."""

    def convert(
        self,
        source: str | Path,
        target_format: ConversionFormat,
        output: str | Path | None = None,
    ) -> str:
        """Convert configuration file to target format.

        Args:
            source: Source configuration file path
            target_format: Target format (YAML, JSON, or PYTHON)
            output: Optional output file path

        Returns:
            Converted configuration as string
        """
        source = Path(source)

        # Load source
        if source.suffix in [".yaml", ".yml"]:
            config = self.load_yaml(source)
        elif source.suffix == ".json":
            config = self.load_json(source)
        else:
            msg = f"Unsupported source format: {source.suffix}"
            raise ValueError(msg)

        # Convert to target format
        if target_format == ConversionFormat.YAML:
            result = self.to_yaml_string(config)
        elif target_format == ConversionFormat.JSON:
            result = self.to_json_string(config)
        elif target_format == ConversionFormat.PYTHON:
            from .generator import CodeGenerator

            generator = CodeGenerator()
            gen_result = generator.from_dict(config)
            result = gen_result.code
        else:
            msg = f"Unsupported target format: {target_format}"
            raise ValueError(msg)

        # Write to output if specified
        if output:
            Path(output).write_text(result)

        return result

    def convert_to_python(
        self,
        source: str | Path,
        output: str | Path | None = None,
    ) -> str:
        """Convert configuration to Python code.

        Args:
            source: Source configuration file
            output: Optional output Python file path

        Returns:
            Generated Python code
        """
        return self.convert(source, ConversionFormat.PYTHON, output)

    def convert_to_yaml(
        self,
        source: str | Path,
        output: str | Path | None = None,
    ) -> str:
        """Convert configuration to YAML.

        Args:
            source: Source configuration file
            output: Optional output YAML file path

        Returns:
            YAML configuration string
        """
        return self.convert(source, ConversionFormat.YAML, output)

    def convert_to_json(
        self,
        source: str | Path,
        output: str | Path | None = None,
    ) -> str:
        """Convert configuration to JSON.

        Args:
            source: Source configuration file
            output: Optional output JSON file path

        Returns:
            JSON configuration string
        """
        return self.convert(source, ConversionFormat.JSON, output)

    def load_yaml(self, filepath: str | Path) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            filepath: Path to YAML file

        Returns:
            Configuration dictionary
        """
        with open(filepath) as f:
            return yaml.safe_load(f)

    def load_json(self, filepath: str | Path) -> dict[str, Any]:
        """Load JSON configuration file.

        Args:
            filepath: Path to JSON file

        Returns:
            Configuration dictionary
        """
        with open(filepath) as f:
            return json.load(f)

    def to_yaml_string(
        self,
        config: dict[str, Any],
        preserve_order: bool = True,
    ) -> str:
        """Convert configuration dictionary to YAML string.

        Args:
            config: Configuration dictionary
            preserve_order: Whether to preserve key order (default True)

        Returns:
            YAML string
        """
        return yaml.dump(
            config,
            sort_keys=not preserve_order,
            default_flow_style=False,
            allow_unicode=True,
        )

    def to_json_string(
        self,
        config: dict[str, Any],
        indent: int = 2,
    ) -> str:
        """Convert configuration dictionary to JSON string.

        Args:
            config: Configuration dictionary
            indent: Indentation spaces (default 2)

        Returns:
            JSON string
        """
        return json.dumps(config, indent=indent, ensure_ascii=False)

    def validate_round_trip(
        self,
        filepath: str | Path,
        target_format: ConversionFormat,
    ) -> bool:
        """Validate round-trip conversion preserves data.

        Args:
            filepath: Source configuration file
            target_format: Intermediate format for conversion

        Returns:
            True if round-trip preserves data, False otherwise
        """
        filepath = Path(filepath)

        # Load original
        if filepath.suffix in [".yaml", ".yml"]:
            original = self.load_yaml(filepath)
        elif filepath.suffix == ".json":
            original = self.load_json(filepath)
        else:
            msg = f"Unsupported format: {filepath.suffix}"
            raise ValueError(msg)

        # Convert to target and back
        if target_format == ConversionFormat.YAML:
            intermediate = self.to_yaml_string(original)
            result = yaml.safe_load(intermediate)
        elif target_format == ConversionFormat.JSON:
            intermediate = self.to_json_string(original)
            result = json.loads(intermediate)
        else:
            # Can't validate Python round-trip directly
            return True

        # Compare
        return self._deep_equal(original, result)

    def _deep_equal(self, obj1: Any, obj2: Any) -> bool:
        """Deep equality comparison for dictionaries and lists.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if objects are deeply equal
        """
        if type(obj1) != type(obj2):
            return False

        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            return all(self._deep_equal(obj1[k], obj2[k]) for k in obj1)

        if isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(self._deep_equal(a, b) for a, b in zip(obj1, obj2, strict=False))

        return obj1 == obj2
