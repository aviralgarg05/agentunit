"""Schema validation for no-code scenario configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Try to import jsonschema, make it optional
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


@dataclass
class ValidationError:
    """Represents a validation error.

    Attributes:
        path: JSONPath to the error location
        message: Error message
        schema_path: Path in schema where validation failed
    """

    path: str
    message: str
    schema_path: str = ""


@dataclass
class ValidationResult:
    """Result of schema validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors (empty if valid)
        warnings: List of non-fatal warnings
        data: Validated data (if valid)
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.valid


class SchemaValidator:
    """Validate YAML/JSON configurations against AgentUnit schema.

    Uses JSON Schema to validate scenario configuration files,
    ensuring they conform to the AgentUnit specification.

    Examples:
        >>> validator = SchemaValidator()
        >>>
        >>> # Validate a YAML file
        >>> result = validator.validate_file("scenario.yaml")
        >>> if result:
        ...     print("Valid!")
        ... else:
        ...     for error in result.errors:
        ...         print(f"{error.path}: {error.message}")
        >>>
        >>> # Validate dict directly
        >>> config = {"name": "test", "adapter": {...}, "dataset": {...}}
        >>> result = validator.validate(config)
    """

    def __init__(self, schema_path: str | Path | None = None):
        """Initialize validator.

        Args:
            schema_path: Path to JSON schema file (uses bundled schema if None)
        """
        if not HAS_JSONSCHEMA:
            msg = (
                "jsonschema is required for schema validation. "
                "Install it with: pip install jsonschema"
            )
            raise ImportError(msg)

        if schema_path is None:
            # Use bundled schema
            schema_path = Path(__file__).parent.parent / "schemas" / "scenario.json"

        self.schema_path = Path(schema_path)

        with open(self.schema_path) as f:
            self.schema = json.load(f)

        # Create validator
        self.validator_cls = jsonschema.Draft7Validator

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary.

        Args:
            data: Configuration dictionary to validate

        Returns:
            ValidationResult with validation outcome
        """
        validator = self.validator_cls(self.schema)
        errors = []
        warnings = []

        # Collect validation errors
        for error in validator.iter_errors(data):
            # Build path string
            path_parts = [str(p) for p in error.absolute_path]
            path = "/".join(path_parts) if path_parts else "<root>"

            # Build schema path
            schema_path_parts = [str(p) for p in error.absolute_schema_path]
            schema_path = "/".join(schema_path_parts)

            errors.append(
                ValidationError(
                    path=path,
                    message=error.message,
                    schema_path=schema_path,
                )
            )

        # Check for warnings (non-fatal issues)
        warnings.extend(self._check_warnings(data))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data=data if len(errors) == 0 else None,
        )

    def validate_file(self, filepath: str | Path) -> ValidationResult:
        """Validate a YAML or JSON file.

        Args:
            filepath: Path to config file

        Returns:
            ValidationResult with validation outcome
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationError(
                        path="<file>",
                        message=f"File not found: {filepath}",
                    )
                ],
            )

        # Load file based on extension
        try:
            with open(filepath) as f:
                if filepath.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif filepath.suffix == ".json":
                    data = json.load(f)
                else:
                    return ValidationResult(
                        valid=False,
                        errors=[
                            ValidationError(
                                path="<file>",
                                message=f"Unsupported file format: {filepath.suffix}",
                            )
                        ],
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationError(
                        path="<file>",
                        message=f"Parse error: {e}",
                    )
                ],
            )

        return self.validate(data)

    def _check_warnings(self, data: dict[str, Any]) -> list[str]:
        """Check for non-fatal warnings.

        Args:
            data: Configuration data

        Returns:
            List of warning messages
        """
        warnings = []

        # Warn if no metrics specified
        if "metrics" not in data or not data["metrics"]:
            warnings.append("No metrics specified - tests will run but not evaluate quality")

        # Warn if using default timeout
        if "timeout" not in data:
            warnings.append("No timeout specified - using default")

        # Warn if dataset is very small
        if "dataset" in data and "cases" in data["dataset"]:
            if len(data["dataset"]["cases"]) < 3:
                warnings.append(
                    "Dataset has fewer than 3 cases - results may not be statistically significant"
                )

        return warnings

    def get_schema_docs(self) -> str:
        """Get human-readable schema documentation.

        Returns:
            Formatted documentation string
        """
        lines = [
            "AgentUnit Scenario Schema",
            "=" * 40,
            "",
            f"Schema version: {self.schema.get('$id', 'unknown')}",
            "",
            "Required fields:",
        ]

        required = self.schema.get("required", [])
        for field in required:
            prop = self.schema.get("properties", {}).get(field, {})
            desc = prop.get("description", "No description")
            lines.append(f"  - {field}: {desc}")

        lines.extend(
            [
                "",
                "Optional fields:",
            ]
        )

        optional = set(self.schema.get("properties", {}).keys()) - set(required)
        for field in sorted(optional):
            prop = self.schema.get("properties", {}).get(field, {})
            desc = prop.get("description", "No description")
            lines.append(f"  - {field}: {desc}")

        return "\n".join(lines)
