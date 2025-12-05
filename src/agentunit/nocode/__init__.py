"""No-code scenario builder with declarative YAML/JSON configuration.

This module enables non-programmers to define test scenarios using simple
YAML or JSON files, which are then automatically converted to Python code.

Features:
- YAML/JSON schema for test scenarios
- JSON Schema validation for config files
- Automatic Python code generation from declarative configs
- CLI tools for format conversion and validation
- Round-trip conversion (Python <-> YAML <-> JSON)
- Template library for common patterns

Examples:
    >>> from agentunit.nocode import ScenarioBuilder, SchemaValidator
    >>>
    >>> # Load scenario from YAML
    >>> builder = ScenarioBuilder()
    >>> scenario = builder.from_yaml("my_test.yaml")
    >>>
    >>> # Validate configuration
    >>> validator = SchemaValidator()
    >>> is_valid = validator.validate_file("my_test.yaml")
    >>>
    >>> # Generate Python code
    >>> python_code = builder.to_python("my_test.yaml")

Classes exported (available via lazy loading):
- ScenarioBuilder: Main interface for building scenarios from config files
- SchemaValidator: Validate YAML/JSON against AgentUnit schema
- CodeGenerator: Generate Python code from declarative configs
- ConfigConverter: Convert between YAML, JSON, and Python formats
- TemplateLibrary: Pre-built templates for common scenarios

All classes use PEP 562 lazy loading.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .builder import ScenarioBuilder
    from .converter import ConfigConverter, ConversionFormat
    from .generator import CodeGenerator, GeneratedCode
    from .templates import ScenarioTemplate, TemplateLibrary
    from .validator import SchemaValidator, ValidationResult

__version__ = "0.4.0"
__author__ = "AgentUnit Contributors"
__description__ = "No-code scenario builder with declarative configuration"

__all__ = [
    "CodeGenerator",
    "ConfigConverter",
    "ConversionFormat",
    "GeneratedCode",
    # Core classes
    "ScenarioBuilder",
    "ScenarioTemplate",
    "SchemaValidator",
    "TemplateLibrary",
    "ValidationResult",
    "__author__",
    "__description__",
    # Version info
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazy load nocode components."""

    if name == "ScenarioBuilder":
        from .builder import ScenarioBuilder

        return ScenarioBuilder

    if name == "SchemaValidator":
        from .validator import SchemaValidator

        return SchemaValidator

    if name == "ValidationResult":
        from .validator import ValidationResult

        return ValidationResult

    if name == "CodeGenerator":
        from .generator import CodeGenerator

        return CodeGenerator

    if name == "GeneratedCode":
        from .generator import GeneratedCode

        return GeneratedCode

    if name == "ConfigConverter":
        from .converter import ConfigConverter

        return ConfigConverter

    if name == "ConversionFormat":
        from .converter import ConversionFormat

        return ConversionFormat

    if name == "TemplateLibrary":
        from .templates import TemplateLibrary

        return TemplateLibrary

    if name == "ScenarioTemplate":
        from .templates import ScenarioTemplate

        return ScenarioTemplate

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    """Return list of available attributes for autocomplete."""
    return __all__
