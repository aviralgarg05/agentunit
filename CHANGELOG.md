# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2025-10-01

### Added
- PEP 562 lazy loading implementation for optional framework dependencies
- Comprehensive `TYPE_CHECKING` imports for static analysis and IDE support
- Graceful degradation for telemetry when OpenTelemetry is not available
- Enhanced dataset registry with optional HuggingFace Hub support
- `.codacy.yml` configuration for linter-friendly lazy loading pattern
- `OPTIONAL_DEPENDENCIES.md` documentation guide

### Changed
- Refactored `src/agentunit/adapters/__init__.py` to use `__getattr__` for lazy imports
- Updated `src/agentunit/telemetry/tracing.py` with no-op fallbacks
- Enhanced module docstrings documenting lazy loading pattern
- Bumped project version to `0.4.0` in `pyproject.toml` and `src/agentunit/__init__.py`

### Fixed
- ModuleNotFoundError cascades from optional dependencies
- Runtime import errors when optional packages not installed
- Adapter availability checks now work correctly without all dependencies

## [0.3.0] - 2025-09-30

### Added
- Expanded scenario helper documentation covering Phidata, PromptFlow, OpenAI Swarm, Anthropic on Bedrock, Mistral Server, and Rasa adapters.
- New framework integration catalog detailing prerequisites, helper signatures, and customization strategies.
- Comprehensive scenario template (`docs/templates/framework_scenarios.py`) showcasing nine helper-driven examples.

### Changed
- Bumped project version to `0.3.0` in `pyproject.toml` and `src/agentunit/__init__.py`.

## [0.2.0] - 2024-??-??

- Initial changelog placeholder.
