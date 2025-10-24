# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-10-24

### Added
- Comprehensive CONTRIBUTING.md with contribution guidelines and issue labels
- SECURITY.md with security policy and disclosure process
- Detailed adapter guide with implementation examples (docs/adapters.md)
- Metrics catalog documenting all available metrics (docs/metrics-catalog.md)
- Telemetry configuration guide for OpenTelemetry (docs/telemetry.md)
- Comparison documentation vs. Ragas, DEEPEVAL, and other tools (docs/comparison.md)
- Sample benchmark suite with canonical RAG dataset (examples/benchmark/)
- Pre-commit configuration with Ruff and formatting tools (.pre-commit-config.yaml)
- Ruff configuration file (ruff.toml) with 25+ enabled rule sets
- py.typed marker for PEP 561 typed package support
- JSON Schema definitions for result artifacts (src/agentunit/schemas/)
- GitHub issue templates for bugs, features, documentation, and questions (.github/ISSUE_TEMPLATE/)
- GitHub issue template config with security contact link (.github/ISSUE_TEMPLATE/config.yml)
- GitHub PR template for standardized contributions (.github/pull_request_template.md)
- GitHub labels configuration with 50+ organized labels (.github/labels.yml)
- Example CI workflow demonstrating AgentUnit integration (.github/workflows/example-ci.yml)
- CI badges to README (PyPI, Python versions, license, CI status, coverage)
- Copy-paste quickstart examples in README (programmatic and YAML)
- Optional extras table documenting all installation options
- Hacktoberfest-ready labels for open source events
- Enhanced CLI help with examples, defaults, and environment variables
- JSON schemas for results, metrics, and datasets

### Changed
- Updated Python version requirement from 3.9+ to 3.10+ (aligned with pyproject.toml)
- Improved README with badges, quickstart examples, and better navigation
- Enhanced documentation map with links to all new guides
- Updated CLI help output with examples and environment variables
- Verification date updated to 2025-10-24
- Bumped version to 0.6.0

### Fixed
- Removed stray code block from README top (PyPI description rendering issue)
- Python version mismatch between README and package metadata

## [0.5.0] - 2025-10-07

### Added
- Regression detection and statistical comparison tools
- A/B testing framework with bootstrap confidence intervals
- Version comparison and change tracking
- Comprehensive comparison reports (Markdown, JSON)
- RegressionDetector for automated quality gate enforcement
- SignificanceAnalyzer with effect size calculations

### Changed
- Improved documentation structure and cross-linking
- Updated README with clearer installation instructions
- Standardized docs/ directory organization

### Fixed
- RegressionDetector now returns native boolean for identity checks
- ScenarioBuilder handles shorthand adapter types correctly
- Consolidated metric instantiation placeholder in no-code builder

### Testing
- All 144 unit tests passing
- 10 tests skipped (optional dependencies)
- 32 warnings from third-party dependencies

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
