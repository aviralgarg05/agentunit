# Contributing to AgentUnit

Thank you for your interest in contributing to AgentUnit! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Labels](#issue-labels)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize for mistakes

## Getting Started

### Prerequisites

- Python 3.10 or later
- Poetry (recommended) or pip
- Git

### Finding Issues to Work On

Look for issues labeled:
- `good-first-issue` - Great for newcomers
- `help-wanted` - Community help needed
- `bug` - Bug fixes welcome
- `enhancement` - Feature requests
- `documentation` - Documentation improvements

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/agentunit.git
cd agentunit
```

3. Add upstream remote:

```bash
git remote add upstream https://github.com/aviralgarg05/agentunit.git
```

4. Install dependencies:

```bash
# Using Poetry (recommended)
poetry install --with dev
poetry shell

# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

5. Install pre-commit hooks:

```bash
pre-commit install
```

## Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions or changes
- `chore/description` - Maintenance tasks

### Code Style

We use the following tools to maintain code quality:

- **Ruff** - Fast Python linter and formatter
- **Black** - Code formatting (via Ruff)
- **isort** - Import sorting (via Ruff)
- **mypy** - Type checking

Run linting and formatting:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type check
mypy src/agentunit
```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Test additions or changes
- `chore` - Maintenance tasks
- `perf` - Performance improvements

Example:

```
feat(adapters): add Anthropic Bedrock adapter

Implement new adapter for AWS Bedrock runtime with Claude models.
Includes support for streaming responses and cost tracking.

Closes #123
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=agentunit --cov-report=html

# Run specific test file
poetry run pytest tests/test_adapters.py

# Run tests matching a pattern
poetry run pytest -k "test_adapter"

# Run with verbose output
poetry run pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use descriptive test names that explain what is being tested
- Include docstrings for complex tests
- Use fixtures for common setup
- Aim for >80% code coverage for new code

Example:

```python
def test_adapter_handles_timeout():
    """Test that adapter properly handles request timeout."""
    adapter = MockAdapter(timeout=1)
    with pytest.raises(TimeoutError):
        adapter.query("test", timeout=0.5)
```

## Documentation

### Documentation Standards

- Update documentation for any API changes
- Add docstrings to all public classes and functions
- Include examples in docstrings
- Update README.md for user-facing changes
- Add entries to docs/ for new features

### Docstring Format

Use Google-style docstrings:

```python
def evaluate_scenario(
    scenario: Scenario,
    metrics: list[Metric],
    timeout: int = 30
) -> Result:
    """Evaluate a scenario against specified metrics.
    
    Args:
        scenario: The scenario to evaluate
        metrics: List of metrics to compute
        timeout: Maximum execution time in seconds
        
    Returns:
        Result object containing evaluation metrics and statistics
        
    Raises:
        TimeoutError: If evaluation exceeds timeout
        ValueError: If scenario or metrics are invalid
        
    Example:
        >>> scenario = Scenario(name="test", ...)
        >>> metrics = [ExactMatch(), Latency()]
        >>> result = evaluate_scenario(scenario, metrics)
        >>> print(result.success_rate)
        0.85
    """
```

### Building Documentation

```bash
# Build docs locally
cd docs
make html

# View docs
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## Pull Request Process

### Before Submitting

1. Update your branch with latest upstream:

```bash
git fetch upstream
git rebase upstream/main
```

2. Run tests and linting:

```bash
poetry run pytest
ruff check .
mypy src/agentunit
```

3. Update documentation if needed
4. Add entry to CHANGELOG.md under "Unreleased"

### Submitting PR

1. Push your branch to your fork:

```bash
git push origin your-branch-name
```

2. Create pull request on GitHub
3. Fill out the PR template completely
4. Link related issues
5. Request review from maintainers

### PR Review Process

- Maintainers will review within 3-5 business days
- Address review comments promptly
- Keep discussions focused and professional
- Be open to feedback and suggestions
- PRs require at least one approval before merging

### After Merge

- Delete your branch
- Close linked issues if resolved
- Thank reviewers and collaborators

## Issue Labels

### Priority Labels

- `priority:critical` - Blocking issues, security vulnerabilities
- `priority:high` - Important features or bugs
- `priority:medium` - Standard priority
- `priority:low` - Nice to have

### Type Labels

- `bug` - Something is broken
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `question` - Questions about usage
- `performance` - Performance improvements
- `technical-debt` - Code quality improvements

### Status Labels

- `status:needs-triage` - Needs initial review
- `status:needs-info` - Awaiting more information
- `status:in-progress` - Being worked on
- `status:blocked` - Blocked by dependency
- `status:wontfix` - Will not be addressed

### Special Labels

- `good-first-issue` - Good for newcomers
- `help-wanted` - Community help needed
- `hacktoberfest` - Valid for Hacktoberfest
- `breaking-change` - Breaks backward compatibility
- `security` - Security related

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible features
- PATCH version for backwards-compatible bug fixes

### Release Steps

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with release notes
3. Create release commit:

```bash
git commit -m "chore(release): bump version to X.Y.Z"
```

4. Create and push tag:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags
```

5. Create GitHub release with changelog
6. CI will automatically publish to PyPI (via trusted publishing)

## Additional Resources

- [Architecture Overview](docs/architecture.md)
- [Writing Scenarios](docs/writing-scenarios.md)
- [Adapter Development Guide](docs/adapters.md)
- [Metrics Catalog](docs/metrics-catalog.md)

## Questions?

- Check existing [GitHub Issues](https://github.com/aviralgarg05/agentunit/issues)
- Ask in [GitHub Discussions](https://github.com/aviralgarg05/agentunit/discussions)
- Review [documentation](docs/)

Thank you for contributing to AgentUnit!
