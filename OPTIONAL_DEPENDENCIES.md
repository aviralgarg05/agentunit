# AgentUnit Optional Dependencies Guide

## Overview

AgentUnit uses a **lazy-loading pattern** (PEP 562) to support multiple agent frameworks without requiring all dependencies to be installed upfront. This keeps the core package lightweight while allowing users to install only the integrations they need.

## How It Works

### Lazy Loading Implementation

The `agentunit.adapters` module uses Python's `__getattr__` hook to defer imports:

```python
# These imports only happen when you actually use the adapter
from agentunit.adapters import LangGraphAdapter  # Imports langgraph only now
from agentunit.adapters import CrewAIAdapter     # Imports crewai only now
```

### Static Type Checking Support

For IDE autocomplete and linters (mypy, pylint, Codacy), we use `TYPE_CHECKING` imports:

```python
if TYPE_CHECKING:
    from .langgraph import LangGraphAdapter
    # These imports are ONLY seen by type checkers, never at runtime
```

## Current Status

### ✅ Working Without Optional Dependencies

The following work **without** installing optional packages:

- **Core Evaluation**: Scenario running, assertions, reporting
- **Built-in Datasets**: JSON/CSV loading, built-in test cases
- **Base Adapters**: `BaseAdapter`, `AdapterOutcome`
- **Telemetry**: Falls back to no-op tracing when OpenTelemetry is absent

### ⚠️ Require Optional Dependencies

| Adapter | Required Package | Install Command |
|---------|-----------------|-----------------|
| LangGraphAdapter | `pyyaml` | `pip install pyyaml` |
| HaystackAdapter | `httpx`, `pyyaml` | `pip install httpx pyyaml` |
| Dataset Registry (HuggingFace) | `huggingface_hub` | `pip install huggingface_hub` |
| OpenTelemetry Tracing | `opentelemetry-api`, `opentelemetry-sdk` | `pip install opentelemetry-api opentelemetry-sdk` |

## Installation Options

### Minimal Install (Core Only)
```bash
pip install agentunit --no-deps
pip install <core-dependencies-only>
```

### Full Install (All Integrations)
```bash
pip install agentunit
# Installs all dependencies from pyproject.toml
```

### Selective Install (Recommended)
```bash
pip install agentunit --no-deps
pip install <core-dependencies>

# Then install only what you need:
pip install pyyaml httpx                    # For LangGraph, Haystack
pip install huggingface_hub                 # For HuggingFace datasets
pip install opentelemetry-api opentelemetry-sdk  # For distributed tracing
```

## Codacy / Linter Warnings

### Expected Warnings

Codacy and similar tools may flag:
```
undefined-all-variable: Names in __all__ are not defined
```

**This is expected and benign.** The lazy-loading pattern defers imports until runtime via `__getattr__`, so names aren't in the module namespace at parse time.

### Why This Is Safe

1. **TYPE_CHECKING imports** provide static type information
2. **__dir__ implementation** ensures autocomplete works
3. **Runtime validation** confirms lazy loading functions correctly
4. **PEP 562** officially supports this pattern

### Suppression Configuration

We've added `.codacy.yml` with file-specific overrides:

```yaml
file_overrides:
  - path: "src/agentunit/adapters/__init__.py"
    tools:
      - pylint:
          disable:
            - undefined-variable
            - undefined-all-variable
```

## Validation Results

### Runtime Tests Passed ✅

- ✓ Base imports work without optional dependencies
- ✓ Lazy loading triggers on first access
- ✓ Missing dependencies raise clear ImportError with install instructions
- ✓ TYPE_CHECKING provides autocomplete without runtime imports
- ✓ No namespace pollution before access
- ✓ Telemetry gracefully degrades without OpenTelemetry

### Example Validation Output

```
✓ Step 1: Base imports successful
✓ Step 2: __all__ contains 20 entries
✓ Step 3: dir() returns 20 names (autocomplete works)
✓ Step 4: No namespace pollution before access
✓ Step 5: Lazy load succeeded for CrewAIAdapter
✓ Step 6: LangGraphAdapter requires yaml (expected)
✓ Step 7: TYPE_CHECKING imports present for linters
✓ Step 8: Module docstring documents lazy-loading pattern
```

## Troubleshooting

### ImportError: No module named 'X'

**Solution**: Install the missing dependency:
```bash
pip install <package-name>
```

### Linter complains about undefined names in __all__

**Solution**: This is expected with lazy loading. Either:
1. Use the provided `.codacy.yml` configuration
2. Add `# noqa` comments for specific lines
3. Configure your linter to ignore this specific pattern

### Autocomplete not working in IDE

**Solution**: Ensure your IDE supports `TYPE_CHECKING` imports. Most modern Python IDEs (VS Code, PyCharm) handle this automatically.

## References

- [PEP 562 – Module __getattr__ and __dir__](https://peps.python.org/pep-0562/)
- [PEP 484 – Type Hints (TYPE_CHECKING)](https://peps.python.org/pep-0484/)
- [Python Import System Documentation](https://docs.python.org/3/reference/import.html)

## Summary

✅ **Everything is working as designed**
- Lazy loading preserves memory and startup time
- Optional dependencies are truly optional
- Static analysis tools have all the info they need via TYPE_CHECKING
- Codacy warnings are expected and documented
- Runtime behavior is validated and correct

**No action required unless you want to use specific adapters** – in which case, install the corresponding optional packages.
