# AgentUnit Pytest Plugin

The AgentUnit pytest plugin allows you to run AgentUnit evaluation scenarios as pytest tests, providing seamless integration with pytest's test discovery, execution, and reporting features.

## Installation

The pytest plugin is automatically available when you install AgentUnit:

```bash
pip install agentunit
```

## Usage

### Basic Usage

1. Create scenario files in the `tests/eval/` directory
2. Run pytest to discover and execute scenarios:

```bash
pytest tests/eval/
```

### Scenario Discovery

The plugin automatically discovers scenarios from files in `tests/eval/`:

- **Python files** (`.py`): Looks for `Scenario` objects and functions starting with `scenario_`
- **Config files** (`.yaml`, `.yml`, `.json`): Loads scenarios using the nocode module

### Python Scenario Files

Create Python files with scenario objects or factory functions:

```python
# tests/eval/my_scenarios.py
from agentunit import Scenario
from agentunit.adapters.base import BaseAdapter, AdapterOutcome
from agentunit.datasets.base import DatasetCase, DatasetSource

class SimpleAdapter(BaseAdapter):
    name = "simple"

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

class MyDataset(DatasetSource):
    def __init__(self):
        super().__init__(name="my-dataset", loader=self._generate_cases)

    def _generate_cases(self):
        return [
            DatasetCase(
                id="test1",
                query="Hello",
                expected_output="Hi there!",
            )
        ]

def my_agent(payload):
    return {"result": "Hi there!"}

# This scenario will be auto-discovered
greeting_scenario = Scenario(
    name="greeting-test",
    adapter=SimpleAdapter(my_agent),
    dataset=MyDataset(),
)

# Factory functions starting with 'scenario_' are also discovered
def scenario_advanced_test():
    return Scenario(
        name="advanced-test",
        adapter=SimpleAdapter(my_agent),
        dataset=MyDataset(),
    )
```

### Pytest Integration Features

#### Markers

The plugin adds pytest markers for filtering:

```bash
# Run only AgentUnit scenarios
pytest -m agentunit

# Run specific scenario by name
pytest -m "scenario('greeting-test')"
```

### Running Scenarios

```bash
# Run all tests (including AgentUnit scenarios)
pytest

# Run only AgentUnit scenarios
pytest tests/eval/

# Run with verbose output
pytest tests/eval/ -v

# Run specific scenario file
pytest tests/eval/basic_scenarios.py

# Filter by markers
pytest -m agentunit
```

## Result Caching

The plugin supports result caching to skip re-running unchanged scenarios, improving performance for repeated test runs.

### How It Works

- Scenario inputs (dataset, adapter config) are hashed to create cache keys
- Results are stored in `.agentunit_cache/` directory
- Cache is automatically invalidated when source files change
- Cache hit/miss is logged for visibility

### Cache Options

```bash
# Force fresh runs (bypass cache)
pytest tests/eval/ --no-cache

# Clear cache before running tests
pytest tests/eval/ --clear-cache

# Combine options
pytest tests/eval/ --clear-cache --no-cache
```

### Cache Behavior

- **Cache hit**: Repeated runs with same inputs use cached results (logged as info)
- **Cache miss**: New or changed scenarios are executed fresh
- **Source change**: Modifying scenario source files invalidates the cache
- **Manual clear**: Use `--clear-cache` to remove all cached results

### Cache Directory

The cache is stored in `.agentunit_cache/` at the project root. Add it to `.gitignore`:

```gitignore
# AgentUnit cache
.agentunit_cache/
```

## Configuration

Add pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "agentunit: marks test as an AgentUnit scenario evaluation",
    "scenario(name): marks test with specific scenario name",
]
testpaths = ["tests", "tests/eval"]
```

## Example Directory Structure

```
project/
├── tests/
│   ├── eval/                    # AgentUnit scenarios
│   │   ├── __init__.py
│   │   ├── basic_scenarios.py
│   │   └── advanced_scenarios.py
│   └── test_regular.py
├── src/
│   └── myproject/
└── pyproject.toml
```
