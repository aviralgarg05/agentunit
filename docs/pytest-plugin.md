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
    """Simple adapter for function-based agents."""
    
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

# Combine with other markers
pytest -m "agentunit and not slow"
```

#### Test Results

- **Passed scenarios**: All test cases in the scenario passed
- **Failed scenarios**: One or more test cases failed (shows detailed failure info)
- **Error scenarios**: Scenario couldn't be loaded or executed

#### Fixtures

AgentUnit scenarios can use pytest fixtures:

```python
import pytest
from agentunit import Scenario

@pytest.fixture
def test_config():
    return {"timeout": 30, "retries": 2}

def scenario_with_fixture(test_config):
    # Use fixture data in scenario creation
    return Scenario(
        name="fixture-test",
        adapter=SimpleAdapter(my_agent),
        dataset=MyDataset(),
        timeout=test_config["timeout"],
        retries=test_config["retries"],
    )
```

### Configuration

Add pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "agentunit: marks test as an AgentUnit scenario evaluation",
    "scenario(name): marks test with specific scenario name",
]
testpaths = ["tests", "tests/eval"]
```

### Example Directory Structure

```
project/
├── tests/
│   ├── eval/                    # AgentUnit scenarios
│   │   ├── __init__.py
│   │   ├── basic_scenarios.py   # Python scenarios
│   │   ├── advanced_scenarios.py
│   │   └── config_scenario.yaml # Config-based scenarios
│   └── test_regular.py          # Regular pytest tests
├── src/
│   └── myproject/
└── pyproject.toml
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

# Run with coverage
pytest tests/eval/ --cov=myproject
```

### Advanced Usage

#### Custom Test Names

Scenarios appear in pytest output with descriptive names:

```
tests/eval/basic_scenarios.py::agentunit::greeting-test PASSED
tests/eval/basic_scenarios.py::agentunit::math-test FAILED
```

#### Parallel Execution

Use pytest-xdist for parallel scenario execution:

```bash
pip install pytest-xdist
pytest tests/eval/ -n auto
```

#### Integration with CI/CD

The plugin works seamlessly with CI/CD systems:

```yaml
# .github/workflows/test.yml
- name: Run AgentUnit scenarios
  run: pytest tests/eval/ --junitxml=scenario-results.xml
```

### Error Handling

The plugin handles various error conditions gracefully:

- **Load errors**: If a scenario file can't be loaded, it appears as a failed test
- **Runtime errors**: Scenario execution errors are reported as test failures
- **Missing dependencies**: Optional dependencies are handled with appropriate skips

### Best Practices

1. **Organize scenarios** by functionality in separate files
2. **Use descriptive names** for scenarios and test cases
3. **Add markers** for easy filtering and organization
4. **Include both positive and negative test cases**
5. **Use fixtures** for shared test configuration
6. **Document scenario purpose** with docstrings

### Troubleshooting

#### Scenarios Not Discovered

- Ensure files are in `tests/eval/` directory
- Check that scenario objects are properly defined
- Verify import statements work correctly

#### Import Errors

- Make sure all dependencies are installed
- Check Python path includes your project
- Verify scenario file syntax is correct

#### Test Failures

- Check scenario agent implementation
- Verify dataset cases have correct expected outputs
- Review error messages in pytest output