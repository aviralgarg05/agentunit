# Integration Tests

This directory contains integration tests that verify AgentUnit works with real framework implementations.

## LangGraph Integration Tests

The LangGraph integration tests verify that AgentUnit can properly evaluate LangGraph agents through a complete evaluation cycle.

### Prerequisites

To run LangGraph integration tests, you need to install LangGraph:

```bash
pip install langgraph
```

Or with poetry:

```bash
poetry add langgraph --group dev
```

### Running Integration Tests

#### Run all integration tests:
```bash
pytest tests/integration/
```

#### Run only LangGraph tests:
```bash
pytest tests/integration/ -m langgraph
```

#### Skip integration tests (run only unit tests):
```bash
pytest -m "not integration"
```

#### Run with verbose output:
```bash
pytest tests/integration/ -v
```

### Test Structure

- `simple_langgraph_agent.py` - Contains a simple LangGraph agent implementation for testing
- `test_langgraph_integration.py` - Integration tests for LangGraph adapter
- `conftest.py` - Test configuration and markers

### What the Tests Cover

1. **Scenario Creation**: Tests creating scenarios from callable agents and Python files
2. **Full Evaluation Cycle**: Tests running complete evaluation cycles with multiple test cases
3. **Metrics Integration**: Tests that metrics can be calculated (when available)
4. **Error Handling**: Tests graceful handling of agent failures
5. **Retry Logic**: Tests scenario retry functionality
6. **Multiple Scenarios**: Tests running multiple scenarios together

### CI Integration

The integration tests are designed to be optionally run in CI:

- Tests are automatically skipped if LangGraph is not installed
- Use pytest markers to selectively run or skip integration tests
- All tests are marked with `@pytest.mark.integration` and `@pytest.mark.langgraph`

### Adding New Integration Tests

When adding integration tests for other frameworks:

1. Create a simple agent implementation in the framework
2. Create test cases that cover the full evaluation cycle
3. Use appropriate pytest markers (e.g., `@pytest.mark.crewai`)
4. Ensure tests are skipped gracefully when dependencies are not available
5. Document the prerequisites and running instructions

### Example Usage

```python
import pytest
from agentunit import Scenario, run_suite
from tests.integration.simple_langgraph_agent import invoke_agent

@pytest.mark.langgraph
@pytest.mark.integration
def test_my_langgraph_scenario():
    scenario = Scenario.load_langgraph(
        path=invoke_agent,
        dataset=my_dataset,
        name="my-test"
    )
    
    result = run_suite([scenario])
    assert len(result.scenarios) == 1
```