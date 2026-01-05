"""CLI commands for pytest plugin setup."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--directory",
    "-d",
    default="tests/eval",
    help="Directory to create for evaluation scenarios",
)
@click.option(
    "--example",
    "-e",
    is_flag=True,
    help="Create example scenario files",
)
def init_eval(directory: str, example: bool) -> None:
    """Initialize directory structure for AgentUnit pytest plugin."""
    eval_dir = Path(directory)

    # Create directory structure
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    init_file = eval_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# AgentUnit evaluation scenarios\n")
        click.echo(f"Created {init_file}")

    if example:
        _create_example_files(eval_dir)

    click.echo(f"\nEvaluation directory initialized at {eval_dir}")
    click.echo("\nNext steps:")
    click.echo(f"1. Add scenario files to {eval_dir}/")
    click.echo("2. Run: pytest tests/eval/")
    click.echo("3. See docs/pytest-plugin.md for more information")


def _create_example_files(eval_dir: Path) -> None:
    """Create example scenario files."""
    example_file = eval_dir / "example_scenarios.py"
    if not example_file.exists():
        example_content = '''"""Example AgentUnit scenarios for pytest plugin."""

from agentunit import Scenario
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
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


class ExampleDataset(DatasetSource):
    """Example dataset for testing."""

    def __init__(self):
        super().__init__(name="example-dataset", loader=self._generate_cases)

    def _generate_cases(self):
        return [
            DatasetCase(
                id="greeting",
                query="Hello, how are you?",
                expected_output="Hello! I'm doing well, thank you.",
                metadata={"category": "greeting"},
            ),
            DatasetCase(
                id="math_simple",
                query="What is 2 + 2?",
                expected_output="4",
                metadata={"category": "math"},
            ),
        ]


def example_agent(payload):
    """Example agent that handles basic queries."""
    query = payload.get("query", "").lower()

    if "hello" in query or "how are you" in query:
        return {"result": "Hello! I'm doing well, thank you."}
    elif "2 + 2" in query or "2+2" in query:
        return {"result": "4"}
    else:
        return {"result": "I don't understand that query."}


# This scenario will be auto-discovered by pytest
example_scenario = Scenario(
    name="example-basic-test",
    adapter=SimpleAdapter(example_agent),
    dataset=ExampleDataset(),
)


def scenario_math_focused():
    """Factory function for math-focused scenario."""

    class MathDataset(DatasetSource):
        def __init__(self):
            super().__init__(name="math-dataset", loader=self._generate_cases)

        def _generate_cases(self):
            return [
                DatasetCase(
                    id="addition",
                    query="What is 5 + 3?",
                    expected_output="8",
                ),
                DatasetCase(
                    id="multiplication",
                    query="What is 4 * 6?",
                    expected_output="24",
                ),
            ]

    def math_agent(payload):
        query = payload.get("query", "")
        if "5 + 3" in query:
            return {"result": "8"}
        elif "4 * 6" in query:
            return {"result": "24"}
        return {"result": "I can only do simple math"}

    return Scenario(
        name="math-focused-test",
        adapter=SimpleAdapter(math_agent),
        dataset=MathDataset(),
    )
'''
        example_file.write_text(example_content)
        click.echo(f"Created {example_file}")

    readme_file = eval_dir / "README.md"
    if not readme_file.exists():
        readme_content = """# AgentUnit Evaluation Scenarios

This directory contains AgentUnit scenarios that can be run as pytest tests.

## Usage

Run all scenarios:
```bash
pytest tests/eval/
```

Run specific scenario file:
```bash
pytest tests/eval/example_scenarios.py
```

Run with AgentUnit marker:
```bash
pytest -m agentunit
```

## Creating Scenarios

1. Create Python files with `Scenario` objects or `scenario_*` functions
2. Use the `DatasetSource` class to define test cases
3. Implement agent functions that process queries and return results

See `example_scenarios.py` for examples.
"""
        readme_file.write_text(readme_content)
        click.echo(f"Created {readme_file}")


if __name__ == "__main__":
    init_eval()
