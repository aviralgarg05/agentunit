"""Example AgentUnit scenarios for pytest plugin."""

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
