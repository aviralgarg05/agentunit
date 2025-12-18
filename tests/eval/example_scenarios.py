"""Example AgentUnit scenarios for pytest plugin demonstration."""

from agentunit import Scenario
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.datasets.base import DatasetCase, DatasetSource


class SimpleTestAdapter(BaseAdapter):
    """Simple adapter for testing."""

    name = "test"

    def __init__(self, agent_func):
        self.agent_func = agent_func

    def prepare(self):
        pass

    def execute(self, case, trace):
        try:
            result = self.agent_func({"query": case.query})
            output = result.get("result", "")
            success = output == case.expected_output
            if success:
                return AdapterOutcome(success=True, output=output)
            else:
                error_msg = f"Expected '{case.expected_output}', got '{output}'"
                return AdapterOutcome(success=False, output=output, error=error_msg)
        except Exception as e:
            return AdapterOutcome(success=False, output=None, error=str(e))


class SimpleTestDataset(DatasetSource):
    """Simple dataset for testing the pytest plugin."""

    def __init__(self):
        super().__init__(name="simple-test", loader=self._generate_cases)

    def _generate_cases(self):
        return [
            DatasetCase(
                id="greeting",
                query="Hello, how are you?",
                expected_output="Hello! I'm doing well, thank you for asking.",
                metadata={"type": "greeting"},
            ),
            DatasetCase(
                id="math", query="What is 2 + 2?", expected_output="4", metadata={"type": "math"}
            ),
        ]


def simple_echo_agent(payload):
    """Simple agent that can handle greetings and basic math."""
    query = payload.get("query", "").lower()
    
    # Handle greeting
    if "hello" in query and "how are you" in query:
        return {"result": "Hello! I'm doing well, thank you for asking."}
    
    # Handle math
    if "what is 2 + 2" in query or "2 + 2" in query:
        return {"result": "4"}
    
    # Default response
    return {"result": f"Echo: {payload.get('query', '')}"}


# Scenario objects that will be auto-discovered
basic_scenario = Scenario(
    name="basic-echo-test",
    adapter=SimpleTestAdapter(simple_echo_agent),
    dataset=SimpleTestDataset(),
)


def scenario_math_test():
    """Scenario factory function (starts with 'scenario_')."""

    def math_agent(payload):
        query = payload.get("query", "").lower()
        
        # Handle greeting
        if "hello" in query and "how are you" in query:
            return {"result": "Hello! I'm doing well, thank you for asking."}
        
        # Handle math
        if "2 + 2" in query or "2+2" in query or "what is 2 + 2" in query:
            return {"result": "4"}
        
        return {"result": "I don't know"}

    return Scenario(
        name="math-test",
        adapter=SimpleTestAdapter(math_agent),
        dataset=SimpleTestDataset(),
    )
