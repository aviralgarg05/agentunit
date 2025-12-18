"""Example of a failing scenario for pytest plugin testing."""

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


class FailingDataset(DatasetSource):
    """Dataset that will cause failures."""

    def __init__(self):
        super().__init__(name="failing-test", loader=self._generate_cases)

    def _generate_cases(self):
        return [
            DatasetCase(
                id="impossible",
                query="What is the meaning of life?",
                expected_output="42",
                metadata={"type": "philosophy"},
            ),
        ]


def always_wrong_agent(payload):
    """Agent that always gives wrong answers."""
    return {"result": "Wrong answer"}


# This scenario will fail when run
failing_scenario = Scenario(
    name="failing-test",
    adapter=SimpleAdapter(always_wrong_agent),
    dataset=FailingDataset(),
)
