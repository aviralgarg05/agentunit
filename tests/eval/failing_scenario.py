"""Example of a failing scenario for pytest plugin testing."""

from agentunit import Scenario
from agentunit.datasets.base import DatasetCase, DatasetSource


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
    agent=always_wrong_agent,
    dataset=FailingDataset(),
)
