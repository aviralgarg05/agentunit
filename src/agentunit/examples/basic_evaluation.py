from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.runner import Runner
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource


class FakeAdapter(BaseAdapter):
    """
    A minimal adapter implementation used for testing and examples.

    FakeAdapter simulates an adapter without performing real computation, it returns predefined response.
    This becomes useful in representing integration of adapters with the AgentUnit.
    """

    def __init__(self, response: str):
        """
        Initialize adapter with static response.

        Args:
            response (str): The output string returns on execution.
        """
        self.response = response

    def prepare(self):
        """
        Prepare adapter before execution.

        FakeAdapter do nor require setup.
        """

    def execute(self, case, trace_log: TraceLog) -> AdapterOutcome:
        """
        Execute the adapter for a given evaluation case.

        Args:
            case(object): Input case of evaluation.
            trace_log (TraceLog): Trace log for recording execution details.
        """
        return AdapterOutcome(success=True, output=self.response, error=None)

    def cleanup(self):
        """
        Cleanup adapter resources.

        No cleanup required for FakeAdapter.
        """


def main() -> None:
    # define simple dataset
    cases = [
        DatasetCase(
            id="case_1",
            input="hello",
            expected_output="hello",
        )
    ]

    # create a scenario using the fake adapter
    scenario = Scenario(
        name="Basic Evaluation Example",
        adapter=FakeAdapter(response="hello"),
        dataset=DatasetSource.from_list(cases),
    )

    # run evaluation
    runner = Runner([scenario])
    result = runner.run()

    scenario_result = result.scenarios[0]
    # print summary
    print("=== Evaluation Summary ===")
    print(f"Scenario: {scenario.name}")
    print(f"Success rate: {scenario_result.success_rate:.0%}")


if __name__ == "__main__":
    main()
