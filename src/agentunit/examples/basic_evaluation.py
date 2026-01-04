from agentunit.adapters.base import BaseAdapter, AdapterOutcome
from agentunit.core.tracelog import TraceLog
from agentunit.core.runner import Runner
from agentunit.core.scenario import Scenario
from agentunit.datasets.base import DatasetCase, DatasetSource


class FakeAdapter(BaseAdapter):
    def __init__(self, response: str):
        self.response = response

    def prepare(self):
        pass

    def execute(self, case, trace_log: TraceLog) -> AdapterOutcome:
        return AdapterOutcome(success=True, output=self.response, error=None)

    def cleanup(self):
        pass


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
