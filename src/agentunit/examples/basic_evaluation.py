from agentunit.adapters.base import BaseAdapter
from agentunit.core.outcome import Outcome
from agentunit.core.runner import Runner
from agentunit.core.scenario import Scenario
from agentunit.dataset import DatasetCase, DatasetSource

class FakeAdapter(BaseAdapter):
    def __init__(self, response: str):
        self.response = response

    def prepare(self):
        pass
    def execute(self, case, trace_log):
        return Outcome(success=True, output=self.response, error=None)
    def cleanup(self):
        pass

def main() -> None:
    # define simple dataset
    cases = [
        DatasetCase(
            input="hello",
            expected="hello",
        )
    ]
    dataset = DatasetSource.from_list(cases)
    
    # create a scenario using the fake adapter
    scenario = Scenario(
        name="Basic Evaluation Example",
        adapter=FakeAdapter(response="hello"),
        dataset=dataset,
    )

    # run evaluation
    runner = Runner([scenario])
    suite_result = runner.run()

    scenario_result = suite_result.scenarios[0]
    # print summary
    print("=== Evaluation Summary ===")
    print(f"Scenario: {scenario.name}")
    print(f"Success rate: {scenario_result.success_rate:.0%}")


if __name__ == "__main__":
    main()
