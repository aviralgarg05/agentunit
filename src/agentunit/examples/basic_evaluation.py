from agentunit import DatasetCase, Runner, Scenario


class FakeAdapter:
    def __init__(self, response: str):
        self.response = response

    def run(self, case: DatasetCase) -> str:
        return self.response


def main() -> None:
    # define simple dataset
    cases = [
        DatasetCase(
            id="echo-task",
            query="Say hello",
            expected_output="hello",
        )
    ]
    # create a scenario using the fake adapter
    scenario = Scenario(
        name="Basic Evaluation Example",
        adapter=FakeAdapter(response="hello"),
        dataset=cases,
    )

    # run evaluation
    runner = Runner()
    result = runner.run(scenario)

    # print summary
    print("=== Evaluation Summary ===")
    print(f"Scenario: {scenario.name}")
    print(f"Success rate: {result.success_rate:.0%}")


if __name__ == "__main__":
    main()
