from agentunit import Evaluation, Task


class FakeAdapter:
    def __init__(self, response: str):
        self.response = response

    def run(self, prompt: str) -> str:
        return self.response


def main():
    # define simple task
    # each task specifies a prompt and expected output
    task = Task(name="echo-task", prompt="Say hello", expected="hello")
    # create a fake model adapter that always outputs "hello"
    fake_model = FakeAdapter(response="hello")

    # build the evaluation
    evaluation = Evaluation(task=[task], model=fake_model)
    # results can be inspected/printed
    results = evaluation.run()

    # print a readable summary
    print("=== Evaluation Summary ===")
    for result in results:
        print(f"Task: {result.task.name}")
        print(f"Prompt: {result.task.prompt}")
        print(f"Model Output: {result.output}")
        print(f"Expected: {result.task.expected}")
        print(f"Passed: {result.passed}")
        print("-" * 30)
    if __name__ == "__main__":
        main()
