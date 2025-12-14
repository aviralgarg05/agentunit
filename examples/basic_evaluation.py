"""
Basic Evaluation Example for AgentUnit
--------------------------------------

This script demonstrates how to run a minimal evaluation using
AgentUnit with a FakeAdapter. It is designed for beginners and does
not require any extra dependencies.
"""

from agentunit.core.adapters import BaseAdapter
from agentunit.core.evaluator import Evaluator


class FakeAdapter(BaseAdapter):
    """
    A simple mock adapter used only for demonstration.
    It returns a predictable output so evaluation is easy to understand.
    """

    def generate(self, prompt: str) -> str:
        # Always returns the same answer for simplicity
        return "Hello, this is a fake response!"


def main():
    # Step 1 — Prepare the adapter
    adapter = FakeAdapter()

    # Step 2 — Create the evaluator
    evaluator = Evaluator(adapter=adapter)

    # Step 3 — Prepare an example prompt
    prompt = "Say hello!"

    # Step 4 — Run the evaluation
    result = evaluator.evaluate(prompt)

    # Step 5 — Print the output
    print("Prompt:", prompt)
    print("Model Output:", result.output)
    print("Evaluation Score:", result.score)


if __name__ == "__main__":
    main()
