from agentunit import Scenario, DatasetCase, Runner
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.datasets.base import DatasetSource

class MockAgent:
    def connect(self):
        return self

    def answer(self, query: str) -> str:
        return "This is a canned FAQ answer."


class MockAdapter(BaseAdapter):
    def __init__(self, agent):
        self.agent = agent

    def prepare(self) -> None:  # pragma: no cover - trivial
        return None

    def execute(self, case, trace) -> AdapterOutcome:
        conn = self.agent.connect()
        resp_text = conn.answer(case.query if hasattr(case, "query") else case.input)
        trace.record("agent_response", content=resp_text)
        trace.record("tool_call", name="knowledge_base", status="success")
        success = case.expected_output is None or resp_text.strip() == case.expected_output.strip()
        return AdapterOutcome(success=success, output=resp_text)

    def cleanup(self) -> None:  # pragma: no cover - trivial
        return None

def test_suite_template_faqadapter_smoke():
    cases = [
        DatasetCase(
            id="faq_1",
            query="How do I reset my password?",
            expected_output="This is a canned FAQ answer."
        )
    ]

    agent = MockAgent()
    adapter = MockAdapter(agent)

    dataset = DatasetSource.from_list(cases, name="test-faq")
    scenario = Scenario(name="FAQAdapter smoke test", adapter=adapter, dataset=dataset)

    runner = Runner([scenario])
    results = runner.run()

    assert results is not None

    # Extract recorded runs from the suite result
    assert len(results.scenarios) == 1
    runs = results.scenarios[0].runs
    matching = [r for r in runs if r.case_id == "faq_1"]
    assert len(matching) == 1, "Expected exactly one recorded response for faq_1"

    recorded_run = matching[0]
    assert recorded_run.success is True
    # Answer correctness metric should be 1.0 for an exact match
    assert recorded_run.metrics.get("answer_correctness") == 1.0
