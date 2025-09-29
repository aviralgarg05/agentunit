"""Template AgentUnit suite wiring a simple agent to canned dataset cases."""
from __future__ import annotations

from typing import Iterable, List, Optional

from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource

from .agent import TemplateAgent


def _template_loader() -> Iterable[DatasetCase]:
    cases: List[DatasetCase] = [
        DatasetCase(
            id="template-001",
            query="What is the capital of France?",
            expected_output="Paris is the capital of France.",
            tools=["knowledge_base"],
            context=["Paris is the capital of France."],
        ),
        DatasetCase(
            id="template-002",
            query="Name two benefits of regular exercise.",
            expected_output="Regular exercise improves cardiovascular health and elevates mood.",
            tools=["knowledge_base"],
            context=["Regular exercise improves cardiovascular health and elevates mood."],
        ),
    ]
    for case in cases:
        yield case


template_dataset = DatasetSource(name="template-project", loader=_template_loader)


class TemplateProjectAdapter(BaseAdapter):
    """Adapter that runs the template agent against dataset cases."""

    name = "template-project"

    def __init__(self, agent: Optional[TemplateAgent] = None) -> None:
        self._agent = agent or TemplateAgent()
        self._prepared = False

    def prepare(self) -> None:
        self._prepared = True

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:  # type: ignore[override]
        if not self._prepared:
            self.prepare()

        trace.record("agent_prompt", input={"query": case.query, "context": case.context})
        trace.record("tool_call", name="knowledge_base", status="success")

        answer = self._agent.answer(case.query, context=case.context)
        trace.record("agent_response", content=answer)

        success = case.expected_output is None or answer.strip().lower() == case.expected_output.strip().lower()
        tool_calls = [{"name": "knowledge_base", "status": "success"}]
        return AdapterOutcome(success=success, output=answer, tool_calls=tool_calls)

    def cleanup(self) -> None:
        self._prepared = False


def create_suite() -> Iterable[Scenario]:
    """Factory returning the template project scenario list."""

    scenario = Scenario(name="template-agent-demo", adapter=TemplateProjectAdapter(), dataset=template_dataset)
    return [scenario]


suite = list(create_suite())
