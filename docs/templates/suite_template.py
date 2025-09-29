"""Starter template for an AgentUnit evaluation suite.

Copy this file into your project (for example `evals/faq_suite.py`) and replace
placeholder sections with your real agent integration. The template demonstrates
how to define a dataset, adapter, and scenario in one place.
"""

from __future__ import annotations

from typing import Iterable

from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource


# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------

def _load_cases() -> Iterable[DatasetCase]:
    """Yield deterministic DatasetCase objects.

    Replace the body with however you want to produce evaluation cases. The
    generator pattern lets you stream large corpora without materialising
    everything in memory.
    """

    yield DatasetCase(
        id="faq-001",
        query="What is the capital of France?",
        expected_output="Paris is the capital of France.",
        context=["Paris is the capital of France."],
        tools=["knowledge_base"],
        metadata={"difficulty": "easy"},
    )

    # Add more DatasetCase entries here


dataset = DatasetSource(name="faq-demo", loader=_load_cases)


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class FAQAdapter(BaseAdapter):
    """Connects AgentUnit to your production agent."""

    name = "faq-adapter"

    def __init__(self, agent) -> None:
        self._agent = agent
        self._ready = False

    def prepare(self) -> None:
        """Optional: warm up connections or load prompts."""

        if not self._ready:
            self._agent.connect()
            self._ready = True

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:  # type: ignore[override]
        """Invoke the agent and record relevant trace events."""

        trace.record("agent_prompt", input={"query": case.query, "context": case.context})
        answer = self._agent.answer(case.query, context=case.context)
        trace.record("agent_response", content=answer)

        # Emit tool usage if your agent uses external systems
        trace.record("tool_call", name="knowledge_base", status="success")

        success = case.expected_output is None or answer.strip() == case.expected_output.strip()
        return AdapterOutcome(success=success, output=answer)

    def cleanup(self) -> None:
        """Optional: close network connections or free resources."""

        if self._ready:
            self._agent.close()
            self._ready = False


# ---------------------------------------------------------------------------
# Suite wiring
# ---------------------------------------------------------------------------

def create_suite() -> list[Scenario]:
    """Return a list of Scenario objects for the CLI to run."""

    agent = ...  # instantiate your agent client here
    adapter = FAQAdapter(agent)
    scenario = Scenario(
        name="faq-demo",
        adapter=adapter,
        dataset=dataset,
        retries=1,
        max_turns=10,
        timeout=60.0,
        tags=["smoke"],
        metadata={"owner": "evals"},
    )
    return [scenario]


# Export `suite` so the CLI can import it directly without calling `create_suite()`.
suite = list(create_suite())
