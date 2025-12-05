from __future__ import annotations

import json
from math import isclose
from typing import TYPE_CHECKING

from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.runner import run_suite
from agentunit.core.scenario import Scenario
from agentunit.datasets.base import DatasetCase, DatasetSource


if TYPE_CHECKING:
    from pathlib import Path

    from agentunit.core.trace import TraceLog


class FakeAdapter(BaseAdapter):
    name = "fake"

    def __init__(self) -> None:
        self.prepare_called = False
        self.cleanup_called = False
        self.executions = 0

    def prepare(self) -> None:
        self.prepare_called = True

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:  # type: ignore[override]
        self.executions += 1
        trace.record("agent_prompt", input={"query": case.query})
        trace.record("tool_call", name="search", status="success")
        content = f"Answer for {case.query}"
        trace.record("agent_response", content=content)
        return AdapterOutcome(success=True, output=content, tool_calls=[{"name": "search", "status": "success"}])

    def cleanup(self) -> None:
        self.cleanup_called = True


def _build_scenario() -> tuple[Scenario, FakeAdapter]:
    case = DatasetCase(
        id="case-1",
        query="What is the capital of France?",
        expected_output="Answer for What is the capital of France?",
        tools=["search"],
        context=["France is in Europe"],
    )
    dataset = DatasetSource.single(case)
    adapter = FakeAdapter()
    scenario = Scenario(name="demo", adapter=adapter, dataset=dataset)
    return scenario, adapter


def test_run_suite_with_fake_adapter(tmp_path: Path) -> None:
    scenario, adapter = _build_scenario()

    result = run_suite(
        [scenario],
        metrics=["faithfulness", "tool_success", "answer_correctness", "hallucination_rate", "retrieval_quality"],
        otel_exporter="console",
        seed=123,
    )

    assert adapter.prepare_called is True
    assert adapter.cleanup_called is True
    assert adapter.executions == 1

    assert isclose(result.scenarios[0].success_rate, 1.0, rel_tol=1e-6)
    run = result.scenarios[0].runs[0]
    assert run.success is True
    assert isclose(run.metrics["tool_success"], 1.0, rel_tol=1e-6)
    assert "faithfulness" in run.metrics
    assert run.trace.last_response() == "Answer for What is the capital of France?"
    assert len(run.trace.events) >= 3

    junit_path = tmp_path / "report.xml"
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    result.to_junit(junit_path)
    result.to_json(json_path)
    result.to_markdown(md_path)

    assert junit_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text())
    first_run = payload["scenarios"][0]["runs"][0]
    assert isclose(first_run["metrics"]["tool_success"], 1.0, rel_tol=1e-6)
    assert first_run["trace"]["events"]

    markdown = md_path.read_text()
    assert "AgentUnit Report" in markdown
    assert "demo" in markdown
