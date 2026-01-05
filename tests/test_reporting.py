import tempfile
from agentunit.reporting.results import ScenarioResult, ScenarioRun, SuiteResult
from datetime import datetime
from pathlib import Path

from agentunit.reporting.results import (
    ScenarioResult,
    ScenarioRun,
    SuiteResult,
)



def test_markdown_contains_emojis():
    passing_run = ScenarioRun(
        scenario_name="test_pass",
        case_id=1,
        success=True,
        metrics={},
        duration_ms=10,
        trace=[],
        error=None
    )

    failing_run = ScenarioRun(
        scenario_name="test_fail",
        case_id=2,
        success=False,
        metrics={},
        duration_ms=10,
        trace=[],
        error="AssertionError"
    )

    scenario_result = ScenarioResult(
        name="emoji-scenario",
        runs=[passing_run, failing_run]
    )

    suite = SuiteResult(
        scenarios=[scenario_result],
        started_at=datetime.now(),
        finished_at=datetime.now()
    )

    # Use TemporaryDirectory to avoid Windows PermissionError
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "suite.md"
        suite.to_markdown(path=tmp_path)
        markdown = output_path.read_text(encoding="utf-8")

    assert "✅" in markdown
    assert "❌" in markdown

    markdown.encode("utf-8")
