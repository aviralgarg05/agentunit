from datetime import datetime, timezone
from pathlib import Path

from agentunit.core.trace import TraceLog
from agentunit.reporting.results import (
    ScenarioResult,
    ScenarioRun,
    SuiteResult,
)


def test_markdown_contains_emojis():
    passing_run = ScenarioRun(
        scenario_name="emoji-suite",
        case_id="test_pass",
        success=True,
        metrics={},
        duration_ms=5,
        trace=TraceLog(),
        error=None,
    )

    failing_run = ScenarioRun(
        scenario_name="emoji-suite",
        case_id="test_fail",
        success=False,
        metrics={},
        duration_ms=6,
        trace=TraceLog(),
        error="AssertionError",
    )

    scenario = ScenarioResult(
        name="emoji-suite",
        runs=[passing_run, failing_run],
    )

    suite = SuiteResult(
        scenarios=[scenario],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    output_path = Path("report.md")
    suite.to_markdown(output_path)

    # FIX: Read the file content
    markdown = output_path.read_text(encoding="utf-8")

    assert "✅" in markdown
