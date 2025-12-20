from datetime import datetime
from pathlib import Path

from agentunit.reporting.results import (
    SuiteResult,
    ScenarioResult,
    ScenarioRun,
)


def test_markdown_contains_emojis():
    passing_run = ScenarioRun(
    scenario_name="emoji-suite",
    case_id="test_pass",
    success=True,
    metrics={},
    duration_ms=5,
    trace=[],
    error=None,
)


    failing_run = ScenarioRun(
    scenario_name="emoji-suite",
    case_id="test_fail",
    success=False,
    metrics={},
    duration_ms=6,
    trace=[],
    error="AssertionError",
    )


    scenario = ScenarioResult(
    name="emoji-suite",
    runs=[passing_run, failing_run],
    )

    suite = SuiteResult(
    scenarios=[scenario],
    started_at=datetime.now(),
    finished_at=datetime.now(),
    )

    output_path = Path("report.md")
    suite.to_markdown(output_path)


    assert "✅" in markdown
    assert "❌" in markdown

    # UTF-8 safety check (important for Windows)
    markdown.encode("utf-8")
