
import sys
from pathlib import Path


# Add the agentunit folder to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "agentunit"))

from agentunit.core.reporting import RunResult, SuiteResult


def test_markdown_contains_emojis():
    # Adjusted to match common RunResult constructor
    passing_run = RunResult(
        name="test_pass",
        status="pass",
        exception=None,
    )

    failing_run = RunResult(
        name="test_fail",
        status="fail",
        exception="AssertionError",
    )

    suite = SuiteResult(
        name="emoji-suite",
        runs=[passing_run, failing_run],
    )

    markdown = suite.to_markdown()

    assert "✅" in markdown
    assert "❌" in markdown

    # UTF-8 safety
    markdown.encode("utf-8")
