# -*- coding: utf-8 -*-
import pytest

from agentunit.core.reporting import SuiteResult, RunResult


def test_markdown_contains_emojis():
    passing_run = RunResult(
        name="test_pass",
        passed=True,
        error=None,
    )

    failing_run = RunResult(
        name="test_fail",
        passed=False,
        error="AssertionError",
    )

    suite = SuiteResult(
        name="emoji-suite",
        runs=[passing_run, failing_run],
    )

    markdown = suite.to_markdown()

    assert "✅" in markdown
    assert "❌" in markdown

    # UTF-8 safety check (important for Windows)
    markdown.encode("utf-8")
