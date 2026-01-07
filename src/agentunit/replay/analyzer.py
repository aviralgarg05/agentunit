"""Analysis tools for session replays."""

from datetime import datetime
from typing import Any

from agentunit.replay.recorder import SessionRecorder


class SessionAnalyzer:
    """Analyze recorded sessions for patterns and issues."""

    def __init__(self, session_path: str):
        self.session = SessionRecorder.load(session_path)

    def detect_loops(self) -> list[dict[str, Any]]:
        """Detect potential execution loops (repeated tools/inputs)."""
        loops = []
        steps = self.session.steps

        # Simple window-based loop detection
        window_size = 3
        if len(steps) < window_size * 2:
            return []

        for i in range(len(steps) - window_size * 2):
            window1 = [s.type + str(s.content) for s in steps[i : i + window_size]]
            window2 = [
                s.type + str(s.content) for s in steps[i + window_size : i + window_size * 2]
            ]

            if window1 == window2:
                loops.append(
                    {
                        "start_step": i,
                        "length": window_size,
                        "pattern": [s.type for s in steps[i : i + window_size]],
                    }
                )

        return loops

    def get_error_rate(self) -> float:
        """Calculate error rate in the session."""
        total_steps = len(self.session.steps)
        if total_steps == 0:
            return 0.0

        errors = sum(1 for s in self.session.steps if s.type == "error")
        return errors / total_steps

    def get_tool_usage(self) -> dict[str, int]:
        """Get counts of tool usage."""
        usage = {}
        for step in self.session.steps:
            if step.type == "tool_call":
                # Assuming content is dict with tool name or string
                tool_name = "unknown"
                if isinstance(step.content, dict):
                    tool_name = step.content.get("name", "unknown")
                elif isinstance(step.content, str):
                    tool_name = step.content

                usage[tool_name] = usage.get(tool_name, 0) + 1
        return usage

    def summarize(self) -> dict[str, Any]:
        """Generate a complete summary analysis."""
        return {
            "session_id": self.session.session_id,
            "duration_seconds": (
                datetime.fromisoformat(self.session.end_time)
                - datetime.fromisoformat(self.session.start_time)
            ).total_seconds()
            if self.session.end_time
            else 0.0,
            "total_steps": len(self.session.steps),
            "error_rate": self.get_error_rate(),
            "loops_detected": len(self.detect_loops()),
            "tool_usage": self.get_tool_usage(),
            "total_tokens": self.session.total_tokens,
            "total_cost": self.session.total_cost,
        }



