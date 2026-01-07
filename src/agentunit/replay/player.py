"""Session player for replaying and analyzing recordings."""

from collections.abc import Iterator
from pathlib import Path

from agentunit.replay.recorder import InteractionStep, SessionRecorder


class SessionPlayer:
    """Plays back and navigates recorded sessions."""

    def __init__(self, session_path: str | Path):
        self.session_path = Path(session_path)
        self.session = SessionRecorder.load(self.session_path)
        self.current_step_index = 0

    @property
    def total_steps(self) -> int:
        return len(self.session.steps)

    @property
    def current_step(self) -> InteractionStep | None:
        if 0 <= self.current_step_index < self.total_steps:
            return self.session.steps[self.current_step_index]
        return None

    def next_step(self) -> InteractionStep | None:
        """Move to next step."""
        if self.current_step_index < self.total_steps - 1:
            self.current_step_index += 1
            return self.current_step
        return None

    def prev_step(self) -> InteractionStep | None:
        """Move to previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            return self.current_step
        return None

    def seek(self, step_index: int) -> InteractionStep | None:
        """Jump to specific step."""
        if 0 <= step_index < self.total_steps:
            self.current_step_index = step_index
            return self.current_step
        return None

    def play(self, speed: float = 1.0) -> Iterator[InteractionStep]:
        """Generator to play through the session."""
        import time

        self.seek(0)
        yield self.current_step

        last_timestamp = self.current_step.timestamp

        while self.next_step():
            current_timestamp = self.current_step.timestamp
            delta = current_timestamp - last_timestamp

            if speed > 0:
                time.sleep(delta / speed)

            yield self.current_step
            last_timestamp = current_timestamp

    def get_context_at_step(self, step_index: int) -> list[InteractionStep]:
        """Get the conversation history leading up to a step."""
        if step_index < 0:
            return []

        # Determine actual index if out of bounds (clamp)
        idx = min(step_index, self.total_steps - 1)
        return self.session.steps[: idx + 1]
