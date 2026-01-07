"""Session recording capabilities."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class InteractionStep:
    """A single step in an agent interaction session."""
    
    step_id: int
    timestamp: float
    type: Literal["input", "output", "tool_call", "tool_result", "thought", "error"]
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class RecordedSession:
    """A complete recorded session."""
    
    session_id: str
    agent_name: str
    start_time: str
    end_time: str | None = None
    steps: list[InteractionStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0


class SessionRecorder:
    """Records agent sessions for replay and analysis.
    
    Features:
    - Step-by-step recording of all interactions
    - Metadata capture (tokens, latency, cost)
    - Serialization to JSON
    """
    
    def __init__(self, session_id: str, agent_name: str, output_dir: str | Path = "sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = RecordedSession(
            session_id=session_id,
            agent_name=agent_name,
            start_time=datetime.utcnow().isoformat()
        )
        self._current_step_id = 0
        self._start_time = time.time()
        
    def record_step(
        self,
        type: Literal["input", "output", "tool_call", "tool_result", "thought", "error"],
        content: Any,
        metadata: dict[str, Any] | None = None,
        duration: float = 0.0
    ):
        """Record a single execution step."""
        step = InteractionStep(
            step_id=self._current_step_id,
            timestamp=time.time() - self._start_time,
            type=type,
            content=content,
            metadata=metadata or {},
            duration=duration
        )
        self.session.steps.append(step)
        self._current_step_id += 1
        
        # Auto-save after each step for crash recovery
        self.save()
        
    def finish(self):
        """Mark session as finished."""
        self.session.end_time = datetime.utcnow().isoformat()
        self.save()
        
    def save(self):
        """Save session to disk."""
        filepath = self.output_dir / f"{self.session.session_id}.json"
        
        # Convert dataclass to dict
        data = asdict(self.session)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str | Path) -> RecordedSession:
        """Load a session from disk."""
        with open(filepath) as f:
            data = json.load(f)
            
        steps = [InteractionStep(**s) for s in data.pop("steps", [])]
        return RecordedSession(steps=steps, **data)
