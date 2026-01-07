"""Definitions for benchmark scenarios and tasks."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BenchmarkTask:
    """A single task within a benchmark."""
    
    id: str
    prompt: str
    expected_answer: Optional[str] = None
    reference_data: Optional[Any] = None
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of executing a single task."""
    
    task_id: str
    prompt: str
    model_output: str
    correct: bool
    latency: float
    cost: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkScenario:
    """A collection of tasks representing a specific benchmark."""
    
    name: str
    description: str
    tasks: List[BenchmarkTask]
    
    @property
    def total_tasks(self) -> int:
        return len(self.tasks)


# --- Standard Default Scenarios ---

MATH_SCENARIO = BenchmarkScenario(
    name="AgentMath-v1",
    description="Standard mathematical reasoning tasks for agents.",
    tasks=[
        BenchmarkTask(
            id="math_1",
            prompt="What is 15% of 850?",
            expected_answer="127.5",
            difficulty="easy",
            tags=["math", "percentage"]
        ),
        BenchmarkTask(
            id="math_2", 
            prompt="Solve for x: 2x + 5 = 15",
            expected_answer="5",
            difficulty="medium",
            tags=["math", "algebra"]
        ),
        BenchmarkTask(
            id="math_3",
            prompt="If a triangle has a base of 10 and height of 5, what is its area?",
            expected_answer="25",
            difficulty="medium",
            tags=["math", "geometry"]
        )
    ]
)

REASONING_SCENARIO = BenchmarkScenario(
    name="AgentReasoning-v1", 
    description="Logic and multi-step reasoning capabilities.",
    tasks=[
        BenchmarkTask(
            id="logic_1",
            prompt="If all cats are mammals and all mammals are animals, are all cats animals?",
            expected_answer="Yes",
            difficulty="easy",
            tags=["logic", "syllogism"]
        ),
        BenchmarkTask(
            id="logic_2",
            prompt="I have 3 apples. I eat one, give one to my friend, and buy 5 more. How many apples do I have now?",
            expected_answer="6",
            difficulty="medium",
            tags=["logic", "arithmetic"]
        )
    ]
)

GENERAL_QA_SCENARIO = BenchmarkScenario(
    name="AgentQA-v1",
    description="General knowledge and question answering.",
    tasks=[
        BenchmarkTask(
            id="qa_1",
            prompt="What is the capital of France?",
            expected_answer="Paris",
            difficulty="easy",
            tags=["knowledge", "geography"]
        ),
        BenchmarkTask(
            id="qa_2",
            prompt="Who wrote 'Romeo and Juliet'?",
            expected_answer="William Shakespeare",
            difficulty="easy",
            tags=["knowledge", "literature"]
        )
    ]
)

DEFAULT_SCENARIOS = [MATH_SCENARIO, REASONING_SCENARIO, GENERAL_QA_SCENARIO]
