"""AgentArena benchmark integration.

AgentArena is a benchmark for evaluating autonomous agents on
realistic multi-step tasks.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ArenaTaskType(Enum):
    """AgentArena task types."""

    WEB_BROWSING = "web_browsing"
    CODE_EXECUTION = "code_execution"
    FILE_MANIPULATION = "file_manipulation"
    DATA_ANALYSIS = "data_analysis"
    MULTI_TOOL = "multi_tool"


@dataclass
class ArenaTask:
    """An AgentArena benchmark task.

    Attributes:
        task_id: Unique task identifier
        task_type: Type of task
        instruction: Task instruction
        success_criteria: Criteria for success
        max_steps: Maximum allowed steps
        required_tools: List of required tools
        initial_state: Initial environment state
        expected_outcome: Expected outcome/result
    """

    task_id: str
    task_type: ArenaTaskType
    instruction: str
    success_criteria: dict[str, Any]
    max_steps: int = 10
    required_tools: list[str] = None
    initial_state: dict[str, Any] = None
    expected_outcome: Any = None

    def __post_init__(self):
        if self.required_tools is None:
            self.required_tools = []
        if self.initial_state is None:
            self.initial_state = {}


class AgentArenaBenchmark:
    """AgentArena benchmark runner.

    This class provides:
    - Loading AgentArena tasks
    - Running multi-step evaluations
    - Tracking agent actions
    - Scoring based on success criteria
    """

    def __init__(self, dataset_path: Path | None = None, task_type: ArenaTaskType | None = None):
        """Initialize AgentArena benchmark.

        Args:
            dataset_path: Path to AgentArena dataset
            task_type: Specific task type to run (None for all types)
        """
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.tasks: list[ArenaTask] = []

    def load_dataset(self, split: str = "test") -> list[ArenaTask]:
        """Load AgentArena dataset.

        Args:
            split: Dataset split ("train", "test")

        Returns:
            List of arena tasks
        """
        if self.dataset_path is None:
            # Return sample tasks for testing
            self.tasks = self._get_sample_tasks()
            return self.tasks

        tasks = []
        dataset_file = self.dataset_path / f"arena_{split}.json"

        if not dataset_file.exists():
            msg = f"Dataset file not found: {dataset_file}"
            raise FileNotFoundError(msg)

        with open(dataset_file) as f:
            data = json.load(f)

        for task_data in data.get("tasks", []):
            task_type = ArenaTaskType(task_data["task_type"])

            # Filter by task type if specified
            if self.task_type and task_type != self.task_type:
                continue

            task = ArenaTask(
                task_id=task_data["task_id"],
                task_type=task_type,
                instruction=task_data["instruction"],
                success_criteria=task_data["success_criteria"],
                max_steps=task_data.get("max_steps", 10),
                required_tools=task_data.get("required_tools", []),
                initial_state=task_data.get("initial_state", {}),
                expected_outcome=task_data.get("expected_outcome"),
            )
            tasks.append(task)

        self.tasks = tasks
        return tasks

    def _get_sample_tasks(self) -> list[ArenaTask]:
        """Get sample AgentArena tasks for testing.

        Returns:
            List of sample tasks
        """
        return [
            ArenaTask(
                task_id="web_001",
                task_type=ArenaTaskType.WEB_BROWSING,
                instruction="Find the current price of Bitcoin on CoinMarketCap.",
                success_criteria={"type": "contains", "value": "BTC"},
                max_steps=5,
                required_tools=["web_browser"],
                expected_outcome="BTC price from CoinMarketCap",
            ),
            ArenaTask(
                task_id="code_001",
                task_type=ArenaTaskType.CODE_EXECUTION,
                instruction="Write and execute Python code to calculate factorial of 10.",
                success_criteria={"type": "equals", "value": 3628800},
                max_steps=3,
                required_tools=["python_interpreter"],
                expected_outcome=3628800,
            ),
            ArenaTask(
                task_id="multi_001",
                task_type=ArenaTaskType.MULTI_TOOL,
                instruction="Search for Python tutorials, summarize the top result, and save to file.",
                success_criteria={"type": "file_exists", "value": "summary.txt"},
                max_steps=8,
                required_tools=["web_search", "file_writer"],
                expected_outcome="summary.txt created",
            ),
        ]

    def to_agentunit_dataset(self) -> list[dict[str, Any]]:
        """Convert AgentArena tasks to AgentUnit dataset format.

        Returns:
            List of test cases in AgentUnit format
        """
        from agentunit.datasets.base import DatasetCase

        cases = []
        for task in self.tasks:
            case = DatasetCase(
                id=task.task_id,
                query=task.instruction,
                expected_output=task.expected_outcome,
                metadata={
                    "task_type": task.task_type.value,
                    "success_criteria": task.success_criteria,
                    "max_steps": task.max_steps,
                    "required_tools": task.required_tools,
                    "initial_state": task.initial_state,
                    "benchmark": "agent_arena",
                },
            )
            cases.append(case)

        return cases

    def evaluate_success(
        self, task: ArenaTask, agent_output: Any, agent_actions: list[dict[str, Any]]
    ) -> bool:
        """Evaluate if agent succeeded on task.

        Args:
            task: The arena task
            agent_output: Final agent output
            agent_actions: List of actions taken by agent

        Returns:
            True if task succeeded
        """
        criteria = task.success_criteria
        criteria_type = criteria.get("type")
        criteria_value = criteria.get("value")

        if criteria_type == "equals":
            return agent_output == criteria_value

        if criteria_type == "contains":
            return criteria_value in str(agent_output)

        if criteria_type == "file_exists":
            # Would check if file exists
            return criteria_value in str(agent_output)

        if criteria_type == "steps_less_than":
            return len(agent_actions) < criteria_value

        if criteria_type == "custom":
            # Custom evaluation function
            return criteria.get("evaluator", lambda x: False)(agent_output)

        return False

    def calculate_score(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate AgentArena scores.

        Args:
            results: Evaluation results

        Returns:
            Dictionary with overall and per-type scores
        """
        scores = {"overall": 0.0, "avg_steps": 0.0, "task_type": {}}

        total_steps = 0
        total_count = 0
        type_counts = {}
        type_scores = {}

        for result in results:
            task_type = result.get("task_type", "unknown")
            passed = result.get("passed", False)
            steps = result.get("steps", 0)

            total_count += 1
            total_steps += steps

            if passed:
                scores["overall"] += 1

            # Per-type scores
            if task_type not in type_counts:
                type_counts[task_type] = 0
                type_scores[task_type] = 0

            type_counts[task_type] += 1
            if passed:
                type_scores[task_type] += 1

        # Calculate percentages
        if total_count > 0:
            scores["overall"] = scores["overall"] / total_count * 100
            scores["avg_steps"] = total_steps / total_count

        for task_type, count in type_counts.items():
            if count > 0:
                scores["task_type"][task_type] = type_scores[task_type] / count * 100

        return scores
