"""GAIA 2.0 benchmark integration.

GAIA (General AI Assistants) is a benchmark for evaluating AI assistants
on real-world tasks across multiple difficulty levels.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class GAIALevel(Enum):
    """GAIA difficulty levels."""
    LEVEL_1 = 1  # Simple tasks
    LEVEL_2 = 2  # Moderate tasks
    LEVEL_3 = 3  # Complex tasks


@dataclass
class GAIATask:
    """A GAIA benchmark task.
    
    Attributes:
        task_id: Unique task identifier
        question: Task question/prompt
        level: Difficulty level
        final_answer: Expected answer
        file_name: Associated file (if any)
        file_path: Path to associated file
        annotator_metadata: Additional metadata
    """
    task_id: str
    question: str
    level: GAIALevel
    final_answer: str
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    annotator_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.annotator_metadata is None:
            self.annotator_metadata = {}


class GAIABenchmark:
    """GAIA 2.0 benchmark runner.
    
    This class provides:
    - Loading GAIA datasets
    - Running evaluations
    - Formatting results for submission
    - Leaderboard compatibility
    """
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        level: Optional[GAIALevel] = None
    ):
        """Initialize GAIA benchmark.
        
        Args:
            dataset_path: Path to GAIA dataset directory
            level: Specific level to run (None for all levels)
        """
        self.dataset_path = dataset_path
        self.level = level
        self.tasks: List[GAIATask] = []
    
    def load_dataset(self, split: str = "validation") -> List[GAIATask]:
        """Load GAIA dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
        
        Returns:
            List of GAIA tasks
        """
        if self.dataset_path is None:
            # Return sample tasks for testing
            self.tasks = self._get_sample_tasks()
            return self.tasks
        
        tasks = []
        dataset_file = self.dataset_path / f"gaia_{split}.jsonl"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                level = GAIALevel(int(data.get("level", 1)))
                
                # Filter by level if specified
                if self.level and level != self.level:
                    continue
                
                task = GAIATask(
                    task_id=data["task_id"],
                    question=data["question"],
                    level=level,
                    final_answer=data["final_answer"],
                    file_name=data.get("file_name"),
                    file_path=self.dataset_path / data["file_name"] if data.get("file_name") else None,
                    annotator_metadata=data.get("annotator_metadata", {})
                )
                tasks.append(task)
        
        self.tasks = tasks
        return tasks
    
    def _get_sample_tasks(self) -> List[GAIATask]:
        """Get sample GAIA tasks for testing.
        
        Returns:
            List of sample tasks
        """
        return [
            GAIATask(
                task_id="sample_1",
                question="What is the capital of France?",
                level=GAIALevel.LEVEL_1,
                final_answer="Paris"
            ),
            GAIATask(
                task_id="sample_2",
                question="Calculate the sum of prime numbers between 1 and 20.",
                level=GAIALevel.LEVEL_2,
                final_answer="77"
            ),
            GAIATask(
                task_id="sample_3",
                question="Analyze the attached spreadsheet and find the top 3 products by revenue.",
                level=GAIALevel.LEVEL_3,
                final_answer="Product A, Product C, Product B",
                file_name="sample_data.csv"
            )
        ]
    
    def to_agentunit_dataset(self) -> List[Dict[str, Any]]:
        """Convert GAIA tasks to AgentUnit dataset format.
        
        Returns:
            List of test cases in AgentUnit format
        """
        from agentunit.datasets.base import DatasetCase
        
        cases = []
        for task in self.tasks:
            case = DatasetCase(
                id=task.task_id,
                query=task.question,
                expected_output=task.final_answer,
                metadata={
                    "level": task.level.value,
                    "file_name": task.file_name,
                    "file_path": str(task.file_path) if task.file_path else None,
                    "annotator_metadata": task.annotator_metadata,
                    "benchmark": "gaia"
                }
            )
            cases.append(case)
        
        return cases
    
    def format_submission(
        self,
        results: List[Dict[str, Any]],
        model_name: str = "custom_model"
    ) -> Dict[str, Any]:
        """Format results for GAIA leaderboard submission.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
        
        Returns:
            Formatted submission dictionary
        """
        submission = {
            "model_name": model_name,
            "results": []
        }
        
        for result in results:
            submission["results"].append({
                "task_id": result["task_id"],
                "model_answer": result.get("output", ""),
                "correct": result.get("passed", False)
            })
        
        return submission
    
    def calculate_score(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate GAIA benchmark scores.
        
        Args:
            results: Evaluation results
        
        Returns:
            Dictionary with overall and per-level scores
        """
        scores = {
            "overall": 0.0,
            "level_1": 0.0,
            "level_2": 0.0,
            "level_3": 0.0
        }
        
        counts = {
            "overall": 0,
            "level_1": 0,
            "level_2": 0,
            "level_3": 0
        }
        
        for result in results:
            level = result.get("level", 1)
            level_key = f"level_{level}"
            passed = result.get("passed", False)
            
            # Overall
            counts["overall"] += 1
            if passed:
                scores["overall"] += 1
            
            # Per-level
            if level_key in counts:
                counts[level_key] += 1
                if passed:
                    scores[level_key] += 1
        
        # Calculate percentages
        for key in scores:
            if counts[key] > 0:
                scores[key] = scores[key] / counts[key] * 100
        
        return scores
