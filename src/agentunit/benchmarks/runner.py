"""Unified benchmark runner."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from .gaia import GAIABenchmark, GAIALevel
from .arena import AgentArenaBenchmark, ArenaTaskType
from .leaderboard import LeaderboardSubmitter, LeaderboardConfig


class BenchmarkType(Enum):
    """Supported benchmark types."""
    GAIA = "gaia"
    AGENT_ARENA = "agent_arena"
    CUSTOM = "custom"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.
    
    Attributes:
        benchmark_type: Type of benchmark
        total_tasks: Total number of tasks
        passed_tasks: Number of passed tasks
        success_rate: Success rate percentage
        scores: Detailed scores
        results: Individual task results
        metadata: Additional metadata
    """
    benchmark_type: BenchmarkType
    total_tasks: int
    passed_tasks: int
    success_rate: float
    scores: Dict[str, float]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BenchmarkRunner:
    """Unified runner for multiple benchmarks.
    
    This runner provides:
    - One-click execution of popular benchmarks
    - Automatic result formatting
    - Leaderboard submission
    - Cross-benchmark comparison
    """
    
    def __init__(
        self,
        leaderboard_config: Optional[LeaderboardConfig] = None
    ):
        """Initialize benchmark runner.
        
        Args:
            leaderboard_config: Configuration for leaderboard submissions
        """
        self.leaderboard_config = leaderboard_config
        self.submitter = None
        if leaderboard_config:
            self.submitter = LeaderboardSubmitter(leaderboard_config)
    
    def run_gaia(
        self,
        dataset_path: Optional[str] = None,
        level: Optional[GAIALevel] = None,
        split: str = "validation",
        submit: bool = False
    ) -> BenchmarkResult:
        """Run GAIA benchmark.
        
        Args:
            dataset_path: Path to GAIA dataset
            level: Specific level to run
            split: Dataset split
            submit: Whether to submit to leaderboard
        
        Returns:
            Benchmark result
        """
        from pathlib import Path
        
        # Initialize GAIA benchmark
        gaia = GAIABenchmark(
            dataset_path=Path(dataset_path) if dataset_path else None,
            level=level
        )
        
        # Load dataset
        tasks = gaia.load_dataset(split)
        
        # Convert to AgentUnit format
        dataset = gaia.to_agentunit_dataset()
        
        # Placeholder for actual evaluation
        # In practice, this would use AgentUnit's runner
        results = [
            {
                "task_id": case.id,
                "passed": True,  # Would be actual evaluation result
                "output": "Sample output",
                "level": case.metadata.get("level", 1)
            }
            for case in dataset
        ]
        
        # Calculate scores
        scores = gaia.calculate_score(results)
        
        # Create result
        benchmark_result = BenchmarkResult(
            benchmark_type=BenchmarkType.GAIA,
            total_tasks=len(tasks),
            passed_tasks=sum(1 for r in results if r["passed"]),
            success_rate=scores["overall"],
            scores=scores,
            results=results,
            metadata={"level": level.value if level else "all", "split": split}
        )
        
        # Submit to leaderboard if requested
        if submit and self.submitter:
            submission = gaia.format_submission(
                results,
                model_name=self.leaderboard_config.model_name
            )
            self.submitter.submit(
                submission["results"],
                benchmark_name="gaia"
            )
        
        return benchmark_result
    
    def run_arena(
        self,
        dataset_path: Optional[str] = None,
        task_type: Optional[ArenaTaskType] = None,
        split: str = "test",
        submit: bool = False
    ) -> BenchmarkResult:
        """Run AgentArena benchmark.
        
        Args:
            dataset_path: Path to AgentArena dataset
            task_type: Specific task type to run
            split: Dataset split
            submit: Whether to submit to leaderboard
        
        Returns:
            Benchmark result
        """
        from pathlib import Path
        
        # Initialize AgentArena benchmark
        arena = AgentArenaBenchmark(
            dataset_path=Path(dataset_path) if dataset_path else None,
            task_type=task_type
        )
        
        # Load dataset
        tasks = arena.load_dataset(split)
        
        # Convert to AgentUnit format
        dataset = arena.to_agentunit_dataset()
        
        # Placeholder for actual evaluation
        results = [
            {
                "task_id": case.id,
                "passed": True,  # Would be actual evaluation result
                "output": "Sample output",
                "task_type": case.metadata.get("task_type", "unknown"),
                "steps": 3  # Would be actual step count
            }
            for case in dataset
        ]
        
        # Calculate scores
        scores = arena.calculate_score(results)
        
        # Create result
        benchmark_result = BenchmarkResult(
            benchmark_type=BenchmarkType.AGENT_ARENA,
            total_tasks=len(tasks),
            passed_tasks=sum(1 for r in results if r["passed"]),
            success_rate=scores["overall"],
            scores=scores,
            results=results,
            metadata={"task_type": task_type.value if task_type else "all", "split": split}
        )
        
        # Submit to leaderboard if requested
        if submit and self.submitter:
            self.submitter.submit(
                results,
                benchmark_name="agent_arena"
            )
        
        return benchmark_result
    
    def compare_benchmarks(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Compare results across multiple benchmarks.
        
        Args:
            results: List of benchmark results
        
        Returns:
            Comparison metrics
        """
        comparison = {
            "benchmarks": [],
            "avg_success_rate": 0.0,
            "total_tasks": 0,
            "total_passed": 0
        }
        
        for result in results:
            comparison["benchmarks"].append({
                "type": result.benchmark_type.value,
                "success_rate": result.success_rate,
                "total_tasks": result.total_tasks,
                "passed_tasks": result.passed_tasks,
                "scores": result.scores
            })
            
            comparison["total_tasks"] += result.total_tasks
            comparison["total_passed"] += result.passed_tasks
        
        if comparison["total_tasks"] > 0:
            comparison["avg_success_rate"] = (
                comparison["total_passed"] / comparison["total_tasks"] * 100
            )
        
        return comparison
