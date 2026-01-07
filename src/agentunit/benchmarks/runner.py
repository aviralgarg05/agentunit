"""Execution engine for running benchmarks against agents."""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from agentunit.benchmarks.definitions import BenchmarkScenario, ScenarioResult


logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """Runs benchmark scenarios against a given agent or model."""

    def __init__(self, submissions_dir: str | Path = "leaderboard_submissions"):
        self.submissions_dir = Path(submissions_dir)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

    def run_scenario(
        self,
        scenario: BenchmarkScenario,
        agent_func: Callable[[str], str],
        model_name: str = "agent-v1",
    ) -> Path:
        """
        Run a specific scenario.

        Args:
            scenario: The benchmark scenario to run.
            agent_func: A function that takes a prompt (str) and returns a response (str).
            model_name: Name of the model/agent being tested.

        Returns:
            Path to the saved submission file.
        """
        results = []
        console.print(f"\n🚀 Starting Benchmark: [bold cyan]{scenario.name}[/bold cyan]")
        console.print(f"[dim]{scenario.description}[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"Running {scenario.total_tasks} tasks...", total=scenario.total_tasks
            )

            for task in scenario.tasks:
                start_time = time.time()
                try:
                    # Execute agent
                    response = agent_func(task.prompt)
                    latency = time.time() - start_time

                    # Simple automated verification (can be upgraded to LLM-as-Judge later)
                    # Checks if expected answer is vaguely present in the response
                    correct = False
                    if task.expected_answer:
                        correct = task.expected_answer.lower() in response.lower()

                    results.append(
                        ScenarioResult(
                            task_id=task.id,
                            prompt=task.prompt,
                            model_output=response,
                            correct=correct,
                            latency=latency,
                        )
                    )

                except Exception as e:
                    latency = time.time() - start_time
                    results.append(
                        ScenarioResult(
                            task_id=task.id,
                            prompt=task.prompt,
                            model_output="",
                            correct=False,
                            latency=latency,
                            error=str(e),
                        )
                    )

                progress.advance(task_id)

        # Save results
        return self._save_submission(scenario, model_name, results)

    def _save_submission(
        self, scenario: BenchmarkScenario, model_name: str, results: list[ScenarioResult]
    ) -> Path:
        """Save results to JSON format compatible with benchmark_viewer."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert objects to dicts
        results_data = [
            {
                "task_id": r.task_id,
                "prompt": r.prompt,
                "response": r.model_output,
                "correct": r.correct,
                "latency": r.latency,
                "cost": r.cost,
                "error": r.error,
            }
            for r in results
        ]

        submission_data = {
            "timestamp": timestamp,
            "benchmark": scenario.name,
            "model_name": model_name,
            "results": results_data,
            "metadata": {"description": scenario.description, "total_tasks": scenario.total_tasks},
        }

        filename = f"submission_{scenario.name}_{int(time.time())}.json"
        filepath = self.submissions_dir / filename

        with open(filepath, "w") as f:
            json.dump(submission_data, f, indent=2, default=str)

        console.print(f"\n✅ Results saved to [bold green]{filepath}[/bold green]")
        return filepath
