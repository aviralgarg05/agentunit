"""CLI tool to view and analyze leaderboard submissions.

Features:
- List all local submissions
- Show detailed stats for a submission
- Compare multiple submissions
- Export leaderboard summary
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table


logger = logging.getLogger(__name__)
console = Console()


@dataclass
class SubmissionSummary:
    """Summary of a benchmark submission."""

    filename: str
    timestamp: str
    benchmark: str
    model: str
    diff_score: float | None
    total_tasks: int
    passed_tasks: int
    score_pct: float
    avg_latency: float
    total_cost: float


class SubmissionViewer:
    """View and analyze local leaderboard submissions."""

    def __init__(self, submissions_dir: Path):
        """Initialize viewer.

        Args:
            submissions_dir: Directory containing JSON submissions
        """
        self.submissions_dir = submissions_dir

    def list_submissions(self) -> list[SubmissionSummary]:
        """List all valid submissions."""
        summaries = []
        if not self.submissions_dir.exists():
            return []

        for f in self.submissions_dir.glob("*.json"):
            try:
                with open(f) as file:
                    data = json.load(file)

                results = data.get("results", [])
                total = len(results)
                passed = sum(1 for r in results if r.get("correct", False))

                # Calculate basic stats
                latencies = [r.get("latency") for r in results if r.get("latency") is not None]
                costs = [r.get("cost") for r in results if r.get("cost") is not None]

                summaries.append(
                    SubmissionSummary(
                        filename=f.name,
                        timestamp=data.get("timestamp", ""),
                        benchmark=data.get("benchmark", "unknown"),
                        model=data.get("model_name", "unknown"),
                        diff_score=None,  # Could be added if diff logic exists
                        total_tasks=total,
                        passed_tasks=passed,
                        score_pct=(passed / total * 100) if total > 0 else 0.0,
                        avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
                        total_cost=sum(costs) if costs else 0.0,
                    )
                )
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")

        # Sort by timestamp descending
        summaries.sort(key=lambda x: x.timestamp, reverse=True)
        return summaries

    def print_leaderboard(self, benchmark: str | None = None):
        """Print rich table of submissions."""
        summaries = self.list_submissions()
        if benchmark:
            summaries = [s for s in summaries if s.benchmark == benchmark]

        table = Table(title=f"Leaderboard Submissions {f'({benchmark})' if benchmark else ''}")
        table.add_column("Timestamp", style="cyan")
        table.add_column("Benchmark", style="magenta")
        table.add_column("Model", style="green")
        table.add_column("Score", justify="right", style="bold yellow")
        table.add_column("Tasks", justify="right")
        table.add_column("Latency (s)", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("Filename", style="dim")

        for s in summaries:
            table.add_row(
                s.timestamp[:19].replace("T", " "),
                s.benchmark,
                s.model,
                f"{s.score_pct:.1f}%",
                f"{s.passed_tasks}/{s.total_tasks}",
                f"{s.avg_latency:.2f}",
                f"${s.total_cost:.4f}",
                s.filename,
            )

        console.print(table)


@click.group()
def cli():
    """Manage leaderboard submissions."""


@cli.command()
@click.option("--dir", default="leaderboard_submissions", help="Submissions directory")
@click.option("--benchmark", help="Filter by benchmark name")
def list(dir, benchmark):
    """List all local submissions."""
    viewer = SubmissionViewer(Path(dir))
    viewer.print_leaderboard(benchmark)


if __name__ == "__main__":
    cli()
