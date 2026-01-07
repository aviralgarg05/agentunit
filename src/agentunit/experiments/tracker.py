"""Experiment tracking for AI agent evaluation.

Standard feature similar to MLflow, DeepEval experiments.
Track runs, compare experiments, version control integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from typing_extensions import Self


logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an experiment run."""

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """Metrics collected during a run."""

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    tokens_total: int | None = None
    custom: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """A single experiment run."""

    run_id: str
    experiment_name: str
    status: str  # pending, running, completed, failed
    config: RunConfig
    metrics: RunMetrics = field(default_factory=RunMetrics)
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    git_commit: str | None = None
    parent_run_id: str | None = None


class ExperimentTracker:
    """Track and manage experiments.

    Standard feature similar to MLflow, DeepEval experiment tracking.

    Features:
    - Create and manage experiment runs
    - Log metrics, parameters, artifacts
    - Compare runs
    - Version control integration
    - Export to various formats

    Example:
        ```python
        tracker = ExperimentTracker("my_experiments")

        with tracker.start_run("baseline_test") as run:
            # Run your evaluation
            run.log_metric("accuracy", 0.95)
            run.log_param("model", "gpt-4o")
            run.log_artifact("results.json")

        # Compare runs
        comparison = tracker.compare_runs(["run1", "run2"])
        ```
    """

    def __init__(
        self,
        experiment_dir: str | Path = "experiments",
        experiment_name: str = "default",
    ):
        """Initialize tracker.

        Args:
            experiment_dir: Directory to store experiments
            experiment_name: Name of this experiment group
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.runs: dict[str, ExperimentRun] = {}
        self.current_run: ExperimentRun | None = None

        # Create experiment directory
        self._experiment_path = self.experiment_dir / experiment_name
        self._experiment_path.mkdir(parents=True, exist_ok=True)

        # Load existing runs
        self._load_runs()

    def _load_runs(self) -> None:
        """Load existing runs from disk."""
        runs_file = self._experiment_path / "runs.json"
        if runs_file.exists():
            try:
                with open(runs_file) as f:
                    data = json.load(f)
                for run_data in data.get("runs", []):
                    run = ExperimentRun(
                        run_id=run_data["run_id"],
                        experiment_name=run_data["experiment_name"],
                        status=run_data["status"],
                        config=RunConfig(**run_data.get("config", {})),
                        metrics=RunMetrics(**run_data.get("metrics", {})),
                        start_time=run_data.get("start_time", ""),
                        end_time=run_data.get("end_time", ""),
                        duration_seconds=run_data.get("duration_seconds", 0),
                        artifacts=run_data.get("artifacts", []),
                        git_commit=run_data.get("git_commit"),
                    )
                    self.runs[run.run_id] = run
            except Exception as e:
                logger.warning(f"Failed to load runs: {e}")

    def _save_runs(self) -> None:
        """Save runs to disk."""
        runs_file = self._experiment_path / "runs.json"
        data = {
            "experiment_name": self.experiment_name,
            "updated_at": datetime.now().isoformat(),
            "runs": [self._run_to_dict(run) for run in self.runs.values()],
        }
        with open(runs_file, "w") as f:
            json.dump(data, f, indent=2)

    def _run_to_dict(self, run: ExperimentRun) -> dict:
        """Convert run to dictionary."""
        return {
            "run_id": run.run_id,
            "experiment_name": run.experiment_name,
            "status": run.status,
            "config": asdict(run.config),
            "metrics": asdict(run.metrics),
            "start_time": run.start_time,
            "end_time": run.end_time,
            "duration_seconds": run.duration_seconds,
            "artifacts": run.artifacts,
            "git_commit": run.git_commit,
        }

    def _generate_run_id(self, name: str) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{timestamp}_{os.getpid()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{short_hash}"

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def start_run(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> RunContext:
        """Start a new experiment run.

        Args:
            name: Run name
            description: Run description
            tags: Optional tags
            parameters: Run parameters

        Returns:
            RunContext for context manager usage
        """
        run_id = self._generate_run_id(name)

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            status="running",
            config=RunConfig(
                name=name,
                description=description,
                tags=tags or [],
                parameters=parameters or {},
            ),
            start_time=datetime.now().isoformat(),
            git_commit=self._get_git_commit(),
        )

        self.runs[run_id] = run
        self.current_run = run
        self._save_runs()

        logger.info(f"Started run: {run_id}")
        return RunContext(self, run)

    def end_run(self, status: str = "completed") -> None:
        """End the current run.

        Args:
            status: Final status (completed, failed)
        """
        if self.current_run is None:
            return

        self.current_run.status = status
        self.current_run.end_time = datetime.now().isoformat()

        # Calculate duration
        start = datetime.fromisoformat(self.current_run.start_time)
        end = datetime.fromisoformat(self.current_run.end_time)
        self.current_run.duration_seconds = (end - start).total_seconds()

        self._save_runs()
        logger.info(f"Ended run: {self.current_run.run_id} ({status})")
        self.current_run = None

    def log_metric(self, key: str, value: float) -> None:
        """Log a metric to current run.

        Args:
            key: Metric name
            value: Metric value
        """
        if self.current_run is None:
            raise RuntimeError("No active run")

        # Check if it's a standard metric
        if hasattr(self.current_run.metrics, key):
            setattr(self.current_run.metrics, key, value)
        else:
            self.current_run.metrics.custom[key] = value

        self._save_runs()

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.current_run is None:
            raise RuntimeError("No active run")
        self.current_run.config.parameters[key] = value
        self._save_runs()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            params: Dictionary of parameters
        """
        if self.current_run is None:
            raise RuntimeError("No active run")
        self.current_run.config.parameters.update(params)
        self._save_runs()

    def log_artifact(self, filepath: str | Path) -> None:
        """Log an artifact file.

        Args:
            filepath: Path to artifact file
        """
        if self.current_run is None:
            raise RuntimeError("No active run")
        self.current_run.artifacts.append(str(filepath))
        self._save_runs()

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Get a specific run.

        Args:
            run_id: Run ID

        Returns:
            ExperimentRun or None
        """
        return self.runs.get(run_id)

    def list_runs(
        self,
        status: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ExperimentRun]:
        """List runs with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags (any match)
            limit: Maximum runs to return

        Returns:
            List of runs
        """
        runs = list(self.runs.values())

        if status:
            runs = [r for r in runs if r.status == status]

        if tags:
            runs = [r for r in runs if any(t in r.config.tags for t in tags)]

        # Sort by start time, newest first
        runs.sort(key=lambda r: r.start_time, reverse=True)

        return runs[:limit]

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Comparison dictionary
        """
        comparison = {
            "runs": [],
            "metrics_comparison": {},
            "best_run": None,
        }

        runs = [self.runs.get(rid) for rid in run_ids if rid in self.runs]

        if not runs:
            return comparison

        for run in runs:
            run_data = {
                "run_id": run.run_id,
                "name": run.config.name,
                "status": run.status,
                "duration": run.duration_seconds,
                "metrics": asdict(run.metrics),
                "parameters": run.config.parameters,
            }
            comparison["runs"].append(run_data)

        # Compare metrics
        metric_names = metrics or ["accuracy", "f1_score", "latency_ms", "cost_usd"]
        for metric in metric_names:
            values = []
            for run in runs:
                value = getattr(run.metrics, metric, None)
                if value is None:
                    value = run.metrics.custom.get(metric)
                values.append((run.run_id, value))

            comparison["metrics_comparison"][metric] = {
                "values": {rid: v for rid, v in values if v is not None},
                "best": max(
                    [(rid, v) for rid, v in values if v is not None],
                    key=lambda x: x[1] if metric != "latency_ms" else -x[1],
                    default=(None, None),
                )[0],
            }

        # Determine best overall run (by accuracy if available)
        if "accuracy" in comparison["metrics_comparison"]:
            comparison["best_run"] = comparison["metrics_comparison"]["accuracy"]["best"]

        return comparison

    def delete_run(self, run_id: str) -> bool:
        """Delete a run.

        Args:
            run_id: Run ID to delete

        Returns:
            True if deleted
        """
        if run_id in self.runs:
            del self.runs[run_id]
            self._save_runs()
            return True
        return False

    def export_runs(self, filepath: str | Path, format: str = "json") -> None:
        """Export runs to file.

        Args:
            filepath: Output file path
            format: Export format (json, csv)
        """
        filepath = Path(filepath)

        if format == "json":
            data = {
                "experiment_name": self.experiment_name,
                "exported_at": datetime.now().isoformat(),
                "runs": [self._run_to_dict(r) for r in self.runs.values()],
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "name",
                        "status",
                        "accuracy",
                        "f1_score",
                        "latency_ms",
                        "cost_usd",
                        "duration_seconds",
                        "start_time",
                    ]
                )
                for run in self.runs.values():
                    writer.writerow(
                        [
                            run.run_id,
                            run.config.name,
                            run.status,
                            run.metrics.accuracy,
                            run.metrics.f1_score,
                            run.metrics.latency_ms,
                            run.metrics.cost_usd,
                            run.duration_seconds,
                            run.start_time,
                        ]
                    )

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(self.runs)} runs to {filepath}")


class RunContext:
    """Context manager for experiment runs."""

    def __init__(self, tracker: ExperimentTracker, run: ExperimentRun):
        """Initialize context."""
        self.tracker = tracker
        self.run = run

    def __enter__(self) -> Self:
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context."""
        status = "failed" if exc_type else "completed"
        self.tracker.end_run(status)

    def log_metric(self, key: str, value: float) -> None:
        """Log metric."""
        self.tracker.log_metric(key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log multiple metrics."""
        self.tracker.log_metrics(metrics)

    def log_param(self, key: str, value: Any) -> None:
        """Log parameter."""
        self.tracker.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.tracker.log_params(params)

    def log_artifact(self, filepath: str | Path) -> None:
        """Log artifact."""
        self.tracker.log_artifact(filepath)


__all__ = [
    "ExperimentRun",
    "ExperimentTracker",
    "RunConfig",
    "RunContext",
    "RunMetrics",
]
