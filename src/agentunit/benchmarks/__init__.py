"""Benchmark integrations for AgentUnit.

This module provides integrations with popular AI agent benchmarks
including GAIA 2.0, AgentArena, and custom leaderboard support.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .arena import AgentArenaBenchmark, ArenaTask, ArenaTaskType
    from .experiments import (
        BenchmarkExperiment,
        ExperimentConfig,
        ExperimentResult,
        ResearchGapAnalyzer,
        TaskResult,
        run_standard_experiment,
    )
    from .gaia import GAIABenchmark, GAIALevel
    from .leaderboard import LeaderboardConfig, LeaderboardSubmitter
    from .runner import BenchmarkResult, BenchmarkRunner

__all__ = [
    "RESEARCH_GAPS",
    "AgentArenaBenchmark",
    "ArenaTask",
    "ArenaTaskType",
    "BenchmarkExperiment",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "GAIABenchmark",
    "GAIALevel",
    "LLMClient",
    "LLMConfig",
    "LeaderboardConfig",
    "LeaderboardSubmitter",
    "RealExperimentRunner",
    "RealTaskResult",
    "ResearchGapAnalyzer",
    "TaskResult",
    "run_real_experiment",
    "run_standard_experiment",
]


def __getattr__(name: str):
    """Lazy loading of benchmark components."""
    if name == "GAIABenchmark":
        from .gaia import GAIABenchmark

        return GAIABenchmark
    if name == "GAIALevel":
        from .gaia import GAIALevel

        return GAIALevel
    if name == "AgentArenaBenchmark":
        from .arena import AgentArenaBenchmark

        return AgentArenaBenchmark
    if name == "ArenaTask":
        from .arena import ArenaTask

        return ArenaTask
    if name == "ArenaTaskType":
        from .arena import ArenaTaskType

        return ArenaTaskType
    if name == "LeaderboardSubmitter":
        from .leaderboard import LeaderboardSubmitter

        return LeaderboardSubmitter
    if name == "LeaderboardConfig":
        from .leaderboard import LeaderboardConfig

        return LeaderboardConfig
    if name == "BenchmarkRunner":
        from .runner import BenchmarkRunner

        return BenchmarkRunner
    if name == "BenchmarkResult":
        from .runner import BenchmarkResult

        return BenchmarkResult

    # New experiment exports
    if name == "BenchmarkExperiment":
        from .experiments import BenchmarkExperiment

        return BenchmarkExperiment
    if name == "ExperimentConfig":
        from .experiments import ExperimentConfig

        return ExperimentConfig
    if name == "ExperimentResult":
        from .experiments import ExperimentResult

        return ExperimentResult
    if name == "TaskResult":
        from .experiments import TaskResult

        return TaskResult
    if name == "ResearchGapAnalyzer":
        from .experiments import ResearchGapAnalyzer

        return ResearchGapAnalyzer
    if name == "run_standard_experiment":
        from .experiments import run_standard_experiment

        return run_standard_experiment

    # Real experiment exports
    if name == "LLMConfig":
        from .real_experiments import LLMConfig

        return LLMConfig
    if name == "LLMClient":
        from .real_experiments import LLMClient

        return LLMClient
    if name == "RealTaskResult":
        from .real_experiments import RealTaskResult

        return RealTaskResult
    if name == "RealExperimentRunner":
        from .real_experiments import RealExperimentRunner

        return RealExperimentRunner
    if name == "run_real_experiment":
        from .real_experiments import run_real_experiment

        return run_real_experiment
    if name == "RESEARCH_GAPS":
        from .real_experiments import RESEARCH_GAPS

        return RESEARCH_GAPS

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
