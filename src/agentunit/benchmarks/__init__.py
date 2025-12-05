"""Benchmark integrations for AgentUnit.

This module provides integrations with popular AI agent benchmarks
including GAIA 2.0, AgentArena, and custom leaderboard support.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .arena import AgentArenaBenchmark, ArenaTask, ArenaTaskType
    from .gaia import GAIABenchmark, GAIALevel
    from .leaderboard import LeaderboardConfig, LeaderboardSubmitter
    from .runner import BenchmarkResult, BenchmarkRunner

__all__ = [
    "AgentArenaBenchmark",
    "ArenaTask",
    "ArenaTaskType",
    "BenchmarkResult",
    "BenchmarkRunner",
    "GAIABenchmark",
    "GAIALevel",
    "LeaderboardConfig",
    "LeaderboardSubmitter",
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
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
