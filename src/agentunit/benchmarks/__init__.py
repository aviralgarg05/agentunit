"""Benchmark integrations for AgentUnit.

This module provides integrations with popular AI agent benchmarks
including GAIA 2.0, AgentArena, and custom leaderboard support.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gaia import GAIABenchmark, GAIALevel
    from .arena import AgentArenaBenchmark, ArenaTask, ArenaTaskType
    from .leaderboard import LeaderboardSubmitter, LeaderboardConfig
    from .runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    "GAIABenchmark",
    "GAIALevel",
    "AgentArenaBenchmark",
    "ArenaTask",
    "ArenaTaskType",
    "LeaderboardSubmitter",
    "LeaderboardConfig",
    "BenchmarkRunner",
    "BenchmarkResult",
]


def __getattr__(name: str):
    """Lazy loading of benchmark components."""
    if name == "GAIABenchmark":
        from .gaia import GAIABenchmark
        return GAIABenchmark
    elif name == "GAIALevel":
        from .gaia import GAIALevel
        return GAIALevel
    elif name == "AgentArenaBenchmark":
        from .arena import AgentArenaBenchmark
        return AgentArenaBenchmark
    elif name == "ArenaTask":
        from .arena import ArenaTask
        return ArenaTask
    elif name == "ArenaTaskType":
        from .arena import ArenaTaskType
        return ArenaTaskType
    elif name == "LeaderboardSubmitter":
        from .leaderboard import LeaderboardSubmitter
        return LeaderboardSubmitter
    elif name == "LeaderboardConfig":
        from .leaderboard import LeaderboardConfig
        return LeaderboardConfig
    elif name == "BenchmarkRunner":
        from .runner import BenchmarkRunner
        return BenchmarkRunner
    elif name == "BenchmarkResult":
        from .runner import BenchmarkResult
        return BenchmarkResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
