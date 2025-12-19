"""AgentUnit - pytest-style evaluation harness for agentic AI and RAG workflows."""

from __future__ import annotations

from .core.runner import Runner, run_suite
from .core.scenario import Scenario
from .datasets.base import DatasetCase, DatasetSource
from .reporting.results import ScenarioResult, SuiteResult


__all__ = [
    "DatasetCase",
    "DatasetSource",
    "Runner",
    "Scenario",
    "ScenarioResult",
    "SuiteResult",
    "run_suite",
]

__version__ = "0.7.0"

