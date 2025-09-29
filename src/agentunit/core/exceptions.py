"""Custom exceptions for AgentUnit."""
from __future__ import annotations


class AgentUnitError(Exception):
    """Base class for AgentUnit exceptions."""


class AdapterNotAvailableError(AgentUnitError):
    """Raised when an adapter cannot be initialized due to missing dependencies."""


class ScenarioExecutionError(AgentUnitError):
    """Raised when a scenario fails during execution."""
