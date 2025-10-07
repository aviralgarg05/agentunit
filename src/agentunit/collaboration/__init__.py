"""Collaborative suite management with git-like version control for test suites.

This module provides version control capabilities for AgentUnit test suites,
enabling collaborative development through branching, merging, and conflict resolution.

Features:
- Git-like versioning for test suites and results
- Suite branching and merging with automatic conflict detection
- Change tracking and history management
- Collaborative editing with conflict resolution strategies
- Integration with DVC/MLflow for data versioning

Examples:
    >>> from agentunit.collaboration import SuiteVersion, VersionManager
    >>> 
    >>> # Version a test suite
    >>> manager = VersionManager(".agentunit")
    >>> version = manager.commit_suite(suite, "Initial suite version", author="user1")
    >>> 
    >>> # Create a branch for experimentation
    >>> manager.create_branch("experiment-metrics")
    >>> 
    >>> # Merge changes back
    >>> result = manager.merge_branch("experiment-metrics", strategy="auto")

Classes exported (available via lazy loading):
- SuiteVersion: Immutable snapshot of a test suite at a point in time
- VersionManager: Main interface for version control operations
- BranchManager: Handle branch creation, switching, and deletion
- MergeStrategy: Conflict resolution strategies for suite merging
- ConflictResolver: Resolve merge conflicts in test suites
- ChangeTracker: Track changes to suites and cases
- CollaborationHub: Multi-user coordination and locking

All classes use PEP 562 lazy loading for optional dependencies (DVC, MLflow).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .version import SuiteVersion, VersionManager
    from .branch import BranchManager
    from .merge import MergeStrategy, MergeResult, ConflictResolver, Conflict
    from .tracking import ChangeTracker, ChangeLog, Change, ChangeType
    from .hub import CollaborationHub, Lock

__version__ = "0.4.0"
__author__ = "AgentUnit Contributors"
__description__ = "Collaborative suite management with version control"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Core classes
    "SuiteVersion",
    "VersionManager",
    "BranchManager",
    "MergeStrategy",
    "MergeResult",
    "ConflictResolver",
    "Conflict",
    "ChangeTracker",
    "ChangeLog",
    "Change",
    "ChangeType",
    "CollaborationHub",
    "Lock",
]

def __getattr__(name: str) -> Any:
    """Lazy load collaboration components to avoid importing optional dependencies."""
    
    if name == "SuiteVersion":
        from .version import SuiteVersion
        return SuiteVersion
    
    if name == "VersionManager":
        from .version import VersionManager
        return VersionManager
    
    if name == "BranchManager":
        from .branch import BranchManager
        return BranchManager
    
    if name == "MergeStrategy":
        from .merge import MergeStrategy
        return MergeStrategy
    
    if name == "MergeResult":
        from .merge import MergeResult
        return MergeResult
    
    if name == "ConflictResolver":
        from .merge import ConflictResolver
        return ConflictResolver
    
    if name == "Conflict":
        from .merge import Conflict
        return Conflict
    
    if name == "ChangeTracker":
        from .tracking import ChangeTracker
        return ChangeTracker
    
    if name == "ChangeLog":
        from .tracking import ChangeLog
        return ChangeLog
    
    if name == "Change":
        from .tracking import Change
        return Change
    
    if name == "ChangeType":
        from .tracking import ChangeType
        return ChangeType
    
    if name == "CollaborationHub":
        from .hub import CollaborationHub
        return CollaborationHub
    
    if name == "Lock":
        from .hub import Lock
        return Lock
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Return list of available attributes for autocomplete."""
    return __all__
