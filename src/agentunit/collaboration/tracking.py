"""Change tracking for test suite modifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .version import SuiteVersion


class ChangeType(Enum):
    """Types of changes that can occur in test suites."""
    
    SCENARIO_ADDED = "scenario_added"
    SCENARIO_REMOVED = "scenario_removed"
    SCENARIO_MODIFIED = "scenario_modified"
    CASE_ADDED = "case_added"
    CASE_REMOVED = "case_removed"
    CASE_MODIFIED = "case_modified"
    CONFIG_CHANGED = "config_changed"


@dataclass
class Change:
    """Represents a single change to a test suite.
    
    Attributes:
        type: Type of change
        path: Path to changed element
        old_value: Previous value (None for additions)
        new_value: New value (None for deletions)
        timestamp: When change occurred
        author: Who made the change
        message: Description of change
    """
    
    type: ChangeType
    path: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = ""
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert change to dictionary."""
        return {
            "type": self.type.value,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "message": self.message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Change:
        """Create change from dictionary."""
        return cls(
            type=ChangeType(data["type"]),
            path=data["path"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data.get("author", ""),
            message=data.get("message", ""),
        )


@dataclass
class ChangeLog:
    """Collection of changes with metadata.
    
    Tracks all changes made between two suite versions.
    
    Attributes:
        from_version: Starting version ID
        to_version: Ending version ID
        changes: List of individual changes
        timestamp: When changelog was created
        summary: High-level summary of changes
    """
    
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    changes: List[Change] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    
    def add_change(self, change: Change) -> None:
        """Add a change to the log."""
        self.changes.append(change)
    
    def filter_by_type(self, change_type: ChangeType) -> List[Change]:
        """Get all changes of a specific type."""
        return [c for c in self.changes if c.type == change_type]
    
    def filter_by_path(self, path_pattern: str) -> List[Change]:
        """Get changes matching a path pattern."""
        return [c for c in self.changes if path_pattern in c.path]
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of changes."""
        summary = {}
        for change_type in ChangeType:
            count = len(self.filter_by_type(change_type))
            if count > 0:
                summary[change_type.value] = count
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert changelog to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changes": [c.to_dict() for c in self.changes],
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary or str(self.get_summary()),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeLog:
        """Create changelog from dictionary."""
        return cls(
            from_version=data.get("from_version"),
            to_version=data.get("to_version"),
            changes=[Change.from_dict(c) for c in data.get("changes", [])],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            summary=data.get("summary", ""),
        )


class ChangeTracker:
    """Track and compute changes between test suite versions.
    
    Examples:
        >>> tracker = ChangeTracker()
        >>> 
        >>> # Compute changes between versions
        >>> changelog = tracker.diff(old_version, new_version)
        >>> 
        >>> # Get summary
        >>> print(changelog.get_summary())
        >>> # {'scenario_added': 2, 'case_modified': 5}
        >>> 
        >>> # Filter specific changes
        >>> scenario_changes = changelog.filter_by_type(ChangeType.SCENARIO_MODIFIED)
    """
    
    def diff(
        self,
        old_version: SuiteVersion,
        new_version: SuiteVersion,
        author: str = "system",
    ) -> ChangeLog:
        """Compute changes between two versions.
        
        Args:
            old_version: Earlier version
            new_version: Later version
            author: Who to attribute changes to
        
        Returns:
            ChangeLog with all detected changes
        """
        changelog = ChangeLog(
            from_version=old_version.commit_id,
            to_version=new_version.commit_id,
        )
        
        # Track scenario changes
        self._diff_scenarios(old_version, new_version, changelog, author)
        
        # Track case changes
        self._diff_cases(old_version, new_version, changelog, author)
        
        # Generate summary
        summary_dict = changelog.get_summary()
        changelog.summary = ", ".join(f"{count} {type}" for type, count in summary_dict.items())
        
        return changelog
    
    def _diff_scenarios(
        self,
        old: SuiteVersion,
        new: SuiteVersion,
        changelog: ChangeLog,
        author: str,
    ) -> None:
        """Detect scenario changes."""
        old_names = set(old.scenarios.keys())
        new_names = set(new.scenarios.keys())
        
        # Added scenarios
        for name in new_names - old_names:
            changelog.add_change(Change(
                type=ChangeType.SCENARIO_ADDED,
                path=f"scenarios/{name}",
                new_value=new.scenarios[name],
                author=author,
                message=f"Added scenario '{name}'",
            ))
        
        # Removed scenarios
        for name in old_names - new_names:
            changelog.add_change(Change(
                type=ChangeType.SCENARIO_REMOVED,
                path=f"scenarios/{name}",
                old_value=old.scenarios[name],
                author=author,
                message=f"Removed scenario '{name}'",
            ))
        
        # Modified scenarios
        for name in old_names & new_names:
            if old.scenarios[name] != new.scenarios[name]:
                changelog.add_change(Change(
                    type=ChangeType.SCENARIO_MODIFIED,
                    path=f"scenarios/{name}",
                    old_value=old.scenarios[name],
                    new_value=new.scenarios[name],
                    author=author,
                    message=f"Modified scenario '{name}'",
                ))
    
    def _diff_cases(
        self,
        old: SuiteVersion,
        new: SuiteVersion,
        changelog: ChangeLog,
        author: str,
    ) -> None:
        """Detect case changes."""
        # Build indices by input
        old_index = {self._case_key(c): c for c in old.cases}
        new_index = {self._case_key(c): c for c in new.cases}
        
        old_keys = set(old_index.keys())
        new_keys = set(new_index.keys())
        
        # Added cases
        for key in new_keys - old_keys:
            case = new_index[key]
            changelog.add_change(Change(
                type=ChangeType.CASE_ADDED,
                path=f"cases/{key}",
                new_value=case,
                author=author,
                message=f"Added case with input '{key}'",
            ))
        
        # Removed cases
        for key in old_keys - new_keys:
            case = old_index[key]
            changelog.add_change(Change(
                type=ChangeType.CASE_REMOVED,
                path=f"cases/{key}",
                old_value=case,
                author=author,
                message=f"Removed case with input '{key}'",
            ))
        
        # Modified cases (same input, different expected)
        for key in old_keys & new_keys:
            if old_index[key] != new_index[key]:
                changelog.add_change(Change(
                    type=ChangeType.CASE_MODIFIED,
                    path=f"cases/{key}",
                    old_value=old_index[key],
                    new_value=new_index[key],
                    author=author,
                    message=f"Modified case with input '{key}'",
                ))
    
    def _case_key(self, case: Dict[str, Any]) -> str:
        """Generate unique key for a case based on input."""
        input_val = case.get("input", "")
        return str(input_val)[:50]  # Truncate for reasonable key length
    
    def get_change_statistics(self, changelog: ChangeLog) -> Dict[str, Any]:
        """Compute detailed statistics from a changelog.
        
        Args:
            changelog: ChangeLog to analyze
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_changes": len(changelog.changes),
            "by_type": changelog.get_summary(),
            "scenarios": {
                "added": len(changelog.filter_by_type(ChangeType.SCENARIO_ADDED)),
                "removed": len(changelog.filter_by_type(ChangeType.SCENARIO_REMOVED)),
                "modified": len(changelog.filter_by_type(ChangeType.SCENARIO_MODIFIED)),
            },
            "cases": {
                "added": len(changelog.filter_by_type(ChangeType.CASE_ADDED)),
                "removed": len(changelog.filter_by_type(ChangeType.CASE_REMOVED)),
                "modified": len(changelog.filter_by_type(ChangeType.CASE_MODIFIED)),
            },
        }
        
        return stats
