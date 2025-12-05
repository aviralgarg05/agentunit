"""Version control for test suites with git-like semantics."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from agentunit.core.scenario import Scenario


@dataclass(frozen=True)
class SuiteVersion:
    """Immutable snapshot of a test suite at a point in time.

    Similar to a git commit, this captures the complete state of a test suite
    including all scenarios, datasets, and configuration.

    Attributes:
        commit_id: Unique hash identifier for this version
        suite_name: Name of the test suite
        timestamp: When this version was created
        author: Who created this version
        message: Commit message describing changes
        parent_id: ID of parent version (None for initial commit)
        branch: Branch this version belongs to
        scenarios: Serialized scenario configurations
        cases: Serialized dataset cases
        metadata: Additional versioning metadata
    """

    commit_id: str
    suite_name: str
    timestamp: datetime
    author: str
    message: str
    parent_id: str | None = None
    branch: str = "main"
    scenarios: dict[str, Any] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.commit_id)

    def serialize(self) -> dict[str, Any]:
        """Serialize version to JSON-compatible dict."""
        return {
            "commit_id": self.commit_id,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "message": self.message,
            "parent_id": self.parent_id,
            "branch": self.branch,
            "scenarios": self.scenarios,
            "cases": self.cases,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SuiteVersion:
        """Deserialize version from dict."""
        return cls(
            commit_id=data["commit_id"],
            suite_name=data["suite_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data["author"],
            message=data["message"],
            parent_id=data.get("parent_id"),
            branch=data.get("branch", "main"),
            scenarios=data.get("scenarios", {}),
            cases=data.get("cases", []),
            metadata=data.get("metadata", {}),
        )


class VersionManager:
    """Main interface for version control operations on test suites.

    Provides git-like commands for committing, branching, merging, and
    tracking changes to AgentUnit test suites.

    Storage uses JSON for metadata and pickle for binary data, organized
    in a .agentunit directory similar to .git:

    .agentunit/
        objects/        # Versioned suite snapshots
        refs/           # Branch references
            heads/      # Local branches
            tags/       # Version tags
        HEAD            # Current branch pointer
        index           # Staging area
        config          # Repository configuration

    Examples:
        >>> manager = VersionManager(".agentunit")
        >>>
        >>> # Commit current state
        >>> version = manager.commit_suite(
        ...     suite,
        ...     "Add new edge case scenarios",
        ...     author="alice@example.com"
        ... )
        >>>
        >>> # List history
        >>> history = manager.get_history(limit=10)
        >>> for v in history:
        ...     print(f"{v.commit_id[:8]} {v.message}")
    """

    def __init__(self, repo_path: str | Path = ".agentunit"):
        """Initialize version manager.

        Args:
            repo_path: Path to version control repository directory
        """
        self.repo_path = Path(repo_path)
        self.objects_dir = self.repo_path / "objects"
        self.refs_dir = self.repo_path / "refs"
        self.heads_dir = self.refs_dir / "heads"
        self.tags_dir = self.refs_dir / "tags"
        self.head_file = self.repo_path / "HEAD"
        self.index_file = self.repo_path / "index"
        self.config_file = self.repo_path / "config"

        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Create repository directory structure if it doesn't exist."""
        for directory in [self.objects_dir, self.heads_dir, self.tags_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        if not self.head_file.exists():
            self.head_file.write_text("ref: refs/heads/main\n")

        if not self.config_file.exists():
            config = {
                "core": {
                    "version": "1",
                    "compression": "pickle",
                },
                "user": {
                    "name": "AgentUnit User",
                    "email": "user@agentunit.dev",
                },
            }
            self.config_file.write_text(json.dumps(config, indent=2))

    def _compute_hash(self, content: Any) -> str:
        """Compute SHA-256 hash of content."""
        serialized = pickle.dumps(content)
        return hashlib.sha256(serialized).hexdigest()

    def _get_current_branch(self) -> str:
        """Get name of currently checked out branch."""
        head_content = self.head_file.read_text().strip()
        if head_content.startswith("ref: refs/heads/"):
            return head_content.replace("ref: refs/heads/", "")
        return "detached"

    def _get_branch_head(self, branch: str) -> str | None:
        """Get commit ID at the head of a branch."""
        branch_file = self.heads_dir / branch
        if branch_file.exists():
            return branch_file.read_text().strip()
        return None

    def _update_branch_head(self, branch: str, commit_id: str) -> None:
        """Update branch to point to a commit."""
        branch_file = self.heads_dir / branch
        branch_file.write_text(commit_id + "\n")

    def commit_suite(
        self,
        scenarios: list[Scenario],
        message: str,
        author: str | None = None,
        suite_name: str = "default",
    ) -> SuiteVersion:
        """Create a new version of the test suite.

        Args:
            scenarios: List of scenarios to version
            message: Commit message describing changes
            author: Author of this version (uses config if not provided)
            suite_name: Name of the suite

        Returns:
            SuiteVersion object representing the commit
        """
        if author is None:
            config = json.loads(self.config_file.read_text())
            author = f"{config['user']['name']} <{config['user']['email']}>"

        current_branch = self._get_current_branch()
        parent_id = self._get_branch_head(current_branch)

        # Serialize scenarios and cases
        scenarios_data = {}
        all_cases = []

        for scenario in scenarios:
            scenarios_data[scenario.name] = {
                "name": scenario.name,
                "adapter": scenario.adapter.name
                if hasattr(scenario.adapter, "name")
                else str(type(scenario.adapter).__name__),
                "config": getattr(scenario, "config", {}),
            }

            # Collect cases from scenario's dataset
            if hasattr(scenario, "dataset"):
                dataset = scenario.dataset
                if hasattr(dataset, "cases"):
                    for case in dataset.cases:
                        case_data = {
                            "input": case.input,
                            "expected": case.expected,
                            "context": getattr(case, "context", None),
                            "metadata": getattr(case, "metadata", {}),
                        }
                        all_cases.append(case_data)

        # Create version object
        version_data = {
            "suite_name": suite_name,
            "author": author,
            "message": message,
            "parent_id": parent_id,
            "branch": current_branch,
            "scenarios": scenarios_data,
            "cases": all_cases,
            "metadata": {
                "num_scenarios": len(scenarios),
                "num_cases": len(all_cases),
            },
        }

        commit_id = self._compute_hash(version_data)

        version = SuiteVersion(
            commit_id=commit_id,
            suite_name=suite_name,
            timestamp=datetime.now(timezone.utc),
            author=author,
            message=message,
            parent_id=parent_id,
            branch=current_branch,
            scenarios=scenarios_data,
            cases=all_cases,
            metadata=version_data["metadata"],
        )

        # Store version object
        object_file = self.objects_dir / commit_id
        with open(object_file, "wb") as f:
            pickle.dump(version, f)

        # Also store JSON metadata for easy inspection
        metadata_file = self.objects_dir / f"{commit_id}.json"
        metadata_file.write_text(json.dumps(version.serialize(), indent=2))

        # Update branch head
        self._update_branch_head(current_branch, commit_id)

        return version

    def checkout(self, target: str) -> SuiteVersion:
        """Checkout a branch or specific commit.

        Args:
            target: Branch name or commit ID to checkout

        Returns:
            SuiteVersion at the checked out state
        """
        # Check if target is a branch
        if (self.heads_dir / target).exists():
            self.head_file.write_text(f"ref: refs/heads/{target}\n")
            commit_id = self._get_branch_head(target)
        else:
            # Assume it's a commit ID
            commit_id = target
            self.head_file.write_text(commit_id + "\n")

        if not commit_id:
            msg = f"Cannot checkout {target}: no commits found"
            raise ValueError(msg)

        return self.load_version(commit_id)

    def load_version(self, commit_id: str) -> SuiteVersion:
        """Load a specific version by commit ID.

        Args:
            commit_id: Commit identifier

        Returns:
            SuiteVersion object
        """
        object_file = self.objects_dir / commit_id

        if not object_file.exists():
            msg = f"Version {commit_id} not found"
            raise FileNotFoundError(msg)

        with open(object_file, "rb") as f:
            return pickle.load(f)

    def get_history(
        self,
        branch: str | None = None,
        limit: int | None = None,
    ) -> list[SuiteVersion]:
        """Get commit history for a branch.

        Args:
            branch: Branch name (uses current branch if None)
            limit: Maximum number of commits to return

        Returns:
            List of SuiteVersion objects in reverse chronological order
        """
        if branch is None:
            branch = self._get_current_branch()

        commit_id = self._get_branch_head(branch)
        history = []

        while commit_id and (limit is None or len(history) < limit):
            try:
                version = self.load_version(commit_id)
                history.append(version)
                commit_id = version.parent_id
            except FileNotFoundError:
                break

        return history

    def list_branches(self) -> list[str]:
        """List all branches in the repository.

        Returns:
            List of branch names
        """
        return [f.name for f in self.heads_dir.iterdir() if f.is_file()]

    def diff(self, commit1: str, commit2: str) -> dict[str, Any]:
        """Compare two versions and return differences.

        Args:
            commit1: First commit ID
            commit2: Second commit ID

        Returns:
            Dictionary describing differences between versions
        """
        v1 = self.load_version(commit1)
        v2 = self.load_version(commit2)

        # Find scenario differences
        scenarios_added = set(v2.scenarios.keys()) - set(v1.scenarios.keys())
        scenarios_removed = set(v1.scenarios.keys()) - set(v2.scenarios.keys())
        scenarios_modified = set()

        for name in set(v1.scenarios.keys()) & set(v2.scenarios.keys()):
            if v1.scenarios[name] != v2.scenarios[name]:
                scenarios_modified.add(name)

        # Find case differences
        cases_diff = {
            "added": len(v2.cases) - len(v1.cases) if len(v2.cases) > len(v1.cases) else 0,
            "removed": len(v1.cases) - len(v2.cases) if len(v1.cases) > len(v2.cases) else 0,
        }

        return {
            "commit1": commit1[:8],
            "commit2": commit2[:8],
            "scenarios": {
                "added": list(scenarios_added),
                "removed": list(scenarios_removed),
                "modified": list(scenarios_modified),
            },
            "cases": cases_diff,
        }

    def tag(self, name: str, commit_id: str | None = None, message: str = "") -> None:
        """Create a tag for a specific commit.

        Args:
            name: Tag name
            commit_id: Commit to tag (uses HEAD if None)
            message: Optional tag message
        """
        if commit_id is None:
            current_branch = self._get_current_branch()
            commit_id = self._get_branch_head(current_branch)

        if not commit_id:
            msg = "No commit to tag"
            raise ValueError(msg)

        tag_file = self.tags_dir / name
        tag_data = {
            "commit_id": commit_id,
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        tag_file.write_text(json.dumps(tag_data, indent=2))

    def list_tags(self) -> list[dict[str, str]]:
        """List all tags in the repository.

        Returns:
            List of tag information dictionaries
        """
        tags = []
        for tag_file in self.tags_dir.iterdir():
            if tag_file.is_file():
                tag_data = json.loads(tag_file.read_text())
                tag_data["name"] = tag_file.name
                tags.append(tag_data)
        return sorted(tags, key=lambda t: t["created_at"], reverse=True)
