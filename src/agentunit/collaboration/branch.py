"""Branch management for collaborative test suite development."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import builtins

    from .version import SuiteVersion, VersionManager


class BranchManager:
    """Handle branch creation, switching, and deletion for test suites.

    Provides git-like branch operations to enable parallel development
    of test suites without conflicts.

    Examples:
        >>> manager = VersionManager(".agentunit")
        >>> branches = BranchManager(manager)
        >>>
        >>> # Create feature branch
        >>> branches.create("feature-new-metrics")
        >>>
        >>> # Switch to branch
        >>> branches.checkout("feature-new-metrics")
        >>>
        >>> # List all branches
        >>> for branch in branches.list():
        ...     current = " *" if branch == branches.current() else "  "
        ...     print(f"{current} {branch}")
    """

    def __init__(self, version_manager: VersionManager):
        """Initialize branch manager.

        Args:
            version_manager: VersionManager instance to use
        """
        self.vm = version_manager

    def create(
        self,
        name: str,
        from_commit: str | None = None,
        checkout: bool = True,
    ) -> str:
        """Create a new branch.

        Args:
            name: Name for the new branch
            from_commit: Commit to branch from (uses HEAD if None)
            checkout: Whether to checkout the new branch immediately

        Returns:
            Commit ID the branch was created from
        """
        if name in self.list():
            msg = f"Branch '{name}' already exists"
            raise ValueError(msg)

        # Get starting commit
        if from_commit is None:
            current_branch = self.current()
            from_commit = self.vm._get_branch_head(current_branch)

        if not from_commit:
            msg = "Cannot create branch: no commits found"
            raise ValueError(msg)

        # Verify commit exists
        _ = self.vm.load_version(from_commit)

        # Create branch reference
        self.vm._update_branch_head(name, from_commit)

        # Checkout if requested
        if checkout:
            self.checkout(name)

        return from_commit

    def checkout(self, name: str) -> SuiteVersion:
        """Switch to a different branch.

        Args:
            name: Branch name to checkout

        Returns:
            SuiteVersion at the branch head
        """
        if name not in self.list():
            msg = f"Branch '{name}' does not exist"
            raise ValueError(msg)

        return self.vm.checkout(name)

    def delete(self, name: str, force: bool = False) -> None:
        """Delete a branch.

        Args:
            name: Branch name to delete
            force: Force deletion even if not merged
        """
        if name == "main":
            msg = "Cannot delete main branch"
            raise ValueError(msg)

        if name == self.current():
            msg = f"Cannot delete current branch '{name}'. Switch branches first."
            raise ValueError(msg)

        if name not in self.list():
            msg = f"Branch '{name}' does not exist"
            raise ValueError(msg)

        branch_file = self.vm.heads_dir / name

        # Check if branch is merged (unless forcing)
        if not force:
            branch_head = self.vm._get_branch_head(name)
            main_history = [v.commit_id for v in self.vm.get_history("main")]

            if branch_head not in main_history:
                msg = f"Branch '{name}' is not merged. Use force=True to delete anyway."
                raise ValueError(msg)

        branch_file.unlink()

    def list(self) -> builtins.list[str]:
        """List all branches.

        Returns:
            List of branch names
        """
        return self.vm.list_branches()

    def current(self) -> str:
        """Get name of currently checked out branch.

        Returns:
            Current branch name
        """
        return self.vm._get_current_branch()

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a branch.

        Args:
            old_name: Current branch name
            new_name: New branch name
        """
        if old_name not in self.list():
            msg = f"Branch '{old_name}' does not exist"
            raise ValueError(msg)

        if new_name in self.list():
            msg = f"Branch '{new_name}' already exists"
            raise ValueError(msg)

        old_file = self.vm.heads_dir / old_name
        new_file = self.vm.heads_dir / new_name

        old_file.rename(new_file)

        # Update HEAD if it pointed to the renamed branch
        head_content = self.vm.head_file.read_text().strip()
        if head_content == f"ref: refs/heads/{old_name}":
            self.vm.head_file.write_text(f"ref: refs/heads/{new_name}\n")

    def status(self) -> dict:
        """Get status of current branch.

        Returns:
            Dictionary with branch status information
        """
        current = self.current()
        head_commit = self.vm._get_branch_head(current)

        if head_commit:
            version = self.vm.load_version(head_commit)
            return {
                "branch": current,
                "head": head_commit[:8],
                "message": version.message,
                "author": version.author,
                "timestamp": version.timestamp.isoformat(),
            }
        return {
            "branch": current,
            "head": None,
            "message": "No commits yet",
        }
