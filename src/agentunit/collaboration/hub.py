"""Collaboration hub for multi-user coordination and resource locking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock as ThreadLock


@dataclass
class Lock:
    """Represents a lock on a test suite resource.

    Prevents concurrent modifications by multiple users.

    Attributes:
        resource: Path or identifier of locked resource
        owner: User who holds the lock
        acquired_at: When lock was acquired
        expires_at: When lock will expire
        message: Optional message about why resource is locked
    """

    resource: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    message: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, str]:
        """Convert lock to dictionary."""
        return {
            "resource": self.resource,
            "owner": self.owner,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Lock:
        """Create lock from dictionary."""
        return cls(
            resource=data["resource"],
            owner=data["owner"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            message=data.get("message", ""),
        )


@dataclass
class UserActivity:
    """Track user activity in collaborative environment.

    Attributes:
        username: User identifier
        last_seen: Last activity timestamp
        current_branch: Branch user is working on
        locked_resources: Resources locked by user
        pending_changes: Number of uncommitted changes
    """

    username: str
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_branch: str = "main"
    locked_resources: set[str] = field(default_factory=set)
    pending_changes: int = 0

    def is_active(self, timeout_minutes: int = 30) -> bool:
        """Check if user is still active (based on timeout)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        return self.last_seen > cutoff

    def to_dict(self) -> dict[str, any]:
        """Convert activity to dictionary."""
        return {
            "username": self.username,
            "last_seen": self.last_seen.isoformat(),
            "current_branch": self.current_branch,
            "locked_resources": list(self.locked_resources),
            "pending_changes": self.pending_changes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> UserActivity:
        """Create activity from dictionary."""
        return cls(
            username=data["username"],
            last_seen=datetime.fromisoformat(data["last_seen"]),
            current_branch=data.get("current_branch", "main"),
            locked_resources=set(data.get("locked_resources", [])),
            pending_changes=data.get("pending_changes", 0),
        )


class CollaborationHub:
    """Multi-user coordination and locking for collaborative suite development.

    Provides pessimistic locking to prevent concurrent modifications and
    tracks active users working on the test suite.

    Examples:
        >>> hub = CollaborationHub(".agentunit")
        >>>
        >>> # Acquire lock
        >>> lock = hub.acquire_lock("scenario/auth_tests", user="alice")
        >>> if lock:
        ...     # Make changes
        ...     pass
        ...     # Release when done
        ...     hub.release_lock("scenario/auth_tests", user="alice")
        >>>
        >>> # Check who's working
        >>> active_users = hub.get_active_users()
        >>> for user in active_users:
        ...     print(f"{user.username} on {user.current_branch}")
    """

    def __init__(self, repo_path: str | Path = ".agentunit"):
        """Initialize collaboration hub.

        Args:
            repo_path: Path to version control repository
        """
        self.repo_path = Path(repo_path)
        self.locks_dir = self.repo_path / "locks"
        self.activity_dir = self.repo_path / "activity"

        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.activity_dir.mkdir(parents=True, exist_ok=True)

        # In-memory lock for thread safety
        self._lock = ThreadLock()

    def acquire_lock(
        self,
        resource: str,
        user: str,
        timeout_minutes: int = 30,
        message: str = "",
    ) -> Lock | None:
        """Acquire lock on a resource.

        Args:
            resource: Resource identifier to lock
            user: User requesting lock
            timeout_minutes: Lock expiration timeout
            message: Optional message about lock purpose

        Returns:
            Lock object if successful, None if resource already locked
        """
        with self._lock:
            # Check if resource is already locked
            existing_lock = self.get_lock(resource)
            if existing_lock and not existing_lock.is_expired:
                if existing_lock.owner != user:
                    return None  # Already locked by someone else

            # Create lock
            now = datetime.now(timezone.utc)
            expires = now + timedelta(minutes=timeout_minutes)

            lock = Lock(
                resource=resource,
                owner=user,
                acquired_at=now,
                expires_at=expires,
                message=message,
            )

            # Store lock
            lock_file = self.locks_dir / f"{self._sanitize_name(resource)}.json"
            lock_file.write_text(json.dumps(lock.to_dict(), indent=2))

            # Update user activity
            self._update_activity(user, locked_resource=resource)

            return lock

    def release_lock(self, resource: str, user: str) -> bool:
        """Release a lock on a resource.

        Args:
            resource: Resource identifier
            user: User releasing lock

        Returns:
            True if lock was released, False if user doesn't own lock
        """
        with self._lock:
            lock = self.get_lock(resource)

            if not lock:
                return False  # No lock exists

            if lock.owner != user:
                return False  # User doesn't own this lock

            # Remove lock file
            lock_file = self.locks_dir / f"{self._sanitize_name(resource)}.json"
            if lock_file.exists():
                lock_file.unlink()

            # Update user activity
            self._update_activity(user, unlocked_resource=resource)

            return True

    def get_lock(self, resource: str) -> Lock | None:
        """Get current lock on a resource.

        Args:
            resource: Resource identifier

        Returns:
            Lock object if locked, None otherwise
        """
        lock_file = self.locks_dir / f"{self._sanitize_name(resource)}.json"

        if not lock_file.exists():
            return None

        try:
            data = json.loads(lock_file.read_text())
            lock = Lock.from_dict(data)

            # Clean up expired locks
            if lock.is_expired:
                lock_file.unlink()
                return None

            return lock
        except Exception:
            return None

    def list_locks(self, user: str | None = None) -> list[Lock]:
        """List all active locks.

        Args:
            user: Optional filter by lock owner

        Returns:
            List of active locks
        """
        locks = []

        for lock_file in self.locks_dir.glob("*.json"):
            try:
                data = json.loads(lock_file.read_text())
                lock = Lock.from_dict(data)

                # Skip expired locks
                if lock.is_expired:
                    lock_file.unlink()
                    continue

                # Filter by user if specified
                if user and lock.owner != user:
                    continue

                locks.append(lock)
            except Exception:
                continue

        return locks

    def force_release_all(self, user: str) -> int:
        """Force release all locks held by a user.

        Useful for cleaning up after crashed sessions.

        Args:
            user: User whose locks to release

        Returns:
            Number of locks released
        """
        user_locks = self.list_locks(user=user)

        for lock in user_locks:
            self.release_lock(lock.resource, user)

        return len(user_locks)

    def heartbeat(self, user: str, branch: str | None = None) -> None:
        """Update user's last activity timestamp.

        Args:
            user: Username
            branch: Current branch user is on
        """
        self._update_activity(user, branch=branch)

    def get_active_users(self, timeout_minutes: int = 30) -> list[UserActivity]:
        """Get list of currently active users.

        Args:
            timeout_minutes: Inactivity timeout for considering user offline

        Returns:
            List of active user activities
        """
        active_users = []

        for activity_file in self.activity_dir.glob("*.json"):
            try:
                data = json.loads(activity_file.read_text())
                activity = UserActivity.from_dict(data)

                if activity.is_active(timeout_minutes):
                    active_users.append(activity)
            except Exception:
                continue

        return active_users

    def _update_activity(
        self,
        user: str,
        branch: str | None = None,
        locked_resource: str | None = None,
        unlocked_resource: str | None = None,
    ) -> None:
        """Update user activity record.

        Args:
            user: Username
            branch: Branch to set (if provided)
            locked_resource: Resource just locked (if provided)
            unlocked_resource: Resource just unlocked (if provided)
        """
        activity_file = self.activity_dir / f"{user}.json"

        # Load existing activity or create new
        if activity_file.exists():
            data = json.loads(activity_file.read_text())
            activity = UserActivity.from_dict(data)
        else:
            activity = UserActivity(username=user)

        # Update fields
        activity.last_seen = datetime.now(timezone.utc)

        if branch:
            activity.current_branch = branch

        if locked_resource:
            activity.locked_resources.add(locked_resource)

        if unlocked_resource:
            activity.locked_resources.discard(unlocked_resource)

        # Save
        activity_file.write_text(json.dumps(activity.to_dict(), indent=2))

    def _sanitize_name(self, name: str) -> str:
        """Sanitize resource name for use as filename."""
        # Replace path separators and special chars
        return name.replace("/", "_").replace("\\", "_").replace(":", "_")
