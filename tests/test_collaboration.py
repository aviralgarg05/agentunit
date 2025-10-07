"""Tests for collaborative suite management and version control."""

import json
import pickle
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from agentunit.collaboration import (
    SuiteVersion,
    VersionManager,
    BranchManager,
    MergeStrategy,
    MergeResult,
    ConflictResolver,
    Conflict,
    ChangeTracker,
    ChangeType,
    ChangeLog,
    CollaborationHub,
    Lock,
)


# Test Fixtures

@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def version_manager(temp_repo):
    """Create a VersionManager with temporary repo."""
    return VersionManager(temp_repo)


@pytest.fixture
def branch_manager(version_manager):
    """Create a BranchManager."""
    return BranchManager(version_manager)


@pytest.fixture
def sample_scenarios():
    """Sample scenario data for testing."""
    return {
        "auth_test": {
            "name": "auth_test",
            "adapter": "MockAdapter",
            "config": {"timeout": 30},
        },
        "payment_test": {
            "name": "payment_test",
            "adapter": "MockAdapter",
            "config": {"timeout": 60},
        },
    }


@pytest.fixture
def sample_cases():
    """Sample test cases for testing."""
    return [
        {"input": "login with valid creds", "expected": "success"},
        {"input": "login with invalid creds", "expected": "error"},
    ]


@pytest.fixture
def mock_scenario():
    """Create mock scenario objects."""
    class MockAdapter:
        name = "MockAdapter"
    
    class MockDataset:
        def __init__(self, cases):
            self.cases = [MockCase(**c) for c in cases]
    
    class MockCase:
        def __init__(self, input, expected, context=None, metadata=None):
            self.input = input
            self.expected = expected
            self.context = context or {}
            self.metadata = metadata or {}
    
    class MockScenario:
        def __init__(self, name, cases):
            self.name = name
            self.adapter = MockAdapter()
            self.config = {}
            self.dataset = MockDataset(cases)
    
    return MockScenario


# Version Control Tests

def test_version_manager_initialization(version_manager, temp_repo):
    """Test VersionManager creates proper directory structure."""
    assert (temp_repo / "objects").exists()
    assert (temp_repo / "refs" / "heads").exists()
    assert (temp_repo / "refs" / "tags").exists()
    assert (temp_repo / "HEAD").exists()
    assert (temp_repo / "config").exists()


def test_create_suite_version(version_manager, mock_scenario, sample_cases):
    """Test creating a suite version."""
    scenarios = [
        mock_scenario("test1", sample_cases[:1]),
        mock_scenario("test2", sample_cases[1:]),
    ]
    
    version = version_manager.commit_suite(
        scenarios,
        "Initial commit",
        author="alice@example.com",
        suite_name="test_suite",
    )
    
    assert version.commit_id
    assert version.suite_name == "test_suite"
    assert version.author == "alice@example.com"
    assert version.message == "Initial commit"
    assert len(version.scenarios) == 2
    assert len(version.cases) == 2


def test_load_version(version_manager, mock_scenario, sample_cases):
    """Test loading a version by commit ID."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version = version_manager.commit_suite(scenarios, "Test commit")
    
    loaded = version_manager.load_version(version.commit_id)
    
    assert loaded.commit_id == version.commit_id
    assert loaded.message == version.message
    assert loaded.scenarios == version.scenarios


def test_version_history(version_manager, mock_scenario, sample_cases):
    """Test getting commit history."""
    scenarios = [mock_scenario("test1", sample_cases)]
    
    v1 = version_manager.commit_suite(scenarios, "First commit")
    v2 = version_manager.commit_suite(scenarios, "Second commit")
    v3 = version_manager.commit_suite(scenarios, "Third commit")
    
    history = version_manager.get_history()
    
    assert len(history) == 3
    assert history[0].commit_id == v3.commit_id
    assert history[1].commit_id == v2.commit_id
    assert history[2].commit_id == v1.commit_id


def test_version_history_with_limit(version_manager, mock_scenario, sample_cases):
    """Test getting limited commit history."""
    scenarios = [mock_scenario("test1", sample_cases)]
    
    version_manager.commit_suite(scenarios, "First")
    version_manager.commit_suite(scenarios, "Second")
    version_manager.commit_suite(scenarios, "Third")
    
    history = version_manager.get_history(limit=2)
    
    assert len(history) == 2


def test_diff_versions(version_manager, mock_scenario, sample_cases):
    """Test computing diff between versions."""
    scenarios1 = [mock_scenario("test1", sample_cases[:1])]
    v1 = version_manager.commit_suite(scenarios1, "First")
    
    scenarios2 = [
        mock_scenario("test1", sample_cases[:1]),
        mock_scenario("test2", sample_cases[1:]),
    ]
    v2 = version_manager.commit_suite(scenarios2, "Second")
    
    diff = version_manager.diff(v1.commit_id, v2.commit_id)
    
    assert "test2" in diff["scenarios"]["added"]
    assert len(diff["scenarios"]["modified"]) == 0


def test_create_tag(version_manager, mock_scenario, sample_cases):
    """Test creating a tag for a commit."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version = version_manager.commit_suite(scenarios, "Test")
    
    version_manager.tag("v1.0.0", version.commit_id, "Release 1.0.0")
    
    tags = version_manager.list_tags()
    assert len(tags) == 1
    assert tags[0]["name"] == "v1.0.0"
    assert tags[0]["commit_id"] == version.commit_id


# Branch Management Tests

def test_create_branch(branch_manager, version_manager, mock_scenario, sample_cases):
    """Test creating a new branch."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version_manager.commit_suite(scenarios, "Initial")
    
    commit_id = branch_manager.create("feature-branch", checkout=False)
    
    assert "feature-branch" in branch_manager.list()
    assert commit_id


def test_checkout_branch(branch_manager, version_manager, mock_scenario, sample_cases):
    """Test checking out a branch."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version_manager.commit_suite(scenarios, "Initial")
    
    branch_manager.create("feature-branch")
    
    assert branch_manager.current() == "feature-branch"


def test_delete_branch(branch_manager, version_manager, mock_scenario, sample_cases):
    """Test deleting a branch."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version_manager.commit_suite(scenarios, "Initial")
    
    branch_manager.create("temp-branch", checkout=False)
    assert "temp-branch" in branch_manager.list()
    
    branch_manager.delete("temp-branch", force=True)
    assert "temp-branch" not in branch_manager.list()


def test_cannot_delete_current_branch(branch_manager, version_manager, mock_scenario, sample_cases):
    """Test that current branch cannot be deleted."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version_manager.commit_suite(scenarios, "Initial")
    
    branch_manager.create("temp-branch")
    
    with pytest.raises(ValueError, match="Cannot delete current branch"):
        branch_manager.delete("temp-branch")


def test_branch_status(branch_manager, version_manager, mock_scenario, sample_cases):
    """Test getting branch status."""
    scenarios = [mock_scenario("test1", sample_cases)]
    version_manager.commit_suite(scenarios, "Test commit")
    
    status = branch_manager.status()
    
    assert status["branch"] == "main"
    assert status["message"] == "Test commit"
    assert "head" in status


# Merge and Conflict Resolution Tests

def test_merge_no_conflicts(mock_scenario, sample_cases):
    """Test merging without conflicts."""
    base = SuiteVersion(
        commit_id="base123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="base",
        scenarios={"test1": {"name": "test1"}},
        cases=[{"input": "a", "expected": "b"}],
    )
    
    ours = SuiteVersion(
        commit_id="ours123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="ours",
        parent_id="base123",
        scenarios={"test1": {"name": "test1"}, "test2": {"name": "test2"}},
        cases=[{"input": "a", "expected": "b"}],
    )
    
    theirs = SuiteVersion(
        commit_id="theirs123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="theirs",
        parent_id="base123",
        scenarios={"test1": {"name": "test1"}},
        cases=[{"input": "a", "expected": "b"}, {"input": "c", "expected": "d"}],
    )
    
    resolver = ConflictResolver(strategy=MergeStrategy.AUTO)
    result = resolver.merge(base, ours, theirs)
    
    assert result.success
    assert len(result.conflicts) == 0
    assert "test2" in result.scenarios


def test_merge_with_conflicts(mock_scenario):
    """Test merging with conflicts."""
    base = SuiteVersion(
        commit_id="base123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="base",
        scenarios={"test1": {"name": "test1", "config": {"timeout": 30}}},
        cases=[],
    )
    
    ours = SuiteVersion(
        commit_id="ours123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="ours",
        parent_id="base123",
        scenarios={"test1": {"name": "test1", "config": {"timeout": 60}}},
        cases=[],
    )
    
    theirs = SuiteVersion(
        commit_id="theirs123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="theirs",
        parent_id="base123",
        scenarios={"test1": {"name": "test1", "config": {"timeout": 90}}},
        cases=[],
    )
    
    resolver = ConflictResolver(strategy=MergeStrategy.AUTO)
    result = resolver.merge(base, ours, theirs)
    
    assert not result.success
    assert len(result.conflicts) > 0
    assert result.has_conflicts


def test_merge_strategy_ours():
    """Test merge with OURS strategy."""
    ours = SuiteVersion(
        commit_id="ours123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="ours",
        scenarios={"test1": {"value": "ours"}},
        cases=[],
    )
    
    theirs = SuiteVersion(
        commit_id="theirs123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="theirs",
        scenarios={"test1": {"value": "theirs"}},
        cases=[],
    )
    
    resolver = ConflictResolver(strategy=MergeStrategy.OURS)
    result = resolver.merge(None, ours, theirs)
    
    assert result.success
    assert result.scenarios["test1"]["value"] == "ours"


# Change Tracking Tests

def test_change_tracker_diff_scenarios():
    """Test tracking scenario changes."""
    old = SuiteVersion(
        commit_id="old123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="old",
        scenarios={"test1": {"name": "test1"}},
        cases=[],
    )
    
    new = SuiteVersion(
        commit_id="new123",
        suite_name="test",
        timestamp=datetime.now(timezone.utc),
        author="test",
        message="new",
        scenarios={"test1": {"name": "test1", "modified": True}, "test2": {"name": "test2"}},
        cases=[],
    )
    
    tracker = ChangeTracker()
    changelog = tracker.diff(old, new)
    
    assert len(changelog.changes) == 2
    added = changelog.filter_by_type(ChangeType.SCENARIO_ADDED)
    modified = changelog.filter_by_type(ChangeType.SCENARIO_MODIFIED)
    
    assert len(added) == 1
    assert len(modified) == 1


def test_change_statistics():
    """Test computing change statistics."""
    tracker = ChangeTracker()
    changelog = ChangeLog()
    
    from agentunit.collaboration.tracking import Change
    
    changelog.add_change(Change(ChangeType.SCENARIO_ADDED, "test1"))
    changelog.add_change(Change(ChangeType.SCENARIO_ADDED, "test2"))
    changelog.add_change(Change(ChangeType.CASE_MODIFIED, "case1"))
    
    stats = tracker.get_change_statistics(changelog)
    
    assert stats["total_changes"] == 3
    assert stats["scenarios"]["added"] == 2
    assert stats["cases"]["modified"] == 1


# Collaboration Hub Tests

def test_acquire_lock(temp_repo):
    """Test acquiring a lock on a resource."""
    hub = CollaborationHub(temp_repo)
    
    lock = hub.acquire_lock("scenario/test1", user="alice")
    
    assert lock is not None
    assert lock.owner == "alice"
    assert lock.resource == "scenario/test1"


def test_lock_prevents_concurrent_access(temp_repo):
    """Test that lock prevents other users from acquiring."""
    hub = CollaborationHub(temp_repo)
    
    lock1 = hub.acquire_lock("scenario/test1", user="alice")
    assert lock1 is not None
    
    lock2 = hub.acquire_lock("scenario/test1", user="bob")
    assert lock2 is None


def test_release_lock(temp_repo):
    """Test releasing a lock."""
    hub = CollaborationHub(temp_repo)
    
    hub.acquire_lock("scenario/test1", user="alice")
    result = hub.release_lock("scenario/test1", user="alice")
    
    assert result is True
    
    # Should be able to acquire again
    lock = hub.acquire_lock("scenario/test1", user="bob")
    assert lock is not None


def test_list_locks(temp_repo):
    """Test listing all locks."""
    hub = CollaborationHub(temp_repo)
    
    hub.acquire_lock("scenario/test1", user="alice")
    hub.acquire_lock("scenario/test2", user="bob")
    
    all_locks = hub.list_locks()
    assert len(all_locks) == 2
    
    alice_locks = hub.list_locks(user="alice")
    assert len(alice_locks) == 1


def test_heartbeat_and_active_users(temp_repo):
    """Test user activity tracking."""
    hub = CollaborationHub(temp_repo)
    
    hub.heartbeat("alice", branch="feature-1")
    hub.heartbeat("bob", branch="feature-2")
    
    active_users = hub.get_active_users()
    
    assert len(active_users) == 2
    usernames = {u.username for u in active_users}
    assert "alice" in usernames
    assert "bob" in usernames
