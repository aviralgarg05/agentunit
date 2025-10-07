"""Merge strategies and conflict resolution for collaborative suite development."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .version import VersionManager, SuiteVersion


class MergeStrategy(Enum):
    """Strategies for merging test suite branches."""
    
    AUTO = "auto"  # Automatic merge with conflict detection
    OURS = "ours"  # Always use current branch's changes
    THEIRS = "theirs"  # Always use merging branch's changes
    UNION = "union"  # Combine both (for additive changes)
    MANUAL = "manual"  # Require manual conflict resolution


@dataclass
class Conflict:
    """Represents a merge conflict between two versions.
    
    Attributes:
        path: Identifier for conflicting element (e.g., scenario name)
        type: Type of conflict (scenario, case, config)
        ours: Value from current branch
        theirs: Value from merging branch
        base: Value from common ancestor (if available)
        resolved: Whether conflict has been resolved
        resolution: Chosen resolution value
    """
    
    path: str
    type: str
    ours: Any
    theirs: Any
    base: Optional[Any] = None
    resolved: bool = False
    resolution: Optional[Any] = None
    
    def resolve_with(self, value: Any) -> None:
        """Resolve conflict with a specific value."""
        self.resolution = value
        self.resolved = True


@dataclass
class MergeResult:
    """Result of a merge operation.
    
    Attributes:
        success: Whether merge completed successfully
        commit_id: ID of merge commit (if successful)
        conflicts: List of conflicts encountered
        message: Merge message or error description
        scenarios: Merged scenario data
        cases: Merged case data
        metadata: Additional merge metadata
    """
    
    success: bool
    commit_id: Optional[str] = None
    conflicts: List[Conflict] = field(default_factory=list)
    message: str = ""
    scenarios: Dict[str, Any] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if merge has unresolved conflicts."""
        return any(not c.resolved for c in self.conflicts)
    
    def get_unresolved(self) -> List[Conflict]:
        """Get list of unresolved conflicts."""
        return [c for c in self.conflicts if not c.resolved]


class ConflictResolver:
    """Resolve merge conflicts using various strategies.
    
    Examples:
        >>> resolver = ConflictResolver(strategy=MergeStrategy.AUTO)
        >>> 
        >>> # Resolve conflicts automatically
        >>> result = resolver.merge(base_version, ours_version, theirs_version)
        >>> 
        >>> if result.has_conflicts:
        ...     # Manual resolution
        ...     for conflict in result.get_unresolved():
        ...         print(f"Conflict in {conflict.path}")
        ...         # User chooses resolution
        ...         conflict.resolve_with(conflict.ours)
    """
    
    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.AUTO,
        custom_resolver: Optional[Callable[[Conflict], Any]] = None,
    ):
        """Initialize conflict resolver.
        
        Args:
            strategy: Default merge strategy to use
            custom_resolver: Optional custom resolution function
        """
        self.strategy = strategy
        self.custom_resolver = custom_resolver
    
    def merge(
        self,
        base: Optional[SuiteVersion],
        ours: SuiteVersion,
        theirs: SuiteVersion,
    ) -> MergeResult:
        """Merge two suite versions using configured strategy.
        
        Args:
            base: Common ancestor version (None for unrelated branches)
            ours: Current branch version
            theirs: Version being merged in
        
        Returns:
            MergeResult with merged data and any conflicts
        """
        result = MergeResult(
            success=False,
            message=f"Merging {theirs.branch} into {ours.branch}",
        )
        
        # Merge scenarios
        scenarios_conflicts = self._merge_scenarios(base, ours, theirs)
        result.conflicts.extend(scenarios_conflicts)
        
        # Build merged scenarios dict
        all_scenario_names = set(ours.scenarios.keys()) | set(theirs.scenarios.keys())
        
        for name in all_scenario_names:
            # Check if there's a conflict for this scenario
            conflict = next((c for c in scenarios_conflicts if c.path == name), None)
            
            if conflict:
                if conflict.resolved:
                    result.scenarios[name] = conflict.resolution
                # Unresolved conflicts are left out
            elif name in ours.scenarios and name in theirs.scenarios:
                # No conflict - use ours (they're the same)
                result.scenarios[name] = ours.scenarios[name]
            elif name in ours.scenarios:
                result.scenarios[name] = ours.scenarios[name]
            else:
                result.scenarios[name] = theirs.scenarios[name]
        
        # Merge cases (union by default for additive data)
        cases_conflicts = self._merge_cases(base, ours, theirs)
        result.conflicts.extend(cases_conflicts)
        
        # Combine cases (with deduplication)
        seen_cases = set()
        for case in ours.cases + theirs.cases:
            case_key = (case.get("input"), case.get("expected"))
            if case_key not in seen_cases:
                result.cases.append(case)
                seen_cases.add(case_key)
        
        # Update metadata
        result.metadata = {
            "base_commit": base.commit_id if base else None,
            "ours_commit": ours.commit_id,
            "theirs_commit": theirs.commit_id,
            "strategy": self.strategy.value,
            "total_conflicts": len(result.conflicts),
            "unresolved_conflicts": len(result.get_unresolved()),
        }
        
        # Mark success if no unresolved conflicts
        result.success = not result.has_conflicts
        
        return result
    
    def _merge_scenarios(
        self,
        base: Optional[SuiteVersion],
        ours: SuiteVersion,
        theirs: SuiteVersion,
    ) -> List[Conflict]:
        """Merge scenario configurations."""
        conflicts = []
        
        # Find scenarios that exist in both
        common_names = set(ours.scenarios.keys()) & set(theirs.scenarios.keys())
        
        for name in common_names:
            ours_scenario = ours.scenarios[name]
            theirs_scenario = theirs.scenarios[name]
            
            # Check if they differ
            if ours_scenario != theirs_scenario:
                base_scenario = base.scenarios.get(name) if base else None
                
                conflict = Conflict(
                    path=name,
                    type="scenario",
                    ours=ours_scenario,
                    theirs=theirs_scenario,
                    base=base_scenario,
                )
                
                # Try to auto-resolve based on strategy
                if self.strategy == MergeStrategy.OURS:
                    conflict.resolve_with(ours_scenario)
                elif self.strategy == MergeStrategy.THEIRS:
                    conflict.resolve_with(theirs_scenario)
                elif self.strategy == MergeStrategy.AUTO:
                    # Auto-resolve if one side matches base (other side changed it)
                    if base_scenario is not None:
                        if ours_scenario == base_scenario:
                            conflict.resolve_with(theirs_scenario)
                        elif theirs_scenario == base_scenario:
                            conflict.resolve_with(ours_scenario)
                        # else: both changed, leave unresolved
                elif self.strategy == MergeStrategy.UNION:
                    # Try to merge configs (dict merge)
                    if isinstance(ours_scenario, dict) and isinstance(theirs_scenario, dict):
                        merged = {**ours_scenario, **theirs_scenario}
                        conflict.resolve_with(merged)
                
                # Apply custom resolver if provided and not resolved
                if not conflict.resolved and self.custom_resolver:
                    try:
                        resolution = self.custom_resolver(conflict)
                        conflict.resolve_with(resolution)
                    except Exception:
                        pass  # Custom resolver failed, leave unresolved
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _merge_cases(
        self,
        base: Optional[SuiteVersion],  # noqa: ARG002 - reserved for future base comparison
        ours: SuiteVersion,
        theirs: SuiteVersion,
    ) -> List[Conflict]:
        """Merge dataset cases."""
        # For now, use union strategy (additive)
        # Could detect conflicts if cases with same input have different expected outputs
        conflicts = []
        
        # Build index of cases by input
        ours_index = {case.get("input"): case for case in ours.cases}
        theirs_index = {case.get("input"): case for case in theirs.cases}
        
        # Find inputs that appear in both
        common_inputs = set(ours_index.keys()) & set(theirs_index.keys())
        
        for input_val in common_inputs:
            ours_case = ours_index[input_val]
            theirs_case = theirs_index[input_val]
            
            # Check if expected outputs differ
            if ours_case.get("expected") != theirs_case.get("expected"):
                conflict = Conflict(
                    path=f"case[{input_val}]",
                    type="case",
                    ours=ours_case,
                    theirs=theirs_case,
                )
                
                # Apply strategy
                if self.strategy == MergeStrategy.OURS:
                    conflict.resolve_with(ours_case)
                elif self.strategy == MergeStrategy.THEIRS:
                    conflict.resolve_with(theirs_case)
                # AUTO and UNION leave unresolved for cases
                
                conflicts.append(conflict)
        
        return conflicts
    
    def find_common_ancestor(
        self,
        vm: VersionManager,
        commit1: str,
        commit2: str,
    ) -> Optional[SuiteVersion]:
        """Find common ancestor of two commits for three-way merge.
        
        Args:
            vm: VersionManager to use
            commit1: First commit ID
            commit2: Second commit ID
        
        Returns:
            Common ancestor SuiteVersion, or None if no common ancestor
        """
        # Get history for both commits
        history1 = set()
        current = commit1
        while current:
            history1.add(current)
            try:
                version = vm.load_version(current)
                current = version.parent_id
            except FileNotFoundError:
                break
        
        # Walk history2 until we find a commit in history1
        current = commit2
        while current:
            if current in history1:
                return vm.load_version(current)
            try:
                version = vm.load_version(current)
                current = version.parent_id
            except FileNotFoundError:
                break
        
        return None


def merge_branches(
    vm: VersionManager,
    source_branch: str,
    target_branch: Optional[str] = None,
    strategy: MergeStrategy = MergeStrategy.AUTO,
    message: Optional[str] = None,
) -> MergeResult:
    """High-level function to merge two branches.
    
    Args:
        vm: VersionManager instance
        source_branch: Branch to merge from
        target_branch: Branch to merge into (uses current if None)
        strategy: Merge strategy to use
        message: Merge commit message
    
    Returns:
        MergeResult with outcome of merge
    """
    # Get target branch
    if target_branch is None:
        target_branch = vm._get_current_branch()
    
    # Get branch heads
    source_commit = vm._get_branch_head(source_branch)
    target_commit = vm._get_branch_head(target_branch)
    
    if not source_commit or not target_commit:
        return MergeResult(
            success=False,
            message="Cannot merge: one or both branches have no commits",
        )
    
    # Load versions
    source_version = vm.load_version(source_commit)
    target_version = vm.load_version(target_commit)
    
    # Find common ancestor
    resolver = ConflictResolver(strategy=strategy)
    base = resolver.find_common_ancestor(vm, source_commit, target_commit)
    
    # Perform merge
    result = resolver.merge(base, target_version, source_version)
    
    # Create merge commit if successful
    if result.success:
        if message is None:
            message = f"Merge branch '{source_branch}' into {target_branch}"
        
        # Create a pseudo-scenario list from merged data (for commit_suite)
        # In real use, you'd reconstruct actual Scenario objects
        # For now, we'll directly create the version
        
        from datetime import datetime, timezone
        import hashlib
        import pickle
        import json
        
        config = json.loads(vm.config_file.read_text())
        author = f"{config['user']['name']} <{config['user']['email']}>"
        
        version_data = {
            "suite_name": target_version.suite_name,
            "author": author,
            "message": message,
            "parent_id": target_commit,
            "branch": target_branch,
            "scenarios": result.scenarios,
            "cases": result.cases,
            "metadata": {
                **result.metadata,
                "merge_parent": source_commit,
            },
        }
        
        commit_id = hashlib.sha256(pickle.dumps(version_data)).hexdigest()
        
        merge_version = SuiteVersion(
            commit_id=commit_id,
            suite_name=target_version.suite_name,
            timestamp=datetime.now(timezone.utc),
            author=author,
            message=message,
            parent_id=target_commit,
            branch=target_branch,
            scenarios=result.scenarios,
            cases=result.cases,
            metadata=version_data["metadata"],
        )
        
        # Store merge commit
        object_file = vm.objects_dir / commit_id
        with open(object_file, 'wb') as f:
            pickle.dump(merge_version, f)
        
        metadata_file = vm.objects_dir / f"{commit_id}.json"
        metadata_file.write_text(json.dumps(merge_version.serialize(), indent=2))
        
        # Update branch head
        vm._update_branch_head(target_branch, commit_id)
        
        result.commit_id = commit_id
        result.message = f"Merge successful: {message}"
    
    return result
