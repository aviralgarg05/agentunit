"""Result caching for AgentUnit pytest plugin."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentunit import Scenario


logger = logging.getLogger(__name__)

CACHE_DIR = ".agentunit_cache"
CACHE_VERSION = "1"


@dataclass
class CachedResult:
    """Cached scenario result."""

    success: bool
    failures: list[str]
    cache_key: str
    source_hash: str | None = None


class ScenarioCache:
    """Cache manager for scenario results."""

    def __init__(self, root_path: Path, enabled: bool = True) -> None:
        self.root_path = root_path
        self.cache_dir = root_path / CACHE_DIR
        self.enabled = enabled

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Add .gitignore to cache directory
            gitignore = self.cache_dir / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*\n")

    def _compute_scenario_hash(self, scenario: Scenario) -> str:
        """Compute hash from scenario inputs (dataset, adapter config)."""
        hash_data: dict[str, Any] = {
            "version": CACHE_VERSION,
            "name": scenario.name,
            "retries": scenario.retries,
            "max_turns": scenario.max_turns,
            "timeout": scenario.timeout,
            "tags": scenario.tags,
            "seed": scenario.seed,
            "metadata": scenario.metadata,
        }

        # Hash adapter configuration
        adapter = scenario.adapter
        hash_data["adapter"] = {
            "name": getattr(adapter, "name", adapter.__class__.__name__),
            "class": adapter.__class__.__name__,
        }

        # Hash dataset cases
        try:
            cases = list(scenario.dataset.iter_cases())
            hash_data["dataset"] = {
                "name": scenario.dataset.name,
                "cases": [
                    {
                        "id": case.id,
                        "query": case.query,
                        "expected_output": case.expected_output,
                    }
                    for case in cases
                ],
            }
        except Exception:
            # If we can't iterate cases, use dataset name only
            hash_data["dataset"] = {"name": scenario.dataset.name}

        # Create deterministic JSON string and hash it
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _compute_source_hash(self, source_path: Path | None) -> str | None:
        """Compute hash of source file for cache invalidation."""
        if source_path is None or not source_path.exists():
            return None

        try:
            content = source_path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return None

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file for given key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self, scenario: Scenario, source_path: Path | None = None
    ) -> CachedResult | None:
        """Get cached result for scenario if available and valid."""
        if not self.enabled:
            logger.debug("Cache disabled, skipping lookup")
            return None

        cache_key = self._compute_scenario_hash(scenario)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss for scenario '{scenario.name}' (key: {cache_key})")
            return None

        try:
            data = json.loads(cache_path.read_text())

            # Check source file hash for invalidation
            current_source_hash = self._compute_source_hash(source_path)
            cached_source_hash = data.get("source_hash")

            if current_source_hash and cached_source_hash:
                if current_source_hash != cached_source_hash:
                    logger.info(
                        f"Cache invalidated for scenario '{scenario.name}' "
                        "(source file changed)"
                    )
                    return None

            logger.info(f"Cache hit for scenario '{scenario.name}' (key: {cache_key})")
            return CachedResult(
                success=data["success"],
                failures=data.get("failures", []),
                cache_key=cache_key,
                source_hash=cached_source_hash,
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to read cache for '{scenario.name}': {e}")
            return None

    def set(
        self,
        scenario: Scenario,
        success: bool,
        failures: list[str],
        source_path: Path | None = None,
    ) -> str:
        """Store scenario result in cache."""
        if not self.enabled:
            return ""

        self._ensure_cache_dir()

        cache_key = self._compute_scenario_hash(scenario)
        source_hash = self._compute_source_hash(source_path)

        cache_data = {
            "success": success,
            "failures": failures,
            "cache_key": cache_key,
            "source_hash": source_hash,
            "scenario_name": scenario.name,
        }

        cache_path = self._get_cache_path(cache_key)
        try:
            cache_path.write_text(json.dumps(cache_data, indent=2))
            logger.debug(f"Cached result for scenario '{scenario.name}' (key: {cache_key})")
        except OSError as e:
            logger.warning(f"Failed to write cache for '{scenario.name}': {e}")

        return cache_key

    def clear(self) -> int:
        """Clear all cached results. Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        logger.info(f"Cleared {count} cached results")
        return count
