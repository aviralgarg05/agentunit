"""Configuration for integration tests."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Configure pytest markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "langgraph: marks tests as requiring LangGraph (skipped if not installed)"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark integration tests."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)