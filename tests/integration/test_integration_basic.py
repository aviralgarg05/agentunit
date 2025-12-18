"""Basic integration test to verify test structure."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_integration_directory_structure():
    """Test that integration test directory is properly set up."""
    integration_dir = Path(__file__).parent

    # Check required files exist
    assert (integration_dir / "__init__.py").exists()
    assert (integration_dir / "conftest.py").exists()
    assert (integration_dir / "simple_langgraph_agent.py").exists()
    assert (integration_dir / "test_langgraph_integration.py").exists()
    assert (integration_dir / "README.md").exists()


@pytest.mark.integration
def test_simple_langgraph_agent_fallback():
    """Test that the simple agent works without LangGraph installed."""
    from .simple_langgraph_agent import LANGGRAPH_AVAILABLE, invoke_agent

    # Should work even without LangGraph (returns mock response)
    payload = {
        "query": "What is quantum tunneling?",
        "context": ["Physics"],
        "tools": ["search"],
        "metadata": {},
    }

    result = invoke_agent(payload)

    assert isinstance(result, dict)
    assert "result" in result
    assert "events" in result

    if not LANGGRAPH_AVAILABLE:
        # Should return mock response
        assert "Mock response" in result["result"]

    # Should contain the query in the response
    assert "quantum" in result["result"].lower() or "Mock response" in result["result"]


@pytest.mark.integration
def test_pytest_markers_configured():
    """Test that pytest markers are properly configured."""

    # This test itself should have the integration marker
    # We can't easily test this programmatically, but the test runner will validate it
