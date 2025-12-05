"""Tests for dashboard module."""

import importlib.util
from pathlib import Path

import pytest

from agentunit.dashboard import (
    DashboardApp,
    DashboardConfig,
    ReportExplorer,
    RunMonitor,
    SuiteEditor,
    TraceViewer,
    start_dashboard,
)


STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


def test_dashboard_imports():
    """Test that dashboard components can be imported."""
    assert DashboardApp is not None
    assert SuiteEditor is not None
    assert RunMonitor is not None
    assert TraceViewer is not None
    assert ReportExplorer is not None
    assert start_dashboard is not None
    assert DashboardConfig is not None


def test_dashboard_config():
    """Test dashboard configuration."""
    config = DashboardConfig(
        workspace_path=Path("/tmp/test"),
        host="0.0.0.0",
        port=9000,
        theme="dark",
        auto_open_browser=False,
    )

    assert config.workspace_path == Path("/tmp/test")
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.theme == "dark"
    assert config.auto_open_browser is False


def test_dashboard_requires_streamlit():
    """Test that dashboard requires streamlit."""
    if STREAMLIT_AVAILABLE:
        pytest.skip("streamlit installed; requirement test not applicable")

    with pytest.raises(ImportError, match="streamlit is required"):
        DashboardApp(workspace_path=Path("/tmp"))


def test_suite_editor_component():
    """Test suite editor component."""
    if STREAMLIT_AVAILABLE:
        pytest.skip("streamlit installed; requirement test not applicable")

    with pytest.raises(ImportError, match="streamlit is required"):
        SuiteEditor()


def test_run_monitor_component():
    """Test run monitor component."""
    if STREAMLIT_AVAILABLE:
        pytest.skip("streamlit installed; requirement test not applicable")

    with pytest.raises(ImportError, match="streamlit is required"):
        RunMonitor()


def test_trace_viewer_component():
    """Test trace viewer component."""
    if STREAMLIT_AVAILABLE:
        pytest.skip("streamlit installed; requirement test not applicable")

    with pytest.raises(ImportError, match="streamlit is required"):
        TraceViewer()


def test_report_explorer_component():
    """Test report explorer component."""
    if STREAMLIT_AVAILABLE:
        pytest.skip("streamlit installed; requirement test not applicable")

    with pytest.raises(ImportError, match="streamlit is required"):
        ReportExplorer()
