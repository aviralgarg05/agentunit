"""Tests for dashboard module."""

import pytest


def test_dashboard_imports():
    """Test that dashboard components can be imported."""
    from agentunit.dashboard import (
        DashboardApp,
        SuiteEditor,
        RunMonitor,
        TraceViewer,
        ReportExplorer,
        start_dashboard,
        DashboardConfig,
    )
    
    assert DashboardApp is not None
    assert SuiteEditor is not None
    assert RunMonitor is not None
    assert TraceViewer is not None
    assert ReportExplorer is not None
    assert start_dashboard is not None
    assert DashboardConfig is not None


def test_dashboard_config():
    """Test dashboard configuration."""
    from pathlib import Path
    from agentunit.dashboard import DashboardConfig
    
    config = DashboardConfig(
        workspace_path=Path("/tmp/test"),
        host="0.0.0.0",
        port=9000,
        theme="dark",
        auto_open_browser=False
    )
    
    assert config.workspace_path == Path("/tmp/test")
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.theme == "dark"
    assert config.auto_open_browser is False


def test_dashboard_requires_streamlit():
    """Test that dashboard requires streamlit."""
    try:
        import streamlit
        HAS_STREAMLIT = True
    except ImportError:
        HAS_STREAMLIT = False
    
    if not HAS_STREAMLIT:
        from agentunit.dashboard import DashboardApp
        from pathlib import Path
        
        with pytest.raises(ImportError, match="streamlit is required"):
            DashboardApp(workspace_path=Path("/tmp"))


def test_suite_editor_component():
    """Test suite editor component."""
    try:
        import streamlit
        HAS_STREAMLIT = True
    except ImportError:
        HAS_STREAMLIT = False
    
    if not HAS_STREAMLIT:
        from agentunit.dashboard import SuiteEditor
        
        with pytest.raises(ImportError, match="streamlit is required"):
            SuiteEditor()


def test_run_monitor_component():
    """Test run monitor component."""
    try:
        import streamlit
        HAS_STREAMLIT = True
    except ImportError:
        HAS_STREAMLIT = False
    
    if not HAS_STREAMLIT:
        from agentunit.dashboard import RunMonitor
        
        with pytest.raises(ImportError, match="streamlit is required"):
            RunMonitor()


def test_trace_viewer_component():
    """Test trace viewer component."""
    try:
        import streamlit
        HAS_STREAMLIT = True
    except ImportError:
        HAS_STREAMLIT = False
    
    if not HAS_STREAMLIT:
        from agentunit.dashboard import TraceViewer
        
        with pytest.raises(ImportError, match="streamlit is required"):
            TraceViewer()


def test_report_explorer_component():
    """Test report explorer component."""
    try:
        import streamlit
        HAS_STREAMLIT = True
    except ImportError:
        HAS_STREAMLIT = False
    
    if not HAS_STREAMLIT:
        from agentunit.dashboard import ReportExplorer
        
        with pytest.raises(ImportError, match="streamlit is required"):
            ReportExplorer()
