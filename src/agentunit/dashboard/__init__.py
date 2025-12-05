"""Interactive web dashboard for AgentUnit.

This module provides a Streamlit-based web interface for:
- Suite authoring and configuration
- Real-time run monitoring
- Trace visualizations
- Interactive report exploration
"""

import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .app import DashboardApp
    from .components import ReportExplorer, RunMonitor, SuiteEditor, TraceViewer
    from .server import DashboardConfig, start_dashboard

__all__ = [
    "DashboardApp",
    "DashboardConfig",
    "ReportExplorer",
    "RunMonitor",
    "SuiteEditor",
    "TraceViewer",
    "start_dashboard",
]


def __getattr__(name: str):
    """Lazy load dashboard components."""
    if name == "DashboardApp":
        from .app import DashboardApp

        return DashboardApp
    if name == "SuiteEditor":
        from .components import SuiteEditor

        return SuiteEditor
    if name == "RunMonitor":
        from .components import RunMonitor

        return RunMonitor
    if name == "TraceViewer":
        from .components import TraceViewer

        return TraceViewer
    if name == "ReportExplorer":
        from .components import ReportExplorer

        return ReportExplorer
    if name == "start_dashboard":
        from .server import start_dashboard

        return start_dashboard
    if name == "DashboardConfig":
        from .server import DashboardConfig

        return DashboardConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Register lazy loader
def __dir__():
    return __all__
