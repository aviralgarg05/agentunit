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
    from .components import SuiteEditor, RunMonitor, TraceViewer, ReportExplorer
    from .server import start_dashboard, DashboardConfig

__all__ = [
    "DashboardApp",
    "SuiteEditor",
    "RunMonitor",
    "TraceViewer",
    "ReportExplorer",
    "start_dashboard",
    "DashboardConfig",
]


def __getattr__(name: str):
    """Lazy load dashboard components."""
    if name == "DashboardApp":
        from .app import DashboardApp
        return DashboardApp
    elif name == "SuiteEditor":
        from .components import SuiteEditor
        return SuiteEditor
    elif name == "RunMonitor":
        from .components import RunMonitor
        return RunMonitor
    elif name == "TraceViewer":
        from .components import TraceViewer
        return TraceViewer
    elif name == "ReportExplorer":
        from .components import ReportExplorer
        return ReportExplorer
    elif name == "start_dashboard":
        from .server import start_dashboard
        return start_dashboard
    elif name == "DashboardConfig":
        from .server import DashboardConfig
        return DashboardConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register lazy loader
if sys.version_info >= (3, 7):
    def __dir__():
        return __all__
