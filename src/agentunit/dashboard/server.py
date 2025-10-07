"""Dashboard server and configuration."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    
    workspace_path: Path
    host: str = "localhost"
    port: int = 8501
    theme: str = "light"
    auto_open_browser: bool = True


def start_dashboard(
    workspace_path: Optional[Path] = None,
    host: str = "localhost",
    port: int = 8501,
    auto_open_browser: bool = True
):
    """Start the AgentUnit dashboard.
    
    Args:
        workspace_path: Path to workspace directory
        host: Host to bind to
        port: Port to bind to
        auto_open_browser: Whether to auto-open browser
        
    Raises:
        ImportError: If streamlit is not installed
    """
    try:
        import streamlit
    except ImportError:
        raise ImportError(
            "streamlit is required for the dashboard. "
            "Install it with: pip install streamlit"
        )
    
    workspace_path = workspace_path or Path.cwd()
    
    # Create dashboard script
    dashboard_script = _create_dashboard_script(workspace_path)
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_script),
        f"--server.port={port}",
        f"--server.address={host}",
    ]
    
    if not auto_open_browser:
        cmd.append("--server.headless=true")
    
    print(f"Starting AgentUnit Dashboard at http://{host}:{port}")
    subprocess.run(cmd)


def _create_dashboard_script(workspace_path: Path) -> Path:
    """Create the dashboard entry script.
    
    Args:
        workspace_path: Path to workspace
        
    Returns:
        Path to created script
    """
    script_content = f'''
import sys
from pathlib import Path

# Add agentunit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentunit.dashboard.app import DashboardApp

# Run dashboard
app = DashboardApp(workspace_path=Path("{workspace_path}"))
app.run()
'''
    
    script_path = workspace_path / ".agentunit" / "dashboard.py"
    script_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path
