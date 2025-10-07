#!/usr/bin/env python3
"""CLI script to run the AgentUnit dashboard."""

import argparse
import sys
from pathlib import Path

# Add agentunit to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentunit.dashboard.server import start_dashboard


def main():
    """Run dashboard CLI."""
    parser = argparse.ArgumentParser(description="Run AgentUnit Dashboard")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Path to workspace directory (default: current directory)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to bind to (default: 8501)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser"
    )
    
    args = parser.parse_args()
    
    start_dashboard(
        workspace_path=args.workspace,
        host=args.host,
        port=args.port,
        auto_open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()
