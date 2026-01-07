"""Time-travel debugger for agent sessions."""

import cmd
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agentunit.replay.player import SessionPlayer

console = Console()
logger = logging.getLogger(__name__)


class TimeTravelDebugger(cmd.Cmd):
    """Interactive debugger for replaying agent sessions."""
    
    intro = "Welcome to AgentUnit Time-Travel Debugger. Type help or ? to list commands."
    prompt = "(pdb-agent) "
    
    def __init__(self, session_path: str | Path):
        super().__init__()
        try:
            self.player = SessionPlayer(session_path)
            self.session_path = session_path
            self._update_prompt()
            print(f"Loaded session '{self.player.session.session_id}' with {self.player.total_steps} steps.")
        except Exception as e:
            print(f"Failed to load session: {e}")
            self.player = None

    def _update_prompt(self):
        """Update prompt with current step info."""
        if self.player:
            self.prompt = f"(step {self.player.current_step_index + 1}/{self.player.total_steps}) "

    def do_next(self, arg):
        """Move to the next step."""
        step = self.player.next_step()
        if step:
            self._print_step(step)
            self._update_prompt()
        else:
            print("End of session reached.")

    def do_prev(self, arg):
        """Move to the previous step."""
        step = self.player.prev_step()
        if step:
            self._print_step(step)
            self._update_prompt()
        else:
            print("Start of session reached.")

    def do_step(self, arg):
        """Show current step details."""
        step = self.player.current_step
        if step:
            self._print_step(step)
        else:
            print("No current step.")

    def do_jump(self, arg):
        """Jump to a specific step number (1-based). Usage: jump 5"""
        try:
            step_num = int(arg)
            step = self.player.seek(step_num - 1)
            if step:
                self._print_step(step)
                self._update_prompt()
            else:
                print("Invalid step number.")
        except ValueError:
            print("Please provide a valid integer step number.")

    def do_context(self, arg):
        """Show conversation history (context) up to current step."""
        history = self.player.get_context_at_step(self.player.current_step_index)
        table = Table(title="Conversation History")
        table.add_column("Step", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Content")
        
        for step in history:
            content_preview = str(step.content)[:100] + "..." if len(str(step.content)) > 100 else str(step.content)
            table.add_row(str(step.step_id + 1), step.type, content_preview)
            
        console.print(table)

    def do_exit(self, arg):
        """Exit the debugger."""
        return True

    def _print_step(self, step):
        """Pretty print a step."""
        console.print(Panel(
            f"[bold cyan]Type:[/bold cyan] {step.type}\n"
            f"[bold yellow]Values:[/bold yellow]\n{step.content}\n"
            f"[bold dim]Metadata:[/bold dim] {step.metadata}",
            title=f"Step {step.step_id + 1} @ {step.timestamp:.3f}s",
            border_style="green"
        ))

    def do_search(self, arg):
        """Search for text in step content. Usage: search tool_name"""
        if not arg:
            print("Usage: search <text>")
            return

        found = []
        for i, step in enumerate(self.player.session.steps):
            if arg.lower() in str(step.content).lower() or arg.lower() in str(step.type).lower():
                found.append(i + 1)
        
        if found:
            print(f"Found matches at steps: {', '.join(map(str, found))}")
        else:
            print("No matches found.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debugger = TimeTravelDebugger(sys.argv[1])
        debugger.cmdloop()
    else:
        print("Usage: python -m agentunit.replay.debugger <session_file.json>")
