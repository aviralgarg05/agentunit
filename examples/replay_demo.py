"""Demo of Session Replay and Time-Travel Debugging.

This script demonstrates:
1. Recording a live agent session
2. Analyzing the session programmatically
3. Replaying the session with the Time-Travel Debugger

Usage:
    export OPENAI_API_KEY=... (optional, for real data)
    python examples/replay_demo.py
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agentunit.replay import SessionRecorder, TimeTravelDebugger
from agentunit.replay.analyzer import SessionAnalyzer

def run_demo_session():
    """Simulate or run a real session."""
    print("🎥 Starting session recording...")
    
    recorder = SessionRecorder(
        session_id=f"demo_{int(time.time())}",
        agent_name="demo_agent",
        output_dir="sessions"
    )
    
    # 1. User Input
    print("Step 1: User Input")
    recorder.record_step("input", "Calculate 15% of 850")
    
    # 2. Agent Thought
    print("Step 2: Agent Thought")
    recorder.record_step("thought", "I need to use a calculator tool.")
    
    # 3. Tool Call
    print("Step 3: Tool Call")
    recorder.record_step("tool_call", {"name": "calculator", "args": "0.15 * 850"})
    
    # 4. Tool Result
    print("Step 4: Tool Result")
    recorder.record_step("tool_result", "127.5")
    
    # 5. Final Output
    print("Step 5: Final Output")
    recorder.record_step("output", "The answer is 127.5")
    
    recorder.finish()
    print(f"✅ Session saved to {recorder.output_dir}/{recorder.session.session_id}.json")
    return recorder.session.session_id

def analyze_session(session_id):
    """Analyze the recorded session."""
    print(f"\n📊 Analyzing session {session_id}...")
    path = f"sessions/{session_id}.json"
    
    analyzer = SessionAnalyzer(path)
    summary = analyzer.summarize()
    
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Tool Usage: {summary['tool_usage']}")
    print("Analysis complete.")

def debug_session(session_id):
    """Launch interactive debugger."""
    print(f"\n🕵️‍♀️ Launching Time-Travel Debugger for {session_id}...")
    print("(Type 'next', 'prev', 'jump <n>', 'context', or 'exit')")
    
    path = f"sessions/{session_id}.json"
    debugger = TimeTravelDebugger(path)
    debugger.cmdloop()

if __name__ == "__main__":
    session_id = run_demo_session()
    analyze_session(session_id)
    
    if os.environ.get("INTERACTIVE_DEMO"):
        debug_session(session_id)
    else:
        print("\n💡 To run interactive debugger:")
        print(f"python -m agentunit.replay.debugger sessions/{session_id}.json")
