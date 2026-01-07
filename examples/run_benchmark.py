"""Run real benchmarks against an agent."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agentunit.benchmarks import BenchmarkRunner, BenchmarkScenario
from agentunit.benchmarks.definitions import DEFAULT_SCENARIOS

def dummy_math_agent(prompt: str) -> str:
    """A dummy agent that knows some math."""
    if "15%" in prompt:
        return "The answer is 127.5"
    if "2x + 5" in prompt:
        return "x = 5"
    if "area" in prompt:
        return "Area is 25"
    return "I don't know."

def dummy_reasoning_agent(prompt: str) -> str:
    """A dummy agent for logic."""
    if "cats" in prompt:
        return "Yes, cats are animals."
    if "apples" in prompt:
        return "You have 6 apples."
    return "Thinking..."

def dummy_qa_agent(prompt: str) -> str:
    """A dummy agent for QA."""
    if "France" in prompt:
        return "Paris"
    if "Romeo" in prompt:
        return "William Shakespeare"
    return "Unknown."

def router_agent(prompt: str) -> str:
    """Simulates a capable agent by routing to specific dummy logic."""
    # Simulate thinking time
    import time
    time.sleep(0.5)
    
    combined = dummy_math_agent(prompt)
    if combined == "I don't know.":
        combined = dummy_reasoning_agent(prompt)
        if combined == "Thinking...":
            combined = dummy_qa_agent(prompt)
    return combined

def main():
    runner = BenchmarkRunner()
    
    print("🤖 Running Benchmarks with 'Simulated-GPT-4'...")
    
    for scenario in DEFAULT_SCENARIOS:
        runner.run_scenario(
            scenario=scenario,
            agent_func=router_agent,
            model_name="Simulated-GPT-4"
        )

if __name__ == "__main__":
    main()
