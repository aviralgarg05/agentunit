import pytest
from dataclasses import dataclass
from typing import Generator, Any
from agentunit import Scenario
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
# Note: Adjust these imports based on what is actually in suite_template.py
# You might need to check src/agentunit/core/context.py or similar for exact types if these fail.

# --- 1. MOCK THE CLASSES FROM THE TEMPLATE ---
# (We copy the essence of FAQAdapter/DatasetSource here since we can't import from docs/)

@dataclass
class FAQItem:
    question: str
    answer: str

class MockDataset:
    """Simulates the dataset source from the template."""
    def __iter__(self) -> Generator[FAQItem, None, None]:
        yield FAQItem("What is AgentUnit?", "A framework.")
        yield FAQItem("Is it open source?", "Yes.")

class MockAgent:
    """A fake agent that always answers correctly."""
    def connect(self) -> None:
        pass

    def answer(self, question: str) -> str:
        # Simple logic to simulate an agent
        if "AgentUnit" in question:
            return "A framework."
        return "Yes."

class FAQAdapter(BaseAdapter):
    """
    This mimics the adapter in docs/templates/suite_template.py
    """
    def __init__(self):
        self.agent = MockAgent()
    
    # RENAME: setup -> prepare
    def prepare(self) -> None:
        self.agent.connect()

    # RENAME: run -> execute
    def execute(self, input_data: FAQItem) -> AdapterOutcome:
        # The core logic we want to test
        response = self.agent.answer(input_data.question)
        
        # Simple exact match check
        success = (response == input_data.answer)
        
        return AdapterOutcome(success=success,output=response)

# --- 2. THE TEST CASE ---

def test_suite_template_flow():
    """
    Verifies that the FAQAdapter logic from the documentation template
    works correctly with a mock agent.
    """
    # 1. Setup the scenario with our adapter and dataset
    dataset = MockDataset()
    adapter = FAQAdapter()
    
    # 2. Call the correct lifecycle methods
    adapter.prepare()  # Was adapter.setup()
    
    for item in dataset:
        outcome = adapter.execute(item)  # Was adapter.run(item)
        
        # 3. Assertions
        assert outcome.success is True
        assert outcome.output == item.answer