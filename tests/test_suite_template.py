import pytest
from dataclasses import dataclass
from typing import Generator, Any
from agentunit import Scenario
from agentunit.adapters.base import AdapterOutcome, BaseAdapter

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
    def prepare(self) -> None:
        self.agent.connect()
    def execute(self, input_data: FAQItem) -> AdapterOutcome:
        # The core logic we want to test
        response = self.agent.answer(input_data.question)
        
        # Simple exact match check
        success = (response == input_data.answer)
        
        return AdapterOutcome(success=success,output=response)

def test_suite_template_flow():
    """
    Verifies that the FAQAdapter logic from the documentation template
    works correctly with a mock agent.
    """
    # 1. Setup the scenario with our adapter and dataset
    dataset = MockDataset()
    adapter = FAQAdapter()
    adapter.prepare()  
    
    for item in dataset:
        outcome = adapter.execute(item)  
        
        # 3. Assertions
        assert outcome.success is True
        assert outcome.output == item.answer