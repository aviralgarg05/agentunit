"""Framework adapters."""
from .base import BaseAdapter, AdapterOutcome
from .langgraph import LangGraphAdapter
from .openai_agents import OpenAIAgentsAdapter
from .crewai import CrewAIAdapter

__all__ = [
    "BaseAdapter",
    "AdapterOutcome",
    "LangGraphAdapter",
    "OpenAIAgentsAdapter",
    "CrewAIAdapter",
]
