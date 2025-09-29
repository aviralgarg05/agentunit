"""Framework adapters."""
from .base import BaseAdapter, AdapterOutcome
from .langgraph import LangGraphAdapter
from .openai_agents import OpenAIAgentsAdapter
from .crewai import CrewAIAdapter
from .autogen import AutoGenAdapter
from .haystack import HaystackAdapter
from .llama_index import LlamaIndexAdapter
from .semantic_kernel import SemanticKernelAdapter
from .phidata import PhidataAdapter
from .promptflow import PromptFlowAdapter
from .openai_swarm import OpenAISwarmAdapter
from .anthropic_bedrock import AnthropicBedrockAdapter
from .mistral_server import MistralServerAdapter
from .rasa import RasaAdapter

__all__ = [
    "BaseAdapter",
    "AdapterOutcome",
    "LangGraphAdapter",
    "OpenAIAgentsAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
    "HaystackAdapter",
    "LlamaIndexAdapter",
    "SemanticKernelAdapter",
    "PhidataAdapter",
    "PromptFlowAdapter",
    "OpenAISwarmAdapter",
    "AnthropicBedrockAdapter",
    "MistralServerAdapter",
    "RasaAdapter",
]
