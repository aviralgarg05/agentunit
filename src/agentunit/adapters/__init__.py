"""Framework adapters.

This module uses PEP 562 lazy loading to defer imports of optional dependencies.
Adapter classes listed in __all__ are imported dynamically via __getattr__ when
first accessed. TYPE_CHECKING imports provide static type information for linters
without triggering runtime imports.

Static analyzers may flag __all__ entries as undefined; this is expected and benign
with the lazy-loading pattern. See PEP 562 for details.
"""

import importlib

from .agentops_adapter import AgentOpsAdapter
from .anthropic_bedrock import AnthropicBedrockAdapter
from .autogen import AutoGenAdapter
from .autogen_ag2 import AG2Adapter
from .base import AdapterOutcome, BaseAdapter
from .crewai import CrewAIAdapter
from .haystack import HaystackAdapter
from .langgraph import LangGraphAdapter
from .langsmith_adapter import LangSmithAdapter
from .llama_index import LlamaIndexAdapter
from .mistral_server import MistralServerAdapter
from .openai_agents import OpenAIAgentsAdapter
from .openai_swarm import OpenAISwarmAdapter
from .phidata import PhidataAdapter
from .promptflow import PromptFlowAdapter
from .rasa import RasaAdapter
from .semantic_kernel import SemanticKernelAdapter
from .swarm_adapter import SwarmAdapter
from .wandb_adapter import WandbAdapter


__all__ = [
    "AG2Adapter",
    "AdapterOutcome",
    "AgentOpsAdapter",
    "AnthropicBedrockAdapter",
    "AutoGenAdapter",
    "BaseAdapter",
    "CrewAIAdapter",
    "HaystackAdapter",
    "LangGraphAdapter",
    "LangSmithAdapter",
    "LlamaIndexAdapter",
    "MistralServerAdapter",
    "OpenAIAgentsAdapter",
    "OpenAISwarmAdapter",
    "PhidataAdapter",
    "PromptFlowAdapter",
    "RasaAdapter",
    "SemanticKernelAdapter",
    "SwarmAdapter",
    "WandbAdapter",
]


# Mapping of adapter names to their import paths and class names
_ADAPTER_IMPORTS = {
    "LangGraphAdapter": ("langgraph", "LangGraphAdapter"),
    "OpenAIAgentsAdapter": ("openai_agents", "OpenAIAgentsAdapter"),
    "CrewAIAdapter": ("crewai", "CrewAIAdapter"),
    "AutoGenAdapter": ("autogen", "AutoGenAdapter"),
    "AG2Adapter": ("autogen_ag2", "AG2Adapter"),
    "HaystackAdapter": ("haystack", "HaystackAdapter"),
    "LlamaIndexAdapter": ("llama_index", "LlamaIndexAdapter"),
    "SemanticKernelAdapter": ("semantic_kernel", "SemanticKernelAdapter"),
    "PhidataAdapter": ("phidata", "PhidataAdapter"),
    "PromptFlowAdapter": ("promptflow", "PromptFlowAdapter"),
    "OpenAISwarmAdapter": ("openai_swarm", "OpenAISwarmAdapter"),
    "SwarmAdapter": ("swarm_adapter", "SwarmAdapter"),
    "LangSmithAdapter": ("langsmith_adapter", "LangSmithAdapter"),
    "AgentOpsAdapter": ("agentops_adapter", "AgentOpsAdapter"),
    "WandbAdapter": ("wandb_adapter", "WandbAdapter"),
    "AnthropicBedrockAdapter": ("anthropic_bedrock", "AnthropicBedrockAdapter"),
    "MistralServerAdapter": ("mistral_server", "MistralServerAdapter"),
    "RasaAdapter": ("rasa", "RasaAdapter"),
}


def __getattr__(name: str):
    """Lazy import adapters to avoid importing dependencies until needed."""
    if name in _ADAPTER_IMPORTS:
        module_name, class_name = _ADAPTER_IMPORTS[name]
        module = importlib.import_module(f".{module_name}", package=__package__)
        return getattr(module, class_name)

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__():
    """Support dir() and autocomplete by listing all available names."""
    return __all__
