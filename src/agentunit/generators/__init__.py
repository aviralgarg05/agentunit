"""Custom dataset generators for synthetic test case creation.

This module provides LLM-powered tools for generating synthetic datasets
with edge case augmentation, adversarial queries, and noisy contexts.
"""

from .augmentation import (
    AdversarialAugmenter,
    DistributionShifter,
    EdgeCaseGenerator,
    NoiseAugmenter,
)
from .llm_generator import LlamaDatasetGenerator, OpenAIDatasetGenerator
from .templates import DatasetTemplate, PromptTemplate


__all__ = [
    "AdversarialAugmenter",
    "DatasetTemplate",
    "DistributionShifter",
    "EdgeCaseGenerator",
    "LlamaDatasetGenerator",
    "NoiseAugmenter",
    "OpenAIDatasetGenerator",
    "PromptTemplate",
]

_GENERATOR_IMPORTS = {
    "LlamaDatasetGenerator": "agentunit.generators.llm_generator",
    "OpenAIDatasetGenerator": "agentunit.generators.llm_generator",
    "AdversarialAugmenter": "agentunit.generators.augmentation",
    "NoiseAugmenter": "agentunit.generators.augmentation",
    "EdgeCaseGenerator": "agentunit.generators.augmentation",
    "DistributionShifter": "agentunit.generators.augmentation",
    "DatasetTemplate": "agentunit.generators.templates",
    "PromptTemplate": "agentunit.generators.templates",
}


def __getattr__(name: str):
    """Lazy loading for generator components."""
    if name in _GENERATOR_IMPORTS:
        import importlib

        module_path = _GENERATOR_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    """Support for dir() and autocomplete."""
    return sorted(__all__)
