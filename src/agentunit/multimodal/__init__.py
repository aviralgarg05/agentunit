"""Multimodal evaluation support for AgentUnit.

This module provides adapters and metrics for evaluating agents that process
vision, audio, and other multimodal inputs. Integrates with GPT-4o, CLIP, and
other multimodal models.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters import MultimodalAdapter, VisionAdapter, AudioAdapter
    from .metrics import (
        CrossModalGroundingMetric,
        ImageCaptionAccuracyMetric,
        VideoResponseRelevanceMetric,
        AudioTranscriptionMetric,
        MultimodalCoherenceMetric,
    )

__all__ = [
    "MultimodalAdapter",
    "VisionAdapter", 
    "AudioAdapter",
    "CrossModalGroundingMetric",
    "ImageCaptionAccuracyMetric",
    "VideoResponseRelevanceMetric",
    "AudioTranscriptionMetric",
    "MultimodalCoherenceMetric",
]

_MULTIMODAL_IMPORTS = {
    "MultimodalAdapter": "agentunit.multimodal.adapters",
    "VisionAdapter": "agentunit.multimodal.adapters",
    "AudioAdapter": "agentunit.multimodal.adapters",
    "CrossModalGroundingMetric": "agentunit.multimodal.metrics",
    "ImageCaptionAccuracyMetric": "agentunit.multimodal.metrics",
    "VideoResponseRelevanceMetric": "agentunit.multimodal.metrics",
    "AudioTranscriptionMetric": "agentunit.multimodal.metrics",
    "MultimodalCoherenceMetric": "agentunit.multimodal.metrics",
}


def __getattr__(name: str):
    """Lazy loading for multimodal components."""
    if name in _MULTIMODAL_IMPORTS:
        import importlib
        module_path = _MULTIMODAL_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Support for dir() and autocomplete."""
    return sorted(__all__)
