"""Multimodal evaluation support for AgentUnit.

This module provides adapters and metrics for evaluating agents that process
vision, audio, and other multimodal inputs. Integrates with GPT-4o, CLIP, and
other multimodal models.
"""

from .adapters import AudioAdapter, MultimodalAdapter, VisionAdapter
from .metrics import (
    AudioTranscriptionMetric,
    CrossModalGroundingMetric,
    ImageCaptionAccuracyMetric,
    MultimodalCoherenceMetric,
    VideoResponseRelevanceMetric,
)


__all__ = [
    "AudioAdapter",
    "AudioTranscriptionMetric",
    "CrossModalGroundingMetric",
    "ImageCaptionAccuracyMetric",
    "MultimodalAdapter",
    "MultimodalCoherenceMetric",
    "VideoResponseRelevanceMetric",
    "VisionAdapter",
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
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    """Support for dir() and autocomplete."""
    return sorted(__all__)
