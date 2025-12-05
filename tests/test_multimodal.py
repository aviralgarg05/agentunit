"""Tests for multimodal evaluation support."""

import pytest

from agentunit import multimodal
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase
from agentunit.multimodal.adapters import MultimodalInput


try:
    from agentunit.multimodal.metrics import (
        CrossModalGroundingMetric,
        ImageCaptionAccuracyMetric,
        MultimodalCoherenceMetric,
    )
except ImportError:
    CrossModalGroundingMetric = None
    ImageCaptionAccuracyMetric = None
    MultimodalCoherenceMetric = None


def test_multimodal_imports():
    """Test that multimodal module can be imported."""
    assert hasattr(multimodal, "__all__")
    assert "MultimodalAdapter" in multimodal.__all__
    assert "VisionAdapter" in multimodal.__all__
    assert "AudioAdapter" in multimodal.__all__


def test_multimodal_input():
    """Test MultimodalInput dataclass."""
    input_data = MultimodalInput(text="Describe this image", image_path="/path/to/image.jpg")

    assert input_data.text == "Describe this image"
    assert input_data.image_path == "/path/to/image.jpg"
    assert input_data.metadata == {}


def test_cross_modal_grounding_metric():
    """Test CrossModalGroundingMetric initialization."""
    if CrossModalGroundingMetric is None:
        pytest.skip("CrossModalGroundingMetric dependencies missing")
    try:
        metric = CrossModalGroundingMetric()
    except ImportError:
        pytest.skip("CrossModalGroundingMetric dependencies missing")

    assert metric.name == "cross_modal_grounding"


def test_image_caption_metric():
    """Test ImageCaptionAccuracyMetric."""
    if ImageCaptionAccuracyMetric is None:
        pytest.skip("ImageCaptionAccuracyMetric dependencies missing")

    metric = ImageCaptionAccuracyMetric(use_semantic=False)

    case = DatasetCase(
        id="test_1",
        query="Describe this image",
        expected_output="A cat sitting on a mat",
        metadata={"image_path": "/fake/path.jpg"},
    )

    trace = TraceLog()
    outcome = "A cat sitting on a mat"

    result = metric.evaluate(case, trace, outcome)

    assert result.name == "image_caption_accuracy"
    assert result.value is not None
    assert result.value > 0.8  # Should have high similarity for exact match


def test_multimodal_coherence_metric():
    """Test MultimodalCoherenceMetric."""
    if MultimodalCoherenceMetric is None:
        pytest.skip("MultimodalCoherenceMetric dependencies missing")

    metric = MultimodalCoherenceMetric()

    case = DatasetCase(
        id="test_2",
        query="Analyze this multimedia content",
        expected_output="coherent response",
        metadata={
            "image_path": "/fake/image.jpg",
            "audio_path": "/fake/audio.mp3",
            "text": "Analyze this content",
        },
    )

    trace = TraceLog()
    outcome = "The image shows a scene while the audio provides narration, creating a coherent multimedia experience"

    result = metric.evaluate(case, trace, outcome)

    assert result.name == "multimodal_coherence"
    assert result.value is not None
    assert "input_modalities" in result.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
