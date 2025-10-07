"""Tests for multimodal evaluation support."""

import pytest
from pathlib import Path

# Test imports work with lazy loading
def test_multimodal_imports():
    """Test that multimodal module can be imported."""
    from agentunit import multimodal
    
    assert hasattr(multimodal, '__all__')
    assert 'MultimodalAdapter' in multimodal.__all__
    assert 'VisionAdapter' in multimodal.__all__
    assert 'AudioAdapter' in multimodal.__all__


def test_multimodal_input():
    """Test MultimodalInput dataclass."""
    from agentunit.multimodal.adapters import MultimodalInput
    
    input_data = MultimodalInput(
        text="Describe this image",
        image_path="/path/to/image.jpg"
    )
    
    assert input_data.text == "Describe this image"
    assert input_data.image_path == "/path/to/image.jpg"
    assert input_data.metadata == {}


def test_cross_modal_grounding_metric():
    """Test CrossModalGroundingMetric initialization."""
    try:
        from agentunit.multimodal.metrics import CrossModalGroundingMetric
        # This will fail if CLIP not installed, which is expected
        metric = CrossModalGroundingMetric()
        assert metric.name == "cross_modal_grounding"
    except ImportError as e:
        # Expected if CLIP not installed
        assert "CLIP required" in str(e)


def test_image_caption_metric():
    """Test ImageCaptionAccuracyMetric."""
    from agentunit.multimodal.metrics import ImageCaptionAccuracyMetric
    from agentunit.datasets.base import DatasetCase
    from agentunit.core.trace import TraceLog
    
    # Test with semantic similarity disabled (no dependencies required)
    metric = ImageCaptionAccuracyMetric(use_semantic=False)
    
    case = DatasetCase(
        id="test_1",
        query="Describe this image",
        expected_output="A cat sitting on a mat",
        metadata={"image_path": "/fake/path.jpg"}
    )
    
    trace = TraceLog()
    outcome = "A cat sitting on a mat"
    
    result = metric.evaluate(case, trace, outcome)
    
    assert result.name == "image_caption_accuracy"
    assert result.value is not None
    assert result.value > 0.8  # Should have high similarity for exact match


def test_multimodal_coherence_metric():
    """Test MultimodalCoherenceMetric."""
    from agentunit.multimodal.metrics import MultimodalCoherenceMetric
    from agentunit.datasets.base import DatasetCase
    from agentunit.core.trace import TraceLog
    
    metric = MultimodalCoherenceMetric()
    
    case = DatasetCase(
        id="test_2",
        query="Analyze this multimedia content",
        expected_output="coherent response",
        metadata={
            "image_path": "/fake/image.jpg",
            "audio_path": "/fake/audio.mp3",
            "text": "Analyze this content"
        }
    )
    
    trace = TraceLog()
    outcome = "The image shows a scene while the audio provides narration, creating a coherent multimedia experience"
    
    result = metric.evaluate(case, trace, outcome)
    
    assert result.name == "multimodal_coherence"
    assert result.value is not None
    assert "input_modalities" in result.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
