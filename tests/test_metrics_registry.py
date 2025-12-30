import pytest

from agentunit.metrics.registry import get_metric, register_metric, resolve_metrics


def test_resolve_default_metrics():
    # When names is None, it should return all default metrics
    metrics = resolve_metrics(None)
    assert len(metrics) == 5  # There are 5 metrics in your DEFAULT_METRICS dict


def test_resolve_unknown_metric():
    # This should raise a KeyError because 'invalid_metric' does not exist
    with pytest.raises(KeyError) as excinfo:
        resolve_metrics(["invalid_metric"])

    # Check if the error message is correct
    assert "Unknown metric 'invalid_metric'" in str(excinfo.value)



def test_register_duplicate_metric():
    # Attempting to register 'faithfulness' again should raise ValueError
    # because it's already in DEFAULT_METRICS
    from agentunit.metrics.builtin import FaithfulnessMetric

    with pytest.raises(ValueError) as excinfo:
        register_metric("faithfulness", FaithfulnessMetric())
    assert "already registered" in str(excinfo.value)


def test_get_metric_success():
    # This should successfully return the metric object
    metric = get_metric("tool_success")
    assert metric is not None
    # Verify it's the correct type (optional but good)
    from agentunit.metrics.builtin import ToolSuccessMetric

    assert isinstance(metric, ToolSuccessMetric)
