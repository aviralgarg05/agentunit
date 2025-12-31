import pytest

from agentunit.metrics.registry import (
    DEFAULT_METRICS,
    get_metric,
    register_metric,
    resolve_metrics,
)


def test_resolve_default_metrics():
    # Use dynamic length instead of hardcoding 5
    metrics = resolve_metrics(None)
    assert len(metrics) == len(DEFAULT_METRICS)


def test_resolve_unknown_metric():
    with pytest.raises(KeyError) as excinfo:
        resolve_metrics(["invalid_metric"])
    assert "Unknown metric 'invalid_metric'" in str(excinfo.value)


def test_register_duplicate_metric():
    from agentunit.metrics.builtin import FaithfulnessMetric

    with pytest.raises(ValueError) as excinfo:
        register_metric("faithfulness", FaithfulnessMetric())
    assert "already registered" in str(excinfo.value)


def test_get_metric_success():
    from agentunit.metrics.builtin import ToolSuccessMetric

    metric = get_metric("tool_success")
    assert isinstance(metric, ToolSuccessMetric)


def test_register_metric_success():
    from agentunit.metrics.builtin import ToolSuccessMetric

    test_name = "test_custom_metric"
    test_metric = ToolSuccessMetric()

    try:
        # Test successful registration
        register_metric(test_name, test_metric)
        retrieved = get_metric(test_name)
        assert retrieved is test_metric
    finally:
        # Cleanup to avoid polluting other tests
        DEFAULT_METRICS.pop(test_name, None)
