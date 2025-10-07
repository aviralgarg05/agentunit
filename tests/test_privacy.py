"""Tests for privacy module."""

import pytest
from agentunit.datasets.base import DatasetCase
from agentunit.core.trace import TraceLog


def test_privacy_imports():
    """Test privacy module imports."""
    from agentunit.privacy import (
        PrivateDatasetWrapper,
        PrivacyConfig,
        PIILeakageMetric,
        PrivacyBudgetMetric,
        DataMinimizationMetric,
        ConsentComplianceMetric,
        FederatedEvaluator,
        PrivacyGuard,
    )
    
    assert PrivateDatasetWrapper is not None
    assert PrivacyConfig is not None
    assert PIILeakageMetric is not None
    assert PrivacyBudgetMetric is not None
    assert DataMinimizationMetric is not None
    assert ConsentComplianceMetric is not None
    assert FederatedEvaluator is not None
    assert PrivacyGuard is not None


def test_pii_leakage_metric():
    """Test PII leakage detection."""
    from agentunit.privacy import PIILeakageMetric
    
    metric = PIILeakageMetric()
    
    # Case with PII in output
    case = DatasetCase(
        id="test_pii",
        query="What is the customer email?"
    )
    
    trace = TraceLog()
    trace.record("agent_response", content="The customer email is john.doe@example.com")
    
    result = metric.evaluate(case, trace, None)
    
    assert result["num_detections"] > 0
    assert result["score"] < 1.0
    assert not result["passed"]
    assert any(d["type"] == "email" for d in result["detections"])


def test_pii_leakage_no_leakage():
    """Test PII metric with no leakage."""
    from agentunit.privacy import PIILeakageMetric
    
    metric = PIILeakageMetric()
    
    case = DatasetCase(
        id="test_no_pii",
        query="What is the weather?"
    )
    
    trace = TraceLog()
    trace.record("agent_response", content="The weather is sunny today.")
    
    result = metric.evaluate(case, trace, None)
    
    assert result["num_detections"] == 0
    assert abs(result["score"] - 1.0) < 0.01
    assert result["passed"]


def test_private_dataset_wrapper():
    """Test private dataset wrapper."""
    from agentunit.privacy import PrivateDatasetWrapper
    
    dataset = [
        DatasetCase(
            id="case1",
            query="Email john.doe@example.com for info",
            metadata={"phone": "555-123-4567"}
        ),
        DatasetCase(
            id="case2",
            query="Call 555-987-6543",
        )
    ]
    
    wrapper = PrivateDatasetWrapper(dataset, enable_pii_masking=True)
    private_dataset = wrapper.get_private_dataset()
    
    assert len(private_dataset) == 2
    assert "[EMAIL]" in private_dataset[0].query
    assert "[PHONE]" in private_dataset[1].query


def test_privacy_config():
    """Test privacy configuration."""
    from agentunit.privacy import PrivacyConfig
    
    config = PrivacyConfig(
        epsilon=0.5,
        delta=1e-6,
        noise_mechanism="gaussian",
        enable_pii_masking=True
    )
    
    assert abs(config.epsilon - 0.5) < 0.01
    assert abs(config.delta - 1e-6) < 1e-7
    assert config.noise_mechanism == "gaussian"
    assert config.enable_pii_masking is True


def test_privacy_budget_metric():
    """Test privacy budget tracking."""
    from agentunit.privacy import PrivacyBudgetMetric
    
    metric = PrivacyBudgetMetric(total_budget=5.0, warn_threshold=0.8)
    
    case = DatasetCase(id="test", query="Test query")
    trace = TraceLog()
    trace.record("privacy_usage", privacy_epsilon=1.0)
    
    # First evaluation
    result1 = metric.evaluate(case, trace, None)
    assert abs(result1["epsilon_used"] - 1.0) < 0.01
    assert abs(result1["total_spent"] - 1.0) < 0.01
    assert not result1["budget_exceeded"]
    
    # Second evaluation
    result2 = metric.evaluate(case, trace, None)
    assert abs(result2["total_spent"] - 2.0) < 0.01


def test_privacy_guard():
    """Test privacy guard constraints."""
    from agentunit.privacy import PrivacyGuard
    
    guard = PrivacyGuard(
        min_batch_size=5,
        max_queries_per_hour=100,
        total_epsilon_budget=10.0
    )
    
    # Check batch size
    assert guard.check_batch_size(10) is True
    assert guard.check_batch_size(3) is False
    
    # Check query limit
    assert guard.check_query_limit() is True
    
    # Check privacy budget
    assert guard.check_privacy_budget(5.0) is True
    assert guard.check_privacy_budget(15.0) is False
    
    # Record queries
    guard.record_query(epsilon=2.0)
    assert abs(guard.epsilon_spent - 2.0) < 0.01
    assert guard.query_count == 1


def test_secure_aggregator():
    """Test secure aggregation."""
    from agentunit.privacy.wrappers import SecureAggregator
    
    aggregator = SecureAggregator(num_parties=3)
    
    # Add shares from parties
    aggregator.add_share(0, 0.8, noise_scale=0.1)
    aggregator.add_share(1, 0.9, noise_scale=0.1)
    aggregator.add_share(2, 0.7, noise_scale=0.1)
    
    # Aggregate
    result = aggregator.aggregate(method="mean")
    
    # Result should be close to 0.8 (with some noise)
    assert 0.5 < result < 1.0


def test_data_minimization_metric():
    """Test data minimization metric."""
    from agentunit.privacy import DataMinimizationMetric
    
    metric = DataMinimizationMetric(context_keys=["user_id", "email"])
    
    # Case with exposed context
    case = DatasetCase(
        id="test",
        query="Get user info",
        metadata={"user_id": "12345", "email": "user@test.com"}
    )
    
    trace = TraceLog()
    trace.record("agent_response", content="User ID: 12345")
    
    result = metric.evaluate(case, trace, None)
    
    assert result["num_leakages"] > 0
    assert not result["passed"]


def test_consent_compliance_metric():
    """Test consent compliance metric."""
    from agentunit.privacy import ConsentComplianceMetric
    
    metric = ConsentComplianceMetric()
    
    # Case with consent restrictions
    case = DatasetCase(
        id="test",
        query="What is my location?",
        metadata={
            "consent": {
                "location": False,
                "email": True
            }
        }
    )
    
    trace = TraceLog()
    trace.record("agent_response", content="Your location is New York")
    
    result = metric.evaluate(case, trace, None)
    
    assert result["num_violations"] > 0
    assert not result["passed"]
