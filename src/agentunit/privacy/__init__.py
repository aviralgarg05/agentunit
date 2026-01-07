"""Privacy and federated testing support.

This module provides:
- Privacy wrappers for datasets
- PII leakage detection metrics
- Privacy-preserving evaluation techniques
"""

import sys

from .federation import FederatedEvaluator, PrivacyGuard
from .metrics import (
    ConsentComplianceMetric,
    DataMinimizationMetric,
    PIILeakageMetric,
    PrivacyBudgetMetric,
)
from .wrappers import PrivacyConfig, PrivateDatasetWrapper


__all__ = [
    "ConsentComplianceMetric",
    "DataMinimizationMetric",
    "FederatedEvaluator",
    "PIILeakageMetric",
    "PrivacyBudgetMetric",
    "PrivacyConfig",
    "PrivacyGuard",
    "PrivateDatasetWrapper",
]


def __getattr__(name: str):
    """Lazy load privacy components."""
    if name == "PrivateDatasetWrapper":
        from .wrappers import PrivateDatasetWrapper

        return PrivateDatasetWrapper
    if name == "PrivacyConfig":
        from .wrappers import PrivacyConfig

        return PrivacyConfig
    if name == "PIILeakageMetric":
        from .metrics import PIILeakageMetric

        return PIILeakageMetric
    if name == "PrivacyBudgetMetric":
        from .metrics import PrivacyBudgetMetric

        return PrivacyBudgetMetric
    if name == "DataMinimizationMetric":
        from .metrics import DataMinimizationMetric

        return DataMinimizationMetric
    if name == "ConsentComplianceMetric":
        from .metrics import ConsentComplianceMetric

        return ConsentComplianceMetric
    if name == "FederatedEvaluator":
        from .federation import FederatedEvaluator

        return FederatedEvaluator
    if name == "PrivacyGuard":
        from .federation import PrivacyGuard

        return PrivacyGuard
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Register lazy loader
def __dir__():
    return __all__
