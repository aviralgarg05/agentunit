"""Privacy and federated testing support.

This module provides:
- Privacy wrappers for datasets
- PII leakage detection metrics
- Privacy-preserving evaluation techniques
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wrappers import PrivateDatasetWrapper, PrivacyConfig
    from .metrics import PIILeakageMetric, PrivacyBudgetMetric, DataMinimizationMetric, ConsentComplianceMetric
    from .federation import FederatedEvaluator, PrivacyGuard

__all__ = [
    "PrivateDatasetWrapper",
    "PrivacyConfig",
    "PIILeakageMetric",
    "PrivacyBudgetMetric",
    "DataMinimizationMetric",
    "ConsentComplianceMetric",
    "FederatedEvaluator",
    "PrivacyGuard",
]


def __getattr__(name: str):
    """Lazy load privacy components."""
    if name == "PrivateDatasetWrapper":
        from .wrappers import PrivateDatasetWrapper
        return PrivateDatasetWrapper
    elif name == "PrivacyConfig":
        from .wrappers import PrivacyConfig
        return PrivacyConfig
    elif name == "PIILeakageMetric":
        from .metrics import PIILeakageMetric
        return PIILeakageMetric
    elif name == "PrivacyBudgetMetric":
        from .metrics import PrivacyBudgetMetric
        return PrivacyBudgetMetric
    elif name == "DataMinimizationMetric":
        from .metrics import DataMinimizationMetric
        return DataMinimizationMetric
    elif name == "ConsentComplianceMetric":
        from .metrics import ConsentComplianceMetric
        return ConsentComplianceMetric
    elif name == "FederatedEvaluator":
        from .federation import FederatedEvaluator
        return FederatedEvaluator
    elif name == "PrivacyGuard":
        from .federation import PrivacyGuard
        return PrivacyGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register lazy loader
if sys.version_info >= (3, 7):
    def __dir__():
        return __all__
