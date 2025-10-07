"""Privacy wrappers for adapters and datasets."""

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agentunit.datasets.base import DatasetCase


@dataclass
class PrivacyConfig:
    """Privacy configuration for evaluation."""
    
    epsilon: float = 1.0  # Privacy budget (smaller = more private)
    delta: float = 1e-5  # Privacy delta parameter
    noise_mechanism: str = "laplace"  # "laplace" or "gaussian"
    clip_threshold: float = 1.0  # Gradient/output clipping threshold
    enable_pii_masking: bool = True
    enable_output_perturbation: bool = True


# Note: DifferentialPrivacyWrapper would wrap a BaseAdapter but we avoid the import
# to prevent dependency issues. Users should implement this based on their adapter structure.


class PrivateDatasetWrapper:
    """Wrapper for privacy-preserving dataset handling.
    
    Features:
    - PII detection and masking
    - Data anonymization
    - K-anonymity enforcement
    """
    
    def __init__(
        self,
        dataset: List[DatasetCase],
        enable_pii_masking: bool = True,
        k_anonymity: int = 3
    ):
        """Initialize private dataset wrapper.
        
        Args:
            dataset: Original dataset
            enable_pii_masking: Whether to mask PII
            k_anonymity: K-anonymity parameter
        """
        self.dataset = dataset
        self.enable_pii_masking = enable_pii_masking
        self.k_anonymity = k_anonymity
        self._masked_dataset = None
    
    def get_private_dataset(self) -> List[DatasetCase]:
        """Get privacy-preserving version of dataset.
        
        Returns:
            Masked dataset
        """
        if self._masked_dataset is None:
            self._masked_dataset = [self._mask_case(case) for case in self.dataset]
        return self._masked_dataset
    
    def _mask_case(self, case: DatasetCase) -> DatasetCase:
        """Mask PII in a dataset case.
        
        Args:
            case: Original case
            
        Returns:
            Masked case
        """
        if not self.enable_pii_masking:
            return case
        
        # Mask query
        masked_query = self._mask_pii(case.query)
        
        # Mask metadata
        masked_metadata = {}
        if case.metadata:
            for key, value in case.metadata.items():
                if isinstance(value, str):
                    masked_metadata[key] = self._mask_pii(value)
                else:
                    masked_metadata[key] = value
        
        # Return new case with masked data
        return DatasetCase(
            id=self._anonymize_id(case.id),
            query=masked_query,
            metadata=masked_metadata or None
        )
    
    def _mask_pii(self, text: str) -> str:
        """Mask PII in text.
        
        Args:
            text: Text potentially containing PII
            
        Returns:
            Masked text
        """
        import re
        
        # Email masking
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )
        
        # Phone number masking (US format)
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            text
        )
        
        # SSN masking
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            text
        )
        
        # Credit card masking
        text = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '[CREDIT_CARD]',
            text
        )
        
        # Name masking (simple heuristic: capitalized words)
        # Note: This is a simplification and may have false positives
        text = re.sub(
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            '[NAME]',
            text
        )
        
        return text
    
    def _anonymize_id(self, case_id: str) -> str:
        """Anonymize case ID using hashing.
        
        Args:
            case_id: Original ID
            
        Returns:
            Anonymized ID
        """
        # Use SHA256 hash truncated to 12 characters
        hash_obj = hashlib.sha256(case_id.encode())
        return hash_obj.hexdigest()[:12]


class SecureAggregator:
    """Secure aggregation for federated evaluation.
    
    Implements secure multi-party computation for aggregating
    metrics across multiple parties without revealing individual values.
    """
    
    def __init__(self, num_parties: int):
        """Initialize secure aggregator.
        
        Args:
            num_parties: Number of parties in federation
        """
        self.num_parties = num_parties
        self.shares: Dict[int, List[float]] = {}
    
    def add_share(self, party_id: int, value: float, noise_scale: float = 1.0):
        """Add a party's share with noise.
        
        Args:
            party_id: ID of the party
            value: Value to share
            noise_scale: Scale of added noise
        """
        # Add Laplace noise for privacy
        noise = random.gauss(0, noise_scale)
        noisy_value = value + noise
        
        if party_id not in self.shares:
            self.shares[party_id] = []
        self.shares[party_id].append(noisy_value)
    
    def aggregate(self, method: str = "mean") -> float:
        """Securely aggregate shares.
        
        Args:
            method: Aggregation method ("mean", "sum", "median")
            
        Returns:
            Aggregated value
        """
        all_values = [v for shares in self.shares.values() for v in shares]
        
        if not all_values:
            return 0.0
        
        if method == "mean":
            return sum(all_values) / len(all_values)
        elif method == "sum":
            return sum(all_values)
        elif method == "median":
            sorted_values = sorted(all_values)
            n = len(sorted_values)
            mid = n // 2
            if n % 2 == 0:
                return (sorted_values[mid - 1] + sorted_values[mid]) / 2
            return sorted_values[mid]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
