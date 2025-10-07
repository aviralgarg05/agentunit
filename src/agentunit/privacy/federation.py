"""Federated evaluation support."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agentunit.datasets.base import DatasetCase
from agentunit.privacy.wrappers import PrivacyConfig, SecureAggregator


@dataclass
class FederatedNode:
    """Represents a node in federated evaluation."""
    
    node_id: str
    dataset: List[DatasetCase]
    privacy_config: Optional[PrivacyConfig] = None
    # Note: adapter reference removed to avoid circular dependencies
    # Users should maintain their own adapter instances


class PrivacyGuard:
    """Guards privacy during federated evaluation.
    
    Enforces:
    - Minimum batch sizes
    - Query rate limits
    - Privacy budget constraints
    """
    
    def __init__(
        self,
        min_batch_size: int = 10,
        max_queries_per_hour: int = 1000,
        total_epsilon_budget: float = 10.0
    ):
        """Initialize privacy guard.
        
        Args:
            min_batch_size: Minimum batch size for queries
            max_queries_per_hour: Max queries per hour
            total_epsilon_budget: Total privacy budget
        """
        self.min_batch_size = min_batch_size
        self.max_queries_per_hour = max_queries_per_hour
        self.total_epsilon_budget = total_epsilon_budget
        self.epsilon_spent = 0.0
        self.query_count = 0
    
    def check_batch_size(self, batch_size: int) -> bool:
        """Check if batch size meets minimum.
        
        Args:
            batch_size: Size of batch
            
        Returns:
            Whether batch size is acceptable
        """
        return batch_size >= self.min_batch_size
    
    def check_query_limit(self) -> bool:
        """Check if query limit is exceeded.
        
        Returns:
            Whether more queries are allowed
        """
        return self.query_count < self.max_queries_per_hour
    
    def check_privacy_budget(self, epsilon: float) -> bool:
        """Check if privacy budget allows operation.
        
        Args:
            epsilon: Epsilon to spend
            
        Returns:
            Whether budget allows operation
        """
        return (self.epsilon_spent + epsilon) <= self.total_epsilon_budget
    
    def record_query(self, epsilon: float = 0.0):
        """Record a query and update budgets.
        
        Args:
            epsilon: Privacy cost of query
        """
        self.query_count += 1
        self.epsilon_spent += epsilon
    
    def reset_hourly_limit(self):
        """Reset hourly query counter."""
        self.query_count = 0


class FederatedEvaluator:
    """Evaluator for federated/distributed evaluation.
    
    Enables privacy-preserving evaluation across multiple
    nodes without sharing raw data.
    """
    
    def __init__(
        self,
        nodes: List[FederatedNode],
        aggregation_method: str = "secure_mean",
        privacy_guard: Optional[PrivacyGuard] = None
    ):
        """Initialize federated evaluator.
        
        Args:
            nodes: List of federated nodes
            aggregation_method: How to aggregate results
            privacy_guard: Privacy enforcement
        """
        self.nodes = nodes
        self.aggregation_method = aggregation_method
        self.privacy_guard = privacy_guard or PrivacyGuard()
        self.aggregator = SecureAggregator(num_parties=len(nodes))
    
    def evaluate_federated(
        self,
        metric_fn: Callable[[DatasetCase, str], float],
        query_fn: Callable[[FederatedNode, DatasetCase], str]
    ) -> Dict[str, Any]:
        """Perform federated evaluation.
        
        Args:
            metric_fn: Metric function to apply
            query_fn: Function to query node adapter
            
        Returns:
            Aggregated results
        """
        # Check privacy constraints
        total_cases = sum(len(node.dataset) for node in self.nodes)
        if not self.privacy_guard.check_batch_size(total_cases):
            raise ValueError(
                f"Total cases {total_cases} below minimum batch size "
                f"{self.privacy_guard.min_batch_size}"
            )
        
        # Evaluate on each node
        node_results = []
        for node_id, node in enumerate(self.nodes):
            node_scores = []
            
            for case in node.dataset:
                # Check query limit
                if not self.privacy_guard.check_query_limit():
                    break
                
                # Get response from node using provided query function
                response = query_fn(node, case)
                
                # Calculate metric
                score = metric_fn(case, response)
                node_scores.append(score)
                
                # Record query
                epsilon = node.privacy_config.epsilon if node.privacy_config else 0.1
                if not self.privacy_guard.check_privacy_budget(epsilon):
                    break
                self.privacy_guard.record_query(epsilon)
            
            # Add node's average to secure aggregator
            if node_scores:
                node_avg = sum(node_scores) / len(node_scores)
                noise_scale = 1.0 / node.privacy_config.epsilon if node.privacy_config else 1.0
                self.aggregator.add_share(node_id, node_avg, noise_scale)
                node_results.append({
                    "node_id": node.node_id,
                    "avg_score": node_avg,
                    "num_cases": len(node_scores)
                })
        
        # Securely aggregate results
        global_score = self.aggregator.aggregate(method="mean")
        
        return {
            "global_score": global_score,
            "num_nodes": len(node_results),
            "total_cases": sum(r["num_cases"] for r in node_results),
            "epsilon_spent": self.privacy_guard.epsilon_spent,
            "node_results": node_results,
        }
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy usage report.
        
        Returns:
            Privacy metrics
        """
        return {
            "epsilon_spent": self.privacy_guard.epsilon_spent,
            "epsilon_budget": self.privacy_guard.total_epsilon_budget,
            "budget_utilization": (
                self.privacy_guard.epsilon_spent / 
                self.privacy_guard.total_epsilon_budget
            ),
            "queries_made": self.privacy_guard.query_count,
            "query_limit": self.privacy_guard.max_queries_per_hour,
        }
