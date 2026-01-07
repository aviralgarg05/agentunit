"""Multi-agent coordination and collaboration metrics.

This module provides metrics specifically designed for evaluating
multi-agent system performance beyond individual agent outcomes.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentunit.metrics.base import Metric, MetricResult


if TYPE_CHECKING:
    from agentunit.core.trace import TraceLog
    from agentunit.datasets.base import DatasetCase
    from agentunit.multiagent import (
        AgentInteraction,
        AgentRole,
        ConflictEvent,
        HandoffEvent,
    )

logger = logging.getLogger(__name__)


@dataclass
class CoordinationMetrics:
    """Aggregated coordination metrics for multi-agent evaluations.

    Attributes:
        handoff_success_rate: Ratio of successful handoffs
        avg_handoff_time: Average time for task handoffs (seconds)
        conflict_rate: Conflicts per 100 interactions
        conflict_resolution_rate: Ratio of resolved conflicts
        avg_resolution_time: Average conflict resolution time (seconds)
        communication_efficiency: Messages per successful outcome
        role_adherence: How well agents follow assigned roles
        load_balance_score: Evenness of work distribution
    """
    handoff_success_rate: float = 0.0
    avg_handoff_time: float = 0.0
    conflict_rate: float = 0.0
    conflict_resolution_rate: float = 0.0
    avg_resolution_time: float = 0.0
    communication_efficiency: float = 0.0
    role_adherence: float = 0.0
    load_balance_score: float = 0.0


@dataclass
class NetworkMetrics:
    """Network topology metrics for agent communication graphs.

    Attributes:
        density: Ratio of actual to possible connections
        centralization: How centralized communication is
        avg_path_length: Average shortest path between agents
        clustering_coefficient: Tendency to form clusters
        hub_agents: Agents with high connectivity
        bottleneck_agents: Agents that are communication bottlenecks
    """
    density: float = 0.0
    centralization: float = 0.0
    avg_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    hub_agents: list[str] = field(default_factory=list)
    bottleneck_agents: list[str] = field(default_factory=list)


@dataclass
class EmergentBehaviorMetrics:
    """Metrics for detecting emergent behaviors in multi-agent systems.

    Attributes:
        self_organization_score: Degree of spontaneous organization
        specialization_emergence: Whether agents developed specializations
        collective_decision_score: Quality of group decisions
        adaptation_rate: How quickly system adapts to changes
        swarm_intelligence_score: Collective problem-solving capability
    """
    self_organization_score: float = 0.0
    specialization_emergence: float = 0.0
    collective_decision_score: float = 0.0
    adaptation_rate: float = 0.0
    swarm_intelligence_score: float = 0.0


class InteractionAnalyzer:
    """Analyzes patterns in agent interactions."""

    def __init__(
        self,
        interactions: list[AgentInteraction],
        handoffs: list[HandoffEvent],
        conflicts: list[ConflictEvent],
    ):
        self.interactions = interactions
        self.handoffs = handoffs
        self.conflicts = conflicts

        # Build interaction graph
        self._build_graphs()

    def _build_graphs(self) -> None:
        """Build adjacency structures from interactions."""
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        self.message_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.agent_activity: dict[str, int] = defaultdict(int)

        for interaction in self.interactions:
            from_agent = interaction.from_agent
            to_agents = (
                interaction.to_agent
                if isinstance(interaction.to_agent, list)
                else [interaction.to_agent]
            )

            for to_agent in to_agents:
                self.adjacency[from_agent].add(to_agent)
                self.message_counts[(from_agent, to_agent)] += 1
                self.agent_activity[from_agent] += 1
                self.agent_activity[to_agent] += 1

    def calculate_handoff_metrics(self) -> dict[str, float]:
        """Calculate handoff-related metrics."""
        if not self.handoffs:
            return {
                "success_rate": 1.0,  # No handoffs = no failures
                "avg_time": 0.0,
                "total_handoffs": 0,
            }

        successful = sum(1 for h in self.handoffs if h.success)
        total_time = sum(h.handoff_time for h in self.handoffs)

        return {
            "success_rate": successful / len(self.handoffs),
            "avg_time": total_time / len(self.handoffs),
            "total_handoffs": len(self.handoffs),
        }

    def calculate_conflict_metrics(self) -> dict[str, float]:
        """Calculate conflict-related metrics."""
        if not self.conflicts:
            return {
                "rate": 0.0,
                "resolution_rate": 1.0,
                "avg_resolution_time": 0.0,
                "total_conflicts": 0,
            }

        interactions_count = max(len(self.interactions), 1)
        resolved = sum(1 for c in self.conflicts if c.resolved)
        resolution_times = [c.resolution_time for c in self.conflicts if c.resolved]
        avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0.0

        return {
            "rate": (len(self.conflicts) / interactions_count) * 100,
            "resolution_rate": resolved / len(self.conflicts),
            "avg_resolution_time": avg_resolution,
            "total_conflicts": len(self.conflicts),
        }

    def calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency score.

        Efficiency is based on:
        - Message success rate
        - Response time
        - Redundant message ratio
        """
        if not self.interactions:
            return 1.0

        successful = sum(1 for i in self.interactions if i.success)
        success_rate = successful / len(self.interactions)

        # Calculate response time efficiency (normalize to 0-1)
        response_times = [i.response_time for i in self.interactions if i.response_time > 0]
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            # Assume 5 seconds is "slow", normalize inversely
            response_efficiency = min(1.0, 5.0 / avg_response) if avg_response > 0 else 1.0
        else:
            response_efficiency = 1.0

        # Calculate redundancy (same sender-receiver pairs with similar messages)
        pair_counts = list(self.message_counts.values())
        if pair_counts:
            max_count = max(pair_counts)
            avg_count = sum(pair_counts) / len(pair_counts)
            redundancy_score = min(1.0, avg_count / max_count) if max_count > 0 else 1.0
        else:
            redundancy_score = 1.0

        return 0.5 * success_rate + 0.3 * response_efficiency + 0.2 * redundancy_score

    def calculate_load_balance(self) -> float:
        """Calculate how evenly work is distributed across agents.

        Returns score from 0 (highly imbalanced) to 1 (perfectly balanced).
        Uses coefficient of variation (lower is better).
        """
        if not self.agent_activity:
            return 1.0

        activities = list(self.agent_activity.values())
        mean_activity = sum(activities) / len(activities)

        if mean_activity == 0:
            return 1.0

        variance = sum((a - mean_activity) ** 2 for a in activities) / len(activities)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_activity  # Coefficient of variation

        # Convert CV to 0-1 score (CV of 0 = perfect balance = 1.0)
        # CV of 1+ = highly imbalanced = 0.0
        return max(0.0, 1.0 - cv)


class NetworkAnalyzer:
    """Analyzes network topology of agent communication."""

    def __init__(self, adjacency: dict[str, set[str]], agents: list[str]):
        self.adjacency = adjacency
        self.agents = agents
        self.n = len(agents)

    def calculate_density(self) -> float:
        """Calculate network density (edges / possible edges)."""
        if self.n <= 1:
            return 1.0

        possible_edges = self.n * (self.n - 1)  # Directed graph
        actual_edges = sum(len(neighbors) for neighbors in self.adjacency.values())

        return actual_edges / possible_edges if possible_edges > 0 else 0.0

    def calculate_centralization(self) -> float:
        """Calculate degree centralization.

        High centralization means communication flows through few agents.
        """
        if self.n <= 2:
            return 0.0

        # Calculate out-degree for each agent
        degrees = [len(self.adjacency.get(agent, set())) for agent in self.agents]
        max_degree = max(degrees) if degrees else 0
        sum_diff = sum(max_degree - d for d in degrees)

        # Maximum possible sum of differences (star graph)
        max_sum_diff = (self.n - 1) * (self.n - 2)

        return sum_diff / max_sum_diff if max_sum_diff > 0 else 0.0

    def calculate_clustering_coefficient(self) -> float:
        """Calculate average clustering coefficient.

        Measures tendency of agents to cluster together.
        """
        if self.n <= 2:
            return 0.0

        coefficients = []
        for agent in self.agents:
            neighbors = self.adjacency.get(agent, set())
            k = len(neighbors)

            if k <= 1:
                coefficients.append(0.0)
                continue

            # Count edges between neighbors
            edges_between = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in self.adjacency.get(n1, set()):
                        edges_between += 1

            # Clustering coefficient for this node
            possible = k * (k - 1)
            coefficients.append(edges_between / possible if possible > 0 else 0.0)

        return sum(coefficients) / len(coefficients) if coefficients else 0.0

    def find_hub_agents(self, threshold: float = 0.7) -> list[str]:
        """Find agents with high connectivity (hubs).

        Args:
            threshold: Percentile threshold for hub detection
        """
        if not self.agents:
            return []

        degrees = [(agent, len(self.adjacency.get(agent, set()))) for agent in self.agents]
        if not degrees:
            return []

        max_degree = max(d for _, d in degrees)
        if max_degree == 0:
            return []

        threshold_degree = threshold * max_degree
        return [agent for agent, degree in degrees if degree >= threshold_degree]

    def find_bottleneck_agents(self) -> list[str]:
        """Find agents that are communication bottlenecks.

        Bottlenecks are agents that many paths go through.
        Uses simplified betweenness centrality estimation.
        """
        if self.n <= 2:
            return []

        # Count how many unique pairs communicate through each agent
        betweenness: dict[str, int] = defaultdict(int)

        for agent in self.agents:
            # Simple heuristic: agents with both high in-degree and out-degree
            # that connect different "communities"
            in_degree = sum(
                1 for other in self.agents
                if agent in self.adjacency.get(other, set())
            )
            out_degree = len(self.adjacency.get(agent, set()))
            betweenness[agent] = in_degree * out_degree

        if not betweenness:
            return []

        max_betweenness = max(betweenness.values())
        if max_betweenness == 0:
            return []

        # Agents with betweenness > 0.5 * max are potential bottlenecks
        threshold = 0.5 * max_betweenness
        return [agent for agent, bc in betweenness.items() if bc >= threshold]

    def calculate_fault_tolerance(self) -> float:
        """Calculate fault tolerance based on network redundancy.

        Higher scores mean the network can tolerate more agent failures.
        """
        if self.n <= 1:
            return 1.0

        density = self.calculate_density()

        # Check for single points of failure
        bottlenecks = self.find_bottleneck_agents()
        bottleneck_ratio = len(bottlenecks) / self.n if self.n > 0 else 0

        # Fault tolerance is higher with:
        # - Higher density (more connections)
        # - Fewer bottlenecks
        return 0.6 * density + 0.4 * (1 - bottleneck_ratio)


class EmergentBehaviorDetector:
    """Detects emergent behaviors in multi-agent systems."""

    def __init__(
        self,
        interactions: list[AgentInteraction],
        agent_roles: dict[str, AgentRole],
        task_allocations: list[Any] | None = None,
    ):
        self.interactions = interactions
        self.agent_roles = agent_roles
        self.task_allocations = task_allocations or []

    def detect_self_organization(self) -> float:
        """Detect spontaneous organization patterns.

        Self-organization is indicated by:
        - Emergence of communication hierarchies
        - Consistent interaction patterns
        - Reduced randomness over time
        """
        if len(self.interactions) < 10:
            return 0.0

        # Split interactions into early and late phases
        mid = len(self.interactions) // 2
        early = self.interactions[:mid]
        late = self.interactions[mid:]

        # Calculate pattern consistency in each phase
        early_patterns = self._calculate_pattern_entropy(early)
        late_patterns = self._calculate_pattern_entropy(late)

        # Self-organization = reduction in entropy (more structured)
        if early_patterns == 0:
            return 0.5  # No baseline to compare

        entropy_reduction = (early_patterns - late_patterns) / early_patterns
        return max(0.0, min(1.0, 0.5 + entropy_reduction))

    def _calculate_pattern_entropy(self, interactions: list[AgentInteraction]) -> float:
        """Calculate entropy of interaction patterns."""
        if not interactions:
            return 0.0

        # Count interaction type frequencies
        type_counts: dict[str, int] = defaultdict(int)
        for interaction in interactions:
            key = f"{interaction.from_agent}->{interaction.interaction_type.value}"
            type_counts[key] += 1

        total = sum(type_counts.values())
        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def detect_specialization(self) -> float:
        """Detect if agents developed task specializations.

        Specialization is indicated by:
        - Agents consistently handling certain message types
        - Division of labor emerging
        """
        if not self.interactions:
            return 0.0

        # Track what types of interactions each agent handles
        agent_specializations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for interaction in self.interactions:
            msg_type = interaction.message_type
            agent_specializations[interaction.from_agent][msg_type] += 1

        if not agent_specializations:
            return 0.0

        # Calculate specialization score for each agent
        # (how focused they are on specific message types)
        specialization_scores = []
        for type_counts in agent_specializations.values():
            total = sum(type_counts.values())
            if total == 0:
                continue

            # Herfindahl-Hirschman Index (HHI) for concentration
            hhi = sum((count / total) ** 2 for count in type_counts.values())
            specialization_scores.append(hhi)

        return sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0

    def detect_collective_decision_making(self) -> float:
        """Detect collective decision-making patterns.

        Indicated by:
        - Negotiation interactions
        - Consensus-building patterns
        - Collaborative problem-solving
        """
        if not self.interactions:
            return 0.0

        # Count collaborative interaction types
        from agentunit.multiagent import InteractionType

        collaborative_types = {
            InteractionType.COLLABORATION,
            InteractionType.NEGOTIATION,
        }

        collaborative_count = sum(
            1 for i in self.interactions
            if i.interaction_type in collaborative_types
        )

        return collaborative_count / len(self.interactions)

    def detect_adaptation_rate(self) -> float:
        """Detect how quickly the system adapts to changes.

        Based on response time trends over the session.
        Faster adaptation = response times decrease over time.
        """
        if len(self.interactions) < 5:
            return 0.5  # Not enough data

        response_times = [
            i.response_time for i in self.interactions
            if i.response_time > 0
        ]

        if len(response_times) < 5:
            return 0.5

        # Compare early vs late response times
        mid = len(response_times) // 2
        early_avg = sum(response_times[:mid]) / mid
        late_avg = sum(response_times[mid:]) / (len(response_times) - mid)

        if early_avg == 0:
            return 0.5

        # Improvement ratio
        improvement = (early_avg - late_avg) / early_avg
        return max(0.0, min(1.0, 0.5 + improvement))

    def calculate_swarm_intelligence(self) -> float:
        """Calculate swarm intelligence score.

        Combines multiple emergent behavior indicators.
        """
        self_org = self.detect_self_organization()
        specialization = self.detect_specialization()
        collective = self.detect_collective_decision_making()
        adaptation = self.detect_adaptation_rate()

        # Weighted combination
        return (
            0.3 * self_org +
            0.25 * specialization +
            0.25 * collective +
            0.2 * adaptation
        )


class MultiAgentMetricsCalculator:
    """Main calculator for multi-agent system metrics."""

    def __init__(
        self,
        interactions: list[AgentInteraction],
        handoffs: list[HandoffEvent],
        conflicts: list[ConflictEvent],
        agent_roles: dict[str, AgentRole],
        task_allocations: list[Any] | None = None,
    ):
        self.interactions = interactions
        self.handoffs = handoffs
        self.conflicts = conflicts
        self.agent_roles = agent_roles
        self.task_allocations = task_allocations or []

        # Initialize analyzers
        self.interaction_analyzer = InteractionAnalyzer(interactions, handoffs, conflicts)
        self.emergent_detector = EmergentBehaviorDetector(
            interactions, agent_roles, task_allocations
        )

        # Build network analyzer
        agents = list(agent_roles.keys())
        self.network_analyzer = NetworkAnalyzer(
            self.interaction_analyzer.adjacency, agents
        )

    def calculate_coordination_metrics(self) -> CoordinationMetrics:
        """Calculate all coordination metrics."""
        handoff_metrics = self.interaction_analyzer.calculate_handoff_metrics()
        conflict_metrics = self.interaction_analyzer.calculate_conflict_metrics()

        return CoordinationMetrics(
            handoff_success_rate=handoff_metrics["success_rate"],
            avg_handoff_time=handoff_metrics["avg_time"],
            conflict_rate=conflict_metrics["rate"],
            conflict_resolution_rate=conflict_metrics["resolution_rate"],
            avg_resolution_time=conflict_metrics["avg_resolution_time"],
            communication_efficiency=self.interaction_analyzer.calculate_communication_efficiency(),
            role_adherence=self._calculate_role_adherence(),
            load_balance_score=self.interaction_analyzer.calculate_load_balance(),
        )

    def calculate_network_metrics(self) -> NetworkMetrics:
        """Calculate network topology metrics."""
        return NetworkMetrics(
            density=self.network_analyzer.calculate_density(),
            centralization=self.network_analyzer.calculate_centralization(),
            clustering_coefficient=self.network_analyzer.calculate_clustering_coefficient(),
            hub_agents=self.network_analyzer.find_hub_agents(),
            bottleneck_agents=self.network_analyzer.find_bottleneck_agents(),
        )

    def calculate_emergent_behavior_metrics(self) -> EmergentBehaviorMetrics:
        """Calculate emergent behavior metrics."""
        return EmergentBehaviorMetrics(
            self_organization_score=self.emergent_detector.detect_self_organization(),
            specialization_emergence=self.emergent_detector.detect_specialization(),
            collective_decision_score=self.emergent_detector.detect_collective_decision_making(),
            adaptation_rate=self.emergent_detector.detect_adaptation_rate(),
            swarm_intelligence_score=self.emergent_detector.calculate_swarm_intelligence(),
        )

    def _calculate_role_adherence(self) -> float:
        """Calculate how well agents follow their assigned roles.

        Based on matching interaction types to role responsibilities.
        """
        if not self.interactions or not self.agent_roles:
            return 1.0

        adherent_count = 0
        total_count = 0

        for interaction in self.interactions:
            from_agent = interaction.from_agent
            if from_agent not in self.agent_roles:
                continue

            role = self.agent_roles[from_agent]
            total_count += 1

            # Check if interaction type aligns with role capabilities
            interaction_type = interaction.interaction_type.value

            # Simple heuristic: DELEGATION requires can_delegate
            if interaction_type == "delegation" and not role.can_delegate:
                continue

            # Higher authority agents should handle more RESPONSE types
            if (interaction_type == "response" and role.authority_level >= 5) or interaction_type != "response":
                adherent_count += 1

        return adherent_count / total_count if total_count > 0 else 1.0

    def calculate_all(self) -> dict[str, Any]:
        """Calculate all multi-agent metrics.

        Returns:
            Dictionary containing all metric categories
        """
        coordination = self.calculate_coordination_metrics()
        network = self.calculate_network_metrics()
        emergent = self.calculate_emergent_behavior_metrics()

        return {
            "coordination": {
                "handoff_success_rate": coordination.handoff_success_rate,
                "avg_handoff_time": coordination.avg_handoff_time,
                "conflict_rate": coordination.conflict_rate,
                "conflict_resolution_rate": coordination.conflict_resolution_rate,
                "avg_resolution_time": coordination.avg_resolution_time,
                "communication_efficiency": coordination.communication_efficiency,
                "role_adherence": coordination.role_adherence,
                "load_balance_score": coordination.load_balance_score,
            },
            "network": {
                "density": network.density,
                "centralization": network.centralization,
                "clustering_coefficient": network.clustering_coefficient,
                "hub_agents": network.hub_agents,
                "bottleneck_agents": network.bottleneck_agents,
            },
            "emergent_behaviors": {
                "self_organization_score": emergent.self_organization_score,
                "specialization_emergence": emergent.specialization_emergence,
                "collective_decision_score": emergent.collective_decision_score,
                "adaptation_rate": emergent.adaptation_rate,
                "swarm_intelligence_score": emergent.swarm_intelligence_score,
            },
            "fault_tolerance": self.network_analyzer.calculate_fault_tolerance(),
        }


# Metric classes for integration with AgentUnit metrics system

class CoordinationEfficiencyMetric(Metric):
    """Metric for overall coordination efficiency in multi-agent systems."""

    name = "coordination_efficiency"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate coordination efficiency."""
        # Extract multi-agent data from trace
        interactions = getattr(trace, "interactions", [])
        handoffs = getattr(trace, "handoffs", [])
        conflicts = getattr(trace, "conflicts", [])
        agent_roles = getattr(trace, "agent_roles", {})

        if not interactions:
            return MetricResult(
                name=self.name,
                value=None,
                detail={"error": "No interaction data in trace"}
            )

        calculator = MultiAgentMetricsCalculator(
            interactions, handoffs, conflicts, agent_roles
        )

        metrics = calculator.calculate_coordination_metrics()

        # Combined score
        score = (
            0.25 * metrics.handoff_success_rate +
            0.20 * metrics.conflict_resolution_rate +
            0.25 * metrics.communication_efficiency +
            0.15 * metrics.role_adherence +
            0.15 * metrics.load_balance_score
        )

        return MetricResult(
            name=self.name,
            value=score,
            detail={
                "handoff_success_rate": metrics.handoff_success_rate,
                "conflict_resolution_rate": metrics.conflict_resolution_rate,
                "communication_efficiency": metrics.communication_efficiency,
                "role_adherence": metrics.role_adherence,
                "load_balance_score": metrics.load_balance_score,
            }
        )


class SwarmIntelligenceMetric(Metric):
    """Metric for emergent swarm intelligence behaviors."""

    name = "swarm_intelligence"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate swarm intelligence indicators."""
        interactions = getattr(trace, "interactions", [])
        agent_roles = getattr(trace, "agent_roles", {})

        if not interactions:
            return MetricResult(
                name=self.name,
                value=None,
                detail={"error": "No interaction data in trace"}
            )

        detector = EmergentBehaviorDetector(interactions, agent_roles)

        metrics = EmergentBehaviorMetrics(
            self_organization_score=detector.detect_self_organization(),
            specialization_emergence=detector.detect_specialization(),
            collective_decision_score=detector.detect_collective_decision_making(),
            adaptation_rate=detector.detect_adaptation_rate(),
            swarm_intelligence_score=detector.calculate_swarm_intelligence(),
        )

        return MetricResult(
            name=self.name,
            value=metrics.swarm_intelligence_score,
            detail={
                "self_organization": metrics.self_organization_score,
                "specialization": metrics.specialization_emergence,
                "collective_decision": metrics.collective_decision_score,
                "adaptation_rate": metrics.adaptation_rate,
            }
        )


class NetworkFaultToleranceMetric(Metric):
    """Metric for network fault tolerance in multi-agent systems."""

    name = "network_fault_tolerance"

    def evaluate(self, case: DatasetCase, trace: TraceLog, outcome: Any) -> MetricResult:
        """Evaluate network fault tolerance."""
        interactions = getattr(trace, "interactions", [])
        agent_roles = getattr(trace, "agent_roles", {})

        if not interactions or not agent_roles:
            return MetricResult(
                name=self.name,
                value=None,
                detail={"error": "Insufficient data for network analysis"}
            )

        # Build adjacency from interactions
        adjacency: dict[str, set[str]] = defaultdict(set)
        for interaction in interactions:
            to_agents = (
                interaction.to_agent
                if isinstance(interaction.to_agent, list)
                else [interaction.to_agent]
            )
            for to_agent in to_agents:
                adjacency[interaction.from_agent].add(to_agent)

        agents = list(agent_roles.keys())
        analyzer = NetworkAnalyzer(adjacency, agents)

        fault_tolerance = analyzer.calculate_fault_tolerance()
        network_metrics = NetworkMetrics(
            density=analyzer.calculate_density(),
            centralization=analyzer.calculate_centralization(),
            clustering_coefficient=analyzer.calculate_clustering_coefficient(),
            hub_agents=analyzer.find_hub_agents(),
            bottleneck_agents=analyzer.find_bottleneck_agents(),
        )

        return MetricResult(
            name=self.name,
            value=fault_tolerance,
            detail={
                "density": network_metrics.density,
                "centralization": network_metrics.centralization,
                "clustering": network_metrics.clustering_coefficient,
                "hub_agents": network_metrics.hub_agents,
                "bottleneck_count": len(network_metrics.bottleneck_agents),
            }
        )


__all__ = [
    # Metric classes
    "CoordinationEfficiencyMetric",
    # Data classes
    "CoordinationMetrics",
    "EmergentBehaviorDetector",
    "EmergentBehaviorMetrics",
    # Analyzers
    "InteractionAnalyzer",
    # Main calculator
    "MultiAgentMetricsCalculator",
    "NetworkAnalyzer",
    "NetworkFaultToleranceMetric",
    "NetworkMetrics",
    "SwarmIntelligenceMetric",
]
