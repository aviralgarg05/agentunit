"""Sustainability metrics for agent evaluations."""

from typing import Any, Dict
from agentunit.metrics.base import Metric, MetricResult
from agentunit.datasets.base import DatasetCase
from agentunit.core.trace import TraceLog


class EnergyMetric:
    """Metric for measuring energy consumption.
    
    Tracks energy usage during agent execution and scores
    based on efficiency thresholds.
    """
    
    def __init__(
        self,
        threshold_kwh: float = 0.1,
        sample_interval: float = 1.0
    ):
        """Initialize energy metric.
        
        Args:
            threshold_kwh: Energy threshold for scoring (kWh)
            sample_interval: Sampling interval in seconds
        """
        self.name = "energy_consumption"
        self.threshold_kwh = threshold_kwh
        self.sample_interval = sample_interval
    
    def evaluate(
        self,
        case: DatasetCase,
        trace: TraceLog,
        outcome: Any
    ) -> MetricResult:
        """Evaluate energy consumption.
        
        Args:
            case: Test case
            trace: Execution trace
            outcome: Execution outcome
        
        Returns:
            MetricResult with energy score
        """
        # Extract energy from trace if available
        energy_kwh = 0.0
        
        if hasattr(trace, 'metadata') and 'energy_kwh' in trace.metadata:
            energy_kwh = trace.metadata['energy_kwh']
        elif isinstance(trace, dict) and 'energy_kwh' in trace:
            energy_kwh = trace['energy_kwh']
        
        # Score: 1.0 if under threshold, scaled down if over
        score = min(1.0, self.threshold_kwh / energy_kwh) if energy_kwh > 0 else 1.0
        
        return MetricResult(
            name=self.name,
            value=score,
            detail={
                "energy_kwh": energy_kwh,
                "threshold_kwh": self.threshold_kwh,
                "under_threshold": energy_kwh <= self.threshold_kwh
            }
        )


class CarbonMetric:
    """Metric for measuring carbon emissions.
    
    Tracks carbon footprint and scores based on
    emissions targets.
    """
    
    def __init__(
        self,
        threshold_kg: float = 0.05,
        grid_intensity: float = 0.475
    ):
        """Initialize carbon metric.
        
        Args:
            threshold_kg: Carbon threshold for scoring (kg CO2eq)
            grid_intensity: Grid carbon intensity (kg CO2/kWh)
        """
        self.name = "carbon_emissions"
        self.threshold_kg = threshold_kg
        self.grid_intensity = grid_intensity
    
    def evaluate(
        self,
        case: DatasetCase,
        trace: TraceLog,
        outcome: Any
    ) -> MetricResult:
        """Evaluate carbon emissions.
        
        Args:
            case: Test case
            trace: Execution trace
            outcome: Execution outcome
        
        Returns:
            MetricResult with carbon score
        """
        # Extract carbon from trace
        carbon_kg = 0.0
        
        if hasattr(trace, 'metadata') and 'carbon_kg' in trace.metadata:
            carbon_kg = trace.metadata['carbon_kg']
        elif isinstance(trace, dict) and 'carbon_kg' in trace:
            carbon_kg = trace['carbon_kg']
        
        # Score: 1.0 if under threshold, scaled down if over
        score = min(1.0, self.threshold_kg / carbon_kg) if carbon_kg > 0 else 1.0
        
        # Calculate equivalents
        km_driven = carbon_kg * 4.6  # Avg car: 0.22 kg CO2/km
        trees_needed = carbon_kg / 21  # Avg tree absorbs 21 kg CO2/year
        
        return MetricResult(
            name=self.name,
            value=score,
            detail={
                "carbon_kg": carbon_kg,
                "threshold_kg": self.threshold_kg,
                "under_threshold": carbon_kg <= self.threshold_kg,
                "equivalents": {
                    "km_driven": km_driven,
                    "trees_needed": trees_needed
                }
            }
        )


class ResourceUtilizationMetric:
    """Metric for measuring resource utilization efficiency.
    
    Evaluates CPU, GPU, and memory usage patterns to assess
    efficiency of agent execution.
    """
    
    def __init__(
        self,
        target_cpu_percent: float = 80.0,
        target_memory_gb: float = 4.0,
        target_gpu_percent: float = 80.0
    ):
        """Initialize resource utilization metric.
        
        Args:
            target_cpu_percent: Target CPU utilization (%)
            target_memory_gb: Target memory usage (GB)
            target_gpu_percent: Target GPU utilization (%)
        """
        self.name = "resource_utilization"
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_gb = target_memory_gb
        self.target_gpu_percent = target_gpu_percent
    
    def evaluate(
        self,
        case: DatasetCase,
        trace: TraceLog,
        outcome: Any
    ) -> MetricResult:
        """Evaluate resource utilization.
        
        Args:
            case: Test case
            trace: Execution trace
            outcome: Execution outcome
        
        Returns:
            MetricResult with utilization score
        """
        # Extract metrics from trace
        cpu_percent = 0.0
        memory_gb = 0.0
        gpu_percent = 0.0
        
        if hasattr(trace, 'metadata'):
            cpu_percent = trace.metadata.get('cpu_percent', 0.0)
            memory_gb = trace.metadata.get('memory_mb', 0.0) / 1024
            gpu_percent = trace.metadata.get('gpu_percent', 0.0)
        elif isinstance(trace, dict):
            cpu_percent = trace.get('cpu_percent', 0.0)
            memory_gb = trace.get('memory_mb', 0.0) / 1024
            gpu_percent = trace.get('gpu_percent', 0.0)
        
        # Score each resource (penalty for over/under utilization)
        cpu_score = self._score_utilization(cpu_percent, self.target_cpu_percent)
        memory_score = self._score_utilization(
            memory_gb, 
            self.target_memory_gb,
            is_absolute=True
        )
        gpu_score = self._score_utilization(gpu_percent, self.target_gpu_percent)
        
        # Combined score (weighted average)
        score = (cpu_score + memory_score + gpu_score) / 3
        
        return MetricResult(
            name=self.name,
            value=score,
            detail={
                "cpu": {
                    "used_percent": cpu_percent,
                    "target_percent": self.target_cpu_percent,
                    "score": cpu_score
                },
                "memory": {
                    "used_gb": memory_gb,
                    "target_gb": self.target_memory_gb,
                    "score": memory_score
                },
                "gpu": {
                    "used_percent": gpu_percent,
                    "target_percent": self.target_gpu_percent,
                    "score": gpu_score
                }
            }
        )
    
    def _score_utilization(
        self,
        actual: float,
        target: float,
        is_absolute: bool = False
    ) -> float:
        """Score resource utilization.
        
        Args:
            actual: Actual usage
            target: Target usage
            is_absolute: Whether values are absolute (vs percentage)
        
        Returns:
            Score from 0.0 to 1.0
        """
        if target == 0:
            return 1.0 if actual == 0 else 0.0
        
        ratio = actual / target
        
        # Penalty for over-utilization (>1.2x target)
        if ratio > 1.2:
            return max(0.0, 1.0 - (ratio - 1.0))
        
        # Penalty for under-utilization (<0.5x target)
        if ratio < 0.5:
            return max(0.0, ratio / 0.5)
        
        # Optimal range: 0.5x to 1.2x target
        return 1.0
