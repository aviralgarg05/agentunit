"""Resource tracking for agent evaluations."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class ResourceMetrics:
    """Resource usage metrics.

    Attributes:
        cpu_percent: CPU utilization percentage
        memory_mb: Memory usage in MB
        gpu_percent: GPU utilization percentage (if available)
        gpu_memory_mb: GPU memory usage in MB (if available)
        energy_kwh: Energy consumed in kWh
        carbon_kg: Carbon emissions in kg CO2eq
        duration_seconds: Measurement duration in seconds
        timestamp: Measurement timestamp
    """

    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    energy_kwh: float = 0.0
    carbon_kg: float = 0.0
    duration_seconds: float = 0.0
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ResourceTracker:
    """Track resource usage during agent evaluations.

    This tracker monitors:
    - CPU and memory usage
    - GPU utilization (if available)
    - Energy consumption estimates
    - Carbon emissions estimates
    """

    def __init__(self, sample_interval: float = 1.0, enable_gpu: bool = True):
        """Initialize resource tracker.

        Args:
            sample_interval: Sampling interval in seconds
            enable_gpu: Whether to track GPU metrics
        """
        self.sample_interval = sample_interval
        self.enable_gpu = enable_gpu

        self.start_time: float | None = None
        self.samples: list[ResourceMetrics] = []

        # Check GPU availability
        self.has_gpu = False
        if enable_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.has_gpu = True
                self.pynvml = pynvml
            except (ImportError, Exception):
                pass

    def start(self):
        """Start tracking resources."""
        self.start_time = time.time()
        self.samples = []

    def sample(self) -> ResourceMetrics:
        """Take a resource usage sample.

        Returns:
            ResourceMetrics snapshot
        """
        metrics = ResourceMetrics()

        if HAS_PSUTIL:
            # CPU and memory
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            metrics.memory_mb = memory.used / (1024 * 1024)

        # GPU metrics (if available)
        if self.has_gpu:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_percent = util.gpu

                memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_mb = memory_info.used / (1024 * 1024)
            except Exception:
                pass

        # Duration
        if self.start_time:
            metrics.duration_seconds = time.time() - self.start_time

        # Energy estimate (simplified)
        # Real implementation would use CodeCarbon or similar
        metrics.energy_kwh = self._estimate_energy(metrics)

        # Carbon estimate
        # Using average grid intensity: 0.475 kg CO2/kWh (global average)
        metrics.carbon_kg = metrics.energy_kwh * 0.475

        self.samples.append(metrics)
        return metrics

    def stop(self) -> ResourceMetrics:
        """Stop tracking and return aggregated metrics.

        Returns:
            Aggregated ResourceMetrics
        """
        if not self.samples:
            self.sample()

        # Aggregate metrics
        agg = ResourceMetrics()

        if self.samples:
            agg.cpu_percent = sum(s.cpu_percent for s in self.samples) / len(self.samples)
            agg.memory_mb = max(s.memory_mb for s in self.samples)
            agg.gpu_percent = sum(s.gpu_percent for s in self.samples) / len(self.samples)
            agg.gpu_memory_mb = max(s.gpu_memory_mb for s in self.samples)
            agg.energy_kwh = sum(s.energy_kwh for s in self.samples)
            agg.carbon_kg = sum(s.carbon_kg for s in self.samples)
            agg.duration_seconds = self.samples[-1].duration_seconds
            agg.timestamp = self.samples[0].timestamp

        return agg

    def get_report(self) -> dict[str, Any]:
        """Get detailed resource report.

        Returns:
            Dictionary with resource usage details
        """
        agg = self.stop()

        return {
            "cpu": {"avg_percent": agg.cpu_percent, "samples": len(self.samples)},
            "memory": {"peak_mb": agg.memory_mb, "peak_gb": agg.memory_mb / 1024},
            "gpu": {
                "avg_percent": agg.gpu_percent,
                "peak_memory_mb": agg.gpu_memory_mb,
                "peak_memory_gb": agg.gpu_memory_mb / 1024,
                "available": self.has_gpu,
            },
            "energy": {"total_kwh": agg.energy_kwh, "total_wh": agg.energy_kwh * 1000},
            "carbon": {"total_kg": agg.carbon_kg, "total_g": agg.carbon_kg * 1000},
            "duration": {"seconds": agg.duration_seconds, "minutes": agg.duration_seconds / 60},
        }

    def _estimate_energy(self, metrics: ResourceMetrics) -> float:
        """Estimate energy consumption.

        Args:
            metrics: Resource metrics

        Returns:
            Energy in kWh
        """
        # Simplified energy estimation
        # Real implementation would use hardware-specific power models

        # Assume average CPU power: 50W, GPU power: 150W
        cpu_power_w = 50 * (metrics.cpu_percent / 100)
        gpu_power_w = 150 * (metrics.gpu_percent / 100) if self.has_gpu else 0

        total_power_w = cpu_power_w + gpu_power_w

        # Convert to kWh for the sample interval
        return (total_power_w * self.sample_interval) / (1000 * 3600)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
