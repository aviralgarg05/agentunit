"""Tests for sustainability tracking."""

import importlib.util

import pytest

from agentunit.sustainability import (
    CarbonMetric,
    CarbonReport,
    EnergyMetric,
    ResourceMetrics,
    ResourceUtilizationMetric,
)


# Skip tracker tests if psutil not available
HAS_PSUTIL = importlib.util.find_spec("psutil") is not None

HAS_TRACKER = False
if HAS_PSUTIL:
    try:
        from agentunit.sustainability import CarbonTracker, ResourceTracker

        HAS_TRACKER = True
    except ImportError:
        pass


# ResourceTracker Tests
@pytest.mark.skipif(not HAS_TRACKER, reason="psutil not available")
def test_resource_tracker_initialization():
    """Test ResourceTracker initialization."""
    tracker = ResourceTracker(sample_interval=2.0, enable_gpu=False)

    assert abs(tracker.sample_interval - 2.0) < 0.001
    assert tracker.enable_gpu is False
    assert tracker.start_time is None
    assert len(tracker.samples) == 0


@pytest.mark.skipif(not HAS_TRACKER, reason="psutil not available")
def test_resource_tracker_sample():
    """Test taking resource samples."""
    tracker = ResourceTracker(enable_gpu=False)
    tracker.start()

    metrics = tracker.sample()

    assert isinstance(metrics, ResourceMetrics)
    assert metrics.cpu_percent >= 0
    assert metrics.memory_mb > 0
    assert metrics.duration_seconds >= 0


@pytest.mark.skipif(not HAS_TRACKER, reason="psutil not available")
def test_resource_tracker_stop():
    """Test stopping tracker and aggregation."""
    tracker = ResourceTracker(enable_gpu=False)
    tracker.start()

    # Take a few samples
    tracker.sample()
    tracker.sample()

    agg = tracker.stop()

    assert isinstance(agg, ResourceMetrics)
    assert agg.cpu_percent >= 0
    assert agg.memory_mb > 0
    assert len(tracker.samples) >= 2


@pytest.mark.skipif(not HAS_TRACKER, reason="psutil not available")
def test_resource_tracker_context_manager():
    """Test ResourceTracker as context manager."""
    with ResourceTracker(enable_gpu=False) as tracker:
        tracker.sample()

    assert len(tracker.samples) >= 1


@pytest.mark.skipif(not HAS_TRACKER, reason="psutil not available")
def test_resource_tracker_report():
    """Test getting detailed resource report."""
    tracker = ResourceTracker(enable_gpu=False)
    tracker.start()
    tracker.sample()

    report = tracker.get_report()

    assert "cpu" in report
    assert "memory" in report
    assert "gpu" in report
    assert "energy" in report
    assert "carbon" in report
    assert "duration" in report

    assert report["cpu"]["avg_percent"] >= 0
    assert report["memory"]["peak_mb"] > 0


# CarbonTracker Tests
@pytest.mark.skipif(not HAS_TRACKER, reason="tracker not available")
def test_carbon_tracker_initialization():
    """Test CarbonTracker initialization."""
    tracker = CarbonTracker(grid_intensity=0.5, use_codecarbon=False)

    assert abs(tracker.grid_intensity - 0.5) < 0.001
    assert tracker.use_codecarbon is False
    assert tracker.start_time is None
    assert abs(tracker.total_energy_kwh) < 0.001


@pytest.mark.skipif(not HAS_TRACKER, reason="tracker not available")
def test_carbon_tracker_update():
    """Test updating energy consumption."""
    tracker = CarbonTracker(use_codecarbon=False)
    tracker.start()

    tracker.update(0.5)
    tracker.update(0.3)

    assert abs(tracker.total_energy_kwh - 0.8) < 0.001


@pytest.mark.skipif(not HAS_TRACKER, reason="tracker not available")
def test_carbon_tracker_stop():
    """Test stopping tracker and generating report."""
    tracker = CarbonTracker(grid_intensity=0.475, use_codecarbon=False)
    tracker.start()
    tracker.update(1.0)

    report = tracker.stop()

    assert isinstance(report, CarbonReport)
    assert abs(report.energy_kwh - 1.0) < 0.001
    assert abs(report.emissions_kg - 0.475) < 0.001
    assert abs(report.grid_intensity - 0.475) < 0.001


@pytest.mark.skipif(not HAS_TRACKER, reason="tracker not available")
def test_carbon_tracker_context_manager():
    """Test CarbonTracker as context manager."""
    with CarbonTracker(use_codecarbon=False) as tracker:
        tracker.update(0.5)

    assert abs(tracker.total_energy_kwh - 0.5) < 0.001


@pytest.mark.skipif(not HAS_TRACKER, reason="tracker not available")
def test_carbon_tracker_report():
    """Test getting detailed carbon report."""
    tracker = CarbonTracker(use_codecarbon=False)
    tracker.start()
    tracker.update(1.0)

    report = tracker.get_report()

    assert "emissions" in report
    assert "energy" in report
    assert "intensity" in report
    assert "duration" in report
    assert "equivalents" in report
    assert "source" in report

    assert report["emissions"]["kg_co2"] > 0
    assert report["equivalents"]["km_driven"] > 0
    assert report["source"] == "estimated"


# EnergyMetric Tests
def test_energy_metric_initialization():
    """Test EnergyMetric initialization."""
    metric = EnergyMetric(threshold_kwh=0.2)

    assert metric.name == "energy_consumption"
    assert metric.threshold_kwh == 0.2


def test_energy_metric_under_threshold():
    """Test energy metric when under threshold."""
    metric = EnergyMetric(threshold_kwh=0.1)

    trace = {"energy_kwh": 0.05}
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 1.0
    assert result.detail["under_threshold"] is True


def test_energy_metric_over_threshold():
    """Test energy metric when over threshold."""
    metric = EnergyMetric(threshold_kwh=0.1)

    trace = {"energy_kwh": 0.2}
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 0.5
    assert result.detail["under_threshold"] is False


def test_energy_metric_with_trace_object():
    """Test energy metric with trace object."""
    metric = EnergyMetric(threshold_kwh=0.1)

    class Trace:
        def __init__(self):
            self.metadata = {"energy_kwh": 0.08}

    trace = Trace()
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 1.0


# CarbonMetric Tests
def test_carbon_metric_initialization():
    """Test CarbonMetric initialization."""
    metric = CarbonMetric(threshold_kg=0.1)

    assert metric.name == "carbon_emissions"
    assert metric.threshold_kg == 0.1


def test_carbon_metric_under_threshold():
    """Test carbon metric when under threshold."""
    metric = CarbonMetric(threshold_kg=0.05)

    trace = {"carbon_kg": 0.03}
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 1.0
    assert result.detail["under_threshold"] is True
    assert "equivalents" in result.detail


def test_carbon_metric_over_threshold():
    """Test carbon metric when over threshold."""
    metric = CarbonMetric(threshold_kg=0.05)

    trace = {"carbon_kg": 0.1}
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 0.5
    assert result.detail["under_threshold"] is False


def test_carbon_metric_equivalents():
    """Test carbon metric equivalents calculation."""
    metric = CarbonMetric(threshold_kg=0.05)

    trace = {"carbon_kg": 1.0}
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.detail["equivalents"]["km_driven"] == 4.6
    assert abs(result.detail["equivalents"]["trees_needed"] - 1.0 / 21) < 0.001


# ResourceUtilizationMetric Tests
def test_resource_utilization_metric_initialization():
    """Test ResourceUtilizationMetric initialization."""
    metric = ResourceUtilizationMetric(
        target_cpu_percent=70.0, target_memory_gb=2.0, target_gpu_percent=75.0
    )

    assert metric.name == "resource_utilization"
    assert metric.target_cpu_percent == 70.0
    assert metric.target_memory_gb == 2.0
    assert metric.target_gpu_percent == 75.0


def test_resource_utilization_metric_optimal():
    """Test resource utilization when in optimal range."""
    metric = ResourceUtilizationMetric(
        target_cpu_percent=80.0, target_memory_gb=4.0, target_gpu_percent=80.0
    )

    trace = {"cpu_percent": 80.0, "memory_mb": 4096.0, "gpu_percent": 80.0}

    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value == 1.0
    assert result.detail["cpu"]["score"] == 1.0
    assert result.detail["memory"]["score"] == 1.0
    assert result.detail["gpu"]["score"] == 1.0


def test_resource_utilization_metric_over():
    """Test resource utilization when over target."""
    metric = ResourceUtilizationMetric(
        target_cpu_percent=50.0, target_memory_gb=2.0, target_gpu_percent=50.0
    )

    trace = {"cpu_percent": 100.0, "memory_mb": 4096.0, "gpu_percent": 100.0}

    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value < 1.0


def test_resource_utilization_metric_under():
    """Test resource utilization when under target."""
    metric = ResourceUtilizationMetric(
        target_cpu_percent=80.0, target_memory_gb=4.0, target_gpu_percent=80.0
    )

    trace = {"cpu_percent": 20.0, "memory_mb": 1024.0, "gpu_percent": 20.0}

    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert result.value < 1.0


def test_resource_utilization_metric_with_trace_object():
    """Test resource utilization metric with trace object."""
    metric = ResourceUtilizationMetric()

    class Trace:
        def __init__(self):
            self.metadata = {"cpu_percent": 80.0, "memory_mb": 4096.0, "gpu_percent": 80.0}

    trace = Trace()
    result = metric.evaluate(case=None, trace=trace, outcome=None)

    assert abs(result.value - 1.0) < 0.001
