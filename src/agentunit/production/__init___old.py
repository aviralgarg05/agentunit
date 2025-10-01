# Production Integration & Monitoring Framework
# Real-time production testing integration with observability platforms

from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import numpy as np
from scipy import stats

# Version and metadata
__version__ = "0.4.0"
__author__ = "AgentUnit Team"
__description__ = "Production integration and monitoring framework"

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
EvaluationID = str
BaselineID = str
AlertID = str

if TYPE_CHECKING:
    from ..core import Scenario


class MonitoringPlatform(Enum):
    """Supported monitoring and observability platforms."""
    LANGSMITH = "langsmith"
    AGENT_OPS = "agent_ops"
    WANDB = "wandb"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    HELICONE = "helicone"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    CUSTOM = "custom"


class EvaluationTrigger(Enum):
    """When to trigger evaluations in production."""
    CONTINUOUS = "continuous"          # Real-time evaluation
    SCHEDULED = "scheduled"           # Time-based triggers (cron)
    THRESHOLD = "threshold"           # Metric threshold triggers
    EVENT_DRIVEN = "event_driven"     # Based on specific events
    DEMAND = "demand"                # On-demand evaluation
    ANOMALY = "anomaly"              # When anomalies detected
    DEPLOYMENT = "deployment"        # After deployments
    ERROR_SPIKE = "error_spike"      # When error rates spike


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA = "data"                    # Input data distribution changes
    CONCEPT = "concept"              # Business logic or requirements changes
    MODEL = "model"                  # Model performance degradation
    PERFORMANCE = "performance"      # System performance changes
    FEATURE = "feature"              # Feature importance changes
    COVARIATE = "covariate"         # Input variable distribution changes
    PRIOR = "prior"                 # Label distribution changes


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProductionMetrics:
    """Comprehensive production system metrics.
    
    Captures performance, cost, usage, and quality metrics
    from production LLM systems.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0           # Requests per second/hour
    
    # Quality metrics
    error_rate: float = 0.0           # Percentage of failed requests
    success_rate: float = 0.0         # Percentage of successful requests
    timeout_rate: float = 0.0         # Percentage of timed out requests
    
    # Cost metrics
    cost_per_request: float = 0.0     # Average cost per request
    total_cost: float = 0.0           # Total cost in time period
    cost_efficiency: float = 0.0      # Cost per successful outcome
    
    # Usage metrics
    token_usage: Dict[str, int] = field(default_factory=dict)  # prompt, completion, total
    request_count: int = 0
    unique_users: int = 0
    
    # Model performance
    model_performance: Dict[str, float] = field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    # User satisfaction
    user_satisfaction: Optional[float] = None
    feedback_score: Optional[float] = None
    
    # System health
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metrics."""
        # Ensure rates are between 0 and 1
        for rate_field in ['error_rate', 'success_rate', 'timeout_rate']:
            value = getattr(self, rate_field)
            if not (0.0 <= value <= 1.0):
                logger.warning(f"{rate_field} should be between 0 and 1, got {value}")


@dataclass
class BaselineMetrics:
    """Baseline metrics for drift detection.
    
    Stores statistical baselines established from historical data
    for comparison with current production metrics.
    """
    baseline_id: BaselineID = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Statistical baselines
    metrics: Dict[str, float] = field(default_factory=dict)
    distributions: Dict[str, np.ndarray] = field(default_factory=dict)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    
    # Metadata
    sample_size: int = 0
    time_range: str = ""              # Description of time range used
    data_sources: List[str] = field(default_factory=list)
    
    # Quality indicators
    stability_score: float = 0.0      # How stable the baseline is
    representativeness: float = 0.0   # How representative of normal operation
    
    # Update tracking
    last_updated: datetime = field(default_factory=datetime.now)
    update_frequency: str = "weekly"  # How often to update baseline
    
    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if baseline is stale and needs updating."""
        age = datetime.now() - self.last_updated
        return age.days > max_age_days


@dataclass
class DriftDetection:
    """Results of drift detection analysis.
    
    Contains detailed information about detected drift including
    type, severity, confidence, and remediation suggestions.
    """
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Drift results
    drift_detected: bool = False
    drift_type: DriftType = DriftType.DATA
    severity: AlertSeverity = AlertSeverity.LOW
    confidence_score: float = 0.0     # 0-1 confidence in detection
    
    # Affected components
    affected_metrics: List[str] = field(default_factory=list)
    affected_models: List[str] = field(default_factory=list)
    affected_features: List[str] = field(default_factory=list)
    
    # Statistical details
    statistical_evidence: Dict[str, Any] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    
    # Comparison data
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    current_vs_baseline: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    remediation_suggestions: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    urgency_level: str = "normal"     # normal, urgent, critical
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationJob:
    """Configuration for production evaluation jobs.
    
    Defines how and when to run evaluations in production,
    including triggers, metrics, and alerting rules.
    """
    job_id: EvaluationID = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Execution configuration
    trigger: EvaluationTrigger = EvaluationTrigger.SCHEDULED
    frequency: str = "0 */1 * * *"    # Cron expression (every hour)
    scenarios: List[str] = field(default_factory=list)
    
    # Metrics to monitor
    target_metrics: List[str] = field(default_factory=list)
    custom_evaluators: List[str] = field(default_factory=list)
    
    # Alerting configuration
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    alert_recipients: List[str] = field(default_factory=list)
    alert_channels: List[str] = field(default_factory=list)  # email, slack, webhook
    
    # Data sampling
    sample_size: int = 100            # Number of samples to evaluate
    sampling_strategy: str = "random" # random, latest, stratified
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_sensitivity: float = 0.95   # Statistical confidence level
    
    # State tracking
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    # Results storage
    store_results: bool = True
    retention_days: int = 90
    
    def is_due(self) -> bool:
        """Check if the job is due to run."""
        if not self.enabled:
            return False
        
        if self.trigger == EvaluationTrigger.DEMAND:
            return False  # On-demand jobs don't run automatically
        
        # For other triggers, implement scheduling logic
        # This is a simplified version
        if not self.last_run:
            return True
        
        # Check based on frequency (simplified)
        return True  # Implement proper cron parsing


@dataclass
class AlertRule:
    """Configuration for alerting rules."""
    rule_id: AlertID = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Trigger conditions
    metric_name: str = ""
    operator: str = ">"              # >, <, >=, <=, ==, !=
    threshold: float = 0.0
    duration: int = 300              # Seconds the condition must persist
    
    # Alert configuration
    severity: AlertSeverity = AlertSeverity.MEDIUM
    recipients: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    
    # State
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Cooldown to prevent spam
    cooldown_seconds: int = 3600     # 1 hour cooldown


class ProductionMonitoringProtocol(Protocol):
    """Protocol for production monitoring integrations."""
    
    def get_production_metrics(self, time_range: str) -> ProductionMetrics:
        """Get current production metrics."""
        ...
    
    def setup_continuous_evaluation(self, job: EvaluationJob) -> str:
        """Setup continuous evaluation pipeline."""
        ...
    
    def check_drift(self, current_data: Dict[str, Any]) -> DriftDetection:
        """Check for drift in current data."""
        ...
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send an alert through the platform."""
        ...


class ProductionIntegration(ABC):
    """Abstract base class for production monitoring integrations.
    
    Provides common functionality for integrating with different
    observability and monitoring platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the production integration.
        
        Args:
            config: Platform-specific configuration
        """
        self.config = config
        self.platform = self._get_platform()
        self.evaluation_jobs: Dict[EvaluationID, EvaluationJob] = {}
        self.alert_rules: Dict[AlertID, AlertRule] = {}
        self.baseline_metrics: Dict[BaselineID, BaselineMetrics] = {}
        self._session_id = str(uuid.uuid4())
        
        # Event callbacks
        self._metric_callbacks: List[Callable[[ProductionMetrics], None]] = []
        self._drift_callbacks: List[Callable[[DriftDetection], None]] = []
        self._alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.platform.value}")
    
    @abstractmethod
    def _get_platform(self) -> MonitoringPlatform:
        """Get the monitoring platform this integration supports."""
        pass
    
    @abstractmethod
    async def get_production_metrics(
        self, 
        time_range: str = "1h",
        filters: Dict[str, Any] = None
    ) -> ProductionMetrics:
        """Get production metrics from the platform.
        
        Args:
            time_range: Time range for metrics (e.g., "1h", "1d", "1w")
            filters: Additional filters for metrics
            
        Returns:
            Production metrics object
        """
        pass
    
    @abstractmethod
    async def setup_continuous_evaluation(
        self,
        scenarios: List['Scenario'],
        job_config: Dict[str, Any]
    ) -> EvaluationJob:
        """Setup continuous evaluation pipeline.
        
        Args:
            scenarios: List of scenarios to evaluate
            job_config: Configuration for the evaluation job
            
        Returns:
            Created evaluation job
        """
        pass
    
    @abstractmethod
    async def send_alert(
        self,
        alert_data: Dict[str, Any],
        recipients: List[str] = None,
        channels: List[str] = None
    ) -> bool:
        """Send an alert through the platform.
        
        Args:
            alert_data: Alert information
            recipients: List of alert recipients
            channels: List of alert channels
            
        Returns:
            True if alert was sent successfully
        """
        pass
    
    def create_evaluation_job(
        self,
        name: str,
        scenarios: List[str],
        **kwargs
    ) -> EvaluationJob:
        """Create a new evaluation job.
        
        Args:
            name: Name for the evaluation job
            scenarios: List of scenario names to evaluate
            **kwargs: Additional job configuration
            
        Returns:
            Created evaluation job
        """
        job = EvaluationJob(
            name=name,
            description=kwargs.get("description", ""),
            trigger=EvaluationTrigger(kwargs.get("trigger", "scheduled")),
            frequency=kwargs.get("frequency", "0 */1 * * *"),
            scenarios=scenarios,
            target_metrics=kwargs.get("target_metrics", ["accuracy", "latency", "cost"]),
            alert_thresholds=kwargs.get("alert_thresholds", {}),
            alert_recipients=kwargs.get("alert_recipients", []),
            sample_size=kwargs.get("sample_size", 100),
            enable_drift_detection=kwargs.get("enable_drift_detection", True),
            drift_sensitivity=kwargs.get("drift_sensitivity", 0.95)
        )
        
        self.evaluation_jobs[job.job_id] = job
        logger.info(f"Created evaluation job {job.job_id}: {name}")
        
        return job
    
    def create_alert_rule(
        self,
        name: str,
        metric_name: str,
        operator: str,
        threshold: float,
        **kwargs
    ) -> AlertRule:
        """Create a new alert rule.
        
        Args:
            name: Name for the alert rule
            metric_name: Name of metric to monitor
            operator: Comparison operator (>, <, >=, <=, ==, !=)
            threshold: Threshold value for alerting
            **kwargs: Additional alert configuration
            
        Returns:
            Created alert rule
        """
        rule = AlertRule(
            name=name,
            description=kwargs.get("description", ""),
            metric_name=metric_name,
            operator=operator,
            threshold=threshold,
            duration=kwargs.get("duration", 300),
            severity=AlertSeverity(kwargs.get("severity", "medium")),
            recipients=kwargs.get("recipients", []),
            channels=kwargs.get("channels", []),
            cooldown_seconds=kwargs.get("cooldown_seconds", 3600)
        )
        
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Created alert rule {rule.rule_id}: {name}")
        
        return rule
    
    def establish_baseline(
        self,
        historical_data: List[Dict[str, Any]],
        metrics: List[str],
        **kwargs
    ) -> BaselineMetrics:
        """Establish baseline metrics from historical data.
        
        Args:
            historical_data: List of historical metric data points
            metrics: List of metrics to establish baselines for
            **kwargs: Additional baseline configuration
            
        Returns:
            Established baseline metrics
        """
        if not historical_data:
            raise ValueError("Historical data required to establish baseline")
        
        # Calculate baseline statistics
        baseline_values = {}
        distributions = {}
        confidence_intervals = {}
        
        for metric in metrics:
            values = [d.get(metric, 0.0) for d in historical_data if metric in d]
            
            if values:
                values_array = np.array(values)
                baseline_values[metric] = float(np.mean(values_array))
                distributions[metric] = values_array
                
                # Calculate confidence intervals
                if len(values) > 1:
                    confidence_level = kwargs.get("confidence_level", 0.95)
                    ci_lower, ci_upper = stats.t.interval(
                        confidence_level,
                        len(values) - 1,
                        loc=np.mean(values_array),
                        scale=stats.sem(values_array)
                    )
                    confidence_intervals[metric] = (float(ci_lower), float(ci_upper))
                else:
                    confidence_intervals[metric] = (baseline_values[metric], baseline_values[metric])
        
        # Calculate quality indicators
        stability_score = self._calculate_stability_score(distributions)
        representativeness = kwargs.get("representativeness", 0.8)
        
        baseline = BaselineMetrics(
            metrics=baseline_values,
            distributions=distributions,
            confidence_intervals=confidence_intervals,
            sample_size=len(historical_data),
            time_range=kwargs.get("time_range", "unknown"),
            data_sources=kwargs.get("data_sources", []),
            stability_score=stability_score,
            representativeness=representativeness,
            update_frequency=kwargs.get("update_frequency", "weekly")
        )
        
        self.baseline_metrics[baseline.baseline_id] = baseline
        logger.info(f"Established baseline {baseline.baseline_id} for {len(metrics)} metrics")
        
        return baseline
    
    def _calculate_stability_score(self, distributions: Dict[str, np.ndarray]) -> float:
        """Calculate stability score for baseline metrics."""
        if not distributions:
            return 0.0
        
        stability_scores = []
        for metric, values in distributions.items():
            if len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                # Lower coefficient of variation = higher stability
                stability = 1.0 / (1.0 + cv)
                stability_scores.append(stability)
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0
    
    async def check_alert_rules(self, metrics: ProductionMetrics) -> List[Dict[str, Any]]:
        """Check if any alert rules are triggered.
        
        Args:
            metrics: Current production metrics
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule is in cooldown
            if rule.last_triggered:
                cooldown_elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                if cooldown_elapsed < rule.cooldown_seconds:
                    continue
            
            # Get metric value
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is None:
                metric_value = metrics.custom_metrics.get(rule.metric_name)
            
            if metric_value is None:
                continue
            
            # Check threshold
            triggered = False
            if rule.operator == ">" and metric_value > rule.threshold:
                triggered = True
            elif rule.operator == "<" and metric_value < rule.threshold:
                triggered = True
            elif rule.operator == ">=" and metric_value >= rule.threshold:
                triggered = True
            elif rule.operator == "<=" and metric_value <= rule.threshold:
                triggered = True
            elif rule.operator == "==" and metric_value == rule.threshold:
                triggered = True
            elif rule.operator == "!=" and metric_value != rule.threshold:
                triggered = True
            
            if triggered:
                alert = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "metric_name": rule.metric_name,
                    "metric_value": metric_value,
                    "threshold": rule.threshold,
                    "operator": rule.operator,
                    "severity": rule.severity.value,
                    "timestamp": datetime.now(),
                    "description": rule.description
                }
                
                triggered_alerts.append(alert)
                
                # Update rule state
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
                
                # Send alert
                await self.send_alert(alert, rule.recipients, rule.channels)
        
        return triggered_alerts
    
    def add_metric_callback(self, callback: Callable[[ProductionMetrics], None]) -> None:
        """Add a callback for metric updates."""
        self._metric_callbacks.append(callback)
    
    def add_drift_callback(self, callback: Callable[[DriftDetection], None]) -> None:
        """Add a callback for drift detection events."""
        self._drift_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback for alert events."""
        self._alert_callbacks.append(callback)
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of the integration status."""
        return {
            "session_id": self._session_id,
            "platform": self.platform.value,
            "evaluation_jobs": len(self.evaluation_jobs),
            "alert_rules": len(self.alert_rules),
            "baselines": len(self.baseline_metrics),
            "active_jobs": len([job for job in self.evaluation_jobs.values() if job.enabled]),
            "active_rules": len([rule for rule in self.alert_rules.values() if rule.enabled])
        }


# Export main classes and enums
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Enums
    "MonitoringPlatform",
    "EvaluationTrigger",
    "DriftType",
    "AlertSeverity",
    
    # Data classes
    "ProductionMetrics",
    "BaselineMetrics",
    "DriftDetection",
    "EvaluationJob", 
    "AlertRule",
    
    # Protocol and base class
    "ProductionMonitoringProtocol",
    "ProductionIntegration",
    
    # Type aliases
    "EvaluationID",
    "BaselineID",
    "AlertID"
]