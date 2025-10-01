# Production Integration & Monitoring Design

*Real-time production testing integration with LangSmith, AgentOps, and continuous evaluation*

## Overview

This document outlines the design for AgentUnit's production integration capabilities, enabling real-time monitoring, drift detection, continuous evaluation, and seamless integration with production LLM systems through LangSmith, AgentOps, and other observability platforms.

## Architecture Overview

### Production Integration Components

```python
# src/agentunit/production/__init__.py
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

class MonitoringPlatform(Enum):
    """Supported monitoring platforms"""
    LANGSMITH = "langsmith"
    AGENT_OPS = "agent_ops"
    WANDB = "wandb"
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    HELICONE = "helicone"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"

class EvaluationTrigger(Enum):
    """When to trigger evaluations"""
    CONTINUOUS = "continuous"          # Real-time evaluation
    SCHEDULED = "scheduled"           # Time-based triggers
    THRESHOLD = "threshold"           # Metric threshold triggers
    EVENT_DRIVEN = "event_driven"     # Based on specific events
    DEMAND = "demand"                # On-demand evaluation
    ANOMALY = "anomaly"              # When anomalies detected

@dataclass
class ProductionMetrics:
    """Production system metrics"""
    timestamp: datetime
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    success_rate: float
    cost_per_request: float
    token_usage: Dict[str, int]
    model_performance: Dict[str, float]
    user_satisfaction: Optional[float] = None

@dataclass
class DriftDetection:
    """Model/data drift detection results"""
    timestamp: datetime
    drift_detected: bool
    drift_type: str  # 'data', 'concept', 'model', 'performance'
    severity: str    # 'low', 'medium', 'high', 'critical'
    affected_metrics: List[str]
    confidence_score: float
    remediation_suggestions: List[str]
    baseline_comparison: Dict[str, Any]

@dataclass
class EvaluationJob:
    """Production evaluation job configuration"""
    job_id: str
    name: str
    trigger: EvaluationTrigger
    scenarios: List[str]
    frequency: str  # cron expression or interval
    target_metrics: List[str]
    alert_thresholds: Dict[str, float]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
```

### LangSmith Integration

```python
# src/agentunit/production/langsmith.py
from typing import Dict, List, Any, Optional, AsyncIterator
import asyncio
from langsmith import Client as LangSmithClient
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from datetime import datetime, timedelta

class LangSmithIntegration:
    """Integration with LangSmith for production monitoring"""
    
    def __init__(self, api_key: str, project_name: str):
        self.client = LangSmithClient(api_key=api_key)
        self.project_name = project_name
        self.evaluation_jobs: Dict[str, EvaluationJob] = {}
        self.drift_detector = DriftDetector()
        
    async def setup_continuous_evaluation(
        self,
        scenarios: List['Scenario'],
        evaluation_config: Dict[str, Any]
    ) -> str:
        """Setup continuous evaluation pipeline"""
        
        job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create dataset from scenarios
        dataset_name = f"agentunit_scenarios_{self.project_name}"
        dataset = await self._create_or_update_dataset(dataset_name, scenarios)
        
        # Setup evaluation job
        job = EvaluationJob(
            job_id=job_id,
            name=f"Continuous Evaluation - {self.project_name}",
            trigger=EvaluationTrigger.CONTINUOUS,
            scenarios=[s.name for s in scenarios],
            frequency=evaluation_config.get("frequency", "*/15 * * * *"),  # Every 15 minutes
            target_metrics=evaluation_config.get("metrics", ["accuracy", "latency", "cost"]),
            alert_thresholds=evaluation_config.get("thresholds", {}),
            enabled=True
        )
        
        self.evaluation_jobs[job_id] = job
        
        # Start continuous monitoring
        asyncio.create_task(self._run_continuous_evaluation(job))
        
        return job_id
    
    async def _run_continuous_evaluation(self, job: EvaluationJob):
        """Run continuous evaluation loop"""
        
        while job.enabled:
            try:
                # Get recent production runs
                recent_runs = await self._get_recent_runs(
                    hours_back=1,  # Evaluate last hour of data
                    limit=100
                )
                
                if recent_runs:
                    # Run evaluation on recent data
                    evaluation_results = await self._evaluate_production_runs(
                        recent_runs, 
                        job.target_metrics
                    )
                    
                    # Check for drift
                    drift_result = await self.drift_detector.check_drift(
                        evaluation_results,
                        baseline_metrics=await self._get_baseline_metrics()
                    )
                    
                    # Store results
                    await self._store_evaluation_results(job.job_id, evaluation_results, drift_result)
                    
                    # Check alert thresholds
                    await self._check_alert_thresholds(job, evaluation_results)
                    
                    # Generate recommendations if drift detected
                    if drift_result.drift_detected:
                        await self._generate_drift_recommendations(drift_result)
                
                # Wait for next evaluation
                await asyncio.sleep(self._parse_frequency_to_seconds(job.frequency))
                
            except Exception as e:
                print(f"Error in continuous evaluation: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _get_recent_runs(self, hours_back: int = 1, limit: int = 100) -> List[Run]:
        """Get recent production runs from LangSmith"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        ))
        
        return runs
    
    async def _evaluate_production_runs(
        self, 
        runs: List[Run], 
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Evaluate production runs against AgentUnit scenarios"""
        
        evaluation_results = {
            "timestamp": datetime.now(),
            "runs_evaluated": len(runs),
            "metrics": {}
        }
        
        # Convert runs to evaluation format
        examples = []
        for run in runs:
            example = Example(
                inputs=run.inputs or {},
                outputs={"actual": run.outputs},
                metadata={"run_id": run.id, "start_time": run.start_time}
            )
            examples.append(example)
        
        # Run evaluations for each metric
        for metric in metrics:
            evaluator = await self._get_evaluator_for_metric(metric)
            
            if evaluator:
                metric_results = await evaluate(
                    lambda inputs: {"prediction": inputs.get("prediction", "")},
                    data=examples,
                    evaluators=[evaluator],
                    experiment_prefix=f"prod_eval_{metric}"
                )
                
                evaluation_results["metrics"][metric] = {
                    "average_score": metric_results.get("average_score", 0.0),
                    "scores": metric_results.get("scores", []),
                    "details": metric_results
                }
        
        return evaluation_results
    
    async def _get_evaluator_for_metric(self, metric: str):
        """Get appropriate evaluator for metric"""
        
        # Import AgentUnit evaluators
        from ..evaluation import get_evaluator
        return get_evaluator(metric)
    
    async def _create_or_update_dataset(
        self, 
        dataset_name: str, 
        scenarios: List['Scenario']
    ):
        """Create or update LangSmith dataset with scenarios"""
        
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
        except:
            dataset = self.client.create_dataset(dataset_name=dataset_name)
        
        # Convert scenarios to examples
        examples = []
        for scenario in scenarios:
            example = Example(
                inputs=scenario.input_data,
                outputs={"expected": scenario.expected_output},
                metadata={
                    "scenario_name": scenario.name,
                    "adapter_type": type(scenario.adapter).__name__,
                    "created_at": datetime.now().isoformat()
                }
            )
            examples.append(example)
        
        # Add examples to dataset
        self.client.create_examples(
            inputs=[ex.inputs for ex in examples],
            outputs=[ex.outputs for ex in examples],
            metadata=[ex.metadata for ex in examples],
            dataset_id=dataset.id
        )
        
        return dataset
    
    async def get_production_metrics(
        self, 
        time_range: timedelta = timedelta(hours=24)
    ) -> ProductionMetrics:
        """Get production metrics from LangSmith"""
        
        end_time = datetime.now()
        start_time = end_time - time_range
        
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=start_time,
            end_time=end_time
        ))
        
        if not runs:
            return ProductionMetrics(
                timestamp=datetime.now(),
                latency_p95=0.0, latency_p99=0.0,
                throughput=0.0, error_rate=0.0,
                success_rate=0.0, cost_per_request=0.0,
                token_usage={}, model_performance={}
            )
        
        # Calculate metrics
        latencies = [run.total_time for run in runs if run.total_time]
        latency_p95 = np.percentile(latencies, 95) if latencies else 0.0
        latency_p99 = np.percentile(latencies, 99) if latencies else 0.0
        
        successful_runs = len([r for r in runs if r.status == "success"])
        success_rate = successful_runs / len(runs) if runs else 0.0
        error_rate = 1.0 - success_rate
        
        throughput = len(runs) / (time_range.total_seconds() / 3600)  # requests per hour
        
        # Calculate token usage and costs
        total_tokens = 0
        total_cost = 0.0
        token_breakdown = {"prompt": 0, "completion": 0}
        
        for run in runs:
            if hasattr(run, 'prompt_tokens'):
                token_breakdown["prompt"] += run.prompt_tokens or 0
            if hasattr(run, 'completion_tokens'):
                token_breakdown["completion"] += run.completion_tokens or 0
            if hasattr(run, 'total_cost'):
                total_cost += run.total_cost or 0.0
        
        total_tokens = sum(token_breakdown.values())
        cost_per_request = total_cost / len(runs) if runs else 0.0
        
        return ProductionMetrics(
            timestamp=datetime.now(),
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate,
            success_rate=success_rate,
            cost_per_request=cost_per_request,
            token_usage=token_breakdown,
            model_performance=await self._calculate_model_performance(runs)
        )
    
    async def _calculate_model_performance(self, runs: List[Run]) -> Dict[str, float]:
        """Calculate model-specific performance metrics"""
        
        model_stats = {}
        
        for run in runs:
            model = getattr(run, 'model', 'unknown')
            
            if model not in model_stats:
                model_stats[model] = {
                    "count": 0,
                    "success_count": 0,
                    "total_latency": 0.0,
                    "total_cost": 0.0
                }
            
            stats = model_stats[model]
            stats["count"] += 1
            
            if run.status == "success":
                stats["success_count"] += 1
            
            if run.total_time:
                stats["total_latency"] += run.total_time
            
            if hasattr(run, 'total_cost') and run.total_cost:
                stats["total_cost"] += run.total_cost
        
        # Calculate averages
        performance = {}
        for model, stats in model_stats.items():
            performance[model] = {
                "success_rate": stats["success_count"] / stats["count"],
                "avg_latency": stats["total_latency"] / stats["count"],
                "avg_cost": stats["total_cost"] / stats["count"]
            }
        
        return performance
```

### AgentOps Integration

```python
# src/agentunit/production/agentops.py
from typing import Dict, List, Any, Optional
import agentops
from datetime import datetime, timedelta

class AgentOpsIntegration:
    """Integration with AgentOps for agent monitoring"""
    
    def __init__(self, api_key: str, project_name: str):
        agentops.init(api_key=api_key, tags=[project_name])
        self.project_name = project_name
        self.active_sessions: Dict[str, Any] = {}
        
    def start_evaluation_session(self, scenario_name: str) -> str:
        """Start an AgentOps session for scenario evaluation"""
        
        session = agentops.start_session(
            tags=[self.project_name, "evaluation", scenario_name]
        )
        
        session_id = session.session_id
        self.active_sessions[session_id] = {
            "scenario": scenario_name,
            "start_time": datetime.now(),
            "session": session
        }
        
        return session_id
    
    def record_agent_action(
        self,
        session_id: str,
        agent_name: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """Record an agent action in AgentOps"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]["session"]
            
            session.record({
                "agent": agent_name,
                "action": action,
                "inputs": inputs,
                "outputs": outputs,
                "timestamp": datetime.now(),
                "metadata": metadata or {}
            })
    
    def end_evaluation_session(
        self,
        session_id: str,
        result: 'ScenarioResult'
    ):
        """End an AgentOps session"""
        
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            session = session_data["session"]
            
            # Record final results
            session.record({
                "event_type": "scenario_completion",
                "scenario": session_data["scenario"],
                "success": result.success,
                "execution_time": result.execution_time,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata
            })
            
            # End session
            agentops.end_session(session_id)
            del self.active_sessions[session_id]
    
    async def get_agent_analytics(
        self,
        time_range: timedelta = timedelta(days=1)
    ) -> Dict[str, Any]:
        """Get agent analytics from AgentOps"""
        
        # This would use AgentOps API to fetch analytics
        # Placeholder implementation
        return {
            "total_sessions": 0,
            "success_rate": 0.0,
            "average_session_duration": 0.0,
            "agent_performance": {},
            "cost_analysis": {},
            "usage_patterns": {}
        }
```

### Drift Detection System

```python
# src/agentunit/production/drift.py
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import jensen_shannon_distance
from dataclasses import dataclass
import logging

@dataclass
class BaselineMetrics:
    """Baseline metrics for drift detection"""
    metrics: Dict[str, float]
    distributions: Dict[str, np.ndarray]
    timestamp: datetime
    sample_size: int
    confidence_intervals: Dict[str, Tuple[float, float]]

class DriftDetector:
    """Detect various types of drift in production systems"""
    
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.baseline_metrics: Optional[BaselineMetrics] = None
        self.drift_history: List[DriftDetection] = []
        
    async def establish_baseline(
        self,
        historical_data: List[Dict[str, Any]],
        metrics: List[str]
    ) -> BaselineMetrics:
        """Establish baseline metrics from historical data"""
        
        if not historical_data:
            raise ValueError("Historical data required to establish baseline")
        
        # Calculate baseline metrics
        baseline_values = {}
        distributions = {}
        confidence_intervals = {}
        
        for metric in metrics:
            values = [d.get(metric, 0.0) for d in historical_data if metric in d]
            
            if values:
                baseline_values[metric] = np.mean(values)
                distributions[metric] = np.array(values)
                
                # Calculate confidence intervals
                ci_lower, ci_upper = stats.t.interval(
                    self.sensitivity,
                    len(values) - 1,
                    loc=np.mean(values),
                    scale=stats.sem(values)
                )
                confidence_intervals[metric] = (ci_lower, ci_upper)
        
        self.baseline_metrics = BaselineMetrics(
            metrics=baseline_values,
            distributions=distributions,
            timestamp=datetime.now(),
            sample_size=len(historical_data),
            confidence_intervals=confidence_intervals
        )
        
        return self.baseline_metrics
    
    async def check_drift(
        self,
        current_data: Dict[str, Any],
        baseline_metrics: Optional[BaselineMetrics] = None
    ) -> DriftDetection:
        """Check for drift in current data"""
        
        baseline = baseline_metrics or self.baseline_metrics
        
        if not baseline:
            raise ValueError("Baseline metrics required for drift detection")
        
        drift_detected = False
        drift_type = "none"
        severity = "low"
        affected_metrics = []
        confidence_score = 0.0
        remediation_suggestions = []
        
        # Check each metric for drift
        for metric, baseline_value in baseline.metrics.items():
            current_value = current_data.get("metrics", {}).get(metric, 0.0)
            
            # Statistical drift detection
            drift_result = await self._detect_statistical_drift(
                metric, current_value, baseline
            )
            
            if drift_result["drift_detected"]:
                drift_detected = True
                affected_metrics.append(metric)
                
                # Determine severity
                drift_magnitude = abs(current_value - baseline_value) / baseline_value
                if drift_magnitude > 0.5:
                    severity = "critical"
                elif drift_magnitude > 0.3:
                    severity = "high"
                elif drift_magnitude > 0.1:
                    severity = "medium"
                
                # Update confidence score
                confidence_score = max(confidence_score, drift_result["confidence"])
        
        # Determine drift type
        if drift_detected:
            drift_type = await self._classify_drift_type(affected_metrics, current_data, baseline)
            remediation_suggestions = await self._generate_remediation_suggestions(
                drift_type, affected_metrics, severity
            )
        
        drift_detection = DriftDetection(
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            affected_metrics=affected_metrics,
            confidence_score=confidence_score,
            remediation_suggestions=remediation_suggestions,
            baseline_comparison={
                "baseline_timestamp": baseline.timestamp,
                "baseline_sample_size": baseline.sample_size,
                "current_metrics": current_data.get("metrics", {}),
                "baseline_metrics": baseline.metrics
            }
        )
        
        self.drift_history.append(drift_detection)
        return drift_detection
    
    async def _detect_statistical_drift(
        self,
        metric: str,
        current_value: float,
        baseline: BaselineMetrics
    ) -> Dict[str, Any]:
        """Detect statistical drift for a specific metric"""
        
        baseline_dist = baseline.distributions.get(metric, np.array([]))
        ci_lower, ci_upper = baseline.confidence_intervals.get(metric, (0.0, 0.0))
        
        # Check if current value is outside confidence interval
        outside_ci = current_value < ci_lower or current_value > ci_upper
        
        # Calculate z-score
        baseline_mean = np.mean(baseline_dist)
        baseline_std = np.std(baseline_dist)
        
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            significant_change = z_score > stats.norm.ppf(self.sensitivity)
        else:
            significant_change = False
            z_score = 0.0
        
        # Combine evidence
        drift_detected = outside_ci or significant_change
        confidence = min(z_score / 3.0, 1.0)  # Normalize to 0-1
        
        return {
            "drift_detected": drift_detected,
            "confidence": confidence,
            "z_score": z_score,
            "outside_ci": outside_ci,
            "current_value": current_value,
            "baseline_mean": baseline_mean
        }
    
    async def _classify_drift_type(
        self,
        affected_metrics: List[str],
        current_data: Dict[str, Any],
        baseline: BaselineMetrics
    ) -> str:
        """Classify the type of drift detected"""
        
        # Performance drift
        performance_metrics = ["latency_p95", "latency_p99", "success_rate", "error_rate"]
        if any(metric in performance_metrics for metric in affected_metrics):
            return "performance"
        
        # Data drift
        data_metrics = ["input_distribution", "output_distribution"]
        if any(metric in data_metrics for metric in affected_metrics):
            return "data"
        
        # Model drift
        model_metrics = ["model_performance", "accuracy", "precision", "recall"]
        if any(metric in model_metrics for metric in affected_metrics):
            return "model"
        
        # Concept drift
        concept_metrics = ["user_satisfaction", "business_metrics"]
        if any(metric in concept_metrics for metric in affected_metrics):
            return "concept"
        
        return "unknown"
    
    async def _generate_remediation_suggestions(
        self,
        drift_type: str,
        affected_metrics: List[str],
        severity: str
    ) -> List[str]:
        """Generate remediation suggestions based on drift type"""
        
        suggestions = []
        
        if drift_type == "performance":
            suggestions.extend([
                "Review recent model changes or deployments",
                "Check infrastructure resources and scaling",
                "Analyze request patterns for unusual load",
                "Consider model optimization or caching strategies"
            ])
        
        elif drift_type == "data":
            suggestions.extend([
                "Validate input data quality and format",
                "Check for changes in data sources",
                "Review data preprocessing pipeline",
                "Consider retraining with recent data"
            ])
        
        elif drift_type == "model":
            suggestions.extend([
                "Evaluate model performance on recent data",
                "Consider fine-tuning or retraining",
                "Review feature importance changes",
                "Implement A/B testing for model variants"
            ])
        
        elif drift_type == "concept":
            suggestions.extend([
                "Review business logic and requirements",
                "Analyze user feedback and satisfaction metrics",
                "Consider updating evaluation criteria",
                "Engage stakeholders for requirement validation"
            ])
        
        # Add severity-specific suggestions
        if severity in ["high", "critical"]:
            suggestions.insert(0, "Consider immediate rollback to previous stable version")
            suggestions.append("Set up enhanced monitoring and alerting")
        
        return suggestions
```

### Production CLI Integration

```python
# Enhancement to CLI for production integration
@click.group()
def production():
    """Production integration commands"""
    pass

@production.command()
@click.option('--platform', type=click.Choice(['langsmith', 'agentops', 'wandb']))
@click.option('--api-key', required=True, help='Platform API key')
@click.option('--project', required=True, help='Project name')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
def setup(platform: str, api_key: str, project: str, config: str):
    """Setup production monitoring integration"""
    
    if platform == 'langsmith':
        integration = LangSmithIntegration(api_key, project)
    elif platform == 'agentops':
        integration = AgentOpsIntegration(api_key, project)
    
    # Save configuration
    config_data = {
        "platform": platform,
        "api_key": api_key,
        "project": project,
        "setup_time": datetime.now().isoformat()
    }
    
    config_path = f".agentunit/{platform}_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    click.echo(f"✓ {platform.title()} integration configured")

@production.command()
@click.option('--scenarios', required=True, help='Scenarios configuration file')
@click.option('--frequency', default='*/15 * * * *', help='Evaluation frequency (cron format)')
@click.option('--metrics', multiple=True, help='Metrics to monitor')
def start_monitoring(scenarios: str, frequency: str, metrics: List[str]):
    """Start continuous production monitoring"""
    
    # Load scenarios
    scenarios_obj = load_scenarios_from_config(scenarios)
    
    # Setup continuous evaluation
    integration = get_configured_integration()
    job_id = integration.setup_continuous_evaluation(
        scenarios_obj,
        {
            "frequency": frequency,
            "metrics": list(metrics) or ["accuracy", "latency", "cost"]
        }
    )
    
    click.echo(f"✓ Started continuous monitoring (Job ID: {job_id})")

@production.command()
@click.option('--hours', default=24, help='Time range in hours')
@click.option('--output', help='Output file for metrics')
def get_metrics(hours: int, output: str):
    """Get production metrics"""
    
    integration = get_configured_integration()
    metrics = integration.get_production_metrics(timedelta(hours=hours))
    
    metrics_dict = {
        "timestamp": metrics.timestamp.isoformat(),
        "latency_p95": metrics.latency_p95,
        "latency_p99": metrics.latency_p99,
        "throughput": metrics.throughput,
        "error_rate": metrics.error_rate,
        "success_rate": metrics.success_rate,
        "cost_per_request": metrics.cost_per_request,
        "token_usage": metrics.token_usage,
        "model_performance": metrics.model_performance
    }
    
    if output:
        with open(output, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    else:
        click.echo(json.dumps(metrics_dict, indent=2))

@production.command()
@click.option('--data-file', required=True, help='Historical data for baseline')
@click.option('--metrics', multiple=True, help='Metrics for drift detection')
def setup_drift_detection(data_file: str, metrics: List[str]):
    """Setup drift detection with historical baseline"""
    
    # Load historical data
    with open(data_file, 'r') as f:
        historical_data = json.load(f)
    
    # Setup drift detector
    detector = DriftDetector()
    baseline = detector.establish_baseline(
        historical_data,
        list(metrics) or ["latency_p95", "success_rate", "error_rate"]
    )
    
    # Save baseline
    baseline_config = {
        "metrics": baseline.metrics,
        "timestamp": baseline.timestamp.isoformat(),
        "sample_size": baseline.sample_size,
        "confidence_intervals": baseline.confidence_intervals
    }
    
    with open('.agentunit/drift_baseline.json', 'w') as f:
        json.dump(baseline_config, f, indent=2)
    
    click.echo("✓ Drift detection baseline established")
```

## Implementation Timeline

### Phase 1: Core Infrastructure (2 weeks)
1. **Base Production Classes**
   - ProductionMetrics, DriftDetection, EvaluationJob data structures
   - Base integration interfaces
   - Monitoring platform enums

2. **LangSmith Integration**
   - Basic integration with LangSmith API
   - Continuous evaluation setup
   - Production metrics collection

### Phase 2: Advanced Monitoring (2 weeks)
1. **AgentOps Integration**
   - Agent action tracking
   - Session management
   - Analytics collection

2. **Drift Detection System**
   - Statistical drift detection
   - Baseline establishment
   - Remediation suggestions

### Phase 3: CLI and Automation (1 week)
1. **Production CLI Commands**
   - Setup and configuration
   - Monitoring management
   - Metrics retrieval

2. **Automated Alerts and Reports**
   - Threshold-based alerting
   - Automated reporting
   - Dashboard integration

This production integration framework enables real-time monitoring, continuous evaluation, and proactive drift detection for LLM applications in production environments.