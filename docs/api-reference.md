# AgentUnit API Reference

## Core Modules

### `agentunit.core`

#### Classes

##### `Scenario`

The main class for defining and running test scenarios.

```python
class Scenario:
    """Represents a reproducible test scenario for multi-agent systems."""
    
    def __init__(
        self,
        name: str,
        adapter: BaseAdapter,
        dataset_source: Union[DatasetSource, str, Dict],
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[CustomMetric]] = None
    ) -> None:
        """
        Initialize a new scenario.
        
        Args:
            name: Unique name for the scenario
            adapter: Platform adapter for running tests
            dataset_source: Test cases data source
            config: Optional configuration overrides
            metrics: Custom evaluation metrics
        """
```

**Methods:**

- `async run() -> ScenarioResult`: Execute the scenario and return results
- `validate() -> bool`: Validate scenario configuration
- `get_config() -> Dict[str, Any]`: Get current configuration
- `set_config(config: Dict[str, Any]) -> None`: Update configuration

**Properties:**

- `name: str`: Scenario name
- `adapter: BaseAdapter`: Platform adapter
- `dataset_source: DatasetSource`: Test data source
- `config: Dict[str, Any]`: Configuration dictionary

##### `ScenarioResult`

Contains the results of a scenario execution.

```python
@dataclass
class ScenarioResult:
    """Results from a scenario execution."""
    
    name: str
    runs: List[ScenarioRun]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate (0.0 to 1.0)."""
        if not self.runs:
            return 0.0
        successful = sum(1 for run in self.runs if run.success)
        return successful / len(self.runs)
    
    @property
    def avg_duration(self) -> float:
        """Average duration in seconds."""
        if not self.runs:
            return 0.0
        total_ms = sum(run.duration_ms for run in self.runs)
        return total_ms / len(self.runs) / 1000.0
    
    def aggregate_metric(self, name: str) -> float:
        """Aggregate a specific metric across all runs."""
        values = [run.metrics.get(name, 0.0) for run in self.runs]
        return sum(values) / len(values) if values else 0.0
    
    def filter_runs(self, predicate: Callable[[ScenarioRun], bool]) -> List[ScenarioRun]:
        """Filter runs based on a predicate function."""
        return [run for run in self.runs if predicate(run)]
    
    def get_failed_runs(self) -> List[ScenarioRun]:
        """Get all failed runs."""
        return self.filter_runs(lambda run: not run.success)
    
    def get_slow_runs(self, threshold_ms: float = 5000) -> List[ScenarioRun]:
        """Get runs slower than threshold."""
        return self.filter_runs(lambda run: run.duration_ms > threshold_ms)
```

##### `ScenarioRun`

Individual test case execution result.

```python
@dataclass
class ScenarioRun:
    """Result of a single test case execution."""
    
    scenario_name: str
    case_id: str
    success: bool
    metrics: Dict[str, float]
    duration_ms: float
    trace: Optional[TraceLog] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a specific metric value."""
        return self.metrics.get(name, default)
    
    def has_error(self) -> bool:
        """Check if run has an error."""
        return self.error is not None
```

##### `DatasetSource`

Manages test case data loading and access.

```python
class DatasetSource:
    """Source of test cases for scenario execution."""
    
    def __init__(self, cases: List[DatasetCase]) -> None:
        """Initialize with a list of test cases."""
        self.cases = cases
    
    @classmethod
    def from_file(cls, file_path: str) -> 'DatasetSource':
        """Load dataset from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSource':
        """Create dataset from dictionary."""
        test_cases = data.get('test_cases', [])
        cases = [DatasetCase(**case) for case in test_cases]
        return cls(cases)
    
    def get_cases(self) -> List[DatasetCase]:
        """Get all test cases."""
        return self.cases
    
    def get_case(self, case_id: str) -> Optional[DatasetCase]:
        """Get specific test case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None
    
    def filter_cases(self, predicate: Callable[[DatasetCase], bool]) -> List[DatasetCase]:
        """Filter cases based on predicate."""
        return [case for case in self.cases if predicate(case)]
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle test cases order."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cases)
```

##### `DatasetCase`

Individual test case definition.

```python
@dataclass
class DatasetCase:
    """A single test case in a dataset."""
    
    id: str
    input: Union[str, Dict[str, Any]]
    expected_output: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_input_text(self) -> str:
        """Get input as text string."""
        if isinstance(self.input, str):
            return self.input
        elif isinstance(self.input, dict):
            return self.input.get('prompt', str(self.input))
        else:
            return str(self.input)
    
    def get_expected_text(self) -> str:
        """Get expected output as text string."""
        if self.expected_output is None:
            return ""
        elif isinstance(self.expected_output, str):
            return self.expected_output
        elif isinstance(self.expected_output, dict):
            return self.expected_output.get('response', str(self.expected_output))
        else:
            return str(self.expected_output)
    
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
```

##### `TraceLog`

Captures detailed execution traces for debugging.

```python
class TraceLog:
    """Captures detailed execution traces."""
    
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start trace logging."""
        self.start_time = datetime.now()
    
    def end(self) -> None:
        """End trace logging."""
        self.end_time = datetime.now()
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the trace."""
        event = TraceEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data
        )
        self.events.append(event)
    
    def get_duration_ms(self) -> float:
        """Get total trace duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0
    
    def get_events_by_type(self, event_type: str) -> List[TraceEvent]:
        """Get events of specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get message events for conversation analysis."""
        message_events = self.get_events_by_type('message')
        return [event.data for event in message_events]
```

#### Functions

##### `run_suite`

Execute multiple scenarios concurrently.

```python
async def run_suite(
    scenarios: List[Scenario],
    max_concurrent: int = 5,
    fail_fast: bool = False
) -> List[ScenarioResult]:
    """
    Run multiple scenarios concurrently.
    
    Args:
        scenarios: List of scenarios to execute
        max_concurrent: Maximum concurrent scenarios
        fail_fast: Stop on first failure
        
    Returns:
        List of scenario results
    """
```

---

## Adapter Framework

### `agentunit.adapters`

#### Base Classes

##### `BaseAdapter`

Abstract base class for all platform adapters.

```python
class BaseAdapter(ABC):
    """Abstract base class for platform adapters."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize adapter with configuration."""
        self.config = config
        self._prepared = False
    
    @abstractmethod
    def prepare(self) -> None:
        """Perform lazy setup (loading models, connecting to services)."""
        pass
    
    @abstractmethod
    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        """Execute a single test case and return results."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources after execution."""
        pass
    
    def supports_replay(self) -> bool:
        """Whether adapter supports replay functionality."""
        return False
    
    def get_capabilities(self) -> List[str]:
        """Get list of adapter capabilities."""
        return []
    
    def validate_config(self) -> bool:
        """Validate adapter configuration."""
        return True
```

##### `MultiAgentAdapter`

Base class for multi-agent platform adapters.

```python
class MultiAgentAdapter(BaseAdapter):
    """Base class for multi-agent platform adapters."""
    
    @abstractmethod
    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Run a complete scenario with multi-agent orchestration."""
        pass
    
    @abstractmethod
    def get_agent_roles(self) -> Dict[AgentID, AgentRole]:
        """Get available agent roles and their capabilities."""
        pass
    
    @abstractmethod  
    def get_communication_modes(self) -> List[CommunicationMode]:
        """Get supported communication patterns."""
        pass
    
    def supports_parallel_execution(self) -> bool:
        """Whether adapter supports parallel agent execution."""
        return False
    
    def get_max_agents(self) -> int:
        """Maximum number of agents supported."""
        return 10
```

#### Platform Adapters

##### `AG2Adapter`

AutoGen AG2 integration adapter.

```python
class AG2Adapter(MultiAgentAdapter):
    """Adapter for AutoGen AG2 multi-agent framework."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize AG2 adapter.
        
        Config options:
            model: LLM model name (default: "gpt-4")
            temperature: Model temperature (default: 0.7)
            max_turns: Maximum conversation turns (default: 10)
            agents: Agent configuration dictionary
            conversation_config: Additional conversation settings
        """
        super().__init__(config)
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_turns = config.get("max_turns", 10)
```

**Methods:**

- `prepare() -> None`: Initialize AG2 agents and LLM config
- `execute(case: DatasetCase, trace: TraceLog) -> AdapterOutcome`: Execute conversation
- `get_agent_roles() -> Dict[AgentID, AgentRole]`: Get configured agent roles
- `create_agents() -> Dict[str, Any]`: Create AG2 agent instances

##### `SwarmAdapter`

OpenAI Swarm integration adapter.

```python
class SwarmAdapter(MultiAgentAdapter):
    """Adapter for OpenAI Swarm multi-agent framework."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Swarm adapter.
        
        Config options:
            model: LLM model name (default: "gpt-4")
            temperature: Model temperature (default: 0.7)
            max_turns: Maximum conversation turns (default: 5)
            agents: Agent configuration with functions
            handoff_config: Agent handoff settings
        """
```

**Methods:**

- `prepare() -> None`: Initialize Swarm client and agents
- `execute(case: DatasetCase, trace: TraceLog) -> AdapterOutcome`: Run conversation
- `get_communication_modes() -> List[CommunicationMode]`: Get handoff patterns
- `register_functions(functions: Dict[str, Callable]) -> None`: Register agent functions

##### `LangSmithAdapter`

LangSmith monitoring and evaluation adapter.

```python
class LangSmithAdapter(BaseAdapter):
    """Adapter for LangSmith monitoring and evaluation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize LangSmith adapter.
        
        Config options:
            project_name: LangSmith project name
            session_name: Optional session name
            trace_level: Trace detail level (DEBUG, INFO, WARNING)
            auto_eval: Enable automatic evaluation
            evaluation_config: Custom evaluator settings
        """
```

**Methods:**

- `start_tracing() -> None`: Start LangSmith trace collection
- `stop_tracing() -> None`: Stop trace collection
- `create_evaluation(dataset_name: str) -> str`: Create evaluation run
- `get_traces(session_id: str) -> List[Dict]`: Retrieve execution traces

##### `AgentOpsAdapter`

AgentOps production monitoring adapter.

```python
class AgentOpsAdapter(BaseAdapter):
    """Adapter for AgentOps production monitoring."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize AgentOps adapter.
        
        Config options:
            environment: Deployment environment (dev, staging, prod)
            auto_start_session: Auto-start monitoring session
            capture_video: Enable video capture
            monitoring_config: Detailed monitoring settings
            alerting_config: Alert thresholds and channels
        """
```

**Methods:**

- `start_session(session_name: str) -> str`: Start monitoring session
- `end_session() -> None`: End current session
- `track_agent_action(action: Dict) -> None`: Track agent action
- `get_session_url() -> str`: Get session dashboard URL

##### `WandbAdapter`

Weights & Biases experiment tracking adapter.

```python
class WandbAdapter(BaseAdapter):
    """Adapter for Wandb experiment tracking."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Wandb adapter.
        
        Config options:
            project: Wandb project name
            entity: Wandb team/user name
            run_config: Run configuration settings
            logging_config: Logging preferences
            hyperparameters: Experiment hyperparameters
        """
```

**Methods:**

- `start_run(run_name: str) -> None`: Start Wandb run
- `log_metrics(metrics: Dict[str, float]) -> None`: Log metrics
- `log_artifact(path: str, name: str) -> None`: Log artifact
- `finish_run() -> None`: Complete Wandb run

---

## Multi-Agent Framework

### `agentunit.multiagent`

#### Classes

##### `AgentRole`

Defines an agent's role and capabilities.

```python
@dataclass
class AgentRole:
    """Defines an agent's role in multi-agent system."""
    
    name: str
    description: str
    responsibilities: List[str]
    capabilities: List[str]
    system_message: Optional[str] = None
    max_turns: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type."""
        return task_type in self.capabilities
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for the agent."""
        if self.system_message:
            return self.system_message
        
        prompt = f"You are {self.name}. {self.description}\n\n"
        prompt += "Your responsibilities include:\n"
        for resp in self.responsibilities:
            prompt += f"- {resp}\n"
        return prompt
```

##### `AgentInteraction`

Represents communication between agents.

```python
@dataclass
class AgentInteraction:
    """Represents a communication between agents."""
    
    source_agent: AgentID
    target_agent: AgentID
    interaction_type: InteractionType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration_since(self, reference_time: datetime) -> float:
        """Get duration since reference time in seconds."""
        delta = self.timestamp - reference_time
        return delta.total_seconds()
    
    def is_broadcast(self) -> bool:
        """Check if interaction is a broadcast to all agents."""
        return self.target_agent == "ALL" or self.target_agent == "*"
```

##### `OrchestrationPattern`

Defines agent coordination patterns.

```python
class OrchestrationPattern(Enum):
    """Agent orchestration patterns."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    HUB_AND_SPOKE = "hub_and_spoke"
    PIPELINE = "pipeline"
    ROUND_ROBIN = "round_robin"
```

---

## Monitoring Framework

### `agentunit.monitoring`

#### Classes

##### `ProductionMonitor`

Main production monitoring orchestrator.

```python
class ProductionMonitor:
    """Production monitoring and observability system."""
    
    def __init__(self, integrations: List[ProductionIntegration]) -> None:
        """Initialize with monitoring integrations."""
        self.integrations = integrations
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.active_sessions: Dict[str, MonitoringSession] = {}
    
    async def start_monitoring(self) -> None:
        """Start monitoring all configured integrations."""
        for integration in self.integrations:
            await integration.start()
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring integrations."""
        for integration in self.integrations:
            await integration.stop()
    
    async def create_session(self, session_name: str) -> SessionID:
        """Create a new monitoring session."""
        session = MonitoringSession(session_name)
        session_id = session.id
        self.active_sessions[session_id] = session
        return session_id
    
    async def track_interaction(self, interaction: AgentInteraction) -> None:
        """Track a single agent interaction."""
        await self.metrics_collector.collect_interaction_metrics(interaction)
        
        # Check for alerts
        alerts = await self.alert_manager.evaluate_alerts(interaction)
        for alert in alerts:
            await self.handle_alert(alert)
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return await self.alert_manager.get_active_alerts()
```

##### `MetricsCollector`

Collects and aggregates performance metrics.

```python
class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
    
    async def collect_interaction_metrics(self, interaction: AgentInteraction) -> None:
        """Collect metrics from an agent interaction."""
        # Response time metrics
        if 'response_time_ms' in interaction.metadata:
            response_time = interaction.metadata['response_time_ms']
            self.histograms['response_time'].append(response_time)
        
        # Count interactions by type
        self.counters[f'interactions_{interaction.interaction_type}'] += 1
        
        # Error tracking
        if interaction.metadata.get('error'):
            self.counters['errors'] += 1
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        values = self.histograms.get(metric_name, [])
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
```

---

## CLI Framework

### `agentunit.cli`

#### Main CLI Interface

```python
def main() -> None:
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        prog='agentunit',
        description='Testing and monitoring framework for multi-agent AI systems'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Multi-agent commands
    multiagent_parser = subparsers.add_parser('multiagent', help='Multi-agent testing')
    setup_multiagent_commands(multiagent_parser)
    
    # Monitoring commands  
    monitoring_parser = subparsers.add_parser('monitoring', help='Production monitoring')
    setup_monitoring_commands(monitoring_parser)
    
    # Analysis commands
    analyze_parser = subparsers.add_parser('analyze', help='Result analysis')
    setup_analysis_commands(analyze_parser)
    
    # Configuration commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    setup_config_commands(config_parser)
```

#### Command Functions

```python
async def run_multiagent_scenario(
    scenario_name: str,
    adapter_name: str,
    dataset_path: str,
    config_path: Optional[str] = None,
    output_dir: str = "./results",
    export_formats: List[str] = None,
    parallel: bool = False,
    verbose: bool = False
) -> None:
    """Run a multi-agent scenario from CLI."""

async def start_monitoring(
    environment: str,
    integrations: List[str],
    config_path: Optional[str] = None,
    real_time: bool = True,
    alerts: bool = True
) -> None:
    """Start production monitoring from CLI."""

async def analyze_results(
    input_path: str,
    compare_with: Optional[str] = None,
    metrics: List[str] = None,
    report_type: str = "summary",
    export_format: str = "json"
) -> None:
    """Analyze test results from CLI."""
```

---

## Utilities and Helpers

### `agentunit.utils`

#### Configuration Management

```python
class ConfigManager:
    """Manages AgentUnit configuration."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager."""
        
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        
    def save_config(self, config: Dict[str, Any], path: str) -> None:
        """Save configuration to file."""
        
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        
    def set_value(self, key: str, value: Any) -> None:
        """Set configuration value."""
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
```

#### Export Utilities

```python
class ResultExporter:
    """Export scenario results to various formats."""
    
    @staticmethod
    def export_json(result: ScenarioResult, path: str) -> None:
        """Export results to JSON format."""
        
    @staticmethod
    def export_html(result: ScenarioResult, path: str) -> None:
        """Export results to HTML report."""
        
    @staticmethod
    def export_xml(result: ScenarioResult, path: str) -> None:
        """Export results to JUnit XML format."""
        
    @staticmethod  
    def export_markdown(result: ScenarioResult, path: str) -> None:
        """Export results to Markdown report."""
```

---

## Custom Extensions

### Creating Custom Adapters

```python
from agentunit.adapters import BaseAdapter
from agentunit.core import DatasetCase, TraceLog, AdapterOutcome

class MyCustomAdapter(BaseAdapter):
    """Custom adapter implementation example."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # Initialize your platform connection
        
    def prepare(self) -> None:
        """Setup your platform connection."""
        self._prepared = True
        
    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        """Execute test case with your platform."""
        trace.start()
        
        try:
            # Your platform-specific logic here
            result = self._run_test_case(case)
            
            outcome = AdapterOutcome(
                success=result.success,
                output=result.output,
                metrics=result.metrics,
                metadata=result.metadata
            )
            
        except Exception as e:
            outcome = AdapterOutcome(
                success=False,
                output="",
                metrics={},
                metadata={"error": str(e)}
            )
        finally:
            trace.end()
            
        return outcome
```

### Creating Custom Metrics

```python
from agentunit.metrics import CustomMetric

class SemanticSimilarityMetric(CustomMetric):
    """Custom semantic similarity metric."""
    
    def __init__(self) -> None:
        super().__init__("semantic_similarity")
        
    def calculate(self, expected: str, actual: str) -> float:
        """Calculate semantic similarity score."""
        # Implementation here
        return similarity_score
        
    def requires_external_service(self) -> bool:
        """Whether metric requires external API calls."""
        return True
```

This API reference provides comprehensive documentation for all major components of the AgentUnit framework, enabling developers to effectively use and extend the system.