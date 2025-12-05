"""
LangSmith integration adapter for AgentUnit.

This module provides integration with LangSmith for comprehensive AI agent monitoring,
evaluation, and production deployment tracking.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4


if TYPE_CHECKING:
    from agentunit.core.scenario import Scenario

from agentunit.multiagent import (
    AgentInteraction,
    AgentMetadata,
    AgentRole,
    CommunicationMode,
    MultiAgentAdapter,
    OrchestrationPattern,
)
from agentunit.production.integrations import MonitoringPlatform, ProductionIntegration
from agentunit.production.monitoring import BaselineMetrics, ProductionMetrics
from agentunit.reporting.results import ScenarioResult


logger = logging.getLogger(__name__)


class LangSmithAdapter(MultiAgentAdapter, ProductionIntegration):
    """
    LangSmith adapter for AgentUnit multi-agent testing and production monitoring.

    Provides integration with LangSmith for:
    - Agent execution tracking and tracing
    - Multi-agent conversation monitoring
    - Performance metrics collection
    - Production deployment monitoring
    - Evaluation dataset management
    """

    def __init__(
        self,
        api_key: str,
        project_name: str = "agentunit-testing",
        endpoint: str | None = None,
        enable_tracing: bool = True,
        enable_feedback: bool = True,
    ):
        """
        Initialize LangSmith adapter.

        Args:
            api_key: LangSmith API key
            project_name: Project name for organizing traces
            endpoint: Optional custom LangSmith endpoint
            enable_tracing: Whether to enable automatic tracing
            enable_feedback: Whether to collect feedback data
        """
        self.api_key = api_key
        self.project_name = project_name
        self.endpoint = endpoint or "https://api.smith.langchain.com"
        self.enable_tracing = enable_tracing
        self.enable_feedback = enable_feedback

        # Initialize LangSmith client
        self._init_langsmith_client()

        # Session tracking
        self.current_session_id: str | None = None
        self.session_agents: dict[str, AgentMetadata] = {}
        self.session_interactions: list[AgentInteraction] = []
        self.session_traces: list[str] = []

        logger.info(f"LangSmith adapter initialized for project: {project_name}")

    def _init_langsmith_client(self):
        """Initialize LangSmith client and verify connection."""
        try:
            # Import LangSmith SDK
            from langsmith import Client

            self.client = Client(api_url=self.endpoint, api_key=self.api_key)

            # Verify connection
            self.client.list_projects(limit=1)
            logger.info("Successfully connected to LangSmith")

        except ImportError:
            logger.error("LangSmith SDK not installed. Install with: pip install langsmith")
            msg = "LangSmith SDK required for LangSmithAdapter"
            raise ImportError(msg)
        except Exception as e:
            logger.error(f"Failed to connect to LangSmith: {e}")
            raise

    @property
    def platform(self) -> MonitoringPlatform:
        """Return the monitoring platform type."""
        return MonitoringPlatform.LANGSMITH

    def create_agent(self, role: AgentRole, agent_id: str | None = None, **kwargs) -> AgentMetadata:
        """
        Create an agent for LangSmith monitoring.

        Args:
            role: Agent role definition
            agent_id: Optional specific agent ID
            **kwargs: Additional agent configuration

        Returns:
            AgentMetadata: Created agent metadata
        """
        agent_id = agent_id or f"langsmith_agent_{uuid4().hex[:8]}"

        # Create agent metadata
        agent_metadata = AgentMetadata(
            agent_id=agent_id,
            name=role.name,
            role=role,
            configuration={
                "adapter_type": "langsmith",
                "project_name": self.project_name,
                "enable_tracing": self.enable_tracing,
                "enable_feedback": self.enable_feedback,
                **kwargs,
            },
        )

        # Register agent in current session if active
        if self.current_session_id:
            self.session_agents[agent_id] = agent_metadata

        logger.info(f"Created LangSmith agent: {agent_id} with role: {role.name}")
        return agent_metadata

    def start_session(
        self,
        session_id: str,
        pattern: OrchestrationPattern,
        communication_mode: CommunicationMode,
        **kwargs,
    ) -> str:
        """
        Start a new multi-agent session with LangSmith tracing.

        Args:
            session_id: Unique session identifier
            pattern: Orchestration pattern to use
            communication_mode: Communication mode for agents
            **kwargs: Additional session configuration

        Returns:
            str: LangSmith session/run ID
        """
        self.current_session_id = session_id
        self.session_agents.clear()
        self.session_interactions.clear()
        self.session_traces.clear()

        # Create LangSmith session run
        session_metadata = {
            "session_id": session_id,
            "orchestration_pattern": pattern.value,
            "communication_mode": communication_mode.value,
            "start_time": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }

        try:
            # Create a session run in LangSmith
            run = self.client.create_run(
                name=f"AgentUnit Session: {session_id}",
                run_type="chain",
                project_name=self.project_name,
                inputs={"session_metadata": session_metadata},
                tags=["agentunit", "multi-agent", pattern.value, communication_mode.value],
            )

            langsmith_session_id = str(run.id)
            logger.info(f"Started LangSmith session: {langsmith_session_id}")
            return langsmith_session_id

        except Exception as e:
            logger.error(f"Failed to start LangSmith session: {e}")
            return session_id

    def send_message(
        self,
        message: str,
        from_agent: str,
        to_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInteraction:
        """
        Send a message between agents with LangSmith tracing.

        Args:
            message: Message content
            from_agent: Sender agent ID
            to_agent: Optional recipient agent ID
            metadata: Optional interaction metadata

        Returns:
            AgentInteraction: Recorded interaction
        """
        interaction_id = f"interaction_{uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc)

        # Create interaction record
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            from_agent=from_agent,
            to_agent=to_agent,
            content=message,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        # Record in session
        if self.current_session_id:
            self.session_interactions.append(interaction)

        # Create LangSmith trace for this interaction
        if self.enable_tracing:
            try:
                run = self.client.create_run(
                    name=f"Agent Message: {from_agent} → {to_agent or 'broadcast'}",
                    run_type="llm",
                    project_name=self.project_name,
                    inputs={
                        "message": message,
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "metadata": metadata,
                    },
                    outputs={"interaction_id": interaction_id},
                    tags=["agentunit", "agent-message"],
                    parent_run_id=self.current_session_id if self.current_session_id else None,
                )

                self.session_traces.append(str(run.id))

            except Exception as e:
                logger.warning(f"Failed to create LangSmith trace: {e}")

        logger.debug(f"Agent message sent: {from_agent} → {to_agent}: {message[:100]}...")
        return interaction

    def end_session(self, session_id: str, final_state: dict[str, Any]) -> dict[str, Any]:
        """
        End the current session and finalize LangSmith tracing.

        Args:
            session_id: Session ID to end
            final_state: Final session state

        Returns:
            Dict[str, Any]: Session summary with LangSmith trace information
        """
        if session_id != self.current_session_id:
            logger.warning(f"Session ID mismatch: {session_id} != {self.current_session_id}")

        # Calculate session metrics
        metrics = self.calculate_coordination_metrics()

        # Finalize LangSmith session run
        session_summary = {
            "session_id": session_id,
            "agents_count": len(self.session_agents),
            "interactions_count": len(self.session_interactions),
            "traces_count": len(self.session_traces),
            "metrics": metrics,
            "final_state": final_state,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        if self.enable_tracing and self.current_session_id:
            try:
                self.client.update_run(
                    run_id=self.current_session_id,
                    outputs=session_summary,
                    end_time=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.warning(f"Failed to finalize LangSmith session: {e}")

        # Reset session state
        self.current_session_id = None
        self.session_agents.clear()
        self.session_interactions.clear()
        self.session_traces.clear()

        logger.info(f"LangSmith session ended: {session_id}")
        return session_summary

    def calculate_coordination_metrics(self) -> dict[str, float]:
        """
        Calculate coordination metrics for the current session.

        Returns:
            Dict[str, float]: Calculated metrics
        """
        if not self.session_interactions:
            return {}

        # Calculate basic metrics
        total_interactions = len(self.session_interactions)
        unique_agents = set()

        for interaction in self.session_interactions:
            unique_agents.add(interaction.from_agent)
            if interaction.to_agent:
                unique_agents.add(interaction.to_agent)

        agent_participation = len(unique_agents)

        # Calculate message distribution
        agent_counts = {}
        for interaction in self.session_interactions:
            from_agent = interaction.from_agent
            agent_counts[from_agent] = agent_counts.get(from_agent, 0) + 1

        # Calculate coordination efficiency
        if agent_participation > 1:
            avg_messages_per_agent = total_interactions / agent_participation
            coordination_efficiency = min(avg_messages_per_agent / 10.0, 1.0)  # Normalize to 0-1
        else:
            coordination_efficiency = 0.0

        # Calculate temporal distribution
        if len(self.session_interactions) > 1:
            timestamps = [interaction.timestamp for interaction in self.session_interactions]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            temporal_density = total_interactions / max(time_span, 1.0)
        else:
            temporal_density = 0.0

        return {
            "total_interactions": float(total_interactions),
            "agent_participation": float(agent_participation),
            "coordination_efficiency": coordination_efficiency,
            "temporal_density": temporal_density,
            "avg_messages_per_agent": sum(agent_counts.values()) / len(agent_counts)
            if agent_counts
            else 0.0,
        }

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Run a scenario with LangSmith integration.

        Args:
            scenario: Scenario to execute

        Returns:
            ScenarioResult: Execution results with LangSmith trace data
        """
        logger.info(f"Running scenario with LangSmith: {scenario.name}")

        # Start LangSmith run for the scenario
        scenario_run_id = None
        if self.enable_tracing:
            try:
                run = self.client.create_run(
                    name=f"Scenario: {scenario.name}",
                    run_type="chain",
                    project_name=self.project_name,
                    inputs={"scenario": scenario.name, "description": scenario.description},
                    tags=["agentunit", "scenario"],
                )
                scenario_run_id = str(run.id)
            except Exception as e:
                logger.warning(f"Failed to create scenario run: {e}")

        # Execute scenario (this would typically involve running the actual test)
        start_time = time.time()

        try:
            # Simulate scenario execution with multi-agent coordination
            session_id = f"scenario_{scenario.name}_{uuid4().hex[:8]}"

            # For demonstration, create a simple multi-agent scenario
            self.start_session(
                session_id=session_id,
                pattern=OrchestrationPattern.HIERARCHICAL,
                communication_mode=CommunicationMode.DIRECT_MESSAGE,
            )

            # Create test agents
            coordinator = self.create_agent(
                AgentRole(
                    name="coordinator",
                    description="Coordinates tasks and monitors progress",
                    responsibilities=["coordinate tasks", "monitor progress"],
                    capabilities=["task_distribution", "progress_tracking"],
                )
            )

            worker = self.create_agent(
                AgentRole(
                    name="worker",
                    description="Executes tasks and reports results",
                    responsibilities=["execute tasks", "report results"],
                    capabilities=["task_execution", "result_reporting"],
                )
            )

            # Simulate interactions
            self.send_message(
                "Start task execution",
                coordinator.agent_id,
                worker.agent_id,
                {"task_type": "test_execution"},
            )

            self.send_message(
                "Task completed successfully",
                worker.agent_id,
                coordinator.agent_id,
                {"status": "completed", "result": "success"},
            )

            # End session
            final_state = {"status": "completed", "agents": 2, "interactions": 2}
            session_summary = self.end_session(session_id, final_state)

            # Calculate results
            execution_time = time.time() - start_time

            # Create trace log
            from agentunit.core.trace import TraceLog

            trace = TraceLog()
            trace.record(
                "scenario_complete", run_id=scenario_run_id, session_summary=session_summary
            )

            # Create scenario run
            from agentunit.reporting.results import ScenarioRun

            scenario_run = ScenarioRun(
                scenario_name=scenario.name,
                case_id=str(scenario_run_id),
                success=True,
                metrics=session_summary.get("metrics", {}),
                duration_ms=execution_time * 1000,
                trace=trace,
            )

            # Create result
            result = ScenarioResult(name=scenario.name)
            result.add_run(scenario_run)

            # Update LangSmith run with results
            if scenario_run_id and self.enable_tracing:
                try:
                    self.client.update_run(
                        run_id=scenario_run_id,
                        outputs={
                            "result": result.passed,
                            "execution_time": execution_time,
                            "details": result.details,
                        },
                        end_time=datetime.now(timezone.utc),
                    )
                except Exception as e:
                    logger.warning(f"Failed to update scenario run: {e}")

            logger.info(f"Scenario completed: {scenario.name}")
            return result

        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")

            # Create trace log
            from agentunit.core.trace import TraceLog

            trace = TraceLog()
            trace.record("scenario_error", error=str(e), run_id=scenario_run_id)

            # Create scenario run
            from agentunit.reporting.results import ScenarioRun

            scenario_run = ScenarioRun(
                scenario_name=scenario.name,
                case_id=str(scenario_run_id),
                success=False,
                metrics={},
                duration_ms=(time.time() - start_time) * 1000,
                trace=trace,
                error=str(e),
            )

            # Create result
            result = ScenarioResult(name=scenario.name)
            result.add_run(scenario_run)

            # Update LangSmith run with error
            if scenario_run_id and self.enable_tracing:
                try:
                    self.client.update_run(
                        run_id=scenario_run_id,
                        outputs={"error": str(e)},
                        end_time=datetime.now(timezone.utc),
                    )
                except Exception as e:
                    logger.warning(f"Failed to update failed scenario run: {e}")

            return result

    def collect_metrics(self, scenario: Any, result: Any, **kwargs) -> ProductionMetrics:
        """
        Collect production metrics from LangSmith.

        Args:
            scenario: The scenario being evaluated
            result: The result of the scenario execution
            **kwargs: Additional metadata

        Returns:
            ProductionMetrics: Current production metrics
        """
        try:
            # Query recent runs from LangSmith
            runs = list(self.client.list_runs(project_name=self.project_name, limit=100))

            if not runs:
                return ProductionMetrics(
                    evaluation_id=f"eval_{uuid4().hex[:8]}",
                    timestamp=datetime.now(timezone.utc),
                    scenario_name=kwargs.get("scenario_name", "langsmith_collection"),
                    performance={},
                    quality={},
                    reliability={},
                    efficiency={},
                    custom_metrics={},
                )

            # Calculate metrics from recent runs
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if not run.error)
            failed_runs = total_runs - successful_runs

            total_duration = sum(
                (run.end_time - run.start_time).total_seconds()
                for run in runs
                if run.end_time and run.start_time
            )
            avg_duration = total_duration / total_runs if total_runs > 0 else 0.0

            # Calculate token usage if available
            total_tokens = 0
            for run in runs:
                if hasattr(run, "outputs") and run.outputs:
                    usage = run.outputs.get("usage", {})
                    total_tokens += usage.get("total_tokens", 0)

            return ProductionMetrics(
                evaluation_id=f"eval_{uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                scenario_name=kwargs.get("scenario_name", "langsmith_collection"),
                performance={
                    "avg_duration": avg_duration,
                    "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0,
                },
                quality={
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "failed_runs": failed_runs,
                },
                efficiency={
                    "total_tokens": total_tokens,
                    "avg_tokens_per_run": total_tokens / total_runs if total_runs > 0 else 0.0,
                },
            )

        except Exception as e:
            logger.error(f"Failed to collect LangSmith metrics: {e}")
            return ProductionMetrics(
                evaluation_id=f"eval_error_{uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                scenario_name=kwargs.get("scenario_name", "langsmith_error"),
                metadata={"error": str(e)},
            )

    def establish_baseline(
        self, historical_data: list[dict[str, Any]], metrics: list[str], **kwargs
    ) -> BaselineMetrics:
        """
        Establish baseline metrics from historical LangSmith data.

        Args:
            historical_data: Historical data for baseline calculation
            metrics: List of metrics to establish baselines for
            **kwargs: Additional configuration including days parameter

        Returns:
            BaselineMetrics: Calculated baseline metrics
        """
        days = kwargs.get("days", 7)
        try:
            from datetime import timedelta

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Query historical runs
            runs = list(
                self.client.list_runs(
                    project_name=self.project_name, start_time=start_date, end_time=end_date
                )
            )

            if not runs:
                logger.warning("No historical data found for baseline calculation")
                return BaselineMetrics(
                    id=f"baseline_{uuid4().hex[:8]}",
                    scenario_name=kwargs.get("scenario_name", "langsmith_baseline"),
                    created_at=datetime.now(timezone.utc),
                    run_count=0,
                )

            # Extract metrics from runs
            durations, token_counts, success_count = self._extract_run_metrics(runs)

            # Calculate baseline statistics
            baseline_metrics = self._calculate_baseline_statistics(
                durations, token_counts, success_count, len(runs)
            )

            return BaselineMetrics(
                id=f"baseline_{uuid4().hex[:8]}",
                scenario_name=kwargs.get("scenario_name", "langsmith_baseline"),
                created_at=datetime.now(timezone.utc),
                run_count=len(runs),
                performance_baseline=baseline_metrics.get("performance", {}),
                quality_baseline=baseline_metrics.get("quality", {}),
                reliability_baseline=baseline_metrics.get("reliability", {}),
                efficiency_baseline=baseline_metrics.get("efficiency", {}),
            )

        except Exception as e:
            logger.error(f"Failed to establish LangSmith baseline: {e}")
            return BaselineMetrics(
                id=f"baseline_error_{uuid4().hex[:8]}",
                scenario_name=kwargs.get("scenario_name", "langsmith_error"),
                created_at=datetime.now(timezone.utc),
                run_count=0,
                metadata={"error": str(e)},
            )

    def _extract_run_metrics(self, runs):
        """Extract metrics from LangSmith runs."""
        durations = []
        token_counts = []
        success_count = 0

        for run in runs:
            if run.end_time and run.start_time:
                duration = (run.end_time - run.start_time).total_seconds()
                durations.append(duration)

            if not run.error:
                success_count += 1

            if hasattr(run, "outputs") and run.outputs:
                usage = run.outputs.get("usage", {})
                token_counts.append(usage.get("total_tokens", 0))

        return durations, token_counts, success_count

    def _calculate_baseline_statistics(self, durations, token_counts, success_count, total_runs):
        """Calculate baseline statistics from extracted metrics."""
        import statistics

        return {
            "avg_duration": statistics.mean(durations) if durations else 0.0,
            "median_duration": statistics.median(durations) if durations else 0.0,
            "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "success_rate": success_count / total_runs,
            "avg_tokens": statistics.mean(token_counts) if token_counts else 0.0,
            "median_tokens": statistics.median(token_counts) if token_counts else 0.0,
            "total_runs": total_runs,
        }

    def create_evaluation_dataset(
        self, name: str, examples: list[dict[str, Any]], description: str | None = None
    ) -> str:
        """
        Create an evaluation dataset in LangSmith.

        Args:
            name: Dataset name
            examples: List of example inputs/outputs
            description: Optional dataset description

        Returns:
            str: Created dataset ID
        """
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description or f"AgentUnit evaluation dataset: {name}",
            )

            # Add examples to dataset
            for example in examples:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    metadata=example.get("metadata", {}),
                )

            logger.info(f"Created LangSmith dataset: {name} with {len(examples)} examples")
            return str(dataset.id)

        except Exception as e:
            logger.error(f"Failed to create LangSmith dataset: {e}")
            raise

    def run_evaluation(self, dataset_id: str, evaluator_function: Any, **kwargs) -> dict[str, Any]:
        """
        Run evaluation on a LangSmith dataset.

        Args:
            dataset_id: Dataset ID to evaluate
            evaluator_function: Function to evaluate examples
            **kwargs: Additional evaluation parameters

        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            from langsmith.evaluation import evaluate

            results = evaluate(
                evaluator_function,
                data=dataset_id,
                project_name=f"{self.project_name}-evaluation",
                **kwargs,
            )

            logger.info(f"Completed LangSmith evaluation on dataset: {dataset_id}")
            return results

        except Exception as e:
            logger.error(f"Failed to run LangSmith evaluation: {e}")
            raise
