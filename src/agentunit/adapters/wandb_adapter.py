"""
Weights & Biases (Wandb) integration adapter for AgentUnit.

This module provides integration with Wandb for comprehensive AI agent monitoring,
experiment tracking, and production deployment analytics.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4


if TYPE_CHECKING:
    from agentunit.core.scenario import Scenario

from agentunit.core.trace import TraceLog
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
from agentunit.reporting.results import ScenarioResult, ScenarioRun


logger = logging.getLogger(__name__)

# Constants for Wandb metric keys
METRIC_SESSION_DURATION = "session/duration"
METRIC_SESSION_INTERACTIONS = "session/interactions_count"
METRIC_COORDINATION_EFFICIENCY = "coordination/coordination_efficiency"


class WandbAdapter(MultiAgentAdapter, ProductionIntegration):
    """
    Wandb adapter for AgentUnit multi-agent testing and production monitoring.

    Provides integration with Weights & Biases for:
    - Agent execution experiment tracking
    - Multi-agent coordination metrics
    - Performance monitoring and visualization
    - Model and artifact management
    - Production deployment analytics
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        api_key: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str = "agentunit-testing",
    ):
        """
        Initialize Wandb adapter.

        Args:
            project: Wandb project name
            entity: Optional Wandb entity (team/user)
            api_key: Optional Wandb API key
            tags: Optional default tags for runs
            group: Optional run group for organization
            job_type: Job type for categorizing runs
        """
        self.project = project
        self.entity = entity
        self.api_key = api_key
        self.default_tags = tags or ["agentunit"]
        self.group = group
        self.job_type = job_type

        # Initialize Wandb client
        self._init_wandb_client()

        # Session tracking
        self.current_session_id: str | None = None
        self.current_run = None
        self.session_agents: dict[str, AgentMetadata] = {}
        self.session_interactions: list[AgentInteraction] = []
        self.session_metrics: dict[str, Any] = {}

        logger.info(f"Wandb adapter initialized for project: {project}")

    def _init_wandb_client(self):
        """Initialize Wandb client and verify connection."""
        try:
            # Import Wandb SDK
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                logger.error("Wandb SDK not installed. Install with: pip install wandb")
                msg = "Wandb SDK required for WandbAdapter"
                raise ImportError(msg)

            # Login if API key provided
            if self.api_key:
                wandb.login(key=self.api_key)

            logger.info("Successfully connected to Wandb")

        except Exception as e:
            logger.error(f"Failed to connect to Wandb: {e}")
            raise

    @property
    def platform(self) -> MonitoringPlatform:
        """Return the monitoring platform type."""
        return MonitoringPlatform.WANDB

    def create_agent(self, role: AgentRole, agent_id: str | None = None, **kwargs) -> AgentMetadata:
        """
        Create an agent for Wandb monitoring.

        Args:
            role: Agent role definition
            agent_id: Optional specific agent ID
            **kwargs: Additional agent configuration

        Returns:
            AgentMetadata: Created agent metadata
        """
        agent_id = agent_id or f"wandb_agent_{uuid4().hex[:8]}"

        # Create agent metadata
        agent_metadata = AgentMetadata(
            agent_id=agent_id,
            name=role.name,
            role=role,
            configuration={
                "adapter_type": "wandb",
                "project": self.project,
                "entity": self.entity,
                "tags": self.default_tags,
                **kwargs,
            },
        )

        # Register agent in current session if active
        if self.current_session_id:
            self.session_agents[agent_id] = agent_metadata

            # Log agent creation to Wandb
            if self.current_run:
                self.current_run.log(
                    {
                        f"agent_created/{agent_id}": {
                            "role": role.name,
                            "responsibilities": role.responsibilities,
                            "capabilities": role.capabilities,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )

        logger.info(f"Created Wandb agent: {agent_id} with role: {role.name}")
        return agent_metadata

    def start_session(
        self,
        session_id: str,
        pattern: OrchestrationPattern,
        communication_mode: CommunicationMode,
        **kwargs,
    ) -> str:
        """
        Start a new multi-agent session with Wandb tracking.

        Args:
            session_id: Unique session identifier
            pattern: Orchestration pattern to use
            communication_mode: Communication mode for agents
            **kwargs: Additional session configuration

        Returns:
            str: Wandb run ID
        """
        self.current_session_id = session_id
        self.session_agents.clear()
        self.session_interactions.clear()
        self.session_metrics.clear()

        # Start Wandb run
        try:
            config = {
                "session_id": session_id,
                "orchestration_pattern": pattern.value,
                "communication_mode": communication_mode.value,
                "start_time": datetime.now(timezone.utc).isoformat(),
                **kwargs,
            }

            self.current_run = self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=f"AgentUnit-{session_id}",
                tags=[*self.default_tags, pattern.value, communication_mode.value],
                group=self.group,
                job_type=self.job_type,
                config=config,
                reinit=True,
            )

            # Log initial session metrics
            self.current_run.log(
                {
                    "session/started": 1,
                    "session/pattern": pattern.value,
                    "session/communication_mode": communication_mode.value,
                }
            )

            wandb_run_id = self.current_run.id
            logger.info(f"Started Wandb session: {wandb_run_id}")
            return wandb_run_id

        except Exception as e:
            logger.error(f"Failed to start Wandb session: {e}")
            return session_id

    def send_message(
        self,
        message: str,
        from_agent: str,
        to_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInteraction:
        """
        Send a message between agents with Wandb tracking.

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

        # Log interaction to Wandb
        if self.current_run:
            message_data = {
                "interactions/count": len(self.session_interactions),
                "interactions/message_length": len(message),
                f"agents/{from_agent}/messages_sent": self._get_agent_message_count(from_agent),
                "interactions/timestamp": timestamp.timestamp(),
            }

            if to_agent:
                message_data[f"agents/{to_agent}/messages_received"] = (
                    self._get_agent_received_count(to_agent)
                )

            # Create interaction table entry
            interaction_table = self.wandb.Table(
                columns=["interaction_id", "from_agent", "to_agent", "message_length", "timestamp"],
                data=[
                    [
                        interaction_id,
                        from_agent,
                        to_agent or "broadcast",
                        len(message),
                        timestamp.isoformat(),
                    ]
                ],
            )

            message_data["interactions/latest"] = interaction_table
            self.current_run.log(message_data)

        logger.debug(f"Agent message sent: {from_agent} → {to_agent}: {message[:100]}...")
        return interaction

    def end_session(self, session_id: str, final_state: dict[str, Any]) -> dict[str, Any]:
        """
        End the current session and finalize Wandb tracking.

        Args:
            session_id: Session ID to end
            final_state: Final session state

        Returns:
            Dict[str, Any]: Session summary with Wandb analytics
        """
        if session_id != self.current_session_id:
            logger.warning(f"Session ID mismatch: {session_id} != {self.current_session_id}")

        # Calculate final session metrics
        metrics = self.calculate_coordination_metrics()

        # Log final metrics to Wandb
        if self.current_run:
            final_metrics = {
                "session/ended": 1,
                "session/agents_count": len(self.session_agents),
                "session/interactions_count": len(self.session_interactions),
                "session/duration": (
                    datetime.now(timezone.utc) - self.current_run.start_time
                ).total_seconds(),
                "session/final_state": final_state.get("status", "unknown"),
                **{f"coordination/{k}": v for k, v in metrics.items()},
            }

            self.current_run.log(final_metrics)

            # Create agent summary table
            if self.session_agents:
                agent_table_data = []
                for agent_id, agent_metadata in self.session_agents.items():
                    agent_table_data.append(
                        [
                            agent_id,
                            agent_metadata.role.name,
                            len(agent_metadata.role.responsibilities),
                            len(agent_metadata.role.capabilities),
                            self._get_agent_message_count(agent_id),
                        ]
                    )

                agent_table = self.wandb.Table(
                    columns=[
                        "agent_id",
                        "role",
                        "responsibilities_count",
                        "capabilities_count",
                        "messages_sent",
                    ],
                    data=agent_table_data,
                )

                self.current_run.log({"session/agents_summary": agent_table})

            # Finish the run
            self.current_run.finish()

        session_summary = {
            "session_id": session_id,
            "wandb_run_id": self.current_run.id if self.current_run else None,
            "agents_count": len(self.session_agents),
            "interactions_count": len(self.session_interactions),
            "metrics": metrics,
            "final_state": final_state,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        # Reset session state
        self.current_session_id = None
        self.current_run = None
        self.session_agents.clear()
        self.session_interactions.clear()
        self.session_metrics.clear()

        logger.info(f"Wandb session ended: {session_id}")
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
            coordination_efficiency = min(avg_messages_per_agent / 10.0, 1.0)
        else:
            coordination_efficiency = 0.0

        # Calculate temporal distribution
        if len(self.session_interactions) > 1:
            timestamps = [interaction.timestamp for interaction in self.session_interactions]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            temporal_density = total_interactions / max(time_span, 1.0)
        else:
            temporal_density = 0.0

        # Calculate message length statistics
        message_lengths = [len(interaction.message) for interaction in self.session_interactions]
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0.0

        # Calculate agent distribution balance (Gini coefficient)
        if agent_counts:
            sorted_counts = sorted(agent_counts.values())
            n = len(sorted_counts)
            cumsum = sum((i + 1) * count for i, count in enumerate(sorted_counts))
            total_sum = sum(sorted_counts)
            gini = (2 * cumsum) / (n * total_sum) - (n + 1) / n if total_sum > 0 else 0.0
            distribution_balance = 1.0 - gini  # Convert to balance score (higher is better)
        else:
            distribution_balance = 0.0

        return {
            "total_interactions": float(total_interactions),
            "agent_participation": float(agent_participation),
            "coordination_efficiency": coordination_efficiency,
            "temporal_density": temporal_density,
            "avg_messages_per_agent": sum(agent_counts.values()) / len(agent_counts)
            if agent_counts
            else 0.0,
            "avg_message_length": avg_message_length,
            "distribution_balance": distribution_balance,
        }

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Run a scenario with Wandb integration.

        Args:
            scenario: Scenario to execute

        Returns:
            ScenarioResult: Execution results with Wandb tracking data
        """
        logger.info(f"Running scenario with Wandb: {scenario.name}")

        start_time = time.time()

        try:
            # Create a dedicated run for the scenario
            scenario_run = self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=f"Scenario-{scenario.name}",
                tags=[*self.default_tags, "scenario", scenario.name],
                group=self.group,
                job_type="scenario-execution",
                config={
                    "scenario_name": scenario.name,
                    "scenario_description": scenario.description,
                },
                reinit=True,
            )

            # Log scenario start
            scenario_run.log({"scenario/started": 1, "scenario/name": scenario.name})

            # Simulate scenario execution with multi-agent coordination
            session_id = f"scenario_{scenario.name}_{uuid4().hex[:8]}"

            # Start session (this will create a separate run)
            original_run = self.current_run

            self.start_session(
                session_id=session_id,
                pattern=OrchestrationPattern.HIERARCHICAL,
                communication_mode=CommunicationMode.DIRECT_MESSAGE,
            )

            # Create test agents
            coordinator = self.create_agent(
                AgentRole(
                    name="coordinator",
                    description="Coordinates tasks and monitors progress in the test scenario",
                    responsibilities=["coordinate tasks", "monitor progress"],
                    capabilities=["task_distribution", "progress_tracking"],
                )
            )

            worker = self.create_agent(
                AgentRole(
                    name="worker",
                    description="Executes tasks and reports results in the test scenario",
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

            # Log scenario completion to scenario run
            scenario_run.log(
                {
                    "scenario/completed": 1,
                    "scenario/execution_time": execution_time,
                    "scenario/passed": True,
                    "scenario/agents_count": 2,
                    "scenario/interactions_count": 2,
                }
            )

            # Create trace log
            trace_log = TraceLog()
            trace_log.record("session_start", session_id=session_id)
            trace_log.record(
                "agent_interaction",
                from_agent="coordinator",
                to_agent="worker",
                message="Start task execution",
            )
            trace_log.record(
                "agent_interaction",
                from_agent="worker",
                to_agent="coordinator",
                message="Task completed successfully",
            )
            trace_log.record(
                "performance_metrics",
                agents_count=2,
                interactions_count=2,
                execution_time=execution_time,
            )
            trace_log.record(
                "session_metadata",
                wandb_scenario_run_id=scenario_run.id,
                wandb_session_run_id=session_summary.get("wandb_run_id"),
                session_summary=session_summary,
                coordination_metrics=session_summary.get("metrics", {}),
            )

            # Create scenario run
            scenario_run_obj = ScenarioRun(
                scenario_name=scenario.name,
                case_id=session_id,
                success=True,
                metrics={
                    "execution_time": execution_time,
                    "agents_count": 2,
                    "interactions_count": 2,
                },
                duration_ms=int(execution_time * 1000),
                trace=trace_log,
            )

            # Create result and add run
            result = ScenarioResult(name=scenario.name)
            result.add_run(scenario_run_obj)

            # Finish scenario run
            scenario_run.finish()

            # Restore original run context
            self.current_run = original_run

            logger.info(f"Scenario completed: {scenario.name}")
            return result

        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")

            # Log scenario failure
            if "scenario_run" in locals():
                scenario_run.log(
                    {
                        "scenario/failed": 1,
                        "scenario/error": str(e),
                        "scenario/execution_time": time.time() - start_time,
                    }
                )
                scenario_run.finish()

            # Create trace log for error case
            error_execution_time = time.time() - start_time
            trace_log = TraceLog()
            trace_log.record("scenario_error", error=str(e), execution_time=error_execution_time)
            trace_log.record(
                "error_metadata",
                wandb_scenario_run_id=scenario_run.id if "scenario_run" in locals() else None,
                error_details=str(e),
            )

            # Create scenario run for error case
            scenario_run_obj = ScenarioRun(
                scenario_name=scenario.name,
                case_id=f"error_{uuid4().hex[:8]}",
                success=False,
                metrics={"execution_time": error_execution_time},
                duration_ms=int(error_execution_time * 1000),
                trace=trace_log,
                error=str(e),
            )

            # Create result and add run
            result = ScenarioResult(name=scenario.name)
            result.add_run(scenario_run_obj)

            return result

    def collect_metrics(self, scenario: Any, result: Any, **kwargs) -> ProductionMetrics:
        """
        Collect production metrics from Wandb.

        Returns:
            ProductionMetrics: Current production metrics
        """
        try:
            # Get runs from Wandb API
            api = self.wandb.Api()
            runs = api.runs(f"{self.entity}/{self.project}" if self.entity else self.project)

            if not runs:
                return ProductionMetrics(
                    evaluation_id=str(uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    scenario_name="wandb_metrics_empty",
                    custom_metrics={},
                )

            # Analyze recent runs (last 100)
            recent_runs = list(runs)[:100]

            total_runs = len(recent_runs)
            successful_runs = sum(1 for run in recent_runs if run.state == "finished")
            failed_runs = sum(1 for run in recent_runs if run.state == "failed")
            running_runs = sum(1 for run in recent_runs if run.state == "running")

            # Calculate duration statistics
            durations = []
            for run in recent_runs:
                if hasattr(run, "summary") and METRIC_SESSION_DURATION in run.summary:
                    durations.append(run.summary[METRIC_SESSION_DURATION])

            avg_duration = sum(durations) / len(durations) if durations else 0.0

            # Calculate interaction statistics
            interaction_counts = []
            for run in recent_runs:
                if hasattr(run, "summary") and METRIC_SESSION_INTERACTIONS in run.summary:
                    interaction_counts.append(run.summary[METRIC_SESSION_INTERACTIONS])

            avg_interactions = (
                sum(interaction_counts) / len(interaction_counts) if interaction_counts else 0.0
            )

            metrics = {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "running_runs": running_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0,
                "avg_duration": avg_duration,
                "avg_interactions": avg_interactions,
            }

            return ProductionMetrics(
                evaluation_id=str(uuid4()),
                timestamp=datetime.now(timezone.utc),
                scenario_name="wandb_metrics_collected",
                custom_metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Failed to collect Wandb metrics: {e}")
            return ProductionMetrics(
                evaluation_id=str(uuid4()),
                timestamp=datetime.now(timezone.utc),
                scenario_name="wandb_metrics_error",
                custom_metrics={"error": str(e)},
            )

    def establish_baseline(
        self, historical_data: list[dict[str, Any]], metrics: list[str], **kwargs
    ) -> BaselineMetrics:
        """
        Establish baseline metrics from historical Wandb data.

        Args:
            historical_data: Historical data for analysis
            metrics: List of metrics to calculate baseline for
            **kwargs: Additional parameters (e.g., days for lookback period)

        Returns:
            BaselineMetrics: Calculated baseline metrics
        """
        try:
            days = kwargs.get("days", 7)  # Default to 7 days if not specified

            # If no historical data provided, get from Wandb
            runs_list = historical_data if historical_data else self._get_historical_runs(days)

            if not runs_list:
                logger.warning("No historical data found for baseline calculation")
                return BaselineMetrics(
                    id=str(uuid4()),
                    scenario_name="wandb_baseline_empty",
                    created_at=datetime.now(timezone.utc),
                    run_count=0,
                    metadata={"period_days": days, "platform": self.platform, "metrics": metrics},
                )

            # Extract and calculate baseline metrics
            baseline_metrics = self._calculate_baseline_from_runs(runs_list)

            return BaselineMetrics(
                id=str(uuid4()),
                scenario_name="wandb_baseline_calculated",
                created_at=datetime.now(timezone.utc),
                run_count=len(runs_list),
                metadata={
                    "period_days": days,
                    "platform": self.platform,
                    "metrics": baseline_metrics,
                },
            )

        except Exception as e:
            logger.error(f"Failed to establish Wandb baseline: {e}")
            days = kwargs.get("days", 7)
            return BaselineMetrics(
                id=str(uuid4()),
                scenario_name="wandb_baseline_error",
                created_at=datetime.now(timezone.utc),
                run_count=0,
                metadata={
                    "period_days": days,
                    "platform": self.platform,
                    "error": str(e),
                    "metrics": metrics,
                },
            )

    def _get_historical_runs(self, days: int):
        """Get historical runs from Wandb within the specified date range."""
        from datetime import timedelta

        api = self.wandb.Api()

        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Query runs in date range
        runs = api.runs(
            f"{self.entity}/{self.project}" if self.entity else self.project,
            filters={"created_at": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}},
        )

        return list(runs)

    def _calculate_baseline_from_runs(self, runs_list):
        """Calculate baseline statistics from a list of runs."""
        # Extract metrics from runs
        durations = []
        interaction_counts = []
        coordination_efficiencies = []
        success_count = 0

        for run in runs_list:
            if run.state == "finished":
                success_count += 1

            if hasattr(run, "summary") and run.summary:
                self._extract_run_metrics(
                    run.summary, durations, interaction_counts, coordination_efficiencies
                )

        # Calculate baseline statistics
        return self._compute_baseline_statistics(
            durations, interaction_counts, coordination_efficiencies, success_count, len(runs_list)
        )

    def _extract_run_metrics(
        self, summary, durations, interaction_counts, coordination_efficiencies
    ):
        """Extract metrics from a single run summary."""
        if METRIC_SESSION_DURATION in summary:
            durations.append(summary[METRIC_SESSION_DURATION])

        if METRIC_SESSION_INTERACTIONS in summary:
            interaction_counts.append(summary[METRIC_SESSION_INTERACTIONS])

        if METRIC_COORDINATION_EFFICIENCY in summary:
            coordination_efficiencies.append(summary[METRIC_COORDINATION_EFFICIENCY])

    def _compute_baseline_statistics(
        self, durations, interaction_counts, coordination_efficiencies, success_count, total_runs
    ):
        """Compute baseline statistics from extracted metrics."""
        import statistics

        return {
            "avg_duration": statistics.mean(durations) if durations else 0.0,
            "median_duration": statistics.median(durations) if durations else 0.0,
            "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "success_rate": success_count / total_runs,
            "avg_interactions": statistics.mean(interaction_counts) if interaction_counts else 0.0,
            "median_interactions": statistics.median(interaction_counts)
            if interaction_counts
            else 0.0,
            "avg_coordination_efficiency": statistics.mean(coordination_efficiencies)
            if coordination_efficiencies
            else 0.0,
            "total_runs": total_runs,
        }

    def _get_agent_message_count(self, agent_id: str) -> int:
        """Get the number of messages sent by an agent."""
        return sum(
            1 for interaction in self.session_interactions if interaction.from_agent == agent_id
        )

    def _get_agent_received_count(self, agent_id: str) -> int:
        """Get the number of messages received by an agent."""
        return sum(
            1 for interaction in self.session_interactions if interaction.to_agent == agent_id
        )

    def create_artifact(self, name: str, artifact_type: str, description: str | None = None) -> Any:
        """
        Create a Wandb artifact.

        Args:
            name: Artifact name
            artifact_type: Type of artifact (e.g., 'model', 'dataset', 'results')
            description: Optional description

        Returns:
            Wandb artifact object
        """
        if not self.current_run:
            msg = "No active Wandb run. Start a session first."
            raise ValueError(msg)

        return self.wandb.Artifact(name=name, type=artifact_type, description=description)

    def log_artifact(self, artifact: Any, aliases: list[str] | None = None):
        """
        Log an artifact to the current run.

        Args:
            artifact: Wandb artifact to log
            aliases: Optional aliases for the artifact
        """
        if not self.current_run:
            msg = "No active Wandb run. Start a session first."
            raise ValueError(msg)

        self.current_run.log_artifact(artifact, aliases=aliases)
        logger.info(f"Logged artifact: {artifact.name}")

    def create_sweep(self, sweep_config: dict[str, Any]) -> str:
        """
        Create a Wandb sweep for hyperparameter optimization.

        Args:
            sweep_config: Sweep configuration

        Returns:
            str: Sweep ID
        """
        sweep_id = self.wandb.sweep(sweep=sweep_config, project=self.project, entity=self.entity)

        logger.info(f"Created Wandb sweep: {sweep_id}")
        return sweep_id
