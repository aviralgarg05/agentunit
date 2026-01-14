# AutoGen AG2 Multi-Agent Adapter
"""
AutoGen AG2 integration for AgentUnit multi-agent testing.

This adapter provides comprehensive integration with Microsoft's AutoGen AG2 framework,
enabling testing of conversational AI systems, multi-agent workflows, and group chats.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentunit.multiagent import (
    AgentInteraction,
    AgentMetadata,
    AgentRole,
    CommunicationMode,
    InteractionType,
    MultiAgentAdapter,
    OrchestrationPattern,
    SessionID,
)
from agentunit.reporting.results import ScenarioResult


if TYPE_CHECKING:
    from agentunit.core import Scenario

# Configure logging
logger = logging.getLogger(__name__)

try:
    import autogen

    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False
    logger.warning("AutoGen not installed. AG2Adapter will have limited functionality.")


class AG2Adapter(MultiAgentAdapter):
    """AutoGen AG2 integration adapter for multi-agent testing."""

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        code_execution_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not HAS_AUTOGEN:
            msg = "AutoGen is required for AG2Adapter. Install with: pip install pyautogen"
            raise ImportError(msg)

        self.llm_config = llm_config or {}
        self.code_execution_config = code_execution_config or {"use_docker": False}
        self.group_chats: dict[str, Any] = {}
        self.conversation_history: dict[SessionID, list[dict[str, Any]]] = {}

        # AutoGen-specific metadata
        self.agent_instances: dict[str, Any] = {}
        self.group_chat_managers: dict[str, Any] = {}

        logger.info("AG2Adapter initialized")

    def setup_agents(
        self, agent_configs: list[dict[str, Any]], **kwargs
    ) -> dict[str, AgentMetadata]:
        """Setup AutoGen agents from configurations."""
        agents = {}

        for config in agent_configs:
            agent_id = config.get("id", f"agent_{uuid.uuid4().hex[:8]}")
            agent_type = config.get("type", "assistant")

            # Create AutoGen agent based on type
            if agent_type == "assistant":
                agent = self._create_assistant_agent(agent_id, config)
            elif agent_type == "user_proxy":
                agent = self._create_user_proxy_agent(agent_id, config)
            elif agent_type == "group_chat_manager":
                agent = self._create_group_chat_manager(agent_id, config)
            else:
                msg = f"Unsupported agent type: {agent_type}"
                raise ValueError(msg)

            # Store agent instance
            self.agent_instances[agent_id] = agent

            # Create metadata
            metadata = AgentMetadata(
                agent_id=agent_id,
                name=config.get("name", agent_id),
                role=AgentRole(
                    name=config.get("role_name", agent_type),
                    description=config.get("role_description", ""),
                    responsibilities=config.get("responsibilities", []),
                    capabilities=config.get("capabilities", []),
                ),
                configuration={
                    "model_config": config.get("llm_config", self.llm_config),
                    "system_message": config.get("system_message", ""),
                    "autogen_type": agent_type,
                    **config.get("metadata", {}),
                },
            )

            agents[agent_id] = metadata

        logger.info(f"Setup {len(agents)} AutoGen agents")
        return agents

    def _create_assistant_agent(self, agent_id: str, config: dict[str, Any]):
        """Create an AutoGen assistant agent."""
        return autogen.AssistantAgent(
            name=config.get("name", agent_id),
            system_message=config.get("system_message", ""),
            llm_config=config.get("llm_config", self.llm_config),
            description=config.get("description", ""),
        )

    def _create_user_proxy_agent(self, agent_id: str, config: dict[str, Any]):
        """Create an AutoGen user proxy agent."""
        return autogen.UserProxyAgent(
            name=config.get("name", agent_id),
            system_message=config.get("system_message", ""),
            code_execution_config=config.get("code_execution_config", self.code_execution_config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 10),
            description=config.get("description", ""),
        )

    def _create_group_chat_manager(self, agent_id: str, config: dict[str, Any]):
        """Create an AutoGen group chat manager."""
        # Group chat manager requires a group chat instance
        agents = config.get("agents", [])
        if not agents:
            msg = "Group chat manager requires agents"
            raise ValueError(msg)

        # Get agent instances
        agent_instances = [self.agent_instances[agent_name] for agent_name in agents]

        group_chat = autogen.GroupChat(
            agents=agent_instances,
            messages=[],
            max_round=config.get("max_round", 10),
            speaker_selection_method=config.get("speaker_selection_method", "auto"),
        )

        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=config.get("llm_config", self.llm_config),
            name=config.get("name", agent_id),
        )

        self.group_chats[agent_id] = group_chat
        return manager

    def initiate_conversation(
        self, scenario: Scenario, initial_message: str, participants: list[str], **kwargs
    ) -> SessionID:
        """Initiate a conversation between agents."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        try:
            # Determine conversation type
            if len(participants) == 2:
                # Two-agent conversation
                session_id = self._initiate_two_agent_chat(
                    session_id, initial_message, participants, **kwargs
                )
            else:
                # Multi-agent group chat
                session_id = self._initiate_group_chat(
                    session_id, initial_message, participants, **kwargs
                )

            # Track session
            self.active_sessions[session_id] = {
                "scenario": scenario,
                "participants": participants,
                "start_time": datetime.now(),
                "status": "active",
                "message_count": 0,
            }

            logger.info(f"Initiated conversation session {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to initiate conversation: {e}")
            raise

    def _initiate_two_agent_chat(
        self, session_id: SessionID, initial_message: str, participants: list[str], **kwargs
    ) -> SessionID:
        """Initiate a two-agent conversation."""
        if len(participants) != 2:
            msg = "Two-agent chat requires exactly 2 participants"
            raise ValueError(msg)

        agent1_id, agent2_id = participants
        agent1 = self.agent_instances[agent1_id]
        agent2 = self.agent_instances[agent2_id]

        # Start conversation
        result = agent1.initiate_chat(
            agent2,
            message=initial_message,
            max_turns=kwargs.get("max_turns", 10),
            summary_method=kwargs.get("summary_method", "last_msg"),
        )

        # Store conversation history
        self.conversation_history[session_id] = result.get("chat_history", [])

        return session_id

    def _initiate_group_chat(
        self, session_id: SessionID, initial_message: str, participants: list[str], **kwargs
    ) -> SessionID:
        """Initiate a group chat conversation."""
        # Find or create group chat manager
        manager_id = kwargs.get("manager_id")
        if not manager_id:
            # Create temporary group chat
            agent_instances = [self.agent_instances[pid] for pid in participants]
            group_chat = autogen.GroupChat(
                agents=agent_instances, messages=[], max_round=kwargs.get("max_round", 10)
            )
            manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=self.llm_config)
        else:
            manager = self.agent_instances[manager_id]

        # Start group conversation
        result = participants[0] and self.agent_instances[participants[0]].initiate_chat(
            manager, message=initial_message
        )

        # Store conversation history
        self.conversation_history[session_id] = result.get("chat_history", [])

        return session_id

    def send_message(
        self,
        session_id: SessionID,
        sender_id: str,
        content: str,
        recipient_id: str | None = None,
        **kwargs,
    ) -> bool:
        """Send a message in an active session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        try:
            # Log interaction
            interaction = AgentInteraction(
                interaction_id=f"interaction_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                from_agent=sender_id,
                to_agent=recipient_id,
                interaction_type=InteractionType.MESSAGE,
                content=content,
                session_id=session_id,
                metadata=kwargs,
            )

            self.interactions.append(interaction)

            # Update session
            self.active_sessions[session_id]["message_count"] += 1

            logger.debug(f"Message sent in session {session_id}: {sender_id} -> {recipient_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def get_conversation_history(self, session_id: SessionID) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])

    def end_session(self, session_id: SessionID) -> ScenarioResult:
        """End a conversation session and return results."""
        if session_id not in self.active_sessions:
            msg = f"Session {session_id} not found"
            raise ValueError(msg)

        session = self.active_sessions[session_id]
        session["status"] = "completed"
        session["end_time"] = datetime.now()

        # Collect metrics
        metrics = self._calculate_session_metrics(session_id)

        # Create trace log
        from agentunit.core.trace import TraceLog

        trace = TraceLog()
        trace.record("session_complete", session_id=session_id, metrics=metrics)

        # Create scenario run
        from agentunit.reporting.results import ScenarioRun

        scenario_run = ScenarioRun(
            scenario_name=session["scenario"].name
            if hasattr(session["scenario"], "name")
            else "ag2_scenario",
            case_id=session_id,
            success=True,
            metrics=metrics,
            duration_ms=((session["end_time"] - session["start_time"]).total_seconds() * 1000),
            trace=trace,
        )

        # Create result
        result = ScenarioResult(
            name=session["scenario"].name
            if hasattr(session["scenario"], "name")
            else "ag2_scenario"
        )
        result.add_run(scenario_run)

        logger.info(f"Session {session_id} ended successfully")
        return result

    def _calculate_session_metrics(self, session_id: SessionID) -> dict[str, Any]:
        """Calculate metrics for a completed session."""
        session = self.active_sessions[session_id]
        session_interactions = [
            interaction for interaction in self.interactions if interaction.session_id == session_id
        ]

        response_times = []

        # Go through messages one by one
        for i in range(1, len(session_interactions)):
            previous_message_time = session_interactions[i - 1].timestamp
            current_message_time = session_interactions[i].timestamp

            time_difference = (current_message_time - previous_message_time).total_seconds()

            response_times.append(time_difference)

        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        # Basic metrics
        metrics = {
            "total_messages": len(session_interactions),
            "unique_speakers": len({i.sender_id for i in session_interactions}),
            "duration_seconds": (
                session.get("end_time", datetime.now()) - session["start_time"]
            ).total_seconds(),
            "average_response_time": average_response_time,
            "conversation_turns": session.get("message_count", 0),
        }

        # Advanced metrics
        if session_interactions:
            metrics.update(
                {
                    "message_types": self._analyze_message_types(session_interactions),
                    "participation_balance": self._calculate_participation_balance(
                        session_interactions
                    ),
                    "interaction_patterns": self._analyze_interaction_patterns(
                        session_interactions
                    ),
                }
            )

        return metrics

    def _analyze_message_types(self, interactions: list[AgentInteraction]) -> dict[str, int]:
        """Analyze types of messages in interactions."""
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction.interaction_type.value
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        return type_counts

    def _calculate_participation_balance(
        self, interactions: list[AgentInteraction]
    ) -> dict[str, float]:
        """Calculate how balanced participation is across agents."""
        speaker_counts = {}
        for interaction in interactions:
            speaker_id = interaction.sender_id
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        total_messages = len(interactions)
        if total_messages == 0:
            return {}

        return {speaker_id: count / total_messages for speaker_id, count in speaker_counts.items()}

    def _analyze_interaction_patterns(self, interactions: list[AgentInteraction]) -> dict[str, Any]:
        """Analyze patterns in agent interactions."""
        patterns = {"turn_taking_regularity": 0.0, "response_chains": [], "interruption_count": 0}

        # Simple pattern analysis
        if len(interactions) > 1:
            speakers = [i.sender_id for i in interactions]

            # Calculate turn-taking regularity
            turn_changes = sum(1 for i in range(1, len(speakers)) if speakers[i] != speakers[i - 1])
            patterns["turn_taking_regularity"] = (
                turn_changes / (len(speakers) - 1) if len(speakers) > 1 else 0
            )

        return patterns

    def get_orchestration_pattern(self) -> OrchestrationPattern:
        """Get the orchestration pattern used by this adapter."""
        return OrchestrationPattern.HIERARCHICAL  # AutoGen typically uses hierarchical patterns

    def get_supported_communication_modes(self) -> list[CommunicationMode]:
        """Get supported communication modes."""
        return [
            CommunicationMode.SYNCHRONOUS,
            CommunicationMode.TURN_BASED,
            CommunicationMode.BROADCAST,
        ]
