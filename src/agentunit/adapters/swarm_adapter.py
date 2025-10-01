# OpenAI Swarm Multi-Agent Adapter
"""
OpenAI Swarm integration for AgentUnit multi-agent testing.

This adapter provides integration with OpenAI's Swarm framework for testing
lightweight multi-agent coordination patterns and handoff workflows.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import logging
import uuid
from datetime import datetime

from ..multiagent import (
    MultiAgentAdapter, AgentRole, AgentMetadata, AgentInteraction,
    OrchestrationPattern, CommunicationMode, InteractionType, SessionID
)

if TYPE_CHECKING:
    from ..core import Scenario
    from ..reporting.results import ScenarioResult

# Configure logging
logger = logging.getLogger(__name__)

try:
    from swarm import Swarm, Agent
    HAS_SWARM = True
except ImportError:
    HAS_SWARM = False
    logger.warning("OpenAI Swarm not installed. SwarmAdapter will have limited functionality.")


class SwarmAdapter(MultiAgentAdapter):
    """OpenAI Swarm integration adapter for lightweight multi-agent testing."""

    def __init__(
        self,
        client_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not HAS_SWARM:
            raise ImportError(
                "OpenAI Swarm is required for SwarmAdapter. "
                "Install with: pip install git+https://github.com/openai/swarm.git"
            )

        self.client_config = client_config or {}
        self.swarm_client = Swarm(**self.client_config)
        self.agent_functions: Dict[str, List[Any]] = {}
        self.handoff_patterns: Dict[str, Dict[str, Any]] = {}

        # Swarm-specific tracking
        self.function_calls: Dict[SessionID, List[Dict[str, Any]]] = {}
        self.agent_handoffs: Dict[SessionID, List[Dict[str, Any]]] = {}

        logger.info("SwarmAdapter initialized")

    def setup_agents(
        self,
        agent_configs: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, AgentMetadata]:
        """Setup Swarm agents from configurations."""
        agents = {}

        for config in agent_configs:
            agent_id = config.get('id', f"agent_{uuid.uuid4().hex[:8]}")

            # Create Swarm agent
            swarm_agent = self._create_swarm_agent(agent_id, config)

            # Store agent instance
            self.agent_instances[agent_id] = swarm_agent

            # Setup functions for this agent
            if 'functions' in config:
                self.agent_functions[agent_id] = config['functions']

            # Setup handoff patterns
            if 'handoffs' in config:
                self.handoff_patterns[agent_id] = config['handoffs']

            # Create metadata
            metadata = AgentMetadata(
                agent_id=agent_id,
                name=config.get('name', agent_id),
                role=AgentRole(
                    name=config.get('role_name', 'swarm_agent'),
                    description=config.get('role_description', ''),
                    responsibilities=config.get('responsibilities', []),
                    capabilities=config.get('capabilities', [])
                ),
                configuration={
                    'model': config.get('model', 'gpt-4'),
                    'instructions': config.get('instructions', ''),
                    'swarm_type': 'agent',
                    'functions': len(config.get('functions', [])),
                    'handoffs': list(config.get('handoffs', {}).keys()),
                    **config.get('metadata', {})
                }
            )

            agents[agent_id] = metadata

        logger.info(f"Setup {len(agents)} Swarm agents")
        return agents

    def _create_swarm_agent(self, agent_id: str, config: Dict[str, Any]):
        """Create a Swarm agent from configuration."""
        # Prepare functions
        functions = []
        if 'functions' in config:
            functions = [self._prepare_function(func) for func in config['functions']]

        # Create agent
        agent = Agent(
            name=config.get('name', agent_id),
            instructions=config.get('instructions', ''),
            functions=functions,
            tool_choice=config.get('tool_choice', None),
            parallel_tool_calls=config.get('parallel_tool_calls', True)
        )

        return agent

    def _prepare_function(self, func_config: Dict[str, Any]) -> Any:
        """Prepare a function for Swarm agent."""
        if callable(func_config):
            return func_config

        # If it's a function definition, create a callable
        def wrapped_function(*args, **kwargs):
            """Wrapped function for tracking."""
            # Execute original function
            if 'implementation' in func_config:
                return func_config['implementation'](*args, **kwargs)
            else:
                return f"Function {func_config.get('name')} called with args: {args}, kwargs: {kwargs}"

        # Add metadata to function
        wrapped_function.__name__ = func_config.get('name', 'unknown_function')
        wrapped_function.__doc__ = func_config.get('description', '')

        return wrapped_function

    def initiate_conversation(
        self,
        scenario: Scenario,
        initial_message: str,
        participants: List[str],
        **kwargs
    ) -> SessionID:
        """Initiate a Swarm conversation."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        try:
            # Get initial agent
            initial_agent_id = participants[0] if participants else None
            if not initial_agent_id or initial_agent_id not in self.agent_instances:
                raise ValueError("Initial agent not found")

            initial_agent = self.agent_instances[initial_agent_id]

            # Prepare messages
            messages = [{"role": "user", "content": initial_message}]

            # Run Swarm conversation
            response = self.swarm_client.run(
                agent=initial_agent,
                messages=messages,
                context_variables=kwargs.get('context_variables', {}),
                max_turns=kwargs.get('max_turns', 10),
                debug=kwargs.get('debug', False)
            )

            # Track session
            self.active_sessions[session_id] = {
                'scenario': scenario,
                'participants': participants,
                'start_time': datetime.now(),
                'status': 'active',
                'messages': response.messages,
                'agent_history': [initial_agent_id],
                'context_variables': response.context_variables
            }

            # Initialize tracking
            self.function_calls[session_id] = []
            self.agent_handoffs[session_id] = []

            # Process response for interactions
            self._process_swarm_response(session_id, response, initial_agent_id)

            logger.info(f"Initiated Swarm conversation session {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to initiate Swarm conversation: {e}")
            raise

    def _create_message_interaction(
        self,
        session_id: SessionID,
        message: dict,
        current_agent: str
    ) -> None:
        """Create an interaction from a Swarm message."""
        interaction = AgentInteraction(
            interaction_id=f"interaction_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            from_agent=current_agent,
            to_agent="",  # Swarm doesn't have explicit recipients
            interaction_type=InteractionType.MESSAGE,
            content=message.get('content', ''),
            session_id=session_id,
            metadata={
                'role': message.get('role'),
                'tool_calls': message.get('tool_calls', []),
                'swarm_response': True
            }
        )
        self.interactions.append(interaction)

    def _track_function_calls(
        self,
        session_id: SessionID,
        message: dict,
        current_agent: str
    ) -> None:
        """Track function calls from a Swarm message."""
        if 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                self.function_calls[session_id].append({
                    'agent_id': current_agent,
                    'function_name': tool_call.get('function', {}).get('name'),
                    'arguments': tool_call.get('function', {}).get('arguments'),
                    'timestamp': datetime.now()
                })

    def _track_agent_handoffs(
        self,
        session_id: SessionID,
        response: Any,
        current_agent: str
    ) -> None:
        """Track agent handoffs in Swarm response."""
        if hasattr(response, 'agent') and response.agent:
            new_agent_name = response.agent.name
            # Find agent ID by name
            new_agent_id = None
            for aid, agent in self.agent_instances.items():
                if agent.name == new_agent_name:
                    new_agent_id = aid
                    break

            if new_agent_id and new_agent_id != current_agent:
                self.agent_handoffs[session_id].append({
                    'from_agent': current_agent,
                    'to_agent': new_agent_id,
                    'timestamp': datetime.now(),
                    'context': response.context_variables
                })

    def _process_swarm_response(
        self,
        session_id: SessionID,
        response: Any,
        initial_agent_id: str
    ) -> None:
        """Process Swarm response and track interactions."""
        current_agent = initial_agent_id

        for message in response.messages:
            # Create interaction
            self._create_message_interaction(session_id, message, current_agent)

            # Track function calls
            self._track_function_calls(session_id, message, current_agent)

        # Track agent changes (handoffs)
        self._track_agent_handoffs(session_id, response, current_agent)

    def send_message(
        self,
        session_id: SessionID,
        content: str,
        **kwargs
    ) -> bool:
        """Send a message in a Swarm session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        try:
            session = self.active_sessions[session_id]

            # Add message to session
            new_message = {"role": "user", "content": content}
            session['messages'].append(new_message)

            # Get current agent
            current_agent_id = session['agent_history'][-1]
            current_agent = self.agent_instances[current_agent_id]

            # Continue conversation
            response = self.swarm_client.run(
                agent=current_agent,
                messages=session['messages'],
                context_variables=session.get('context_variables', {}),
                max_turns=kwargs.get('max_turns', 1),
                debug=kwargs.get('debug', False)
            )

            # Update session
            session['messages'] = response.messages
            session['context_variables'] = response.context_variables

            # Process new response
            self._process_swarm_response(session_id, response, current_agent_id)

            return True

        except Exception as e:
            logger.error(f"Failed to send message in Swarm session: {e}")
            return False

    def end_session(self, session_id: SessionID) -> ScenarioResult:
        """End a Swarm session and return results."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session['status'] = 'completed'
        session['end_time'] = datetime.now()

        # Collect metrics
        metrics = self._calculate_swarm_metrics(session_id)

        # Create trace log
        from ..core.trace import TraceLog
        trace = TraceLog()
        trace.record('session_complete', session_id=session_id, metrics=metrics)

        # Create scenario run
        from ..reporting.results import ScenarioRun
        scenario_run = ScenarioRun(
            scenario_name=session['scenario'].name if hasattr(session['scenario'], 'name') else 'swarm_scenario',
            case_id=session_id,
            success=True,
            metrics=metrics,
            duration_ms=0.0,  # TODO: Track actual duration
            trace=trace
        )

        # Create result
        result = ScenarioResult(name=session['scenario'].name if hasattr(session['scenario'], 'name') else 'swarm_scenario')
        result.add_run(scenario_run)

        logger.info(f"Swarm session {session_id} ended successfully")
        return result

    def _calculate_swarm_metrics(self, session_id: SessionID) -> Dict[str, Any]:
        """Calculate Swarm-specific metrics."""
        session = self.active_sessions[session_id]
        session_interactions = [
            interaction for interaction in self.interactions
            if interaction.session_id == session_id
        ]

        function_calls = self.function_calls.get(session_id, [])
        handoffs = self.agent_handoffs.get(session_id, [])

        metrics = {
            # Basic conversation metrics
            'total_messages': len(session['messages']),
            'total_interactions': len(session_interactions),
            'duration_seconds': (
                session.get('end_time', datetime.now()) - session['start_time']
            ).total_seconds(),

            # Swarm-specific metrics
            'function_calls_count': len(function_calls),
            'agent_handoffs_count': len(handoffs),
            'unique_agents_used': len(set(session.get('agent_history', []))),
            'context_variables_count': len(session.get('context_variables', {})),

            # Function usage analysis
            'function_usage': self._analyze_function_usage(function_calls),
            'handoff_patterns': self._analyze_handoff_patterns(handoffs),
            'agent_utilization': self._calculate_agent_utilization(session_id)
        }

        return metrics

    def _analyze_function_usage(
        self,
        function_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze function usage patterns."""
        if not function_calls:
            return {}

        usage = {}
        for call in function_calls:
            func_name = call.get('function_name', 'unknown')
            usage[func_name] = usage.get(func_name, 0) + 1

        return {
            'call_counts': usage,
            'unique_functions': len(usage),
            'most_used': max(usage.items(), key=lambda x: x[1]) if usage else None
        }

    def _analyze_handoff_patterns(
        self,
        handoffs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze agent handoff patterns."""
        if not handoffs:
            return {}

        patterns = {}
        for handoff in handoffs:
            pattern = f"{handoff['from_agent']} -> {handoff['to_agent']}"
            patterns[pattern] = patterns.get(pattern, 0) + 1

        return {
            'handoff_counts': patterns,
            'unique_patterns': len(patterns),
            'most_common': max(patterns.items(), key=lambda x: x[1]) if patterns else None
        }

    def _calculate_agent_utilization(
        self,
        session_id: SessionID
    ) -> Dict[str, float]:
        """Calculate how much each agent was used in the session."""
        session_interactions = [
            interaction for interaction in self.interactions
            if interaction.session_id == session_id
        ]

        if not session_interactions:
            return {}

        agent_counts = {}
        for interaction in session_interactions:
            from_agent = interaction.from_agent
            agent_counts[from_agent] = agent_counts.get(from_agent, 0) + 1

        total_interactions = len(session_interactions)
        return {
            agent_id: count / total_interactions
            for agent_id, count in agent_counts.items()
        }

    def get_orchestration_pattern(self) -> OrchestrationPattern:
        """Get the orchestration pattern used by Swarm."""
        return OrchestrationPattern.SWARM

    def get_supported_communication_modes(self) -> List[CommunicationMode]:
        """Get supported communication modes for Swarm."""
        return [
            CommunicationMode.SYNCHRONOUS,
            CommunicationMode.FUNCTION_CALLING,
            CommunicationMode.CONTEXT_SHARING
        ]