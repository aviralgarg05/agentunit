# Multi-Agent Orchestration Framework
# Advanced testing framework for multi-agent systems and swarm intelligence

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from pathlib import Path

# Version and metadata
__version__ = "0.4.0"
__author__ = "AgentUnit Team"
__description__ = "Multi-agent system testing and orchestration framework"

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for better readability
AgentID = str
TaskID = str
MessageID = str
SessionID = str

if TYPE_CHECKING:
    from ..core import Scenario
    from ..reporting.results import ScenarioResult


class OrchestrationPattern(Enum):
    """Different patterns of multi-agent orchestration.
    
    Each pattern represents a different way agents can be organized
    and coordinate with each other in a multi-agent system.
    """
    HIERARCHICAL = "hierarchical"      # Tree-like command structure with clear authority levels
    PEER_TO_PEER = "peer_to_peer"     # Equal agents collaborating without hierarchy
    MARKETPLACE = "marketplace"        # Auction/bidding based task allocation
    PIPELINE = "pipeline"              # Sequential processing with defined stages
    SWARM = "swarm"                   # Collective intelligence with emergent coordination
    FEDERATION = "federation"         # Loosely coupled groups with local autonomy
    MESH = "mesh"                     # Fully connected network with direct communication
    HYBRID = "hybrid"                 # Combination of multiple patterns


class CommunicationMode(Enum):
    """Modes of inter-agent communication.
    
    Defines how agents can communicate with each other
    within the multi-agent system.
    """
    DIRECT_MESSAGE = "direct_message"       # Point-to-point messaging
    BROADCAST = "broadcast"                 # One-to-many messaging
    PUBLISH_SUBSCRIBE = "publish_subscribe" # Topic-based messaging
    SHARED_MEMORY = "shared_memory"         # Shared data structures
    EVENT_DRIVEN = "event_driven"           # Event-based communication
    BLACKBOARD = "blackboard"               # Shared knowledge base
    QUEUE_BASED = "queue_based"             # Message queue systems
    RPC = "rpc"                            # Remote procedure calls


class AgentState(Enum):
    """Possible states of an agent during execution."""
    IDLE = "idle"                    # Agent is waiting for tasks
    ACTIVE = "active"                # Agent is processing tasks
    WAITING = "waiting"              # Agent is waiting for responses
    BLOCKED = "blocked"              # Agent is blocked by dependencies
    ERROR = "error"                  # Agent encountered an error
    TERMINATED = "terminated"        # Agent has finished execution
    SUSPENDED = "suspended"          # Agent is temporarily suspended


class InteractionType(Enum):
    """Types of interactions between agents."""
    REQUEST = "request"              # Agent requesting action/information
    RESPONSE = "response"            # Agent responding to request
    NOTIFICATION = "notification"    # Agent notifying others of events
    DELEGATION = "delegation"        # Agent delegating task to another
    COLLABORATION = "collaboration"  # Agents working together
    NEGOTIATION = "negotiation"      # Agents negotiating terms
    CONFLICT = "conflict"            # Agents in disagreement
    HANDOFF = "handoff"             # Task transfer between agents


@dataclass(frozen=True)
class AgentRole:
    """Definition of an agent's role in the multi-agent system.
    
    Attributes:
        name: Unique name identifying the role
        description: Human-readable description of the role
        responsibilities: List of what this role is responsible for
        capabilities: List of what this role can do
        authority_level: Authority level (0-10 scale, 10 being highest)
        specialization: Primary area of expertise
        can_delegate: Whether this role can delegate tasks to others
        can_escalate: Whether this role can escalate issues
        required_skills: Skills required for this role
        communication_protocols: Allowed communication modes for this role
    """
    name: str
    description: str
    responsibilities: List[str]
    capabilities: List[str]
    authority_level: int = 5  # 0-10 scale
    specialization: str = "general"
    can_delegate: bool = False
    can_escalate: bool = True
    required_skills: List[str] = field(default_factory=list)
    communication_protocols: List[CommunicationMode] = field(
        default_factory=lambda: [CommunicationMode.DIRECT_MESSAGE]
    )
    
    def __post_init__(self):
        """Validate role configuration."""
        if not (0 <= self.authority_level <= 10):
            raise ValueError("Authority level must be between 0 and 10")
        if not self.name.strip():
            raise ValueError("Role name cannot be empty")


@dataclass
class AgentMetadata:
    """Extended metadata about an agent in the system.
    
    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name
        role: Role definition for this agent
        state: Current state of the agent
        created_at: When the agent was created
        last_active: Last time agent was active
        performance_metrics: Real-time performance data
        configuration: Agent-specific configuration
        tags: Tags for categorization and filtering
    """
    agent_id: AgentID
    name: str
    role: AgentRole
    state: AgentState = AgentState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def update_activity(self):
        """Update the last active timestamp."""
        self.last_active = datetime.now()
    
    def update_state(self, new_state: AgentState):
        """Update agent state and activity timestamp."""
        self.state = new_state
        self.update_activity()


@dataclass
class AgentInteraction:
    """Represents a single interaction between agents.
    
    This captures all the details of communication or coordination
    between agents in the multi-agent system.
    """
    interaction_id: MessageID = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    from_agent: AgentID = ""
    to_agent: Union[AgentID, List[AgentID]] = ""  # Support for broadcast
    interaction_type: InteractionType = InteractionType.REQUEST
    message_type: str = "generic"
    content: Any = None
    success: bool = True
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_interaction_id: Optional[MessageID] = None  # For conversation threading
    session_id: Optional[SessionID] = None
    
    def __post_init__(self):
        """Validate interaction data."""
        if not self.from_agent:
            raise ValueError("from_agent is required")
        if not self.to_agent:
            raise ValueError("to_agent is required")


@dataclass
class HandoffEvent:
    """Represents a task handoff between agents.
    
    Captures the details when one agent transfers responsibility
    for a task to another agent.
    """
    handoff_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    from_agent: AgentID = ""
    to_agent: AgentID = ""
    task_id: TaskID = ""
    task_context: Dict[str, Any] = field(default_factory=dict)
    handoff_reason: str = "unspecified"
    success: bool = True
    handoff_time: float = 0.0
    completion_status: str = "in_progress"  # in_progress, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_completed(self, success: bool = True, metadata: Dict[str, Any] = None):
        """Mark the handoff as completed."""
        self.completion_status = "completed" if success else "failed"
        self.success = success
        if metadata:
            self.metadata.update(metadata)


@dataclass
class ConflictEvent:
    """Represents a conflict between agents.
    
    Captures disagreements, resource conflicts, or coordination
    issues between agents in the system.
    """
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agents_involved: List[AgentID] = field(default_factory=list)
    conflict_type: str = "unspecified"  # 'resource', 'priority', 'strategy', 'information'
    description: str = ""
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'
    resolution_method: str = "unresolved"
    resolution_time: float = 0.0
    resolved: bool = False
    resolution_details: Dict[str, Any] = field(default_factory=dict)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self, method: str, details: Dict[str, Any] = None):
        """Mark the conflict as resolved."""
        self.resolved = True
        self.resolution_method = method
        self.resolution_time = (datetime.now() - self.timestamp).total_seconds()
        if details:
            self.resolution_details = details


@dataclass
class TaskAllocation:
    """Represents task allocation in multi-agent system."""
    task_id: TaskID = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    assigned_agent: AgentID = ""
    task_description: str = ""
    priority: int = 5  # 1-10 scale
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    dependencies: List[TaskID] = field(default_factory=list)
    status: str = "assigned"  # assigned, in_progress, completed, failed, cancelled
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentProtocol(Protocol):
    """Protocol defining the interface for multi-agent systems."""
    
    def get_agent_roles(self) -> Dict[AgentID, AgentRole]:
        """Get all agent roles in the system."""
        ...
    
    def get_orchestration_pattern(self) -> OrchestrationPattern:
        """Get the orchestration pattern used."""
        ...
    
    def get_communication_modes(self) -> List[CommunicationMode]:
        """Get supported communication modes."""
        ...
    
    def start_monitoring(self) -> None:
        """Start monitoring agent interactions."""
        ...
    
    def stop_monitoring(self) -> None:
        """Stop monitoring agent interactions."""
        ...


class MultiAgentAdapter(ABC):
    """Abstract base class for multi-agent system adapters.
    
    This class provides the foundation for integrating different
    multi-agent frameworks with AgentUnit's testing capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the multi-agent adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config or {}
        self.agent_metadata: Dict[AgentID, AgentMetadata] = {}
        self.interaction_history: List[AgentInteraction] = []
        self.handoff_history: List[HandoffEvent] = []
        self.conflict_history: List[ConflictEvent] = []
        self.task_allocations: List[TaskAllocation] = []
        self._monitoring_active = False
        self._session_id = str(uuid.uuid4())
        self._created_at = datetime.now()
        
        # Event callbacks for real-time monitoring
        self._interaction_callbacks: List[Callable[[AgentInteraction], None]] = []
        self._handoff_callbacks: List[Callable[[HandoffEvent], None]] = []
        self._conflict_callbacks: List[Callable[[ConflictEvent], None]] = []
        
        logger.info(f"Initialized {self.__class__.__name__} with session {self._session_id}")
    
    @abstractmethod
    def get_agent_roles(self) -> Dict[AgentID, AgentRole]:
        """Get all agent roles in the system.
        
        Returns:
            Dictionary mapping agent IDs to their roles
        """
        raise NotImplementedError("Subclasses must implement get_active_agents")
    
    @abstractmethod
    def get_orchestration_pattern(self) -> OrchestrationPattern:
        """Get the orchestration pattern used by this system.
        
        Returns:
            The orchestration pattern enum value
        """
        raise NotImplementedError("Subclasses must implement get_orchestration_pattern")
    
    @abstractmethod
    def get_communication_modes(self) -> List[CommunicationMode]:
        """Get supported communication modes.
        
        Returns:
            List of supported communication modes
        """
        raise NotImplementedError("Subclasses must implement get_communication_modes")
    
    @abstractmethod
    async def create_scenario(self, scenario_config: Dict[str, Any]) -> 'Scenario':
        """Create a multi-agent scenario from configuration.
        
        Args:
            scenario_config: Configuration for the scenario
            
        Returns:
            Created scenario object
        """
        raise NotImplementedError("Subclasses must implement create_scenario")
    
    @abstractmethod
    async def run_scenario(self, scenario: 'Scenario') -> 'ScenarioResult':
        """Run a multi-agent scenario.
        
        Args:
            scenario: The scenario to run
            
        Returns:
            Results of the scenario execution
        """
        raise NotImplementedError("Subclasses must implement run_scenario")
    
    @abstractmethod
    async def _setup_interaction_hooks(self) -> None:
        """Setup hooks to monitor interactions.
        
        This method should be implemented by subclasses to hook into
        the specific multi-agent framework's communication system.
        """
        raise NotImplementedError("Subclasses must implement _setup_interaction_hooks")
    
    async def start_monitoring(self) -> None:
        """Start monitoring agent interactions."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        await self._setup_interaction_hooks()
        logger.info(f"Started monitoring for session {self._session_id}")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring agent interactions."""
        if not self._monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self._monitoring_active = False
        logger.info(f"Stopped monitoring for session {self._session_id}")
    
    def register_agent(self, agent_id: AgentID, role: AgentRole, **kwargs) -> AgentMetadata:
        """Register an agent with the system.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role definition for the agent
            **kwargs: Additional metadata for the agent
            
        Returns:
            Created agent metadata
        """
        if agent_id in self.agent_metadata:
            logger.warning(f"Agent {agent_id} is already registered")
            return self.agent_metadata[agent_id]
        
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=kwargs.get("name", agent_id),
            role=role,
            configuration=kwargs.get("configuration", {}),
            tags=kwargs.get("tags", [])
        )
        
        self.agent_metadata[agent_id] = metadata
        logger.info(f"Registered agent {agent_id} with role {role.name}")
        
        return metadata
    
    def _record_interaction(self, interaction: AgentInteraction) -> None:
        """Record an agent interaction.
        
        Args:
            interaction: The interaction to record
        """
        if not self._monitoring_active:
            return
        
        # Add session ID if not set
        if not interaction.session_id:
            interaction.session_id = self._session_id
        
        self.interaction_history.append(interaction)
        
        # Update agent activity
        if interaction.from_agent in self.agent_metadata:
            self.agent_metadata[interaction.from_agent].update_activity()
        
        # Trigger callbacks
        for callback in self._interaction_callbacks:
            try:
                callback(interaction)
            except Exception as e:
                logger.error(f"Error in interaction callback: {e}")
        
        logger.debug(f"Recorded interaction {interaction.interaction_id}")
    
    def _record_handoff(self, handoff: HandoffEvent) -> None:
        """Record a task handoff.
        
        Args:
            handoff: The handoff event to record
        """
        if not self._monitoring_active:
            return
        
        self.handoff_history.append(handoff)
        
        # Update agent states
        if handoff.from_agent in self.agent_metadata:
            self.agent_metadata[handoff.from_agent].update_state(AgentState.ACTIVE)
        if handoff.to_agent in self.agent_metadata:
            self.agent_metadata[handoff.to_agent].update_state(AgentState.ACTIVE)
        
        # Trigger callbacks
        for callback in self._handoff_callbacks:
            try:
                callback(handoff)
            except Exception as e:
                logger.error(f"Error in handoff callback: {e}")
        
        logger.debug(f"Recorded handoff {handoff.handoff_id}")
    
    def _record_conflict(self, conflict: ConflictEvent) -> None:
        """Record a conflict event.
        
        Args:
            conflict: The conflict event to record
        """
        if not self._monitoring_active:
            return
        
        self.conflict_history.append(conflict)
        
        # Trigger callbacks
        for callback in self._conflict_callbacks:
            try:
                callback(conflict)
            except Exception as e:
                logger.error(f"Error in conflict callback: {e}")
        
        logger.info(f"Recorded conflict {conflict.conflict_id} involving {len(conflict.agents_involved)} agents")
    
    def add_interaction_callback(self, callback: Callable[[AgentInteraction], None]) -> None:
        """Add a callback for interaction events.
        
        Args:
            callback: Function to call when interactions occur
        """
        self._interaction_callbacks.append(callback)
    
    def add_handoff_callback(self, callback: Callable[[HandoffEvent], None]) -> None:
        """Add a callback for handoff events.
        
        Args:
            callback: Function to call when handoffs occur
        """
        self._handoff_callbacks.append(callback)
    
    def add_conflict_callback(self, callback: Callable[[ConflictEvent], None]) -> None:
        """Add a callback for conflict events.
        
        Args:
            callback: Function to call when conflicts occur
        """
        self._conflict_callbacks.append(callback)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session.
        
        Returns:
            Dictionary containing session statistics
        """
        return {
            "session_id": self._session_id,
            "created_at": self._created_at.isoformat(),
            "monitoring_active": self._monitoring_active,
            "total_agents": len(self.agent_metadata),
            "total_interactions": len(self.interaction_history),
            "total_handoffs": len(self.handoff_history),
            "total_conflicts": len(self.conflict_history),
            "agent_states": {
                agent_id: metadata.state.value 
                for agent_id, metadata in self.agent_metadata.items()
            },
            "orchestration_pattern": self.get_orchestration_pattern().value,
            "communication_modes": [mode.value for mode in self.get_communication_modes()]
        }
    
    def export_session_data(self, filepath: Union[str, Path]) -> None:
        """Export session data to a file.
        
        Args:
            filepath: Path to save the session data
        """
        import json
        
        def datetime_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        data = {
            "session_summary": self.get_session_summary(),
            "agent_metadata": {
                agent_id: {
                    "agent_id": metadata.agent_id,
                    "name": metadata.name,
                    "role": {
                        "name": metadata.role.name,
                        "description": metadata.role.description,
                        "responsibilities": metadata.role.responsibilities,
                        "capabilities": metadata.role.capabilities,
                        "authority_level": metadata.role.authority_level,
                        "specialization": metadata.role.specialization
                    },
                    "state": metadata.state.value,
                    "created_at": metadata.created_at,
                    "last_active": metadata.last_active,
                    "performance_metrics": metadata.performance_metrics,
                    "tags": metadata.tags
                }
                for agent_id, metadata in self.agent_metadata.items()
            },
            "interactions": [
                {
                    "interaction_id": interaction.interaction_id,
                    "timestamp": interaction.timestamp,
                    "from_agent": interaction.from_agent,
                    "to_agent": interaction.to_agent,
                    "interaction_type": interaction.interaction_type.value,
                    "message_type": interaction.message_type,
                    "success": interaction.success,
                    "response_time": interaction.response_time,
                    "metadata": interaction.metadata
                }
                for interaction in self.interaction_history
            ],
            "handoffs": [
                {
                    "handoff_id": handoff.handoff_id,
                    "timestamp": handoff.timestamp,
                    "from_agent": handoff.from_agent,
                    "to_agent": handoff.to_agent,
                    "task_id": handoff.task_id,
                    "handoff_reason": handoff.handoff_reason,
                    "success": handoff.success,
                    "handoff_time": handoff.handoff_time,
                    "completion_status": handoff.completion_status
                }
                for handoff in self.handoff_history
            ],
            "conflicts": [
                {
                    "conflict_id": conflict.conflict_id,
                    "timestamp": conflict.timestamp,
                    "agents_involved": conflict.agents_involved,
                    "conflict_type": conflict.conflict_type,
                    "description": conflict.description,
                    "severity": conflict.severity,
                    "resolved": conflict.resolved,
                    "resolution_method": conflict.resolution_method,
                    "resolution_time": conflict.resolution_time
                }
                for conflict in self.conflict_history
            ]
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=datetime_serializer)
        
        logger.info(f"Exported session data to {filepath}")


# Export main classes and enums
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__description__",
    
    # Enums
    "OrchestrationPattern",
    "CommunicationMode", 
    "AgentState",
    "InteractionType",
    
    # Data classes
    "AgentRole",
    "AgentMetadata", 
    "AgentInteraction",
    "HandoffEvent",
    "ConflictEvent",
    "TaskAllocation",
    
    # Protocol and base class
    "MultiAgentProtocol",
    "MultiAgentAdapter",
    
    # Type aliases
    "AgentID",
    "TaskID", 
    "MessageID",
    "SessionID"
]