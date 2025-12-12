"""Simple LangGraph agent for integration testing."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback types for when LangGraph is not available
    class StateGraph:
        pass
    
    class BaseMessage:
        pass
    
    class HumanMessage:
        pass
    
    class AIMessage:
        pass
    
    END = "END"
    
    def add_messages(a, b):
        return a + b


class AgentState(TypedDict):
    """State for the simple agent."""
    messages: List[BaseMessage]
    query: str
    context: List[str]
    tools: List[str]
    metadata: Dict[str, Any]


def create_simple_agent():
    """Create a simple LangGraph agent for testing."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not available")
    
    def process_query(state: AgentState) -> AgentState:
        """Process the user query and generate a response."""
        query = state.get("query", "")
        context = state.get("context", [])
        
        # Simple response generation based on query
        if "quantum" in query.lower():
            response = "Quantum tunneling is a quantum mechanical phenomenon where particles can pass through energy barriers."
        elif "python" in query.lower():
            response = "Python is a high-level programming language known for its simplicity and readability."
        elif "weather" in query.lower():
            response = "I would need access to weather APIs to provide current weather information."
        else:
            response = f"I understand you're asking about: {query}"
        
        # Add context if available
        if context:
            response += f" Based on the context: {', '.join(context)}"
        
        # Create response message
        ai_message = AIMessage(content=response)
        
        return {
            **state,
            "messages": state.get("messages", []) + [ai_message]
        }
    
    def should_continue(state: AgentState) -> str:
        """Determine if the agent should continue processing."""
        return END
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process", process_query)
    
    # Set entry point
    workflow.set_entry_point("process")
    
    # Add edges
    workflow.add_conditional_edges(
        "process",
        should_continue,
        {END: END}
    )
    
    # Compile the graph
    return workflow.compile()


def invoke_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the agent with a payload compatible with AgentUnit."""
    if not LANGGRAPH_AVAILABLE:
        # Return a mock response when LangGraph is not available
        return {
            "result": f"Mock response for: {payload.get('query', 'unknown query')}",
            "events": []
        }
    
    agent = create_simple_agent()
    
    # Convert payload to agent state
    initial_state = {
        "messages": [HumanMessage(content=payload.get("query", ""))],
        "query": payload.get("query", ""),
        "context": payload.get("context", []),
        "tools": payload.get("tools", []),
        "metadata": payload.get("metadata", {})
    }
    
    # Run the agent
    result = agent.invoke(initial_state)
    
    # Extract the final response
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    final_response = final_message.content if final_message else "No response generated"
    
    return {
        "result": final_response,
        "events": [
            {"type": "agent_start", "query": payload.get("query")},
            {"type": "agent_response", "content": final_response}
        ]
    }


# Create a graph instance for direct use
if LANGGRAPH_AVAILABLE:
    graph = create_simple_agent()
else:
    graph = None