# Platform Integration Guides

## Overview

AgentUnit supports multiple AI platforms through dedicated adapters. Each adapter provides seamless integration with platform-specific features while maintaining consistency through the AgentUnit interface.

## AutoGen AG2 Integration

### Overview
AutoGen AG2 (AutoGen Second Generation) is Microsoft's advanced multi-agent conversation framework. The AG2Adapter enables AgentUnit to orchestrate and test complex agent conversations.

### Installation & Setup

```bash
# Install AutoGen AG2
pip install autogen-ag2

# Configure API keys
export OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_KEY="your-azure-key"  # Optional for Azure OpenAI
```

### Configuration

```python
from agentunit import Scenario
from agentunit.adapters import AG2Adapter

# Basic configuration
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_turns": 10,
    "timeout": 60
}

adapter = AG2Adapter(config)
```

### Advanced Configuration

```python
# Advanced AG2 configuration
advanced_config = {
    "model": "gpt-4",
    "model_config": {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    },
    "agents": {
        "user_proxy": {
            "name": "UserProxy",
            "system_message": "You are a helpful assistant.",
            "human_input_mode": "NEVER",
            "max_consecutive_auto_reply": 3
        },
        "assistant": {
            "name": "Assistant", 
            "system_message": "You are an AI assistant specialized in problem solving.",
            "llm_config": {
                "model": "gpt-4",
                "temperature": 0.5
            }
        }
    },
    "conversation_config": {
        "max_turns": 10,
        "silent": False,
        "cache_seed": None
    }
}

adapter = AG2Adapter(advanced_config)
```

### Usage Examples

```python
# Create a scenario with AG2 adapter
scenario = Scenario(
    name="multi_agent_conversation",
    adapter=adapter,
    dataset_source="conversation_prompts.json"
)

# Run the scenario
result = await scenario.run()
print(f"Success rate: {result.success_rate:.2%}")
```

### AG2-Specific Features

- **Group Chat**: Support for multi-agent group conversations
- **Code Execution**: Safe code execution with Docker
- **Function Calling**: Tool use and function execution
- **Memory**: Conversation history and context management
- **Workflow**: Complex multi-step agent workflows

### Best Practices

1. **Agent Design**: Create specialized agents with clear roles
2. **Turn Limits**: Set appropriate max_turns to prevent infinite loops
3. **Error Handling**: Implement robust error handling for network issues
4. **Resource Management**: Use timeouts and cleanup for long-running conversations
5. **Testing**: Start with simple two-agent conversations before complex groups

---

## OpenAI Swarm Integration

### Overview
OpenAI Swarm is an experimental framework for multi-agent orchestration. The SwarmAdapter provides integration with Swarm's lightweight agent coordination.

### Installation & Setup

```bash
# Install OpenAI Swarm
pip install openai-swarm

# Configure API key
export OPENAI_API_KEY="your-api-key"
```

### Configuration

```python
from agentunit.adapters import SwarmAdapter

# Basic Swarm configuration
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_turns": 5
}

adapter = SwarmAdapter(config)
```

### Advanced Configuration

```python
# Advanced Swarm configuration with custom agents
advanced_config = {
    "model": "gpt-4-turbo",
    "agents": {
        "coordinator": {
            "name": "Coordinator",
            "instructions": "You coordinate between other agents to solve complex tasks.",
            "functions": ["transfer_to_analyst", "transfer_to_executor"]
        },
        "analyst": {
            "name": "Analyst", 
            "instructions": "You analyze problems and provide detailed insights.",
            "functions": ["analyze_data", "generate_insights"]
        },
        "executor": {
            "name": "Executor",
            "instructions": "You execute tasks based on analysis and coordination.",
            "functions": ["execute_plan", "report_results"]
        }
    },
    "handoff_config": {
        "max_handoffs": 10,
        "context_variables": {}
    }
}

adapter = SwarmAdapter(advanced_config)
```

### Usage Examples

```python
# Define agent functions
def transfer_to_analyst():
    """Transfer conversation to the analyst agent."""
    return agents["analyst"]

def analyze_data(data: str):
    """Analyze the provided data."""
    return f"Analysis of {data}: [detailed analysis]"

# Create scenario with custom functions
scenario = Scenario(
    name="swarm_coordination",
    adapter=adapter,
    dataset_source="coordination_tasks.json"
)

result = await scenario.run()
```

### Swarm-Specific Features

- **Agent Handoffs**: Seamless transfer between specialized agents
- **Function Calling**: Rich function execution with context passing
- **Context Variables**: Shared state across agent interactions
- **Lightweight Design**: Minimal overhead for simple multi-agent tasks

### Best Practices

1. **Agent Specialization**: Create focused agents with specific expertise
2. **Handoff Strategy**: Design clear handoff conditions and logic
3. **Function Design**: Keep functions simple and well-documented
4. **Context Management**: Use context variables effectively for state sharing
5. **Testing**: Test individual agents before complex orchestrations

---

## LangSmith Integration

### Overview
LangSmith provides observability and evaluation for LLM applications. The LangSmithAdapter enables comprehensive monitoring and debugging of agent interactions.

### Installation & Setup

```bash
# Install LangSmith
pip install langsmith

# Configure API key and project
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="your-project-name"
export LANGCHAIN_TRACING_V2=true
```

### Configuration

```python
from agentunit.adapters import LangSmithAdapter

# Basic LangSmith configuration
config = {
    "project_name": "agentunit-testing",
    "trace_level": "INFO",
    "auto_eval": True
}

adapter = LangSmithAdapter(config)
```

### Advanced Configuration

```python
# Advanced LangSmith configuration
advanced_config = {
    "project_name": "production-monitoring",
    "session_name": "multi-agent-testing",
    "trace_config": {
        "trace_level": "DEBUG",
        "include_metadata": True,
        "trace_sampling_rate": 1.0
    },
    "evaluation_config": {
        "auto_eval": True,
        "eval_chains": ["correctness", "helpfulness", "conciseness"],
        "custom_evaluators": ["domain_accuracy", "response_quality"]
    },
    "feedback_config": {
        "collect_user_feedback": True,
        "feedback_scale": "thumbs",
        "automatic_scoring": True
    }
}

adapter = LangSmithAdapter(advanced_config)
```

### Usage Examples

```python
# Create scenario with LangSmith monitoring
scenario = Scenario(
    name="monitored_conversation",
    adapter=adapter,
    dataset_source="evaluation_dataset.json"
)

# Run with automatic tracing
result = await scenario.run()

# Access detailed traces
for run in result.runs:
    trace_url = run.metadata.get("langsmith_trace_url")
    print(f"View trace: {trace_url}")
```

### LangSmith-Specific Features

- **Automatic Tracing**: Full execution traces with timing and metadata
- **Evaluation Chains**: Built-in and custom evaluation metrics
- **Dataset Management**: Manage test datasets and examples
- **Feedback Collection**: User feedback and human evaluation
- **Performance Analytics**: Latency, cost, and usage analytics

### Best Practices

1. **Project Organization**: Use clear project and session names
2. **Trace Sampling**: Adjust sampling rates for production vs development
3. **Custom Evaluators**: Create domain-specific evaluation metrics
4. **Dataset Curation**: Maintain high-quality evaluation datasets
5. **Continuous Monitoring**: Set up alerts for performance degradation

---

## AgentOps Integration

### Overview
AgentOps provides production monitoring and observability for AI agents. The AgentOpsAdapter enables real-time monitoring and performance analytics.

### Installation & Setup

```bash
# Install AgentOps
pip install agentops

# Configure API key
export AGENTOPS_API_KEY="your-api-key"
```

### Configuration

```python
from agentunit.adapters import AgentOpsAdapter

# Basic AgentOps configuration
config = {
    "environment": "production",
    "auto_start_session": True,
    "capture_video": False
}

adapter = AgentOpsAdapter(config)
```

### Advanced Configuration

```python
# Advanced AgentOps configuration
advanced_config = {
    "environment": "production",
    "session_config": {
        "auto_start_session": True,
        "session_tags": ["agentunit", "multi-agent", "testing"],
        "capture_video": True,
        "capture_screenshots": True
    },
    "monitoring_config": {
        "track_llm_calls": True,
        "track_agent_actions": True,
        "track_tools": True,
        "track_errors": True
    },
    "analytics_config": {
        "cost_tracking": True,
        "performance_metrics": True,
        "usage_analytics": True
    },
    "alerting_config": {
        "error_threshold": 0.05,
        "latency_threshold": 5000,  # milliseconds
        "cost_threshold": 100.0     # dollars
    }
}

adapter = AgentOpsAdapter(advanced_config)
```

### Usage Examples

```python
# Create scenario with AgentOps monitoring
scenario = Scenario(
    name="production_monitoring",
    adapter=adapter,
    dataset_source="production_cases.json"
)

# Run with full monitoring
result = await scenario.run()

# Access monitoring data
session_url = result.metadata.get("agentops_session_url")
print(f"View session: {session_url}")
```

### AgentOps-Specific Features

- **Real-time Monitoring**: Live dashboard with agent activity
- **Video Capture**: Visual recordings of agent interactions
- **Cost Tracking**: Detailed cost analysis and budgeting
- **Error Detection**: Automatic error detection and alerting
- **Performance Analytics**: Latency, throughput, and efficiency metrics

### Best Practices

1. **Environment Tags**: Use clear environment and session tags
2. **Video Capture**: Enable for critical test scenarios
3. **Cost Monitoring**: Set appropriate cost thresholds and alerts
4. **Error Tracking**: Monitor error rates and implement auto-recovery
5. **Performance Optimization**: Use analytics to identify bottlenecks

---

## Weights & Biases (Wandb) Integration

### Overview
Wandb provides experiment tracking and model management. The WandbAdapter enables comprehensive experiment tracking for agent testing and evaluation.

### Installation & Setup

```bash
# Install Wandb
pip install wandb

# Login to Wandb
wandb login
```

### Configuration

```python
from agentunit.adapters import WandbAdapter

# Basic Wandb configuration
config = {
    "project": "agentunit-experiments",
    "entity": "your-team",
    "job_type": "evaluation"
}

adapter = WandbAdapter(config)
```

### Advanced Configuration

```python
# Advanced Wandb configuration
advanced_config = {
    "project": "multi-agent-research",
    "entity": "ai-research-team",
    "run_config": {
        "job_type": "hyperparameter-sweep",
        "group": "model-comparison",
        "tags": ["multi-agent", "gpt-4", "evaluation"]
    },
    "logging_config": {
        "log_frequency": 1,
        "log_gradients": False,
        "log_parameters": True,
        "log_artifacts": True
    },
    "experiment_config": {
        "save_code": True,
        "monitor_system": True,
        "track_env": True
    },
    "hyperparameters": {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "num_agents": 3
    }
}

adapter = WandbAdapter(advanced_config)
```

### Usage Examples

```python
# Create scenario with Wandb tracking
scenario = Scenario(
    name="experiment_tracking",
    adapter=adapter,
    dataset_source="research_dataset.json"
)

# Run experiment with full tracking
result = await scenario.run()

# Log custom metrics
wandb.log({
    "success_rate": result.success_rate,
    "avg_response_time": result.avg_response_time,
    "total_cost": result.total_cost
})

# Save artifacts
wandb.save("results.json")
wandb.save("conversation_logs.txt")
```

### Wandb-Specific Features

- **Experiment Tracking**: Complete experiment history and comparison
- **Hyperparameter Sweeps**: Automated hyperparameter optimization
- **Artifact Management**: Version control for datasets and models
- **Collaborative Features**: Team sharing and collaboration tools
- **Rich Visualizations**: Charts, tables, and custom visualizations

### Best Practices

1. **Project Organization**: Use clear project and experiment naming
2. **Hyperparameter Tracking**: Log all relevant configuration parameters
3. **Artifact Versioning**: Version datasets and model checkpoints
4. **Collaborative Workflows**: Share experiments with team members
5. **Custom Metrics**: Define domain-specific evaluation metrics

---

## Integration Comparison

| Feature | AG2 | Swarm | LangSmith | AgentOps | Wandb |
|---------|-----|--------|-----------|----------|-------|
| **Multi-Agent** | ✅ Advanced | ✅ Lightweight | ❌ Monitoring | ❌ Monitoring | ❌ Tracking |
| **Observability** | ⚠️ Basic | ⚠️ Basic | ✅ Advanced | ✅ Advanced | ✅ Experiments |
| **Production Ready** | ✅ Yes | ⚠️ Experimental | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cost Tracking** | ❌ No | ❌ No | ⚠️ Basic | ✅ Advanced | ⚠️ Basic |
| **Real-time Monitor** | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Limited |
| **Collaboration** | ⚠️ Limited | ⚠️ Limited | ✅ Yes | ✅ Yes | ✅ Advanced |

## Best Practices for Multi-Platform Usage

### 1. Adapter Selection
- **Development**: Use AG2 or Swarm for agent development
- **Testing**: Use LangSmith for evaluation and debugging
- **Production**: Use AgentOps for monitoring and alerting
- **Research**: Use Wandb for experiment tracking

### 2. Configuration Management
```python
# Environment-specific configurations
configs = {
    "development": {
        "ag2": {"model": "gpt-3.5-turbo", "max_turns": 5},
        "langsmith": {"trace_level": "DEBUG"}
    },
    "production": {
        "ag2": {"model": "gpt-4", "max_turns": 10},
        "agentops": {"capture_video": True},
        "langsmith": {"trace_level": "INFO"}
    }
}
```

### 3. Error Handling
```python
try:
    result = await scenario.run()
except AdapterError as e:
    # Handle adapter-specific errors
    logger.error(f"Adapter error: {e}")
except TimeoutError as e:
    # Handle timeout errors
    logger.error(f"Scenario timeout: {e}")
```

### 4. Performance Optimization
- Use connection pooling for multiple scenarios
- Implement caching for repeated operations
- Monitor resource usage across adapters
- Set appropriate timeouts and limits

This comprehensive guide ensures successful integration with all supported platforms while maximizing the benefits of each adapter's unique capabilities.