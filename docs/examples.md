# AgentUnit Examples and Templates

## Quick Start Examples

### 1. Basic Multi-Agent Conversation

```python
# basic_conversation.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def basic_conversation_example():
    """Simple two-agent conversation example."""
    
    # Configure the AG2 adapter
    config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_turns": 5
    }
    adapter = AG2Adapter(config)
    
    # Create a simple dataset
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": "conversation_1",
                "input": "Please help me plan a team meeting agenda.",
                "expected_output": "A structured meeting agenda with time allocations"
            }
        ]
    })
    
    # Create and run scenario
    scenario = Scenario(
        name="basic_conversation",
        adapter=adapter,
        dataset_source=dataset
    )
    
    result = await scenario.run()
    print(f"✅ Success rate: {result.success_rate:.2%}")
    print(f"📊 Total runs: {len(result.runs)}")
    
    for run in result.runs:
        print(f"  Case {run.case_id}: {'✅' if run.success else '❌'}")

if __name__ == "__main__":
    asyncio.run(basic_conversation_example())
```

### 2. Multi-Platform Monitoring

```python
# multi_platform_monitoring.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import SwarmAdapter, LangSmithAdapter, AgentOpsAdapter

async def multi_platform_example():
    """Example using multiple monitoring platforms."""
    
    # Configure multiple adapters
    swarm_config = {
        "model": "gpt-4",
        "max_turns": 3
    }
    
    langsmith_config = {
        "project_name": "agentunit-demo",
        "trace_level": "INFO"
    }
    
    agentops_config = {
        "environment": "development",
        "capture_video": True
    }
    
    # Create adapters
    swarm_adapter = SwarmAdapter(swarm_config)
    langsmith_adapter = LangSmithAdapter(langsmith_config)
    agentops_adapter = AgentOpsAdapter(agentops_config)
    
    # Load dataset from file
    dataset = DatasetSource.from_file("examples/demo_dataset.json")
    
    # Run scenarios with different adapters
    scenarios = [
        Scenario("swarm_test", swarm_adapter, dataset),
        Scenario("langsmith_test", langsmith_adapter, dataset),
        Scenario("agentops_test", agentops_adapter, dataset)
    ]
    
    results = []
    for scenario in scenarios:
        result = await scenario.run()
        results.append(result)
        print(f"📋 {scenario.name}: {result.success_rate:.2%} success")
    
    # Compare results
    for i, result in enumerate(results):
        print(f"\n📊 Scenario {i+1} ({scenarios[i].name}):")
        print(f"  Success Rate: {result.success_rate:.2%}")
        print(f"  Avg Duration: {result.avg_duration:.2f}s")

if __name__ == "__main__":
    asyncio.run(multi_platform_example())
```

### 3. Custom Evaluation Metrics

```python
# custom_metrics.py
import asyncio
from agentunit import Scenario, DatasetSource, run_suite
from agentunit.adapters import AG2Adapter
from agentunit.metrics import CustomMetric

class SemanticSimilarityMetric(CustomMetric):
    """Custom metric for semantic similarity evaluation."""
    
    def __init__(self):
        super().__init__("semantic_similarity")
        
    def calculate(self, expected: str, actual: str) -> float:
        """Calculate semantic similarity between expected and actual output."""
        # Simplified example - in practice, use sentence transformers
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embeddings = model.encode([expected, actual])
        # similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Simple word overlap for demo
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        overlap = len(expected_words.intersection(actual_words))
        total = len(expected_words.union(actual_words))
        return overlap / total if total > 0 else 0.0

async def custom_metrics_example():
    """Example using custom evaluation metrics."""
    
    # Configure adapter
    adapter = AG2Adapter({
        "model": "gpt-4",
        "temperature": 0.5,
        "max_turns": 3
    })
    
    # Create dataset with expected outputs
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": "summary_task",
                "input": "Summarize the benefits of renewable energy.",
                "expected_output": "Renewable energy reduces carbon emissions, decreases dependency on fossil fuels, and provides sustainable power generation."
            },
            {
                "id": "explanation_task", 
                "input": "Explain machine learning in simple terms.",
                "expected_output": "Machine learning teaches computers to learn patterns from data without explicit programming."
            }
        ]
    })
    
    # Create scenario with custom metrics
    scenario = Scenario(
        name="custom_evaluation",
        adapter=adapter,
        dataset_source=dataset,
        metrics=[SemanticSimilarityMetric()]
    )
    
    result = await scenario.run()
    
    print(f"📊 Custom Metrics Results:")
    print(f"  Success Rate: {result.success_rate:.2%}")
    
    for run in result.runs:
        semantic_score = run.metrics.get("semantic_similarity", 0)
        print(f"  Case {run.case_id}: {semantic_score:.3f} similarity")

if __name__ == "__main__":
    asyncio.run(custom_metrics_example())
```

## Advanced Examples

### 4. Multi-Agent Coordination

```python
# multi_agent_coordination.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter

async def coordination_example():
    """Advanced multi-agent coordination scenario."""
    
    # Configure multi-agent setup
    config = {
        "model": "gpt-4",
        "agents": {
            "project_manager": {
                "name": "ProjectManager",
                "system_message": """You are a project manager. Your role is to:
                1. Break down complex tasks into smaller components
                2. Assign tasks to appropriate team members
                3. Monitor progress and coordinate between team members
                4. Ensure project deadlines are met""",
                "max_consecutive_auto_reply": 3
            },
            "developer": {
                "name": "Developer",
                "system_message": """You are a software developer. Your role is to:
                1. Analyze technical requirements
                2. Propose implementation approaches
                3. Identify potential technical challenges
                4. Provide time estimates for development tasks""",
                "max_consecutive_auto_reply": 2
            },
            "qa_engineer": {
                "name": "QAEngineer", 
                "system_message": """You are a QA engineer. Your role is to:
                1. Review requirements for testability
                2. Identify potential edge cases and risks
                3. Propose testing strategies
                4. Ensure quality standards are met""",
                "max_consecutive_auto_reply": 2
            }
        },
        "conversation_config": {
            "max_turns": 15,
            "speaker_selection_method": "auto"
        }
    }
    
    adapter = AG2Adapter(config)
    
    # Complex coordination dataset
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": "project_planning",
                "input": """Plan the development of a new user authentication system with the following requirements:
                - Support email/password and OAuth login
                - Multi-factor authentication
                - Password reset functionality
                - User profile management
                - Security audit logging
                
                The timeline is 8 weeks. Please coordinate between team members to create a detailed plan.""",
                "expected_output": "A comprehensive project plan with task breakdown, assignments, timeline, and risk assessment"
            }
        ]
    })
    
    scenario = Scenario(
        name="multi_agent_coordination",
        adapter=adapter,
        dataset_source=dataset
    )
    
    result = await scenario.run()
    
    print(f"🚀 Multi-Agent Coordination Results:")
    print(f"  Success Rate: {result.success_rate:.2%}")
    
    for run in result.runs:
        print(f"\n📋 Case: {run.case_id}")
        print(f"  Duration: {run.duration_ms/1000:.1f}s")
        print(f"  Success: {'✅' if run.success else '❌'}")
        
        # Display conversation summary
        if hasattr(run, 'trace') and run.trace:
            messages = run.trace.get_messages()
            print(f"  Messages exchanged: {len(messages)}")
            for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                speaker = msg.get('speaker', 'Unknown')
                content = msg.get('content', '')[:100] + '...'
                print(f"    {speaker}: {content}")

if __name__ == "__main__":
    asyncio.run(coordination_example())
```

### 5. Production Monitoring Setup

```python
# production_monitoring.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AgentOpsAdapter
from agentunit.monitoring import ProductionMonitor

async def production_monitoring_example():
    """Production monitoring with alerts and dashboards."""
    
    # Configure production monitoring
    config = {
        "environment": "production",
        "session_config": {
            "auto_start_session": True,
            "session_tags": ["user-support", "chatbot", "production"],
            "capture_video": True
        },
        "monitoring_config": {
            "track_llm_calls": True,
            "track_agent_actions": True,
            "track_errors": True,
            "real_time_analytics": True
        },
        "alerting_config": {
            "error_threshold": 0.02,      # 2% error rate
            "latency_threshold": 3000,    # 3 seconds
            "cost_threshold": 50.0,       # $50 per hour
            "alert_channels": ["email", "slack"]
        }
    }
    
    adapter = AgentOpsAdapter(config)
    
    # Production dataset simulating real user interactions
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": "customer_support_1",
                "input": "I can't log into my account. I've tried resetting my password but didn't receive an email.",
                "expected_output": "Troubleshooting steps and account recovery assistance"
            },
            {
                "id": "billing_inquiry",
                "input": "I was charged twice for my subscription this month. Can you help me understand why?",
                "expected_output": "Billing investigation and resolution steps"
            },
            {
                "id": "feature_request",
                "input": "Can you add a dark mode to the mobile app? Many users have been asking for it.",
                "expected_output": "Feature request acknowledgment and process explanation"
            }
        ]
    })
    
    # Set up production monitor
    monitor = ProductionMonitor([adapter])
    await monitor.start_monitoring()
    
    scenario = Scenario(
        name="production_customer_support",
        adapter=adapter,
        dataset_source=dataset
    )
    
    try:
        result = await scenario.run()
        
        print(f"🎯 Production Monitoring Results:")
        print(f"  Success Rate: {result.success_rate:.2%}")
        print(f"  Total Interactions: {len(result.runs)}")
        print(f"  Average Response Time: {result.avg_duration:.2f}s")
        
        # Check for alerts
        alerts = await monitor.get_active_alerts()
        if alerts:
            print(f"🚨 Active Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"    {alert.type}: {alert.message}")
        else:
            print("✅ No active alerts")
            
        # Get session URL for detailed analysis
        session_url = result.metadata.get("agentops_session_url")
        if session_url:
            print(f"📊 View detailed session: {session_url}")
            
    finally:
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(production_monitoring_example())
```

## Templates

### 1. Basic Scenario Template

```python
# scenario_template.py
import asyncio
from agentunit import Scenario, DatasetSource
from agentunit.adapters import AG2Adapter  # Change adapter as needed

async def scenario_template():
    """Template for creating new AgentUnit scenarios."""
    
    # TODO: Configure your adapter
    config = {
        "model": "gpt-4",              # Your model choice
        "temperature": 0.7,            # Adjust creativity
        "max_turns": 5                 # Conversation length
    }
    adapter = AG2Adapter(config)
    
    # TODO: Define your test cases
    dataset = DatasetSource.from_dict({
        "test_cases": [
            {
                "id": "test_case_1",
                "input": "Your test input here",
                "expected_output": "Expected output here"
            }
            # Add more test cases...
        ]
    })
    
    # TODO: Create scenario with your name
    scenario = Scenario(
        name="your_scenario_name",
        adapter=adapter,
        dataset_source=dataset
    )
    
    # Run scenario
    result = await scenario.run()
    
    # TODO: Customize result analysis
    print(f"Results for {scenario.name}:")
    print(f"  Success Rate: {result.success_rate:.2%}")
    print(f"  Total Cases: {len(result.runs)}")
    
    for run in result.runs:
        status = "✅" if run.success else "❌"
        print(f"  {status} {run.case_id}: {run.duration_ms}ms")

if __name__ == "__main__":
    asyncio.run(scenario_template())
```

### 2. Configuration Template

```yaml
# config_template.yaml
# AgentUnit Configuration Template

# Global Settings
global:
  timeout: 60                    # Default timeout in seconds
  retries: 3                     # Number of retries on failure
  log_level: INFO               # DEBUG, INFO, WARNING, ERROR
  output_dir: "./results"       # Results output directory

# Adapter Configurations
adapters:
  autogen_ag2:
    model: "gpt-4"
    temperature: 0.7
    max_turns: 10
    timeout: 30
    
  openai_swarm:
    model: "gpt-4-turbo"
    temperature: 0.5
    max_turns: 5
    
  langsmith:
    project_name: "agentunit-testing"
    trace_level: "INFO"
    auto_eval: true
    
  agentops:
    environment: "development"
    capture_video: false
    auto_start_session: true
    
  wandb:
    project: "agentunit-experiments"
    entity: "your-team"
    job_type: "evaluation"

# Dataset Configuration
datasets:
  default_format: "json"
  validation: true
  max_cases_per_run: 100

# Monitoring Configuration  
monitoring:
  enabled: true
  real_time: true
  alerts:
    error_threshold: 0.05
    latency_threshold: 5000
    cost_threshold: 100.0
  integrations:
    - langsmith
    - agentops

# Evaluation Configuration
evaluation:
  default_metrics:
    - accuracy
    - response_time
    - cost_efficiency
  custom_metrics: []
  export_formats:
    - json
    - html
    - markdown

# Environment-Specific Overrides
environments:
  development:
    log_level: DEBUG
    monitoring:
      real_time: false
      
  staging:
    timeout: 120
    retries: 5
    
  production:
    log_level: WARNING
    monitoring:
      alerts:
        error_threshold: 0.01
        latency_threshold: 3000
```

### 3. Dataset Template

```json
{
  "metadata": {
    "name": "Dataset Template",
    "description": "Template for creating AgentUnit datasets",
    "version": "1.0.0",
    "created_date": "2024-01-01",
    "tags": ["template", "example"]
  },
  "test_cases": [
    {
      "id": "example_case_1",
      "category": "conversation",
      "description": "Basic conversation test",
      "input": {
        "prompt": "Hello, how can you help me?",
        "context": "User is new to the platform",
        "user_profile": {
          "experience_level": "beginner",
          "preferences": ["clear explanations", "step-by-step guidance"]
        }
      },
      "expected_output": {
        "response_type": "helpful_explanation",
        "key_elements": ["greeting", "capability_overview", "next_steps"],
        "tone": "friendly_professional"
      },
      "evaluation_criteria": {
        "helpfulness": {"min_score": 0.8, "weight": 0.4},
        "clarity": {"min_score": 0.7, "weight": 0.3},
        "completeness": {"min_score": 0.6, "weight": 0.3}
      },
      "metadata": {
        "difficulty": "easy",
        "estimated_duration": 30,
        "tags": ["greeting", "onboarding"]
      }
    },
    {
      "id": "example_case_2", 
      "category": "problem_solving",
      "description": "Complex problem-solving scenario",
      "input": {
        "prompt": "I'm facing a complex technical issue with integrating multiple APIs. Can you help me design a solution?",
        "context": "Developer working on enterprise integration project",
        "constraints": [
          "Must support real-time data sync",
          "Budget limitations for external services",
          "Security compliance requirements"
        ]
      },
      "expected_output": {
        "response_type": "technical_solution",
        "key_elements": ["problem_analysis", "solution_options", "implementation_plan"],
        "deliverables": ["architecture_diagram", "step_by_step_guide", "risk_assessment"]
      },
      "evaluation_criteria": {
        "technical_accuracy": {"min_score": 0.9, "weight": 0.5},
        "practicality": {"min_score": 0.8, "weight": 0.3},
        "completeness": {"min_score": 0.7, "weight": 0.2}
      },
      "metadata": {
        "difficulty": "advanced",
        "estimated_duration": 300,
        "tags": ["api_integration", "architecture", "enterprise"]
      }
    }
  ],
  "evaluation_config": {
    "default_timeout": 60,
    "retry_failed_cases": true,
    "parallel_execution": false,
    "metrics": [
      {
        "name": "response_quality",
        "type": "custom",
        "config": {"model": "gpt-4", "evaluation_prompt": "Rate the quality of this response..."}
      }
    ]
  }
}
```

### 4. CLI Usage Template

```bash
#!/bin/bash
# cli_template.sh - AgentUnit CLI Usage Examples

echo "🚀 AgentUnit CLI Usage Examples"

# Basic scenario execution
echo "Running basic scenario..."
agentunit multiagent run \
  --scenario "basic_conversation" \
  --adapter "autogen_ag2" \
  --dataset "examples/conversation_tests.json" \
  --output "results/"

# Advanced scenario with custom config
echo "Running advanced scenario with custom configuration..."
agentunit multiagent run \
  --scenario "complex_coordination" \
  --adapter "autogen_ag2" \
  --config "configs/production.yaml" \
  --dataset "datasets/coordination_tests.json" \
  --metrics "accuracy,response_time,cost" \
  --export "json,html" \
  --parallel \
  --verbose

# Start monitoring session
echo "Starting production monitoring..."
agentunit monitoring start \
  --environment "production" \
  --integrations "agentops,langsmith" \
  --alerts \
  --real-time

# Analyze results
echo "Analyzing test results..."
agentunit analyze results \
  --input "results/latest_run.json" \
  --compare-with "results/baseline.json" \
  --metrics "all" \
  --report "detailed" \
  --export "html"

# Generate scenario template
echo "Generating new scenario template..."
agentunit multiagent template \
  --name "my_new_scenario" \
  --adapter "openai_swarm" \
  --agents 3 \
  --output "scenarios/"

# Configuration management
echo "Managing configuration..."
agentunit config set adapters.autogen_ag2.model "gpt-4-turbo"
agentunit config get monitoring.alerts.error_threshold
agentunit config reset --confirm

echo "✅ All examples completed!"
```

### 5. Integration Test Template

```python
# integration_test_template.py
import asyncio
import pytest
from agentunit import Scenario, DatasetSource, run_suite
from agentunit.adapters import AG2Adapter, SwarmAdapter, LangSmithAdapter

class TestAgentUnitIntegration:
    """Integration test template for AgentUnit."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        return DatasetSource.from_dict({
            "test_cases": [
                {
                    "id": "integration_test_1",
                    "input": "Perform a simple calculation: 15 + 27",
                    "expected_output": "42"
                },
                {
                    "id": "integration_test_2", 
                    "input": "Explain the concept of machine learning briefly",
                    "expected_output": "A brief explanation of machine learning concepts"
                }
            ]
        })
    
    @pytest.mark.asyncio
    async def test_ag2_adapter_integration(self, sample_dataset):
        """Test AG2 adapter integration."""
        config = {
            "model": "gpt-3.5-turbo",  # Use cheaper model for tests
            "temperature": 0.1,
            "max_turns": 3
        }
        
        adapter = AG2Adapter(config)
        scenario = Scenario("ag2_test", adapter, sample_dataset)
        
        result = await scenario.run()
        
        assert result is not None
        assert len(result.runs) == 2
        assert all(run.duration_ms > 0 for run in result.runs)
        assert result.success_rate >= 0.0  # Should be between 0 and 1
    
    @pytest.mark.asyncio 
    async def test_swarm_adapter_integration(self, sample_dataset):
        """Test Swarm adapter integration."""
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_turns": 3
        }
        
        adapter = SwarmAdapter(config)
        scenario = Scenario("swarm_test", adapter, sample_dataset)
        
        result = await scenario.run()
        
        assert result is not None
        assert len(result.runs) == 2
        assert all(run.case_id in ["integration_test_1", "integration_test_2"] 
                  for run in result.runs)
    
    @pytest.mark.asyncio
    async def test_multi_scenario_suite(self, sample_dataset):
        """Test running multiple scenarios together."""
        adapters = [
            AG2Adapter({"model": "gpt-3.5-turbo", "max_turns": 2}),
            SwarmAdapter({"model": "gpt-3.5-turbo", "max_turns": 2})
        ]
        
        scenarios = [
            Scenario(f"suite_test_{i}", adapter, sample_dataset)
            for i, adapter in enumerate(adapters)
        ]
        
        results = await run_suite(scenarios)
        
        assert len(results) == 2
        assert all(result.success_rate >= 0.0 for result in results)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recovery."""
        # Test with invalid configuration
        invalid_config = {"model": "nonexistent-model"}
        adapter = AG2Adapter(invalid_config)
        
        dataset = DatasetSource.from_dict({
            "test_cases": [{"id": "error_test", "input": "test", "expected_output": "test"}]
        })
        
        scenario = Scenario("error_test", adapter, dataset)
        
        # Should handle errors gracefully
        result = await scenario.run()
        assert result is not None
        # May have low success rate due to invalid config, but shouldn't crash
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_dataset):
        """Test performance benchmarks."""
        config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_turns": 2
        }
        
        adapter = AG2Adapter(config)
        scenario = Scenario("performance_test", adapter, sample_dataset)
        
        # Measure execution time
        import time
        start_time = time.time()
        result = await scenario.run()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Basic performance assertions
        assert execution_time < 60  # Should complete within 1 minute
        assert result.avg_duration < 30000  # Average response under 30 seconds
        assert all(run.duration_ms < 60000 for run in result.runs)  # No run over 1 minute

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

These examples and templates provide a comprehensive foundation for getting started with AgentUnit, from basic scenarios to advanced production monitoring setups. Each example is self-contained and can be customized for specific use cases.