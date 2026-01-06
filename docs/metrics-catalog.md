# AgentUnit Metrics Catalog

This document provides a comprehensive reference for all evaluation metrics available in AgentUnit. Metrics are organized by category and include detailed specifications for inputs, outputs, and usage patterns.

---

## Overview

AgentUnit provides a layered metrics system designed to evaluate AI agents across multiple dimensions:

| Category | Focus Area | Metrics Count |
|----------|------------|---------------|
| **Core Quality** | Response correctness, faithfulness, hallucination | 5 |
| **Operational** | Tool usage, cost, token consumption | 3 |
| **Multimodal** | Cross-modal grounding, video/audio/image metrics | 5 |
| **Privacy** | PII detection, consent compliance, data minimization | 4 |
| **Sustainability** | Energy, carbon, resource utilization | 3 |
| **Multi-Agent** | Coordination, communication, emergent behaviors | Planned |

---

## Metric Interface

All metrics implement the `Metric` protocol:

```python
from agentunit.metrics.base import Metric, MetricResult

class Metric(Protocol):
    name: str
    
    def evaluate(
        self, 
        case: DatasetCase, 
        trace: TraceLog, 
        outcome: Any
    ) -> MetricResult: ...
```

### MetricResult Structure

```python
@dataclass
class MetricResult:
    name: str           # Metric identifier
    value: float | None # Normalized score (0.0-1.0 for quality, raw for operational)
    detail: dict        # Additional metric-specific information
```

---

## Core Quality Metrics

### FaithfulnessMetric

**Purpose**: Measures how factually consistent the agent's response is with the provided context.

| Attribute | Value |
|-----------|-------|
| **Name** | `faithfulness` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | RAGAS (optional) |
| **Score Range** | 0.0 - 1.0 |

**Inputs**:
- `case.context`: List of reference context strings
- `outcome.output`: Agent's generated response

**Algorithm**:
1. If RAGAS available: Uses `ragas.metrics.faithfulness` for LLM-based evaluation
2. Fallback: Calculates ratio of context strings appearing in the response

```python
from agentunit.metrics.builtin import FaithfulnessMetric

metric = FaithfulnessMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.85 (85% of context grounded)
# result.detail: {"answer": "...", "references": [...]}
```

---

### AnswerCorrectnessMetric

**Purpose**: Evaluates semantic correctness of the agent's response against expected output.

| Attribute | Value |
|-----------|-------|
| **Name** | `answer_correctness` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | RAGAS (optional) |
| **Score Range** | 0.0 - 1.0 |

**Inputs**:
- `case.expected_output`: Ground truth answer
- `outcome.output`: Agent's generated response

**Algorithm**:
1. Exact match: Returns 1.0 if strings are identical
2. RAGAS available: Uses `ragas.metrics.answer_correctness` for semantic similarity
3. Fallback: Returns 0.0 for non-matching responses

```python
from agentunit.metrics.builtin import AnswerCorrectnessMetric

metric = AnswerCorrectnessMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.92
# result.detail: {"expected": "...", "answer": "..."}
```

---

### HallucinationRateMetric

**Purpose**: Detects responses containing information not grounded in the provided context.

| Attribute | Value |
|-----------|-------|
| **Name** | `hallucination_rate` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | RAGAS (optional) |
| **Score Range** | 0.0 - 1.0 (lower is better) |

**Inputs**:
- `case.context`: Reference context strings
- `outcome.output`: Agent's generated response

**Algorithm**:
1. RAGAS available: Uses `ragas.metrics.hallucination`, inverts score (1 - hallucination_score)
2. Fallback: Returns 1.0 if any context missing from response, 0.0 otherwise

```python
from agentunit.metrics.builtin import HallucinationRateMetric

metric = HallucinationRateMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.15 (15% hallucinated content)
```

> **Note**: A hallucination rate of 0.0 indicates a fully grounded response.

---

### RetrievalQualityMetric

**Purpose**: Measures precision of retrieved context for RAG pipelines.

| Attribute | Value |
|-----------|-------|
| **Name** | `retrieval_quality` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | RAGAS (optional) |
| **Score Range** | 0.0 - 1.0 |

**Inputs**:
- `case.context`: Retrieved context documents
- `outcome.output`: Agent's response

**Algorithm**:
1. RAGAS available: Uses `ragas.metrics.context_precision`
2. Fallback: Calculates ratio of context strings mentioned in response

```python
from agentunit.metrics.builtin import RetrievalQualityMetric

metric = RetrievalQualityMetric()
result = metric.evaluate(case, trace, outcome)
# result.detail: {"references": [...], "answer": "..."}
```

---

### ToolSuccessMetric

**Purpose**: Tracks success rate of tool/function calls during agent execution.

| Attribute | Value |
|-----------|-------|
| **Name** | `tool_success` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Inputs**:
- `trace.events`: Events with type `"tool_call"`
- `outcome.success`: Overall execution success

**Algorithm**:
1. Extracts all `tool_call` events from trace
2. Counts events with `status == "success"`
3. Returns success_count / total_tool_calls

```python
from agentunit.metrics.builtin import ToolSuccessMetric

metric = ToolSuccessMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.90 (9/10 tools succeeded)
# result.detail: {"tool_calls": [{...}, ...]}
```

---

## Operational Metrics

### CostMetric

**Purpose**: Tracks monetary cost of agent execution (API calls, tokens, tools).

| Attribute | Value |
|-----------|-------|
| **Name** | `cost` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | None |
| **Score Range** | Raw USD value |

**Inputs**:
- `trace.metadata["cost"]`: Total execution cost
- `outcome.cost`: Cost from outcome object
- `trace.events`: Tool call costs

**Algorithm**:
1. Checks `trace.metadata` for `"cost"` key
2. Falls back to `outcome.cost` attribute
3. Sums individual tool call costs from events

```python
from agentunit.metrics.builtin import CostMetric

metric = CostMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.0032 (USD)
# result.detail: {"cost": 0.0032}
```

---

### TokenUsageMetric

**Purpose**: Tracks token consumption across prompt and completion.

| Attribute | Value |
|-----------|-------|
| **Name** | `token_usage` |
| **Module** | `agentunit.metrics.builtin` |
| **Dependencies** | None |
| **Score Range** | Raw token count |

**Inputs**:
- `trace.metadata["usage"]`: Token usage dict
- `outcome.usage`: Usage from outcome object

**Output Detail**:
```python
{
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
}
```

```python
from agentunit.metrics.builtin import TokenUsageMetric

metric = TokenUsageMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 225.0 (total tokens)
```

---

## Multimodal Metrics

### CrossModalGroundingMetric

**Purpose**: Measures how well text responses are grounded in visual inputs using CLIP embeddings.

| Attribute | Value |
|-----------|-------|
| **Name** | `cross_modal_grounding` |
| **Module** | `agentunit.multimodal.metrics` |
| **Dependencies** | `clip`, `torch`, `PIL` |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
CrossModalGroundingMetric(
    clip_model_name="ViT-B/32",  # CLIP model variant
    threshold=0.25               # Minimum similarity threshold
)
```

**Inputs**:
- `case.metadata["image_path"]`: Path to input image
- `outcome`: Text response

**Algorithm**:
1. Encodes image using CLIP vision encoder
2. Encodes response text using CLIP text encoder
3. Computes cosine similarity between embeddings
4. Scores based on threshold comparison

---

### ImageCaptionAccuracyMetric

**Purpose**: Evaluates quality of image caption generation using semantic similarity.

| Attribute | Value |
|-----------|-------|
| **Name** | `image_caption_accuracy` |
| **Module** | `agentunit.multimodal.metrics` |
| **Dependencies** | `sentence-transformers` (optional) |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
ImageCaptionAccuracyMetric(
    use_semantic=True,
    model_name="all-MiniLM-L6-v2"
)
```

**Algorithm**:
1. Semantic similarity (70% weight): Sentence transformer embeddings
2. Keyword overlap (30% weight): Precision, recall, F1 of word overlap

---

### VideoResponseRelevanceMetric

**Purpose**: Measures relevance of responses to video content by analyzing key frames.

| Attribute | Value |
|-----------|-------|
| **Name** | `video_response_relevance` |
| **Module** | `agentunit.multimodal.metrics` |
| **Dependencies** | `clip`, `opencv-python`, `PIL` |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
VideoResponseRelevanceMetric(
    num_frames=8,              # Frames to sample
    clip_model_name="ViT-B/32"
)
```

**Algorithm**:
1. Extracts evenly-spaced frames from video
2. Computes CLIP similarity for each frame
3. Returns weighted combination: 0.6 * avg + 0.4 * max

---

### AudioTranscriptionMetric

**Purpose**: Evaluates audio transcription quality using Word Error Rate (WER).

| Attribute | Value |
|-----------|-------|
| **Name** | `audio_transcription_quality` |
| **Module** | `agentunit.multimodal.metrics` |
| **Dependencies** | `jiwer` (optional) |
| **Score Range** | 0.0 - 1.0 |

**Output Detail**:
```python
{
    "wer": 0.12,        # Word Error Rate
    "cer": 0.08,        # Character Error Rate
    "accuracy": 0.88    # 1 - WER
}
```

---

### MultimodalCoherenceMetric

**Purpose**: Measures coherence when responses reference multiple input modalities.

| Attribute | Value |
|-----------|-------|
| **Name** | `multimodal_coherence` |
| **Module** | `agentunit.multimodal.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Algorithm**:
1. Detects input modalities from metadata (image, audio, video, text)
2. Scans response for modality-specific keywords
3. Coverage score: referenced / total modalities
4. Coherence score: presence of transition words
5. Final: 0.6 * coverage + 0.4 * coherence

---

## Privacy Metrics

### PIILeakageMetric

**Purpose**: Detects Personally Identifiable Information (PII) leakage in outputs.

| Attribute | Value |
|-----------|-------|
| **Name** | `pii_leakage` |
| **Module** | `agentunit.privacy.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 (higher = less leakage) |

**Configuration**:
```python
PIILeakageMetric(
    check_input=False,
    check_output=True,
    severity_weights={
        "email": 0.5,
        "phone": 0.6,
        "ssn": 1.0,
        "credit_card": 1.0,
        "name": 0.3,
        "address": 0.7
    }
)
```

**Detected PII Types**:
| Type | Pattern | Confidence |
|------|---------|------------|
| Email | RFC 5322 regex | 0.95 |
| Phone | XXX-XXX-XXXX variants | 0.90 |
| SSN | XXX-XX-XXXX | 0.98 |
| Credit Card | 16-digit with separators | 0.85 |
| Name | Capitalized word pairs | 0.60 |

---

### PrivacyBudgetMetric

**Purpose**: Tracks differential privacy budget consumption (epsilon).

| Attribute | Value |
|-----------|-------|
| **Name** | `privacy_budget` |
| **Module** | `agentunit.privacy.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 (remaining budget ratio) |

**Configuration**:
```python
PrivacyBudgetMetric(
    total_budget=10.0,    # Total epsilon available
    warn_threshold=0.8    # Warning at 80% utilization
)
```

**Output Detail**:
```python
{
    "epsilon_used": 0.5,
    "total_spent": 3.2,
    "remaining": 6.8,
    "utilization": 0.32,
    "budget_exceeded": False,
    "warning": False
}
```

---

### DataMinimizationMetric

**Purpose**: Verifies responses only include necessary information.

| Attribute | Value |
|-----------|-------|
| **Name** | `data_minimization` |
| **Module** | `agentunit.privacy.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Checked Keys** (default):
```python
["user_id", "email", "phone", "address", "ssn", 
 "credit_card", "password", "api_key", "token"]
```

---

### ConsentComplianceMetric

**Purpose**: Verifies data usage adheres to consent preferences.

| Attribute | Value |
|-----------|-------|
| **Name** | `consent_compliance` |
| **Module** | `agentunit.privacy.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 or 1.0 (binary) |

**Usage**:
```python
case.metadata = {
    "consent": {
        "email": True,      # Allowed to use
        "location": False   # Not allowed
    }
}
```

---

## Sustainability Metrics

### EnergyMetric

**Purpose**: Measures energy consumption during agent execution.

| Attribute | Value |
|-----------|-------|
| **Name** | `energy_consumption` |
| **Module** | `agentunit.sustainability.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
EnergyMetric(
    threshold_kwh=0.1,      # Energy threshold (kWh)
    sample_interval=1.0     # Sampling interval (seconds)
)
```

---

### CarbonMetric

**Purpose**: Tracks carbon emissions from agent execution.

| Attribute | Value |
|-----------|-------|
| **Name** | `carbon_emissions` |
| **Module** | `agentunit.sustainability.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
CarbonMetric(
    threshold_kg=0.05,      # CO2eq threshold (kg)
    grid_intensity=0.475    # Grid carbon intensity (kg CO2/kWh)
)
```

**Output Detail**:
```python
{
    "carbon_kg": 0.023,
    "threshold_kg": 0.05,
    "under_threshold": True,
    "equivalents": {
        "km_driven": 0.106,
        "trees_needed": 0.0011
    }
}
```

---

### ResourceUtilizationMetric

**Purpose**: Evaluates efficiency of CPU, GPU, and memory usage.

| Attribute | Value |
|-----------|-------|
| **Name** | `resource_utilization` |
| **Module** | `agentunit.sustainability.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Configuration**:
```python
ResourceUtilizationMetric(
    target_cpu_percent=80.0,
    target_memory_gb=4.0,
    target_gpu_percent=80.0
)
```

**Scoring Algorithm**:
- Optimal range: 50% - 120% of target
- Penalty for under-utilization (< 50%)
- Penalty for over-utilization (> 120%)

---

## Composite Metrics

### CompositeMetric

**Purpose**: Combines multiple metrics into a single aggregated score.

```python
from agentunit.metrics.base import CompositeMetric
from agentunit.metrics.builtin import (
    FaithfulnessMetric,
    AnswerCorrectnessMetric,
    ToolSuccessMetric
)

composite = CompositeMetric([
    FaithfulnessMetric(),
    AnswerCorrectnessMetric(),
    ToolSuccessMetric()
])

result = composite.evaluate(case, trace, outcome)
# result.value: Average of all non-None metric values
# result.detail: {metric_name: metric_detail, ...}
```

---

## Custom Metric Development

### Creating a Custom Metric

```python
from agentunit.metrics.base import Metric, MetricResult
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase

class LatencyMetric(Metric):
    name = "latency"
    
    def __init__(self, threshold_ms: float = 5000.0):
        self.threshold_ms = threshold_ms
    
    def evaluate(
        self, 
        case: DatasetCase, 
        trace: TraceLog, 
        outcome: Any
    ) -> MetricResult:
        # Calculate total latency from trace
        start = trace.events[0].timestamp if trace.events else 0
        end = trace.events[-1].timestamp if trace.events else 0
        latency_ms = (end - start) * 1000
        
        # Score: 1.0 if under threshold, scaled otherwise
        score = min(1.0, self.threshold_ms / latency_ms) if latency_ms > 0 else 1.0
        
        return MetricResult(
            name=self.name,
            value=score,
            detail={
                "latency_ms": latency_ms,
                "threshold_ms": self.threshold_ms,
                "under_threshold": latency_ms <= self.threshold_ms
            }
        )
```

### Registering Custom Metrics

```python
from agentunit.metrics.registry import register_metric

register_metric("latency", LatencyMetric)
```

---

## Multi-Agent Metrics

The following metrics are available for evaluating multi-agent system coordination and collaboration.

### CoordinationEfficiencyMetric

**Purpose**: Evaluates overall coordination efficiency in multi-agent systems.

| Attribute | Value |
|-----------|-------|
| **Name** | `coordination_efficiency` |
| **Module** | `agentunit.multiagent.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Inputs** (from trace):
- `trace.interactions`: List of `AgentInteraction` events
- `trace.handoffs`: List of `HandoffEvent` events
- `trace.conflicts`: List of `ConflictEvent` events
- `trace.agent_roles`: Dict mapping agent IDs to `AgentRole`

**Scoring Components**:
| Component | Weight | Description |
|-----------|--------|-------------|
| Handoff success rate | 25% | Ratio of successful task handoffs |
| Conflict resolution rate | 20% | Ratio of resolved conflicts |
| Communication efficiency | 25% | Message success rate + response time |
| Role adherence | 15% | How well agents follow assigned roles |
| Load balance | 15% | Evenness of work distribution |

```python
from agentunit.multiagent import CoordinationEfficiencyMetric

metric = CoordinationEfficiencyMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.85
# result.detail: {
#     "handoff_success_rate": 1.0,
#     "conflict_resolution_rate": 0.9,
#     "communication_efficiency": 0.95,
#     "role_adherence": 0.8,
#     "load_balance_score": 0.75
# }
```

---

### SwarmIntelligenceMetric

**Purpose**: Detects emergent swarm intelligence behaviors in multi-agent systems.

| Attribute | Value |
|-----------|-------|
| **Name** | `swarm_intelligence` |
| **Module** | `agentunit.multiagent.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Emergent Behavior Components**:
| Behavior | Weight | Detection Method |
|----------|--------|------------------|
| Self-organization | 30% | Entropy reduction over time |
| Specialization | 25% | HHI concentration of message types |
| Collective decisions | 25% | Negotiation/collaboration interaction count |
| Adaptation rate | 20% | Response time improvement over session |

```python
from agentunit.multiagent import SwarmIntelligenceMetric

metric = SwarmIntelligenceMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.65
# result.detail: {
#     "self_organization": 0.7,
#     "specialization": 0.5,
#     "collective_decision": 0.8,
#     "adaptation_rate": 0.6
# }
```

---

### NetworkFaultToleranceMetric

**Purpose**: Evaluates network fault tolerance based on communication topology.

| Attribute | Value |
|-----------|-------|
| **Name** | `network_fault_tolerance` |
| **Module** | `agentunit.multiagent.metrics` |
| **Dependencies** | None |
| **Score Range** | 0.0 - 1.0 |

**Network Analysis Components**:
| Metric | Description |
|--------|-------------|
| Density | Ratio of actual to possible connections |
| Centralization | How concentrated communication is |
| Clustering | Tendency to form communication clusters |
| Bottlenecks | Agents that are single points of failure |

**Scoring Formula**:
```
fault_tolerance = 0.6 * density + 0.4 * (1 - bottleneck_ratio)
```

```python
from agentunit.multiagent import NetworkFaultToleranceMetric

metric = NetworkFaultToleranceMetric()
result = metric.evaluate(case, trace, outcome)
# result.value: 0.82
# result.detail: {
#     "density": 0.65,
#     "centralization": 0.3,
#     "clustering": 0.45,
#     "hub_agents": ["coordinator_agent"],
#     "bottleneck_count": 1
# }
```

---

### MultiAgentMetricsCalculator

**Purpose**: Main calculator for comprehensive multi-agent metrics (programmatic use).

```python
from agentunit.multiagent import MultiAgentMetricsCalculator

calculator = MultiAgentMetricsCalculator(
    interactions=interaction_list,
    handoffs=handoff_list,
    conflicts=conflict_list,
    agent_roles={"agent_a": role_a, "agent_b": role_b}
)

# Calculate individual metric categories
coordination = calculator.calculate_coordination_metrics()
network = calculator.calculate_network_metrics()
emergent = calculator.calculate_emergent_behavior_metrics()

# Or calculate all at once
all_metrics = calculator.calculate_all()
```

**CoordinationMetrics Dataclass**:
```python
@dataclass
class CoordinationMetrics:
    handoff_success_rate: float   # 0.0 - 1.0
    avg_handoff_time: float       # seconds
    conflict_rate: float          # per 100 interactions
    conflict_resolution_rate: float
    avg_resolution_time: float    # seconds
    communication_efficiency: float
    role_adherence: float
    load_balance_score: float
```

**NetworkMetrics Dataclass**:
```python
@dataclass
class NetworkMetrics:
    density: float                # 0.0 - 1.0
    centralization: float         # 0.0 - 1.0
    avg_path_length: float
    clustering_coefficient: float
    hub_agents: list[str]
    bottleneck_agents: list[str]
```

**EmergentBehaviorMetrics Dataclass**:
```python
@dataclass
class EmergentBehaviorMetrics:
    self_organization_score: float
    specialization_emergence: float
    collective_decision_score: float
    adaptation_rate: float
    swarm_intelligence_score: float
```

---

### Analyzer Classes

For advanced analysis, use the underlying analyzer classes:

```python
from agentunit.multiagent import (
    InteractionAnalyzer,
    NetworkAnalyzer,
    EmergentBehaviorDetector
)

# Analyze interactions
interaction_analyzer = InteractionAnalyzer(interactions, handoffs, conflicts)
handoff_metrics = interaction_analyzer.calculate_handoff_metrics()
conflict_metrics = interaction_analyzer.calculate_conflict_metrics()
efficiency = interaction_analyzer.calculate_communication_efficiency()
load_balance = interaction_analyzer.calculate_load_balance()

# Analyze network topology
network_analyzer = NetworkAnalyzer(adjacency_dict, agent_list)
density = network_analyzer.calculate_density()
centralization = network_analyzer.calculate_centralization()
hubs = network_analyzer.find_hub_agents(threshold=0.7)
bottlenecks = network_analyzer.find_bottleneck_agents()

# Detect emergent behaviors
detector = EmergentBehaviorDetector(interactions, agent_roles)
self_org = detector.detect_self_organization()
specialization = detector.detect_specialization()
collective = detector.detect_collective_decision_making()
adaptation = detector.detect_adaptation_rate()
swarm_score = detector.calculate_swarm_intelligence()
```

---

## Metric Resolution

AgentUnit provides automatic metric resolution from string names:

```python
from agentunit.metrics import resolve_metrics

# Resolve from string names
metrics = resolve_metrics(["faithfulness", "tool_success", "cost"])

# Use DEFAULT_METRICS for standard set
from agentunit.metrics import DEFAULT_METRICS
```

**Default Metrics**:
- `faithfulness`
- `answer_correctness`
- `tool_success`
- `cost`
- `token_usage`

---

## Best Practices

1. **Combine quality and operational metrics** for holistic evaluation
2. **Use RAGAS integration** for production-grade quality metrics
3. **Enable privacy metrics** when handling user data
4. **Track sustainability** for cost-conscious deployments
5. **Create scenario-specific composite metrics** for domain needs
6. **Set appropriate thresholds** based on use case requirements

---

## Migration Notes

### From v0.3.x to v0.4.x

- Metric `compute()` method renamed to `evaluate()`
- Added `TraceLog` parameter to evaluation signature
- Privacy and sustainability metrics moved to separate modules
