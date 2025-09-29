# AgentUnit Architecture Expansion Plan

This document captures the planned architecture upgrades requested for AgentUnit. Each major capability is decomposed into subsystems with owning modules, interfaces, and cross-cutting concerns.

## 1. Adapter Ecosystem

### Goals
- Support additional agent frameworks: AutoGen, Haystack, LlamaIndex, Semantic Kernel.
- Preserve existing adapter contract to keep suites portable.

### Key Components
- `src/agentunit/adapters/registry.py` → central registration of adapters.
- New adapter modules under `src/agentunit/adapters/` following `BaseAdapter` API.
- Shared utilities for context conversion (`src/agentunit/adapters/utils.py`).

### Interfaces
```python
class BaseAdapter(ABC):
    @abstractmethod
    async def run(self, scenario: Scenario, context: RunContext) -> StepResult: ...
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...
```

### Interactions
- Suites declare adapters by identifier; registry resolves to implementation.
- Metrics and telemetry consume `StepResult` emitted by adapters.

## 2. Multimodal Evaluators

### Goals
- Add evaluators for vision/audio pipelines (CLIP, Whisper, etc.).
- Provide metrics for cross-modal accuracy, hallucination, grounding.

### Components
- `src/agentunit/metrics/multimodal.py` (new).
- Optional model providers under `src/agentunit/providers/` with pluggable backends.
- Dataset extensions to carry multimodal artifacts (images, audio).

### Interfaces
```python
class MultimodalMetric(Metric):
    modality: Literal["vision", "audio", "multimodal"]
```

## 3. Custom Metric Plugin System

### Goals
- Allow users to ship bespoke metrics via entry points or filesystem drop-ins.

### Components
- `src/agentunit/metrics/plugins.py` for discovery/registration.
- Config schema in `pyproject.toml` (`[tool.agentunit.metrics]`).

### Lifecyle
1. Load plugin modules via entry points.
2. Validate `Metric` subclasses.
3. Register into metric registry.

## 4. Parallel & Distributed Runner

### Goals
- Execute suites concurrently across nodes or GPUs.
- Provide load balancing, fault tolerance, checkpointing.

### Components
- New runner `src/agentunit/core/distributed_runner.py` built atop Ray or Dask.
- `src/agentunit/core/partitioner.py` for sharding scenarios.
- CLI flag `--distributed backend=ray address=auto`.

### Cross-Cutting Concerns
- Telemetry must aggregate metrics from workers.
- Cost tracking aggregated per scenario.

## 5. Drift Detection

### Goals
- Compare metrics over time and detect regressions.

### Components
- `src/agentunit/analytics/drift.py` for statistical tests (e.g., KL divergence).
- Persistence layer (s3/local) for historical benchmark snapshots.
- Alerts emitted through CLI or CI gates.

## 6. Error Analysis Dashboard

### Goals
- Interactive HTML report (Plotly) with failure modes, trace diffs, heatmaps.

### Components
- `src/agentunit/reporting/dashboard.py` to materialize HTML.
- Additional data schema in `results.py` to capture hallucination hotspots.

## 7. Mocking & Simulation

### Goals
- Deterministic offline evaluation of tools/APIs.

### Components
- `src/agentunit/simulation/mocks.py` set of fake services.
- Scenario DSL to reference mocks.

## 8. Benchmark Integrations

### Goals
- Export results to AgentBench & HELM schemas; auto-submit via APIs.

### Components
- `src/agentunit/integrations/agentbench.py`
- `src/agentunit/integrations/helm.py`
- CLI commands `agentunit benchmarks submit --target agentbench`.

## 9. Cost & Efficiency Metrics

### Goals
- Track tokens, latency, cost; provide optimization suggestions.

### Components
- `src/agentunit/metrics/cost.py`
- Integration with telemetry tracing.
- Recommendation engine in `analytics/optimizer.py`.

## 10. Security & Robustness Testing

### Goals
- Built-in red-team suites (jailbreak, PII, adversarial).

### Components
- `src/agentunit/scenarios/security/`
- `metrics/security.py` scoring policy violations.

## 11. Version Control for Tests

### Goals
- Git-like operations for scenario definitions.

### Components
- `src/agentunit/versioning/` with commit objects stored in `.agentunit`.
- CLI commands `agentunit suites commit`, `diff`, `merge`.

## 12. CI/CD Hooks

### Goals
- First-class integration with GitHub Actions, Jenkins, CircleCI.

### Components
- Reusable workflow templates under `.github/workflows/templates/`.
- CLI `agentunit ci generate --target github`.

## 13. Dataset Augmentation

### Goals
- Generate paraphrases, perturbations for robustness.

### Components
- `src/agentunit/datasets/augmentation.py` with LLM + rule-based augmenters.

## 14. Real-Time Monitoring Mode

### Goals
- Stream metrics from production through OpenTelemetry.

### Components
- `src/agentunit/telemetry/streaming.py`
- CLI `agentunit monitor --endpoint <otlp>`.

## 15. Community Extensions

### Goals
- Allow third-party plugins for adapters/metrics/datasets.

### Components
- Extension manifest format `agentunit_extension.toml`.
- Registry metadata in `docs/extensions/registry.md`.

## Documentation Strategy
- Update `README.md` with capability matrix.
- Per-feature guides under `docs/` mirroring module structure.
- API reference generated via `mkdocs` extension.

## Testing Strategy
- Unit tests per module under `tests/` mirroring hierarchy.
- Integration suites for distributed runner, drift detection, dashboards.
- CI gating using new GitHub workflow templates.

## Timeline Phasing
1. **Phase 1**: Adapters, multimodal metrics, plugin system.
2. **Phase 2**: Distributed runner, cost metrics, mocks.
3. **Phase 3**: Drift detection, dashboards, security suites.
4. **Phase 4**: Benchmark integrations, dataset augmentation, community registry.

This roadmap will inform the implementation tasks tracked in the repository todo list.
