# AgentUnit

[![Python Versions](https://img.shields.io/pypi/pyversions/agentunit)](https://pypi.org/project/agentunit/)
[![PyPI version](https://img.shields.io/pypi/v/agentunit)](https://pypi.org/project/agentunit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/aviralgarg05/agentunit/ci.yml?branch=main)](https://github.com/aviralgarg05/agentunit/actions)
[![codecov](https://codecov.io/gh/aviralgarg05/agentunit/branch/main/graph/badge.svg)](https://codecov.io/gh/aviralgarg05/agentunit)

AgentUnit is a framework for evaluating, monitoring, and benchmarking multi-agent systems. It standardises how teams define scenarios, run experiments, and report outcomes across adapters, model providers, and deployment targets.

## Overview

- **Scenario-centric design** – describe datasets, adapters, and policies once, then reuse them in local runs, CI jobs, and production monitors.
- **Extensible adapters** – plug into LangGraph, CrewAI, PromptFlow, OpenAI Swarm, Anthropic Bedrock, Phidata, and custom agents through a consistent interface.
- **Comprehensive metrics** – combine exact-match assertions, RAGAS quality scores, and operational metrics with optional OpenTelemetry traces.
- **Production-first tooling** – export JSON, Markdown, and JUnit reports, gate releases with regression detection, and surface telemetry in existing observability stacks.

## Installation

AgentUnit requires Python 3.10 or later. The recommended workflow uses Poetry for dependency management.

```bash
git clone https://github.com/aviralgarg05/agentunit.git
cd agentunit
poetry install
poetry shell
```

To use pip instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional integrations are published as extras; install only what you need:

```bash
poetry install --with promptflow,crewai,langgraph
# or with pip
pip install agentunit[promptflow,crewai,langgraph]
```
##  Run CI locally
Prerequisites: Python 3.10+,Poetry
- poetry install --with dev
- poetry check
- poetry run ruff check .
- poetry run ruff format --check .
- poetry run pytest

### Optional Extras

| Extra | Includes | Use Case |
|-------|----------|----------|
| `promptflow` | `promptflow>=1.0.0` | Azure PromptFlow integration |
| `crewai` | `crewai>=0.201.1` | CrewAI multi-agent orchestration |
| `langgraph` | `langgraph>=1.0.0a4` | LangGraph state machines |
| `openai` | `openai>=1.0.0` | OpenAI models and Swarm |
| `anthropic` | `anthropic>=0.18.0` | Claude/Bedrock integration |
| `phidata` | `phidata>=2.0.0` | Phidata agents |
| `all` | All above extras | Complete installation |

Refer to the [adapters guide](docs/adapters.md) for per-adapter requirements and feature support matrices.

## Quickstart

### 2-Minute Copy-Paste Example

Create a file `example_suite.py`:

```python
from agentunit import Scenario, DatasetCase, Runner
from agentunit.adapters import MockAdapter
from agentunit.metrics import ExactMatch

# Define test cases
cases = [
    DatasetCase(
        id="math_1",
        query="What is 2 + 2?",
        expected_output="4"
    ),
    DatasetCase(
        id="capital_1",
        query="What is the capital of France?",
        expected_output="Paris"
    )
]

# Create scenario
scenario = Scenario(
    name="Basic Q&A Test",
    adapter=MockAdapter(),  # Replace with your adapter
    dataset=cases,
    metrics=[ExactMatch()]
)

# Run evaluation
runner = Runner()
results = runner.run(scenario)

# Print results
print(f"Success rate: {results.success_rate:.1%}")
print(f"Average latency: {results.avg_latency:.2f}s")
```

Run it:

```bash
python example_suite.py
```

### YAML Configuration Example

Create `example_suite.yaml`:

```yaml
name: "Customer Support Q&A"
description: "Evaluate customer support agent responses"

adapter:
  type: "openai"
  config:
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 500

dataset:
  cases:
    - input: "How do I reset my password?"
      expected: "Use the 'Forgot Password' link on the login page"
      metadata:
        category: "account"
    
    - input: "What are your business hours?"
      expected: "Monday-Friday 9AM-5PM EST"
      metadata:
        category: "general"

metrics:
  - "exact_match"
  - "semantic_similarity"
  - "latency"

timeout: 30
retries: 2
```

Run it with the CLI:

```bash
agentunit example_suite.yaml \
  --json results.json \
  --markdown results.md \
  --junit results.xml
```

### Getting started

1. Follow the [Quickstart](#quickstart) above for a 2-minute runnable example.
2. Review [Writing Scenarios](docs/writing-scenarios.md) for dataset and adapter templates plus helper constructors for popular frameworks.
3. Consult the [CLI reference](docs/cli.md) to orchestrate suites from the command line and export results for CI, dashboards, or audits.
4. Explore the [adapters guide](docs/adapters.md) for concrete adapter implementations and feature support.
5. Check the [metrics catalog](docs/metrics-catalog.md) for all available evaluation metrics.

### CLI Usage

AgentUnit exposes an `agentunit` CLI entry point once installed. Typical usage:

```bash
agentunit path.to.suite \
  --metrics faithfulness answer_correctness \
  --json reports/results.json \
  --markdown reports/results.md \
  --junit reports/results.xml
```

Programmatic runners are available through `agentunit.core.Runner` for notebook- or script-driven workflows.

## Documentation map

| Topic | Reference |
| --- | --- |
| Quick evaluation walkthrough | [Quickstart](#quickstart) |
| Scenario and adapter authoring | [docs/writing-scenarios.md](docs/writing-scenarios.md) |
| Adapter implementations guide | [docs/adapters.md](docs/adapters.md) |
| Metrics catalog and reference | [docs/metrics-catalog.md](docs/metrics-catalog.md) |
| CLI options and examples | [docs/cli.md](docs/cli.md) |
| Architecture overview | [docs/architecture.md](docs/architecture.md) |
| Framework-specific guides | [docs/platform-guides.md](docs/platform-guides.md) |
| No-code builder guide | [docs/nocode-quickstart.md](docs/nocode-quickstart.md) |
| OpenTelemetry integration | [docs/telemetry.md](docs/telemetry.md) |
| Performance testing | [docs/performance-testing.md](docs/performance-testing.md) |
| Comparison to other tools | [docs/comparison.md](docs/comparison.md) |
| Templates | [docs/templates/README.md](docs/templates/README.md) |

Use the table above as the canonical navigation surface; every document cross-links back to related topics for clarity.

## Development workflow

1. Install dependencies (Poetry or pip).
2. Run the unit and integration suite:

```bash
poetry run python3 -m pytest tests -v
```

3. Execute targeted suites during active development, then run the full matrix before opening a pull request.

Latest verification (2025-10-24): 144 passed, 10 skipped, 32 warnings. Warnings originate from third-party dependencies (`langchain` pydantic shim deprecations and `datetime.utcnow` usage). Track upstream fixes or pin patched releases as needed.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup and workflow
- Code style and linting guidelines
- Testing requirements
- Pull request process
- Issue labels and tags for open source events

Security disclosures and sensitive topics should follow responsible disclosure guidelines outlined in [SECURITY.md](SECURITY.md).

## License

AgentUnit is released under the MIT License. See [LICENSE](LICENSE) for the full text.

---

Need an overview for stakeholders? Start with [docs/architecture.md](docs/architecture.md). Ready to extend the platform? Explore the templates under [docs/templates/README.md](docs/templates/README.md).
