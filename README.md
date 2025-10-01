# AgentUnit

**AgentUnit** is a comprehensive framework for testing, monitoring, and validating multi-agent AI systems across different platforms. It provides a unified interface to test agent interactions, measure performance, and ensure reliability of conversational AI workflows.

## 🚀 Features

### Multi-Platform Support
- **AutoGen AG2**: Microsoft's conversational AI framework
- **OpenAI Swarm**: Multi-agent coordination and orchestration
- **LangSmith**: Advanced language model monitoring and evaluation
- **AgentOps**: Production monitoring and observability
- **Wandb**: Experiment tracking and performance analytics

### Core Capabilities
- **Multi-Agent Testing**: Comprehensive testing of agent interactions and workflows
- **Production Monitoring**: Real-time monitoring of agent performance and behavior
- **Performance Analytics**: Detailed metrics and insights into agent system performance
- **Scenario Management**: Create, run, and manage test scenarios across platforms
- **Reporting & Export**: Generate detailed reports in multiple formats (JSON, XML, HTML)

### Architecture Highlights
- **Modular Design**: Platform-agnostic architecture with pluggable adapters
- **Async Processing**: Fully asynchronous execution for high-performance testing
- **Extensible Framework**: Easy to add new platforms and monitoring capabilities
- **Production Ready**: Built for enterprise-scale deployment and monitoring

> Looking for a crash course? Jump to the [five-minute quickstart](#five-minute-quickstart).

## Table of contents

1. [Installation](#installation)
2. [Five-minute quickstart](#five-minute-quickstart)
3. [Workflow overview](#workflow-overview)
4. [Authoring evaluation suites](#authoring-evaluation-suites)
5. [Command-line interface](#command-line-interface)
6. [Metrics & reporting](#metrics--reporting)
7. [Observability with OpenTelemetry](#observability-with-opentelemetry)
8. [Local development](#local-development)
9. [Additional resources](#additional-resources)

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentunit.git
cd agentunit

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install AgentUnit
pip install -e .

# Verify installation
agentunit --help
```

### Platform-Specific Dependencies

Install additional dependencies for specific platforms:

```bash
# For AutoGen AG2 support
pip install pyautogen[ag2]

# For OpenAI Swarm support
pip install openai-swarm

# For LangSmith integration
pip install langsmith

# For AgentOps monitoring
pip install agentops

# For Wandb tracking
pip install wandb
```

## 🏃‍♂️ Quick Start

### 1. Basic Multi-Agent Test

```python
from agentunit.core import Scenario, DatasetSource, DatasetCase
from agentunit.adapters.autogen_ag2 import AG2Adapter

# Create a test case
test_case = DatasetCase(
    id="greeting_test",
    query="Hello, can you help me plan a meeting?",
    expected_output="I'd be happy to help you plan a meeting.",
    metadata={"category": "greeting", "complexity": "simple"}
)

# Create dataset
dataset = DatasetSource("meeting_scenarios", lambda: [test_case])

# Configure adapter
adapter = AG2Adapter({
    "model": "gpt-4",
    "max_turns": 10,
    "timeout": 60
})

# Create and run scenario
scenario = Scenario(
    name="meeting_planning_test",
    adapter=adapter,
    dataset=dataset
)

# Execute the test
from agentunit.core import Runner
runner = Runner()
results = await runner.run_scenario(scenario)

print(f"Success rate: {results.success_rate}")
```

### 2. CLI Usage

```bash
# Run a multi-agent test scenario
agentunit multiagent run --scenario meeting_test.json --adapter autogen_ag2

# Monitor production deployment
agentunit monitoring start --platform langsmith --project my_agents

# Generate analysis report
agentunit analyze --results results.json --output report.html

# Configure AgentUnit settings
agentunit config set adapter.default autogen_ag2
agentunit config set monitoring.enabled true
```

### 3. Production Monitoring

```python
from agentunit.monitoring import ProductionMonitor
from agentunit.adapters.langsmith_adapter import LangSmithAdapter

# Setup production monitoring
monitor = ProductionMonitor()
adapter = LangSmithAdapter({
    "project_name": "production_agents",
    "api_key": "your_langsmith_key"
})

# Start monitoring
await monitor.start_monitoring(adapter)

# Monitor specific agent interactions
session_id = await monitor.create_session("customer_support")
# Your agent interactions here...
await monitor.end_session(session_id)

# Generate monitoring report
report = await monitor.generate_report()
```

Or add it to a Poetry project:

```bash
poetry add agentunit
```

The CLI entry point `agentunit` becomes available immediately after installation.

## Five-minute quickstart

1. **Install the package** (see above) inside a virtual environment.
2. **Bootstrap a template project** using the bundled example suite:

   ```bash
   agentunit agentunit.examples.template_project.suite \
	 --json reports/template.json \
	 --markdown reports/template.md \
	 --junit reports/template.xml
   ```

   This runs the canned template agent against two sample questions and emits three report formats under `reports/`.

3. **Inspect the reports** for aggregated pass/fail signals, metric scores, and tool usage breakdowns.

Ready to wire in your own agent? Proceed to the next section.

## Workflow overview

AgentUnit breaks evaluations into four concepts:

| Concept | Purpose |
| --- | --- |
| **Dataset** | Supplies deterministic prompts, ground-truth expectations, and contextual metadata for each case. |
| **Adapter** | Knows how to call your agent (LangGraph, CrewAI, custom code) and translate results into `AdapterOutcome` objects. |
| **Scenario** | Couples an adapter with a dataset, retries, and runtime limits. Scenarios are collected into suites. |
| **Metrics** | Score each run with heuristic checks (exact match, tool success) or RAGAS-powered evaluations when dependencies are available. |

You author suites in plain Python, then execute them with the CLI or your own runner.

## Authoring evaluation suites

Use the scaffolding below to create your own suite module. Place the file anywhere in your codebase (for example `evals/my_suite.py`) and point the CLI to it.

```python
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource


def build_dataset() -> DatasetSource:
	"""Return a DatasetSource yielding deterministic DatasetCase objects."""

	def _loader():
		yield DatasetCase(
			id="faq-001",
			query="What is the capital of France?",
			expected_output="Paris is the capital of France.",
			context=["Paris is the capital of France."],
			tools=["knowledge_base"],
		)

	return DatasetSource(name="faq-demo", loader=_loader)


class MyAdapter(BaseAdapter):
	"""Calls your agent stack and returns AdapterOutcome objects."""

	name = "faq-adapter"

	def __init__(self, agent):
		self._agent = agent

	def prepare(self) -> None:
		...  # warm up connections, load prompts, etc.

	def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
		trace.record("agent_prompt", input={"query": case.query, "context": case.context})
		answer = self._agent.answer(case.query, context=case.context)
		trace.record("agent_response", content=answer)
		success = case.expected_output is None or answer.strip() == case.expected_output.strip()
		return AdapterOutcome(success=success, output=answer)

	def cleanup(self) -> None:
		...


dataset = build_dataset()


def create_suite():
	agent = ...  # instantiate your production agent here
	adapter = MyAdapter(agent)
	scenario = Scenario(name="faq-demo", adapter=adapter, dataset=dataset)
	return [scenario]


suite = list(create_suite())
```

### Helpful utilities

- `Scenario.load_langgraph(path, dataset=...)`: build scenarios directly from LangGraph graphs.
- `Scenario.from_openai_agents(flow, dataset=...)`: plug into the OpenAI Agents SDK.
- `Scenario.from_crewai(crew, dataset=...)`: evaluate any CrewAI `Crew` object.
- `Scenario.with_dataset(new_dataset)`: reuse adapters across multiple datasets.

See the [template suite](docs/templates/suite_template.py) for a fully commented example.

## Command-line interface

Run suites with the `agentunit` CLI:

```bash
agentunit path.to.your.suite \
  --metrics faithfulness answer_correctness \
  --otel-exporter otlp \
  --json reports/results.json \
  --markdown reports/results.md \
  --junit reports/results.xml
```

### Key flags

- `suite`: Either a module path (`package.module`) or a Python file ending in `.py` that exports `suite` or `create_suite`.
- `--metrics`: Optional subset of metric names. Omit to run everything registered in `agentunit.metrics`.
- `--seed`: Force deterministic shuffling for stochastic datasets.
- `--otel-exporter`: Choose `console` (default pretty-print) or `otlp` (send spans to OTLP endpoint).
- `--json`, `--markdown`, `--junit`: Export evaluation artifacts in your preferred formats.

Full CLI documentation lives under [`docs/cli.md`](docs/cli.md).

## Metrics & reporting

- **Answer correctness**: Exact-match with RAGAS-backed fuzzy scoring when available.
- **Faithfulness**, **retrieval quality**, **hallucination rate**: Guard against fabricated responses by aligning answers with provided context.
- **Tool success**: Verifies end-to-end workflows by aggregating tool call outcomes from traces.

Exports contain per-scenario, per-case, and aggregate metric summaries. JUnit reports integrate with CI platforms, while JSON/Markdown support custom dashboards.

## Observability with OpenTelemetry

AgentUnit emits structured spans through OpenTelemetry. Set `--otel-exporter otlp` to forward traces to an OTLP-compatible collector (such as OpenTelemetry Collector, Honeycomb, or Datadog). The default `console` exporter prints spans to stdout for quick inspection.

You can also hook into `TraceLog` within adapters to record domain-specific events (tool prompts, retrieval hits, etc.).

## Local development

Clone the repository if you plan to contribute:

```bash
git clone https://github.com/aviralgarg05/agentunit.git
cd agentunit
poetry install
poetry run pytest
```

The template suite lives under `src/agentunit/examples/template_project/` to help you smoke-test the framework.

## Additional resources

- [Quickstart guide](docs/quickstart.md)
- [How to write scenarios](docs/writing-scenarios.md)
- [CLI reference](docs/cli.md)
- [Template suite skeleton](docs/templates/suite_template.py)

If you build something cool with AgentUnit, let us know by opening an issue or discussion on GitHub!
