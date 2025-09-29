# AgentUnit

AgentUnit is a pytest-inspired evaluation harness for autonomous agents and retrieval-augmented generation (RAG) workflows. It helps you describe repeatable scenarios, connect them to your agent stack, and score results with both heuristic and LLM-backed metrics.

## At a glance

- 📦 **Installable package**: `pip install agentunit` brings the CLI, adapters, metrics, and scenario primitives into your project.
- 🧪 **Scenario-driven evaluations**: Define deterministic test suites for LangGraph, OpenAI Agents, CrewAI, or custom adapters.
- 📊 **Built-in metrics & reports**: Export Markdown, JSON, and JUnit summaries. Optional OpenTelemetry spans keep your observability stack in sync.
- 🧱 **Composable templates**: Copy-ready skeletons help you scaffold datasets, adapters, and suites in minutes.

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

## Installation

AgentUnit targets Python 3.10 and newer. Install from PyPI:

```bash
pip install agentunit
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
