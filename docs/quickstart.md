# Quickstart

This guide walks through the fastest path to exercising AgentUnit: install the package, run the bundled template suite, and connect your own agent.

## Prerequisites

- Python 3.10 or higher
- A virtual environment (recommended)
- Access to the agent you want to evaluate (LangGraph graph, LLM client, CrewAI crew, etc.)

## 1. Install AgentUnit

```bash
python -m venv .venv
source .venv/bin/activate
pip install agentunit
```

The CLI entry point `agentunit` should now be available on your `$PATH`. If you are developing against the repository clone, use `poetry install` instead (see the top-level [README](../README.md#installation)).

## 2. Run the template suite

AgentUnit ships with a deterministic template agent so you can confirm end-to-end execution.

```bash
agentunit agentunit.examples.template_project.suite \
  --json reports/template.json \
  --markdown reports/template.md \
  --junit reports/template.xml
```

Expect output similar to the following:

```
Scenario template-agent-demo: 2 cases
✓ template-001
✓ template-002
```

Check the `reports/` directory for Markdown, JSON, and JUnit exports that can feed dashboards or CI gates.

from agentunit.core.trace import TraceLog

```python
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource


+
    def _loader():
        yield DatasetCase(
            id="faq-001",
            query="What is the capital of France?",
            expected_output="Paris is the capital of France.",
            context=["Paris is the capital of France."],
            tools=["knowledge_base"],
        )

    return DatasetSource(name="faq-demo", loader=_loader)


+Add the CLI invocation to your continuous integration pipeline. JUnit exports let you fail PRs automatically when scenarios regress. Markdown or JSON can power dashboards for stakeholders.


class MyAdapter(BaseAdapter):
    name = "faq-adapter"

    def __init__(self, agent):
        self._agent = agent

    def prepare(self) -> None:
        self._agent.connect()  # optional

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        trace.record("agent_prompt", input={"query": case.query})
        answer = self._agent.answer(case.query, context=case.context)
        trace.record("agent_response", content=answer)
        return AdapterOutcome(success=answer.strip() == case.expected_output.strip(), output=answer)

    def cleanup(self) -> None:
        self._agent.close()  # optional


+
    agent = ...  # instantiate your agent here
    adapter = MyAdapter(agent)
    scenario = Scenario(name="faq-demo", adapter=adapter, dataset=dataset)
    return [scenario]


+Need more depth? Continue with the [Writing Scenarios guide](writing-scenarios.md).
```

Run your suite with:

```bash
agentunit evals.my_suite --markdown reports/my-suite.md
```

Refer to [Writing Scenarios](writing-scenarios.md) for additional dataset adapters and helper constructors.

## 4. Tailor metrics and outputs

- Add `--metrics faithfulness answer_correctness` to restrict evaluation to specific metrics.
- Use `--otel-exporter otlp` to stream spans into an OpenTelemetry collector.
- Export reports in multiple formats by combining `--json`, `--markdown`, and `--junit` flags.

## 5. Iterate in CI/CD

Integrate the CLI command into your continuous integration pipeline. JUnit exports allow you to fail pull requests automatically when scenarios regress, while JSON and Markdown outputs can inform dashboards.

Need more depth? Continue with the [Writing Scenarios guide](writing-scenarios.md) or explore the [CLI reference](cli.md).
