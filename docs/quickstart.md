# Quickstart

This guide walks you through the fastest way to evaluate an agent with AgentUnit. You will install the package, run the bundled template suite, and wire in your own adapter.

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

The CLI entry point `agentunit` should now be on your `$PATH`.

## 2. Run the template suite

AgentUnit ships with a deterministic template agent so you can verify that everything works end-to-end.

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

Check the `reports/` directory for Markdown, JSON, and JUnit exports you can share with stakeholders or CI systems.

## 3. Swap in your own agent

Open a new Python module (for example `evals/my_suite.py`) and paste the template below. Replace the `...` sections with your agent initialization and adapter logic.

```python
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.scenario import Scenario
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase, DatasetSource


def build_dataset() -> DatasetSource:
    def _loader():
        yield DatasetCase(
            id="faq-001",
            query="What is the capital of France?",
            expected_output="Paris is the capital of France.",
            context=["Paris is the capital of France."],
            tools=["knowledge_base"],
        )

    return DatasetSource(name="faq-demo", loader=_loader)


dataset = build_dataset()


+class MyAdapter(BaseAdapter):
+    name = "faq-adapter"
+
+    def __init__(self, agent):
+        self._agent = agent
+
+    def prepare(self) -> None:
+        self._agent.connect()  # optional
+
+    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
+        trace.record("agent_prompt", input={"query": case.query})
+        answer = self._agent.answer(case.query, context=case.context)
+        trace.record("agent_response", content=answer)
+        return AdapterOutcome(success=answer.strip() == case.expected_output.strip(), output=answer)
+
+    def cleanup(self) -> None:
+        self._agent.close()  # optional
+
+
+def create_suite():
+    agent = ...  # instantiate your agent here
+    adapter = MyAdapter(agent)
+    scenario = Scenario(name="faq-demo", adapter=adapter, dataset=dataset)
+    return [scenario]
+
+
+suite = list(create_suite())
+```
+
+Run your suite with:
+
+```bash
+agentunit evals.my_suite --markdown reports/my-suite.md
+```
+
+## 4. Tailor metrics and outputs
+
+- Add `--metrics faithfulness answer_correctness` to restrict evaluation to specific metrics.
+- Use `--otel-exporter otlp` when you want spans to stream into an OpenTelemetry collector.
+- Export reports in multiple formats simultaneously by combining `--json`, `--markdown`, and `--junit` flags.
+
+## 5. Iterate in CI/CD
+
+Add the CLI invocation to your continuous integration pipeline. JUnit exports let you fail PRs automatically when scenarios regress. Markdown or JSON can power dashboards for stakeholders.
+
+Need more depth? Continue with the [Writing Scenarios guide](writing-scenarios.md).
