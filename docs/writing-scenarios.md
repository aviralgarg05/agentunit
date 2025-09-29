# Writing Scenarios

This guide covers the building blocks that make up an AgentUnit evaluation suite. By the end you will understand how datasets, adapters, and scenarios cooperate, and you will have reusable templates to adapt for your project.

## Core data structures

| Component | Description | Implemented via |
| --- | --- | --- |
| **DatasetSource** | Lazily yields `DatasetCase` objects. Each case encapsulates a unique prompt, target answer, tool hints, and optional context documents. | `agentunit.datasets.base.DatasetSource` |
| **BaseAdapter** | Abstract base class your integration inherits from. Responsible for preparing resources, executing a case, and returning an `AdapterOutcome`. | `agentunit.adapters.base.BaseAdapter` |
| **Scenario** | Couples an adapter with a dataset and execution policy (timeouts, retries, tags). Suites are simply iterables of `Scenario` instances. | `agentunit.core.scenario.Scenario` |

## Template: dataset module

Create `datasets/faq_dataset.py`:

```python
from agentunit.datasets.base import DatasetCase, DatasetSource


def load_cases():
    yield DatasetCase(
        id="faq-001",
        query="What is the capital of France?",
        expected_output="Paris is the capital of France.",
        context=["Paris is the capital of France."],
        tools=["knowledge_base"],
    )


dataset = DatasetSource(name="faq", loader=load_cases)
```

- Use generators to stream large datasets without loading everything into memory.
- Add additional metadata in `DatasetCase.metadata` if your adapter needs it (for example retrieval corpus IDs).

## Template: adapter module

Create `adapters/faq_adapter.py`:

```python
from agentunit.adapters.base import AdapterOutcome, BaseAdapter
from agentunit.core.trace import TraceLog
from agentunit.datasets.base import DatasetCase


class FAQAdapter(BaseAdapter):
    name = "faq-adapter"

    def __init__(self, agent):
        self._agent = agent
        self._ready = False

    def prepare(self) -> None:
        if not self._ready:
            self._agent.connect()
            self._ready = True

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:  # type: ignore[override]
        trace.record("agent_prompt", input={"query": case.query, "context": case.context})
        answer = self._agent.answer(case.query, context=case.context)
        trace.record("agent_response", content=answer)
        success = case.expected_output is None or answer.strip() == case.expected_output.strip()
        return AdapterOutcome(success=success, output=answer)

    def cleanup(self) -> None:
        self._agent.close()
        self._ready = False
```

- Use `TraceLog.record` liberally: every event is available to downstream metrics and telemetry.
- Return tool call metadata in `AdapterOutcome.tool_calls` when your agent interacts with external systems.

## Template: scenario suite

Create `evals/faq_suite.py`:

```python
from adapters.faq_adapter import FAQAdapter
from datasets.faq_dataset import dataset
from agentunit.core.scenario import Scenario


def create_suite(agent) -> list[Scenario]:
    adapter = FAQAdapter(agent)
    scenario = Scenario(name="faq-demo", adapter=adapter, dataset=dataset, retries=1, max_turns=10)
    return [scenario]


suite = create_suite(agent=...)  # pragma: no cover - replace with actual agent
```

### Using convenience constructors

AgentUnit ships with adapter helpers when you use popular frameworks:

```python
from agentunit.core.scenario import Scenario

# LangGraph (path to graph file or Python object)
langgraph_scenario = Scenario.load_langgraph("graphs/customer_support.py", dataset="faq")

# OpenAI Agents SDK
from my_flows import support_flow
openai_scenario = Scenario.from_openai_agents(support_flow, dataset="faq", name="support-flow")

# CrewAI
from my_crewai_setup import crew
crewai_scenario = Scenario.from_crewai(crew, dataset="faq", retries=2)
```

Mix and match scenarios in a plain list or generator; the CLI accepts anything iterable.

## Organizing suites

A common layout for teams with multiple agents:

```
project/
├─ adapters/
├─ datasets/
├─ evals/
│  ├─ customer_support.py
│  └─ financial_assistant.py
└─ tests/
```

- Keep reusable datasets in `datasets/` and reference them across suites.
- Export both `create_suite()` and `suite` so the CLI can import whichever it finds first.
- If scenarios share adapters, use `Scenario.with_dataset` to avoid re-instantiating your agent.

## Tips for reliable evaluations

1. **Seed randomness** – pass `seed=1234` when your dataset shuffles candidates.
2. **Limit retries** – flaky agents can hide regressions; keep `retries` small in CI.
3. **Record context** – populate `DatasetCase.context` so faithfulness metrics can operate.
4. **Log tool usage** – metrics like `tool_success` rely on the trace events you emit.
5. **Generate reports** – combine `--junit` for CI and `--markdown` for human-friendly summaries.

Continue with the [CLI reference](cli.md) to see all runtime options.
