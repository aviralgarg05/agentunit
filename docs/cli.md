# CLI Reference

The `agentunit` command executes one or more evaluation scenarios and exports summaries. This reference documents every flag and shows concrete recipes.

## Synopsis

```bash
agentunit <suite> [options]
```

- `<suite>` – a Python module path (`package.module`) or a filesystem path to a `.py` file. The module must expose either `suite` (an iterable of `Scenario`) or a callable `create_suite()` that returns one.

## Global options

| Flag | Description |
| --- | --- |
| `--metrics METRIC [METRIC ...]` | Optional subset of registered metrics. Default runs everything registered in `agentunit.metrics`. |
| `--otel-exporter {console,otlp}` | Choose the telemetry exporter. `console` (default) pretty prints spans; `otlp` streams to the active OTLP endpoint. |
| `--seed INT` | Seed used to initialize any random generators before iterating dataset cases. |
| `--json PATH` | Write evaluation output as structured JSON. |
| `--markdown PATH` | Write a human-readable Markdown summary. |
| `--junit PATH` | Write JUnit XML compatible with most CI dashboards. |

## Module resolution

AgentUnit looks for suites in two ways:

1. **Python module import** – run `agentunit my_project.evals.product_suite`. Python must be able to import the module (package installed or available on `PYTHONPATH`).
2. **Direct file execution** – run `agentunit evals/product_suite.py`. The file must have a `.py` extension and export `suite` or `create_suite`.

If both `suite` and `create_suite` exist, the CLI prefers `suite`. Use `create_suite()` to keep initialization logic fresh on every CLI invocation.

## Export formats

- **JSON**: includes aggregate metrics, per-scenario records, and per-case diagnostics. Ideal for dashboards or custom scripts.
- **Markdown**: readable summary for pull requests and changelogs.
- **JUnit**: standard CI integration. Failed cases surface as failed tests.

You can supply any combination of export flags during a single run.

## Metrics

Metrics are discovered from the plugin registry (`agentunit.metrics.registry`). To list available metrics:

```python
from agentunit.metrics.registry import list_metric_names
print(list_metric_names())
```

Specify the subset you care about via `--metrics faithfulness tool_success`.

## Telemetry

Set `--otel-exporter otlp` to forward spans to your collector. Set the following environment variables to configure the OTLP endpoint before launching the CLI:

- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_HEADERS`
- `OTEL_RESOURCE_ATTRIBUTES`

If omitted, the SDK default (localhost gRPC) is used. Use `console` exporter when you want to debug spans locally.

## Reproducibility controls

- `--seed 42` – seeds Python's `random` module before iterating cases.
- Use deterministic datasets where possible (sorting inputs, removing randomness from retrieval).
- Pair seeds with pinned model versions to reduce nondeterministic responses.

## Example commands

Run a suite with all metrics and Markdown export:

```bash
agentunit my_project.evals.qa_suite --markdown reports/qa-suite.md
```

Limit metrics to correctness and hallucination:

```bash
agentunit my_project.evals.qa_suite --metrics answer_correctness hallucination_rate
```

Send telemetry to OTLP collector and emit all export formats:

```bash
agentunit my_project.evals.qa_suite \
  --otel-exporter otlp \
  --json reports/qa.json \
  --markdown reports/qa.md \
  --junit reports/qa.xml
```

## Exit codes

- `0` – CLI executed successfully (individual scenario failures still produce exit code 0; inspect reports for results).
- Non-zero – CLI encountered a configuration or runtime error (invalid suite, import failure, unhandled adapter exception).

Use JUnit exports or JSON parsing to fail CI when cases regress.
