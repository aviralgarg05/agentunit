# AgentUnit

AgentUnit is a framework for evaluating, monitoring, and benchmarking multi-agent systems. It standardises how teams define scenarios, run experiments, and report outcomes across adapters, model providers, and deployment targets.

## Overview

- **Scenario-centric design** – describe datasets, adapters, and policies once, then reuse them in local runs, CI jobs, and production monitors.
- **Extensible adapters** – plug into LangGraph, CrewAI, PromptFlow, OpenAI Swarm, Anthropic Bedrock, Phidata, and custom agents through a consistent interface.
- **Comprehensive metrics** – combine exact-match assertions, RAGAS quality scores, and operational metrics with optional OpenTelemetry traces.
- **Production-first tooling** – export JSON, Markdown, and JUnit reports, gate releases with regression detection, and surface telemetry in existing observability stacks.

## Installation

AgentUnit targets Python 3.9+. The recommended workflow uses Poetry for dependency management.

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

Refer to the [framework integrations catalog](docs/framework-integrations.md) for per-adapter requirements.

## Getting started

1. Follow the [Quickstart](docs/quickstart.md) to run the bundled template suite and swap in your own adapter.
2. Review [Writing Scenarios](docs/writing-scenarios.md) for dataset and adapter templates plus helper constructors for popular frameworks.
3. Consult the [CLI reference](docs/cli.md) to orchestrate suites from the command line and export results for CI, dashboards, or audits.

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
| Quick evaluation walkthrough | [docs/quickstart.md](docs/quickstart.md) |
| Scenario and adapter authoring | [docs/writing-scenarios.md](docs/writing-scenarios.md) |
| CLI options and examples | [docs/cli.md](docs/cli.md) |
| Architecture overview | [docs/architecture.md](docs/architecture.md) |
| Framework-specific guides | [docs/platform-guides.md](docs/platform-guides.md) |
| No-code builder guide | [docs/nocode-quickstart.md](docs/nocode-quickstart.md) |
| Templates | [docs/templates/](docs/templates/) |
| Performance testing | [docs/performance-testing.md](docs/performance-testing.md) |

Use the table above as the canonical navigation surface; every document cross-links back to related topics for clarity.

## Development workflow

1. Install dependencies (Poetry or pip).
2. Run the unit and integration suite:

```bash
poetry run python3 -m pytest tests -v
```

3. Execute targeted suites during active development, then run the full matrix before opening a pull request.

Latest verification (2025-10-07): 144 passed, 10 skipped, 32 warnings. Warnings originate from third-party dependencies (`langchain` pydantic shim deprecations and `datetime.utcnow` usage). Track upstream fixes or pin patched releases as needed.

## Contributing

- Fork the repository and target the `main` branch for pull requests.
- Include tests for new features or behavioural changes.
- Update documentation when public APIs change; use the navigation table above to keep references synchronized.
- Adhere to the existing code style and run `pytest` before submitting changes.

Security disclosures and discussions are managed through GitHub issues; sensitive topics should follow responsible disclosure guidelines outlined in `SECURITY.md` (if unavailable, open a private issue via GitHub).

## License

AgentUnit is released under the MIT License. See [LICENSE](LICENSE) for the full text.

---

Need an overview for stakeholders? Start with [docs/architecture.md](docs/architecture.md). Ready to extend the platform? Explore the templates under [docs/templates/](docs/templates/).
