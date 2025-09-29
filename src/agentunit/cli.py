"""Command line interface for AgentUnit."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Iterable, List
import sys

from .core.scenario import Scenario
from .core.runner import run_suite


def entrypoint(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentUnit CLI")
    parser.add_argument("suite", help="Python module or file defining a 'suite' list of scenarios")
    parser.add_argument("--metrics", nargs="*", help="Metric names to evaluate", default=None)
    parser.add_argument("--otel-exporter", choices=["console", "otlp"], default="console")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--junit", type=str, default=None, help="Path for JUnit XML export")
    parser.add_argument("--json", dest="json_path", type=str, default=None, help="Path for JSON export")
    parser.add_argument("--markdown", type=str, default=None, help="Path for Markdown export")
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenarios = _load_scenarios(args.suite)
    if not scenarios:
        parser.error("Suite did not resolve to any Scenario instances")

    result = run_suite(scenarios, metrics=args.metrics, otel_exporter=args.otel_exporter, seed=args.seed)

    if args.junit:
        result.to_junit(args.junit)
    if args.json_path:
        result.to_json(args.json_path)
    if args.markdown:
        result.to_markdown(args.markdown)

    return 0


def _load_scenarios(target: str) -> List[Scenario]:
    path = Path(target)
    module = None
    if path.exists():
        if path.suffix != ".py":
            raise SystemExit("Only Python files are supported at the moment")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Unable to import module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    suite = getattr(module, "suite", None)
    if suite is None and hasattr(module, "create_suite"):
        suite = module.create_suite()
    if suite is None:
        raise SystemExit("Module must export 'suite' list or 'create_suite' callable")
    scenarios = list(suite)
    if not all(isinstance(item, Scenario) for item in scenarios):
        raise SystemExit("Suite must contain Scenario instances")
    return scenarios


if __name__ == "__main__":  # pragma: no cover
    sys.exit(entrypoint())
