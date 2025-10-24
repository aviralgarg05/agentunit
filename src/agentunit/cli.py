"""Command line interface for AgentUnit."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Iterable, List
import sys

from .core.scenario import Scenario
from .core.runner import run_suite


class EnhancedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that preserves formatting and adds defaults."""
    
    def _get_help_string(self, action):
        help_text = action.help or ""
        if action.default is not argparse.SUPPRESS and action.default is not None:
            if action.default != "":
                help_text += f" (default: {action.default})"
        return help_text


def entrypoint(argv: Iterable[str] | None = None) -> int:
    epilog = """
examples:
  # Run evaluations from a Python file
  agentunit suite.py
  
  # Run with specific metrics
  agentunit suite.py --metrics accuracy relevance
  
  # Export results in multiple formats
  agentunit suite.py --junit results.xml --json results.json --markdown report.md
  
  # Run with OTLP telemetry export
  agentunit suite.py --otel-exporter otlp
  
  # Set random seed for reproducibility
  agentunit suite.py --seed 42

environment variables:
  AGENTUNIT_OTEL_EXPORTER    Default OpenTelemetry exporter (console or otlp)
  AGENTUNIT_SEED             Default random seed for reproducibility
  OTEL_EXPORTER_OTLP_ENDPOINT OTLP endpoint (e.g., http://localhost:4317)
  OTEL_SERVICE_NAME          Service name for telemetry (default: agentunit)
  OPENAI_API_KEY             OpenAI API key for LLM-based metrics
  ANTHROPIC_API_KEY          Anthropic API key for Claude models
  
exit codes:
  0    All scenarios passed successfully
  1    One or more scenarios failed
  2    Configuration or usage error
    """
    
    parser = argparse.ArgumentParser(
        description="AgentUnit - Agentic AI Evaluation Framework",
        epilog=epilog,
        formatter_class=EnhancedHelpFormatter
    )
    
    parser.add_argument(
        "suite",
        help="Python module or file defining a 'suite' list of scenarios. "
             "The file must export a 'suite' variable (list of Scenario objects) "
             "or a 'create_suite()' callable that returns the suite."
    )
    
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        metavar="METRIC",
        help="Metric names to evaluate. If not specified, all metrics defined in "
             "scenarios will be evaluated. Available metrics: accuracy, faithfulness, "
             "relevance, coherence, hallucination, toxicity, bias, latency, cost, etc."
    )
    
    parser.add_argument(
        "--otel-exporter",
        choices=["console", "otlp"],
        default=os.environ.get("AGENTUNIT_OTEL_EXPORTER", "console"),
        help="OpenTelemetry exporter backend. 'console' prints to stdout, "
             "'otlp' sends to OTLP endpoint specified by OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("AGENTUNIT_SEED", "0")) or None,
        metavar="INT",
        help="Random seed for reproducible evaluations. Affects sampling, "
             "LLM temperature, and other stochastic operations"
    )
    
    parser.add_argument(
        "--junit",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results as JUnit XML to the specified path. "
             "Useful for CI/CD integration with test reporting tools"
    )
    
    parser.add_argument(
        "--json",
        dest="json_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results as JSON to the specified path. "
             "Includes detailed metrics, timings, and metadata"
    )
    
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        metavar="PATH",
        help="Export results as Markdown report to the specified path. "
             "Human-readable format with tables and summaries"
    )
    
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=False,
        help="Exit with code 1 if any scenario metrics fall below their thresholds. "
             "Enables using AgentUnit as a quality gate in CI/CD pipelines"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output with detailed execution information"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.6.0"
    )
    
    args = parser.parse_args(list(argv) if argv is not None else None)

    scenarios = _load_scenarios(args.suite)
    if not scenarios:
        parser.error("Suite did not resolve to any Scenario instances")

    result = run_suite(
        scenarios,
        metrics=args.metrics,
        otel_exporter=args.otel_exporter,
        seed=args.seed
    )

    if args.junit:
        result.to_junit(args.junit)
        if args.verbose:
            print(f"JUnit XML exported to: {args.junit}")
    
    if args.json_path:
        result.to_json(args.json_path)
        if args.verbose:
            print(f"JSON results exported to: {args.json_path}")
    
    if args.markdown:
        result.to_markdown(args.markdown)
        if args.verbose:
            print(f"Markdown report exported to: {args.markdown}")

    # Check for failures or regressions
    if args.fail_on_regression and hasattr(result, "has_regressions"):
        if result.has_regressions():
            print("ERROR: Quality regressions detected")
            return 1
    
    if hasattr(result, "has_failures") and result.has_failures():
        print("ERROR: One or more scenarios failed")
        return 1

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
