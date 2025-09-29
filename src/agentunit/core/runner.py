"""Scenario runner orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from time import perf_counter
from datetime import datetime, timezone
import logging
import random

from .scenario import Scenario
from ..metrics.registry import resolve_metrics
from ..metrics.base import Metric, MetricResult
from ..reporting.results import ScenarioResult, ScenarioRun, SuiteResult
from ..core.trace import TraceLog
from ..telemetry.tracing import configure_tracer, span

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunnerConfig:
    metrics: Sequence[str] | None = None
    otel_exporter: str | None = None
    seed: int | None = None


class Runner:
    def __init__(self, scenarios: Iterable[Scenario], config: RunnerConfig | None = None) -> None:
        self._scenarios = list(scenarios)
        self._config = config or RunnerConfig()
        if self._config.seed is not None:
            random.seed(self._config.seed)

    def run(self) -> SuiteResult:
        configure_tracer(self._config.otel_exporter)
        metrics = resolve_metrics(self._config.metrics)
        started = datetime.now(timezone.utc)
        scenario_results: List[ScenarioResult] = []
        for scenario in self._scenarios:
            scenario_result = self._run_scenario(scenario, metrics)
            scenario_results.append(scenario_result)
        finished = datetime.now(timezone.utc)
        return SuiteResult(scenarios=scenario_results, started_at=started, finished_at=finished)

    def _run_scenario(self, scenario: Scenario, metrics: Sequence[Metric]) -> ScenarioResult:
        scenario.adapter.prepare()
        scenario_result = ScenarioResult(name=scenario.name)
        for case in scenario.iter_cases():
            trace_log = TraceLog()
            attempts = scenario.retries + 1
            final_outcome = None
            error = None
            start = perf_counter()
            for attempt in range(attempts):
                with span("agentunit.scenario", scenario=scenario.name, case=case.id, attempt=attempt):
                    final_outcome = scenario.adapter.execute(case, trace_log)
                if final_outcome.success:
                    break
                error = final_outcome.error
            duration_ms = (perf_counter() - start) * 1000.0
            metric_values = _evaluate_metrics(metrics, case, trace_log, final_outcome)
            run = ScenarioRun(
                scenario_name=scenario.name,
                case_id=case.id,
                success=bool(final_outcome and final_outcome.success),
                metrics=metric_values,
                duration_ms=duration_ms,
                trace=trace_log,
                error=error,
            )
            scenario_result.add_run(run)
        scenario.adapter.cleanup()
        return scenario_result


def _evaluate_metrics(metrics: Sequence[Metric], case, trace, outcome) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for metric in metrics:
        try:
            result: MetricResult = metric.evaluate(case, trace, outcome)
            values[result.name] = result.value
        except Exception:  # pragma: no cover
            logger.exception("Metric %s failed", metric.name)
            values[metric.name] = None
    return values


def run_suite(
    suite: Iterable[Scenario],
    metrics: Sequence[str] | None = None,
    otel_exporter: str | None = None,
    seed: int | None = None,
) -> SuiteResult:
    runner = Runner(scenarios=suite, config=RunnerConfig(metrics=metrics, otel_exporter=otel_exporter, seed=seed))
    return runner.run()
