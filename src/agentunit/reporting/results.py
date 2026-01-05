"""Result containers and exporters."""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agentunit.reporting.html import render_html_report


if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from agentunit.core.trace import TraceLog


@dataclass(slots=True)
class ScenarioRun:
    scenario_name: str
    case_id: str
    success: bool
    metrics: dict[str, float | None]
    duration_ms: float
    trace: TraceLog
    error: str | None = None


@dataclass(slots=True)
class ScenarioResult:
    name: str
    runs: list[ScenarioRun] = field(default_factory=list)

    def add_run(self, run: ScenarioRun) -> None:
        self.runs.append(run)

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.success) / len(self.runs)

    def aggregate_metric(self, name: str) -> float | None:
        values = [run.metrics.get(name) for run in self.runs if run.metrics.get(name) is not None]
        if not values:
            return None
        return float(statistics.fmean(values))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "success_rate": self.success_rate,
            "runs": [
                {
                    "case_id": run.case_id,
                    "success": run.success,
                    "metrics": run.metrics,
                    "duration_ms": run.duration_ms,
                    "error": run.error,
                    "trace": run.trace.to_dict(),
                }
                for run in self.runs
            ],
        }


@dataclass(slots=True)
class SuiteResult:
    scenarios: list[ScenarioResult]
    started_at: datetime
    finished_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def to_markdown(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# AgentUnit Report", ""]
        for scenario in self.scenarios:
            lines.extend(_render_markdown_scenario(scenario))
        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    def to_junit(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        testsuites = ET.Element(
            "testsuite",
            attrib={
                "name": "agentunit",
                "tests": str(sum(len(s.runs) for s in self.scenarios)),
                "failures": str(sum(1 for s in self.scenarios for r in s.runs if not r.success)),
                "time": f"{(self.finished_at - self.started_at).total_seconds():.4f}",
            },
        )
        for scenario in self.scenarios:
            for run in scenario.runs:
                testcase = ET.SubElement(
                    testsuites,
                    "testcase",
                    attrib={
                        "classname": scenario.name,
                        "name": run.case_id,
                        "time": f"{run.duration_ms / 1000.0:.4f}",
                    },
                )
                if not run.success:
                    failure = ET.SubElement(
                        testcase,
                        "failure",
                        attrib={"message": run.error or "Scenario failed"},
                    )
                    failure.text = json.dumps(run.metrics)
        tree = ET.ElementTree(testsuites)
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target

    def to_html(self, path: str | Path) -> Path:
        """
        Export results as a self-contained HTML report.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        html = render_html_report(self)
        target.write_text(html, encoding="utf-8")
        return target


def merge_results(results: Iterable[SuiteResult]) -> SuiteResult:
    results = list(results)
    scenarios: dict[str, ScenarioResult] = {}
    for result in results:
        for scenario in result.scenarios:
            existing = scenarios.setdefault(scenario.name, ScenarioResult(name=scenario.name))
            for run in scenario.runs:
                existing.add_run(run)
    started = min(result.started_at for result in results)
    finished = max(result.finished_at for result in results)
    return SuiteResult(scenarios=list(scenarios.values()), started_at=started, finished_at=finished)


def _render_markdown_scenario(scenario: ScenarioResult) -> list[str]:
    lines = [f"## {scenario.name}", f"Success rate: {scenario.success_rate:.2%}", ""]
    for run in scenario.runs:
        lines.append(f"- **{run.case_id}**: {'✅' if run.success else '❌'}")
        metrics_repr = ", ".join(
            f"{name}={value:.2f}" for name, value in run.metrics.items() if value is not None
        )
        if metrics_repr:
            lines.append(f"  - Metrics: {metrics_repr}")
        if run.error:
            lines.append(f"  - Error: {run.error}")
    lines.append("")
    return lines
