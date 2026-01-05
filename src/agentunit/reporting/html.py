from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from agentunit.reporting.results import SuiteResult


def render_html_report(result: SuiteResult) -> str:
    scenarios = result.scenarios

    total_runs = sum(len(s.runs) for s in scenarios)
    failed_runs = sum(1 for s in scenarios for r in s.runs if not r.success)
    passed_runs = total_runs - failed_runs
    duration = (result.finished_at - result.started_at).total_seconds()

    scenario_rows: list[str] = []

    for scenario in scenarios:
        scenario_rows.append(
            f"""
            <tr>
              <td>{escape(scenario.name)}</td>
              <td>{scenario.success_rate:.2%}</td>
              <td>{len(scenario.runs)}</td>
            </tr>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AgentUnit Report</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      padding: 24px;
    }}
    h1, h2 {{
      margin-bottom: 0.5em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }}
    th {{
      background-color: #f4f4f4;
    }}
    .bar {{
      height: 18px;
      margin: 6px 0;
    }}
    .passed {{
      background-color: #4CAF50;
      width: {passed_runs * 20}px;
    }}
    .failed {{
      background-color: #F44336;
      width: {failed_runs * 20}px;
    }}
    .meta {{
      color: #555;
      font-size: 0.9em;
    }}
  </style>
</head>

<body>

<h1>AgentUnit HTML Report</h1>
<p class="meta">
  Duration: {duration:.2f}s · Total runs: {total_runs}
</p>

<h2>Summary</h2>
<p>Passed: {passed_runs} · Failed: {failed_runs}</p>

<div class="bar passed"></div>
<div class="bar failed"></div>

<h2>Scenarios</h2>
<table>
  <tr>
    <th>Name</th>
    <th>Success rate</th>
    <th>Runs</th>
  </tr>
  {"".join(scenario_rows)}
</table>

</body>
</html>
"""
