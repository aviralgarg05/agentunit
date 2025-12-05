"""Dashboard UI components."""

from typing import Any


try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None


class SuiteEditor:
    """Component for editing evaluation suites."""

    def __init__(self):
        """Initialize suite editor."""
        if not HAS_STREAMLIT:
            msg = "streamlit is required for dashboard components"
            raise ImportError(msg)

    def render(self, suite_data: dict[str, Any] | None = None):
        """Render suite editor interface.

        Args:
            suite_data: Existing suite data to edit
        """
        suite_data = suite_data or {}

        # Suite metadata
        st.subheader("Suite Configuration")
        name = st.text_input("Name", value=suite_data.get("name", ""))
        description = st.text_area("Description", value=suite_data.get("description", ""))

        # Adapter settings
        st.subheader("Adapter Settings")
        adapter_type = st.selectbox(
            "Type",
            ["openai", "anthropic", "custom"],
            index=self._get_adapter_index(suite_data.get("adapter", {})),
        )

        # Dataset management
        st.subheader("Dataset")
        dataset_action = st.radio("Action", ["Load Existing", "Upload New", "Generate with LLM"])

        return {
            "name": name,
            "description": description,
            "adapter": {"type": adapter_type},
            "dataset_action": dataset_action,
        }

    def _get_adapter_index(self, adapter: dict[str, Any]) -> int:
        """Get adapter type index."""
        adapter_types = ["openai", "anthropic", "custom"]
        adapter_type = adapter.get("type", "openai")
        return adapter_types.index(adapter_type) if adapter_type in adapter_types else 0


class RunMonitor:
    """Component for monitoring evaluation runs."""

    def __init__(self):
        """Initialize run monitor."""
        if not HAS_STREAMLIT:
            msg = "streamlit is required for dashboard components"
            raise ImportError(msg)

    def render(self, runs: list[dict[str, Any]]):
        """Render run monitor interface.

        Args:
            runs: List of run data dictionaries
        """
        st.subheader("Active Runs")

        for run in runs:
            status_icon = self._get_status_icon(run.get("status", "unknown"))

            with st.expander(f"{status_icon} Run {run.get('id', 'Unknown')}"):
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", run.get("total", 0))
                with col2:
                    st.metric("Passed", run.get("passed", 0))
                with col3:
                    st.metric("Failed", run.get("failed", 0))
                with col4:
                    st.metric("Progress", f"{run.get('progress', 0):.0f}%")

                # Progress bar
                progress = run.get("progress", 0) / 100
                st.progress(progress)

                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View Traces", key=f"traces_{run.get('id')}"):
                        return {"action": "view_traces", "run_id": run.get("id")}
                with col2:
                    if run.get("status") == "running":
                        if st.button("Stop", key=f"stop_{run.get('id')}"):
                            return {"action": "stop_run", "run_id": run.get("id")}

        return None

    def _get_status_icon(self, status: str) -> str:
        """Get icon for run status."""
        icons = {"running": "⏳", "completed": "✅", "failed": "❌", "stopped": "⏸️"}
        return icons.get(status, "❓")


class TraceViewer:
    """Component for viewing execution traces."""

    def __init__(self):
        """Initialize trace viewer."""
        if not HAS_STREAMLIT:
            msg = "streamlit is required for dashboard components"
            raise ImportError(msg)

    def render(self, trace: dict[str, Any]):
        """Render trace viewer interface.

        Args:
            trace: Trace data dictionary
        """
        # Trace header
        status = trace.get("status", "unknown")
        status_color = "green" if status == "passed" else "red"

        st.markdown(
            f"""
        <div style="padding: 10px; background-color: {status_color}20; border-left: 4px solid {status_color};">
            <strong>Case ID:</strong> {trace.get("case_id", "Unknown")}<br>
            <strong>Status:</strong> {status.upper()}<br>
            <strong>Duration:</strong> {trace.get("duration", 0):.2f}s
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tool Calls", "Metrics", "Raw Data"])

        with tab1:
            self._render_overview(trace)

        with tab2:
            self._render_tool_calls(trace.get("tool_calls", []))

        with tab3:
            self._render_metrics(trace.get("metrics", {}))

        with tab4:
            st.json(trace)

    def _render_overview(self, trace: dict[str, Any]):
        """Render trace overview."""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            st.code(trace.get("input", ""), language="json")

        with col2:
            st.subheader("Output")
            st.code(trace.get("output", ""), language="json")

        # Expected output (if available)
        if "expected" in trace:
            st.subheader("Expected Output")
            st.code(trace.get("expected", ""), language="json")

    def _render_tool_calls(self, tool_calls: list[dict[str, Any]]):
        """Render tool call timeline."""
        if not tool_calls:
            st.info("No tool calls in this trace")
            return

        st.subheader(f"Tool Calls ({len(tool_calls)})")

        for i, call in enumerate(tool_calls, 1):
            with st.container():
                st.markdown(
                    f"""
                <div style="border-left: 3px solid #4CAF50; padding-left: 15px; margin: 10px 0;">
                    <strong>Step {i}:</strong> {call.get("name", "Unknown")}<br>
                    <em>Duration: {call.get("duration", 0):.3f}s</em>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                with st.expander("Details"):
                    st.json(call)

    def _render_metrics(self, metrics: dict[str, Any]):
        """Render metric results."""
        if not metrics:
            st.info("No metrics available")
            return

        st.subheader("Metric Results")

        for metric_name, metric_data in metrics.items():
            with st.container():
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.text(metric_name)
                    if isinstance(metric_data, dict) and "details" in metric_data:
                        st.caption(metric_data.get("details", ""))

                with col2:
                    score = (
                        metric_data.get("score") if isinstance(metric_data, dict) else metric_data
                    )
                    if isinstance(score, (int, float)):
                        st.metric("Score", f"{score:.2f}")


class ReportExplorer:
    """Component for exploring evaluation reports."""

    def __init__(self):
        """Initialize report explorer."""
        if not HAS_STREAMLIT:
            msg = "streamlit is required for dashboard components"
            raise ImportError(msg)

    def render(self, report: dict[str, Any]):
        """Render report explorer interface.

        Args:
            report: Report data dictionary
        """
        report_type = report.get("type", "summary")

        if report_type == "summary":
            self._render_summary_report(report)
        elif report_type == "comparison":
            self._render_comparison_report(report)
        elif report_type == "regression":
            self._render_regression_report(report)

    def _render_summary_report(self, report: dict[str, Any]):
        """Render summary report."""
        st.header("Summary Report")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Rate", f"{report.get('success_rate', 0):.1f}%")
        with col2:
            st.metric("Avg Latency", f"{report.get('avg_latency', 0):.2f}s")
        with col3:
            st.metric("Total Cost", f"${report.get('total_cost', 0):.4f}")
        with col4:
            st.metric("Cases", report.get("total_cases", 0))

        # Metric breakdown
        st.subheader("Metric Breakdown")
        for metric_name, metric_stats in report.get("metrics", {}).items():
            with st.expander(metric_name):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{metric_stats.get('mean', 0):.2f}")
                with col2:
                    st.metric("Median", f"{metric_stats.get('median', 0):.2f}")
                with col3:
                    st.metric("Std Dev", f"{metric_stats.get('std', 0):.2f}")

    def _render_comparison_report(self, report: dict[str, Any]):
        """Render comparison report."""
        st.header("Comparison Report")

        # Version info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Version A")
            st.text(report.get("version_a", "Unknown"))
        with col2:
            st.subheader("Version B")
            st.text(report.get("version_b", "Unknown"))

        # Comparison metrics
        st.subheader("Metric Comparisons")
        for metric_name, comparison in report.get("comparisons", {}).items():
            with st.expander(metric_name):
                delta = comparison.get("delta", 0)
                delta_pct = comparison.get("delta_pct", 0)
                is_significant = comparison.get("is_significant", False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Version A", f"{comparison.get('a_mean', 0):.2f}")
                with col2:
                    st.metric("Version B", f"{comparison.get('b_mean', 0):.2f}")
                with col3:
                    st.metric(
                        "Delta",
                        f"{delta:.2f} ({delta_pct:+.1f}%)",
                        delta=delta if is_significant else None,
                    )

    def _render_regression_report(self, report: dict[str, Any]):
        """Render regression report."""
        st.header("Regression Report")

        # Regression summary
        regressions = report.get("regressions", [])

        if not regressions:
            st.success("No regressions detected!")
            return

        # Severity counts
        col1, col2, col3 = st.columns(3)
        severity_counts = {}
        for reg in regressions:
            severity = reg.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        with col1:
            st.metric("Critical", severity_counts.get("critical", 0))
        with col2:
            st.metric("Major", severity_counts.get("major", 0))
        with col3:
            st.metric("Minor", severity_counts.get("minor", 0))

        # Regression details
        st.subheader("Regression Details")
        for reg in regressions:
            severity = reg.get("severity", "unknown")
            severity_color = {"critical": "red", "major": "orange", "minor": "yellow"}.get(
                severity, "gray"
            )

            with st.expander(f"{reg.get('metric', 'Unknown')} - {severity.upper()}"):
                st.markdown(
                    f"""
                <div style="border-left: 4px solid {severity_color}; padding-left: 15px;">
                    <strong>Metric:</strong> {reg.get("metric", "Unknown")}<br>
                    <strong>Baseline:</strong> {reg.get("baseline", 0):.2f}<br>
                    <strong>Current:</strong> {reg.get("current", 0):.2f}<br>
                    <strong>Delta:</strong> {reg.get("delta", 0):.2f} ({reg.get("delta_pct", 0):+.1f}%)
                </div>
                """,
                    unsafe_allow_html=True,
                )
