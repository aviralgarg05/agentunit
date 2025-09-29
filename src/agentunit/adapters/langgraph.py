"""Adapter for LangGraph v1.x graphs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import importlib
import importlib.util
import logging

import yaml

from .base import BaseAdapter, AdapterOutcome
from ..core.exceptions import AdapterNotAvailableError, AgentUnitError
from ..core.trace import TraceLog
from ..datasets.base import DatasetCase

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import langgraph
except Exception:  # pragma: no cover
    langgraph = None


class LangGraphAdapter(BaseAdapter):
    name = "langgraph"

    def __init__(self, source: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self._source = source
        self._config = config or {}
        self._graph_callable: Optional[Callable[[Dict[str, Any]], Any]] = None

    @classmethod
    def from_source(cls, source: str | Path | Callable[..., Any] | Any, **config: Any) -> "LangGraphAdapter":
        return cls(source=source, config=config)

    def prepare(self) -> None:
        if self._graph_callable is not None:
            return
        if callable(self._source):
            self._graph_callable = self._wrap_callable(self._source)
            return
        if langgraph is None:
            raise AdapterNotAvailableError("langgraph>=1.0.0a4 is required for LangGraphAdapter")
        elif isinstance(self._source, (str, Path)):
            self._graph_callable = self._load_from_path(Path(self._source))
        else:
            self._graph_callable = self._wrap_callable(self._source)

    def execute(self, case: DatasetCase, trace: TraceLog) -> AdapterOutcome:
        if self._graph_callable is None:
            self.prepare()
        assert self._graph_callable is not None  # mypy guard

        payload = {
            "query": case.query,
            "context": case.context,
            "tools": case.tools,
            "metadata": case.metadata,
        }
        trace.record("agent_prompt", input=payload)
        try:
            response = self._graph_callable(payload)
            if isinstance(response, dict) and "events" in response:
                for event in response["events"]:
                    trace.record(event.get("type", "event"), **event)
                final = response.get("result")
            else:
                final = response
            trace.record("agent_response", content=final)
            return AdapterOutcome(success=True, output=final)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LangGraph execution failed")
            trace.record("error", message=str(exc))
            return AdapterOutcome(success=False, output=None, error=str(exc))

    # Helpers -----------------------------------------------------------------
    def _wrap_callable(self, candidate: Any) -> Callable[[Dict[str, Any]], Any]:
        if hasattr(candidate, "invoke"):
            return lambda payload: candidate.invoke(payload)
        if hasattr(candidate, "run"):
            return lambda payload: candidate.run(payload)
        if callable(candidate):
            return candidate
        raise AgentUnitError("Unsupported LangGraph source; expected callable or graph instance")

    def _load_from_path(self, path: Path) -> Callable[[Dict[str, Any]], Any]:
        if not path.exists():
            raise AgentUnitError(f"LangGraph file does not exist: {path}")
        if path.suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(path.read_text())
            return self._load_from_config(config)
        if path.suffix == ".json":
            config = yaml.safe_load(path.read_text())
            return self._load_from_config(config)
        if path.suffix == ".py":
            return self._load_callable_from_python(path)
        raise AgentUnitError(f"Unsupported LangGraph file type: {path.suffix}")

    def _load_from_config(self, config: Dict[str, Any]) -> Callable[[Dict[str, Any]], Any]:
        module_name = config.get("module")
        attr = config.get("object") or config.get("callable")
        if not module_name or not attr:
            raise AgentUnitError("LangGraph config must define 'module' and 'object' keys")
        module = importlib.import_module(module_name)
        candidate = getattr(module, attr)
        return self._wrap_callable(candidate)

    def _load_callable_from_python(self, path: Path) -> Callable[[Dict[str, Any]], Any]:
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise AgentUnitError(f"Unable to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target_name = self._config.get("callable", "graph")
        if not hasattr(module, target_name):
            raise AgentUnitError(f"Module {module_name} does not expose attribute '{target_name}'")
        candidate = getattr(module, target_name)
        return self._wrap_callable(candidate)
