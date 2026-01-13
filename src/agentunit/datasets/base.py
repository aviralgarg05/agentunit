"""Dataset abstraction for AgentUnit scenarios."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agentunit.core.exceptions import AgentUnitError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


@dataclass(slots=True)
class DatasetCase:
    """Represents a single test case for an agent scenario."""

    id: str
    query: str
    expected_output: str | None = None
    tools: list[str] | None = None
    context: list[str] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class DatasetSource:
    """Container that produces dataset cases on demand."""

    def __init__(self, name: str, loader: Callable[[], Iterable[DatasetCase]]) -> None:
        self._name = name
        self._loader = loader

    @property
    def name(self) -> str:
        return self._name

    def iter_cases(self) -> Iterator[DatasetCase]:
        yield from self._loader()

    @classmethod
    def single(cls, case: DatasetCase) -> DatasetSource:
        return cls(name=f"single:{case.id}", loader=lambda: [case])

    @classmethod
    def from_list(cls, cases: list[DatasetCase], name: str = "from_list") -> DatasetSource:
        return cls(name=name, loader=lambda: cases)

    @classmethod
    def empty(cls, name: str = "empty") -> DatasetSource:
        return cls(name=name, loader=list)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"DatasetSource(name={self._name!r})"


def load_local_json(path: str | Path) -> DatasetSource:
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Dataset file not found: {file_path}"
        raise AgentUnitError(msg)

    def _loader() -> Iterable[DatasetCase]:
        content = json.loads(file_path.read_text())
        items = content.get("items", []) if isinstance(content, dict) else content
        for idx, row in enumerate(items):
            yield DatasetCase(
                id=row.get("id") or f"case-{idx}",
                query=row["query"],
                expected_output=row.get("expected_output"),
                tools=row.get("tools"),
                context=row.get("context"),
                metadata=row.get("metadata", {}),
            )

    return DatasetSource(name=str(file_path.stem), loader=_loader)


def _parse_list_field(value: str | None, delimiter: str) -> list[str] | None:
    """Safely parse a delimited list field from CSV."""
    if not value or not isinstance(value, str):
        return None

    if not delimiter:
        # Empty delimiter is invalid for str.split
        return [value.strip()] if value.strip() else None

    parts = [item.strip() for item in value.split(delimiter)]
    cleaned = [p for p in parts if p]
    return cleaned or None



def load_local_csv(
    path: str | Path,
    tools_delimiter: str = ";",
    context_delimiter: str = "||",
) -> DatasetSource:
    file_path = Path(path)
    if not file_path.exists():
        msg = f"Dataset file not found: {file_path}"
        raise AgentUnitError(msg)

    def _loader() -> Iterable[DatasetCase]:
        with file_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for idx, row in enumerate(reader):
                try:
                    yield DatasetCase(
                        id=row.get("id") or f"case-{idx}",
                        query=row["query"],
                        expected_output=row.get("expected_output"),
                        tools=_parse_list_field(row.get("tools"), tools_delimiter),
                        context=_parse_list_field(row.get("context"), context_delimiter),
                        metadata={
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "id",
                                "query",
                                "expected_output",
                                "tools",
                                "context",
                            }
                        },
                    )
                except Exception as exc:
                    raise AgentUnitError(f"Malformed CSV row at index {idx}: {row}") from exc

    return DatasetSource(name=str(file_path.stem), loader=_loader)
