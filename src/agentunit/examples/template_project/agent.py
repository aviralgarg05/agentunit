"""Simplistic agent used by the template project."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


class TemplateAgent:
    """Very small agent implementation returning canned responses."""

    def answer(self, query: str, context: Iterable[str] | None = None) -> str:
        normalized = query.lower().strip()
        if "capital of france" in normalized:
            return "Paris is the capital of France."
        if "benefits of regular exercise" in normalized:
            return "Regular exercise improves cardiovascular health and elevates mood."
        return "I'm not sure yet, but I'll find out!"
