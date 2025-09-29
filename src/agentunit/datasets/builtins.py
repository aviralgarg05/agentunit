"""Built-in dataset subsets shipping with AgentUnit."""
from __future__ import annotations

from typing import Iterable, Dict, List

from .base import DatasetCase, DatasetSource


_GAIA_L1_SHOPPING: List[Dict[str, object]] = [
    {
        "id": "gaia-shopping-001",
        "query": "Find the best price for a pack of AA rechargeable batteries with at least 2500mAh capacity.",
        "expected_output": "Provide product links and summarized pricing for AA rechargeable batteries >=2500mAh.",
        "tools": ["search", "browser"],
        "context": ["User prioritizes trusted retailers."]
    },
    {
        "id": "gaia-shopping-002",
        "query": "Compare grocery delivery options for lactose-free milk in San Francisco.",
        "expected_output": "List top delivery options, estimated prices, and delivery windows for lactose-free milk in SF.",
        "tools": ["search", "maps"],
        "context": ["Budget sensitive buyer."]
    },
]

_SWE_BENCH_LITE: List[Dict[str, object]] = [
    {
        "id": "swe-lite-001",
        "query": "Fix the bug where the API returns HTTP 500 when the username is missing.",
        "expected_output": "Code diff that handles missing username with a 400 response.",
        "tools": ["repo_browser", "unit_tests"],
        "context": ["Project uses FastAPI"],
        "metadata": {"repo": "example/webapp"}
    },
    {
        "id": "swe-lite-002",
        "query": "Add a unit test covering the `calculate_total` helper for empty carts.",
        "expected_output": "Unit test file name and snippet verifying zero total.",
        "tools": ["repo_browser"],
        "context": ["Primary language: Python"],
        "metadata": {"repo": "example/cart"}
    },
]


def _build_loader(rows: List[Dict[str, object]]) -> Iterable[DatasetCase]:
    for row in rows:
        yield DatasetCase(
            id=row["id"],
            query=row["query"],
            expected_output=row.get("expected_output"),
            tools=row.get("tools"),
            context=row.get("context"),
            metadata=row.get("metadata", {}),
        )


BUILTIN_DATASETS: Dict[str, DatasetSource] = {
    "gaia:l1:shopping": DatasetSource("gaia:l1:shopping", lambda: _build_loader(_GAIA_L1_SHOPPING)),
    "swe-bench:lite": DatasetSource("swe-bench:lite", lambda: _build_loader(_SWE_BENCH_LITE)),
}
