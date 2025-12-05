"""Dataset registry capable of resolving built-in and external datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None

from agentunit.core.exceptions import AgentUnitError

from .base import DatasetCase, DatasetSource, load_local_csv, load_local_json
from .builtins import BUILTIN_DATASETS


if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetRequest:
    identifier: str
    limit: int | None = None


def resolve_dataset(spec: str | DatasetSource | None) -> DatasetSource:
    if spec is None:
        return DatasetSource.empty()
    if isinstance(spec, DatasetSource):
        return spec
    if spec in BUILTIN_DATASETS:
        return BUILTIN_DATASETS[spec]
    if spec.startswith("hf://"):
        return _load_from_huggingface(spec)
    if spec.endswith(".json"):
        return load_local_json(spec)
    if spec.endswith(".csv"):
        return load_local_csv(spec)
    msg = f"Unsupported dataset specifier: {spec}"
    raise AgentUnitError(msg)


def _load_from_huggingface(spec: str) -> DatasetSource:
    _, repo_id = spec.split("//", maxsplit=1)
    if not repo_id:
        msg = f"Invalid Hugging Face dataset spec: {spec}"
        raise AgentUnitError(msg)
    repo_and_file = repo_id.split(":", maxsplit=1)
    repo = repo_and_file[0]
    filename = repo_and_file[1] if len(repo_and_file) > 1 else "data.json"

    def _loader() -> Iterable[DatasetCase]:
        if not HF_HUB_AVAILABLE:
            msg = "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
            raise AgentUnitError(msg)
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:  # pragma: no cover - depends on network
            msg = f"Failed to download dataset {spec}: {exc}"
            raise AgentUnitError(msg) from exc
        path = Path(downloaded)
        source = load_local_csv(path) if filename.endswith(".csv") else load_local_json(path)
        yield from source.iter_cases()

    return DatasetSource(name=f"hf:{repo}/{filename}", loader=_loader)
