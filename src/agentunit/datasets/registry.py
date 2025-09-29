"""Dataset registry capable of resolving built-in and external datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from .base import DatasetSource, DatasetCase, load_local_json, load_local_csv
from .builtins import BUILTIN_DATASETS
from ..core.exceptions import AgentUnitError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetRequest:
    identifier: str
    limit: Optional[int] = None


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
    raise AgentUnitError(f"Unsupported dataset specifier: {spec}")


def _load_from_huggingface(spec: str) -> DatasetSource:
    _, repo_id = spec.split("//", maxsplit=1)
    if not repo_id:
        raise AgentUnitError(f"Invalid Hugging Face dataset spec: {spec}")
    repo_and_file = repo_id.split(":", maxsplit=1)
    repo = repo_and_file[0]
    filename = repo_and_file[1] if len(repo_and_file) > 1 else "data.json"

    def _loader() -> Iterable[DatasetCase]:
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:  # pragma: no cover - depends on network
            raise AgentUnitError(f"Failed to download dataset {spec}: {exc}") from exc
        path = Path(downloaded)
        if filename.endswith(".csv"):
            source = load_local_csv(path)
        else:
            source = load_local_json(path)
        yield from source.iter_cases()

    return DatasetSource(name=f"hf:{repo}/{filename}", loader=_loader)
