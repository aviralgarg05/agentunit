"""Registry utilities for discovering and instantiating adapters."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Type, TypeVar

from .base import BaseAdapter

TAdapter = TypeVar("TAdapter", bound=BaseAdapter)

_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(
    adapter_cls: Type[TAdapter],
    *,
    aliases: Iterable[str] | None = None,
    override: bool = False,
) -> Type[TAdapter]:
    """Register an adapter class by its canonical name and optional aliases."""

    names = {adapter_cls.name.lower()}
    if aliases:
        names.update(alias.lower() for alias in aliases)

    for name in names:
        if not override and name in _REGISTRY and _REGISTRY[name] is not adapter_cls:
            raise ValueError(
                f"Adapter '{name}' already registered as {_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = adapter_cls
    return adapter_cls


def resolve_adapter(name: str) -> Type[BaseAdapter]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        registered = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Adapter '{name}' not found. Registered adapters: {registered}"
        ) from exc


def create_adapter(name: str, *args: object, **kwargs: object) -> BaseAdapter:
    return resolve_adapter(name)(*args, **kwargs)


def registered_adapters() -> List[Tuple[str, Type[BaseAdapter]]]:
    return sorted(_REGISTRY.items())


def clear_registry() -> None:
    _REGISTRY.clear()