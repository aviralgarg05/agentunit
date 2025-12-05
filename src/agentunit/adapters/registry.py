"""Registry utilities for discovering and instantiating adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from .base import BaseAdapter


if TYPE_CHECKING:
    from collections.abc import Iterable


TAdapter = TypeVar("TAdapter", bound=BaseAdapter)

_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(
    adapter_cls: type[TAdapter],
    *,
    aliases: Iterable[str] | None = None,
    override: bool = False,
) -> type[TAdapter]:
    """Register an adapter class by its canonical name and optional aliases."""

    names = {adapter_cls.name.lower()}
    if aliases:
        names.update(alias.lower() for alias in aliases)

    for name in names:
        if not override and name in _REGISTRY and _REGISTRY[name] is not adapter_cls:
            msg = f"Adapter '{name}' already registered as {_REGISTRY[name].__name__}"
            raise ValueError(msg)
        _REGISTRY[name] = adapter_cls
    return adapter_cls


def resolve_adapter(name: str) -> type[BaseAdapter]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        registered = ", ".join(sorted(_REGISTRY)) or "<none>"
        msg = f"Adapter '{name}' not found. Registered adapters: {registered}"
        raise KeyError(msg) from exc


def create_adapter(name: str, *args: object, **kwargs: object) -> BaseAdapter:
    return resolve_adapter(name)(*args, **kwargs)


def registered_adapters() -> list[tuple[str, type[BaseAdapter]]]:
    return sorted(_REGISTRY.items())


def clear_registry() -> None:
    _REGISTRY.clear()
