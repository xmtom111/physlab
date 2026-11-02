"""Name-to-factory registry.

A :class:`Registry` holds factories for one product type. Duplicate names are
rejected. Instances are passed into collaborators; nothing here is global.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from physlab.core.errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Factories addressed by name.

    ``names`` returns names in registration order.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        """Register ``factory`` under ``name``.

        Args:
            name: Non-empty unique name.
            factory: Callable invoked by :meth:`create`.

        Raises:
            ConfigError: ``name`` is empty or already registered.
        """
        if not name:
            raise ConfigError("Registry name must be non-empty")
        if name in self._factories:
            raise ConfigError(
                f"Duplicate registry name {name!r}",
                details={"name": name},
            )
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> T:
        """Build an object with the factory registered as ``name``.

        ``Any`` is required on ``kwargs``: factories stored in one registry do
        not share a single parameter list.

        Args:
            name: Registered factory name.
            **kwargs: Arguments forwarded to the factory.

        Returns:
            Object produced by the factory.

        Raises:
            ConfigError: ``name`` is not registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise ConfigError(
                f"Unknown registry name {name!r}",
                details={"name": name, "known": list(self._factories)},
            )
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order.

        Returns:
            Tuple of names. Empty when nothing has been registered.
        """
        return tuple(self._factories)
