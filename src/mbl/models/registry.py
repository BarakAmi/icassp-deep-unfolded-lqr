"""Generic name -> constructor registry for controller selection (Registry/Strategy pattern)."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ControllerRegistry(Generic[T]):
    """Maps a model-type name to a constructor, so callers can select a controller by name
    instead of importing and hardcoding concrete classes."""

    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., T]] = {}

    def register(self, name: str, constructor: Callable[..., T]) -> None:
        """Register `constructor` under `name`.

        Args:
            name: The unique model-type name to register under.
            constructor: A callable (typically a class) invoked as
                ``constructor(**kwargs)`` by `create`.

        Raises:
            ValueError: If `name` is already registered.
        """
        if name in self._constructors:
            raise ValueError(f"Controller type '{name}' is already registered.")
        self._constructors[name] = constructor

    def get(self, name: str) -> Callable[..., T]:
        """Look up the constructor registered under `name`.

        Args:
            name: The model-type name to look up.

        Returns:
            The registered constructor.

        Raises:
            KeyError: If `name` is not registered; the message lists the
                currently `available` names.
        """
        if name not in self._constructors:
            raise KeyError(
                f"Unknown model type '{name}'. Available: {self.available()}"
            )
        return self._constructors[name]

    def create(self, name: str, **kwargs: Any) -> T:
        """Construct a new instance of the model type registered under `name`.

        Args:
            name: The model-type name to look up.
            **kwargs: Forwarded to the registered constructor.

        Returns:
            The constructed instance.

        Raises:
            KeyError: If `name` is not registered (via `get`).
        """
        return self.get(name)(**kwargs)

    def available(self) -> tuple[str, ...]:
        """Return every currently registered model-type name.

        Returns:
            A tuple of registered names, in registration order.
        """
        return tuple(self._constructors)


MODEL_REGISTRY: ControllerRegistry = ControllerRegistry()


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register the decorated class under `name` in the global
    MODEL_REGISTRY as a side effect of the class being defined (i.e. of its module
    being imported). This is how new controller families get added without
    touching this file (Open/Closed Principle) -- see analytic/riccati.py,
    neural/nerual.py, and unfolded/base.py for usage."""

    def decorator(cls: type[T]) -> type[T]:
        """Register `cls` under the enclosing `name` and return it unchanged.

        Args:
            cls: The controller class to register.

        Returns:
            `cls`, unchanged.

        Raises:
            ValueError: If `name` is already registered.
        """
        MODEL_REGISTRY.register(name, cls)
        return cls

    return decorator


def build_default_controller_registry() -> ControllerRegistry:
    """Import this project's built-in controller modules -- triggering their
    @register_model decorators -- and return the populated global registry.

    Returns:
        The global `MODEL_REGISTRY`, populated with every built-in controller
        (``"analytic"``, ``"truncated_riccati"``, ``"neural"``, ``"unfolded"``,
        ``"cocp"``).
    """
    from . import (
        analytic,
    )  # analytic/__init__.py imports riccati.py and truncated_riccati.py
    from .neural import nerual  # noqa: F401  registers "neural"
    from .unfolded import base  # noqa: F401  registers "unfolded"
    from .constrained import cocp  # noqa: F401  registers "cocp"

    del analytic, nerual, base, cocp  # imported for their registration side effect only
    return MODEL_REGISTRY
