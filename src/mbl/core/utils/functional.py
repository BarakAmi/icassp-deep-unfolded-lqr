"""Small, dependency-free functional helpers shared across layers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

T = TypeVar("T")


def prefix_keys(prefix: str, mapping: Mapping[str, T]) -> dict[str, T]:
    """Return a new dict with ``prefix`` prepended to every key.

    Args:
        prefix: The string to prepend to each key.
        mapping: The source mapping; values are copied by reference.

    Returns:
        A new ``dict`` with the same values as `mapping`, keyed by
        ``f"{prefix}{key}"``.
    """
    return {f"{prefix}{key}": value for key, value in mapping.items()}


def call_if_present(obj: object, method_name: str, *args: Any, **kwargs: Any) -> bool:
    """Call ``obj.method_name(*args, **kwargs)`` iff it exists and is callable.

    Lets callers duck-type optional hooks (e.g. ``.train()``/``.eval()`` on a
    possibly-non-``nn.Module`` model) without hasattr/callable boilerplate at
    each site.

    Args:
        obj: The object to probe for `method_name`.
        method_name: The attribute name to look up on `obj`.
        *args: Positional arguments forwarded to the method, if called.
        **kwargs: Keyword arguments forwarded to the method, if called.

    Returns:
        ``True`` if `obj.method_name` existed and was called; ``False`` if the
        attribute is absent or not callable (in which case nothing happens).
    """
    method = getattr(obj, method_name, None)
    if callable(method):
        method(*args, **kwargs)
        return True
    return False
