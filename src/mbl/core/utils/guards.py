"""Uniform, tiered guard infrastructure.

Every runtime validation in the project routes through this one module, so
expensive checks can be globally or *selectively* bypassed for performance
(e.g. large Monte-Carlo sweeps) via a single, well-defined mechanism instead of
scattered ad-hoc ``if``/``assert`` walls.

Two decorators, two very deliberately different resolution times:

* :func:`guard` wraps a *pure validator* (``(...) -> None``, raises on failure)
  that is invoked at **object-construction time** -- not inside a hot loop.  It
  keeps a cheap runtime tier check so tiers can be toggled at runtime (tests,
  :func:`guards_disabled`).  Under ``python -O`` the validator is stripped
  entirely at decoration time.

* :func:`enforce_tensor_shapes` (see ``shapes.py``) wraps a *hot / boundary
  business function* and therefore resolves its on/off decision **once, at
  decoration/import time**: when its tier is inactive it returns the original,
  unwrapped function so there is provably zero per-call overhead and nothing for
  ``torch.compile`` / ``torch.jit.script`` to trip over.

Bypass surfaces, in precedence order:

1. ``python -O`` -> ``__debug__ is False`` -> guards compile out.
2. env ``SC_DISABLE_GUARDS=1`` / ``SC_GUARD_LEVEL=SHAPE`` (read once at import).
3. programmatic :func:`disable_guards` / :func:`set_guard_tier`.
4. scoped :func:`guards_disabled` / :func:`guard_tier` around a hot region.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from functools import wraps
from typing import Any, TypeVar

import numpy as np

Validator = TypeVar("Validator", bound=Callable[..., None])
AnyFunc = TypeVar("AnyFunc", bound=Callable[..., Any])


class GuardTier(IntEnum):
    """Cost tiers, so "bypass for performance" can be selective rather than
    all-or-nothing. A guard runs iff its tier ``<=`` the active ``max_tier``."""

    SHAPE = 10  # cheap: rank / axis-size / dtype checks
    DOMAIN = 20  # moderate: positivity, bounds, dimensional coherence
    EXPENSIVE = 30  # costly: eigenvalue PSD/PD, controllability / PBH tests


_TIER_BY_NAME = {tier.name: tier for tier in GuardTier}
_TRUTHY = {"1", "true", "yes", "on"}


def _read_env_tier(default: GuardTier) -> GuardTier:
    raw = os.environ.get("SC_GUARD_LEVEL")
    if raw is None:
        return default
    key = raw.strip().upper()
    if key in _TIER_BY_NAME:
        return _TIER_BY_NAME[key]
    try:
        return GuardTier(int(raw))
    except (ValueError, TypeError):
        return default


class _GuardState:
    """Process-global guard configuration, seeded once from the environment."""

    __slots__ = ("enabled", "max_tier")

    def __init__(self) -> None:
        self.enabled = (
            os.environ.get("SC_DISABLE_GUARDS", "").strip().lower() not in _TRUTHY
        )
        self.max_tier = _read_env_tier(GuardTier.EXPENSIVE)


GUARDS = _GuardState()


def is_tier_active(tier: GuardTier) -> bool:
    """Check whether guards at `tier` should currently run.

    Args:
        tier: The tier to check.

    Returns:
        ``True`` iff running under normal (non ``-O``) Python, guards are
        globally enabled, and `tier` is at or below the active `GUARDS.max_tier`.
    """
    return __debug__ and GUARDS.enabled and tier <= GUARDS.max_tier


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def guard(tier: GuardTier = GuardTier.DOMAIN) -> Callable[[Validator], Validator]:
    """Decorate a construction-time validator so it runs only when `tier` is
    active.

    The tier check lives inside the returned wrapper, but this is invoked once
    per *object construction* -- never inside a rollout/inner loop -- so it does
    not contradict the zero-overhead-in-hot-loops rule that governs
    :func:`enforce_tensor_shapes`. Under ``python -O`` the validator body is
    dropped at decoration time.

    Args:
        tier: The cost tier this validator belongs to; it only runs while
            `tier <= GUARDS.max_tier` (see `GuardTier`).

    Returns:
        A decorator that wraps a ``(...) -> None`` validator (any exception it
        raises propagates to the caller when the guard fires; it is silently
        skipped otherwise).
    """

    def decorate(validator: Validator) -> Validator:
        if not __debug__:
            return _noop  # type: ignore[return-value]

        @wraps(validator)
        def wrapper(*args: object, **kwargs: object) -> None:
            if GUARDS.enabled and tier <= GUARDS.max_tier:
                validator(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


# --- runtime / scoped configuration -----------------------------------------


def set_guard_tier(tier: GuardTier) -> None:
    """Set the process-global maximum active `GuardTier` (persists until changed
    again; prefer `guard_tier` for a scoped override).

    Args:
        tier: Guards at this tier and below will run; higher tiers are skipped.
    """
    GUARDS.max_tier = tier


def disable_guards() -> None:
    """Globally disable all `@guard`-decorated validators (persists until
    `enable_guards` is called; prefer `guards_disabled` for a scoped override)."""
    GUARDS.enabled = False


def enable_guards() -> None:
    """Globally re-enable `@guard`-decorated validators after `disable_guards`."""
    GUARDS.enabled = True


@contextmanager
def guards_disabled() -> Iterator[None]:
    """Scoped bypass of every ``@guard`` validator (e.g. around a hot region).

    Note: :func:`enforce_tensor_shapes` decorators resolve at import time and
    are intentionally *not* affected -- their bypass surface is ``-O`` / env.

    Yields:
        Control, with guards disabled; restores the previous enabled/disabled
        state on exit (even if the block raises).
    """
    previous = GUARDS.enabled
    GUARDS.enabled = False
    try:
        yield
    finally:
        GUARDS.enabled = previous


@contextmanager
def guard_tier(tier: GuardTier) -> Iterator[None]:
    """Scoped ``max_tier`` override for ``@guard`` validators.

    Args:
        tier: The maximum tier to allow for the duration of the ``with`` block.

    Yields:
        Control, with `GUARDS.max_tier` set to `tier`; restores the previous
        tier on exit (even if the block raises).
    """
    previous = GUARDS.max_tier
    GUARDS.max_tier = tier
    try:
        yield
    finally:
        GUARDS.max_tier = previous


# --- content-addressed caching for expensive static-matrix checks ------------


_MISS = object()


def array_content_key(array: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    """A hashable key uniquely identifying a numpy array's *contents*, so an
    expensive check (PSD, controllability, ...) on a static matrix computes
    exactly once even across unrelated call sites.

    Shared primitive: this is also the basis for ``core.utils.signing.hash_array``
    (which wraps the same ``(shape, dtype, bytes)`` triple into a persisted
    SHA256 digest instead of an in-memory dict key) -- ``guards.py`` is the
    lowest-level module in ``core.utils`` (nothing here imports the rest of
    the package), so it is the safe, cycle-free home for this primitive.

    Args:
        array: The array to key (converted to a contiguous byte buffer).

    Returns:
        A ``(shape, dtype_str, raw_bytes)`` tuple, hashable and unique per
        distinct array content.
    """
    contiguous = np.ascontiguousarray(array)
    return (contiguous.shape, contiguous.dtype.str, contiguous.tobytes())


def _key_part(value: object) -> object:
    """Return `value`'s cache key contribution: its content key if it is a
    ``numpy.ndarray`` (see `array_content_key`), otherwise `value` itself."""
    return array_content_key(value) if isinstance(value, np.ndarray) else value


def content_cached(maxsize: int = 256) -> Callable[[AnyFunc], AnyFunc]:
    """Memoize a check whose arguments include one or more ``np.ndarray`` on the
    arrays' *contents* (bounded, FIFO-evicted).

    Numpy arrays are unhashable, so ``functools.lru_cache`` cannot key on them
    directly; this is the equivalent for the static cost/system matrices that
    get validated repeatedly during a run (PSD checks, controllability, ...).
    A call with an unhashable non-array argument simply bypasses the cache.

    Args:
        maxsize: Maximum number of distinct call signatures cached before the
            oldest entry is evicted (FIFO).

    Returns:
        A decorator producing a memoized wrapper; the wrapper additionally
        exposes ``.cache_clear()`` to empty the cache.
    """

    def decorate(func: AnyFunc) -> AnyFunc:
        cache: OrderedDict[tuple, Any] = OrderedDict()

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            try:
                key = (
                    tuple(_key_part(arg) for arg in args),
                    tuple((name, _key_part(v)) for name, v in sorted(kwargs.items())),
                )
                hash(key)
            except TypeError:
                return func(*args, **kwargs)
            cached = cache.get(key, _MISS)
            if cached is _MISS:
                cached = func(*args, **kwargs)
                cache[key] = cached
                if len(cache) > maxsize:
                    cache.popitem(last=False)
            return cached

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate
