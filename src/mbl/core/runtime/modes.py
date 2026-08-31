"""The global dual-mode validation policy: STRICT vs. FAST (T1.5.b).

One mode, one place, driving every runtime-validation surface coherently:

* ``STRICT`` — comprehensive fail-fast: research debugging, CI,
  new-feature bring-up. Guard tiers run at ``EXPENSIVE``; jaxtyping
  runtime checking is active on decorated surfaces.
* ``FAST`` — zero-overhead: publication sweeps, benchmarks, Monte-Carlo
  campaigns. Guard tiers drop to ``SHAPE``; decorator-based validation
  surfaces obey the **Import-Time Identity Law** (see ``validators.py``):
  they evaluate to the original, undecorated function object at
  decoration time.

Resolution precedence (mirroring ``core.utils.guards``):

1. ``python -O`` (``__debug__ is False``) — forced ``FAST``.
2. env ``SC_VALIDATION_MODE=strict|fast``, parsed once at import through
   the T1.c settings surface.
3. programmatic `set_validation_mode` — call **before** importing/defining
   decorated modules; decorator surfaces burn the mode at decoration time.
4. scoped `validation_mode` context manager — governs only surfaces that
   retain runtime toggles (``@guard`` validators); already-decorated
   identity-law surfaces are intentionally unaffected.

Provenance exemption (T1.5.c): the mode must **never** enter a signature
or cache key — a STRICT run and a FAST run of the same experiment are the
same experiment. Nothing in this module is signable, by design.

Guard-tier coherence: setting the mode (env at import, programmatic, or
scoped) synchronizes ``GUARDS.max_tier`` per the mode table
(STRICT -> ``EXPENSIVE``, FAST -> ``SHAPE``). An explicitly set
``SC_GUARD_LEVEL`` wins over the env-mode seeding — the dedicated knob is
more specific than the blanket mode.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from mbl.core.utils.guards import GUARDS, GuardTier

from .settings import env_str

_ENV_VAR = "SC_VALIDATION_MODE"


class ValidationMode(StrEnum):
    """The two project-wide validation regimes."""

    STRICT = "strict"
    FAST = "fast"


_GUARD_TIER_BY_MODE = {
    ValidationMode.STRICT: GuardTier.EXPENSIVE,
    ValidationMode.FAST: GuardTier.SHAPE,
}


def _read_env_mode() -> ValidationMode | None:
    """Parse ``SC_VALIDATION_MODE`` once, at import.

    Returns:
        The configured `ValidationMode`, or ``None`` when unset.

    Raises:
        ValueError: On an unrecognized value — a typo'd mode silently
            running in the wrong regime is exactly the failure class this
            layer exists to prevent, so ingress is loud.
    """
    raw = env_str(_ENV_VAR)
    if raw is None:
        return None
    try:
        return ValidationMode(raw.lower())
    except ValueError:
        valid = ", ".join(mode.value for mode in ValidationMode)
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is not a valid validation mode "
            f"(expected one of: {valid})."
        ) from None


class _ModeState:
    """Process-global mode, seeded once from the environment (STRICT default:
    reproduces pre-S1 behavior exactly)."""

    __slots__ = ("mode",)

    def __init__(self) -> None:
        self.mode = _read_env_mode() or ValidationMode.STRICT


_STATE = _ModeState()

# Env-seeded coherence: an explicit mode in the environment drives the guard
# tier too -- unless the dedicated SC_GUARD_LEVEL knob was itself set, which
# is more specific and wins.
if env_str(_ENV_VAR) is not None and env_str("SC_GUARD_LEVEL") is None:
    GUARDS.max_tier = _GUARD_TIER_BY_MODE[_STATE.mode]


def get_validation_mode() -> ValidationMode:
    """The currently active `ValidationMode`.

    Decorator surfaces consult this at decoration time (Import-Time
    Identity Law); runtime-toggleable surfaces may consult it per call.

    Returns:
        ``FAST`` unconditionally under ``python -O``; otherwise the current
        process-global mode.
    """
    if not __debug__:
        return ValidationMode.FAST
    return _STATE.mode


def is_strict() -> bool:
    """Whether the active mode is ``STRICT`` (convenience predicate).

    Returns:
        ``get_validation_mode() is ValidationMode.STRICT``.
    """
    return get_validation_mode() is ValidationMode.STRICT


def set_validation_mode(mode: ValidationMode | str) -> None:
    """Set the process-global validation mode and synchronize the guard tier.

    Call **before** importing modules whose decorators must honor the new
    mode: identity-law surfaces resolve at decoration time, so functions
    decorated earlier keep their already-burned behavior.

    Args:
        mode: The mode to activate (`ValidationMode` or its string value).
    """
    resolved = ValidationMode(mode)
    _STATE.mode = resolved
    GUARDS.max_tier = _GUARD_TIER_BY_MODE[resolved]


@contextmanager
def validation_mode(mode: ValidationMode | str) -> Iterator[None]:
    """Scoped mode override for the runtime-toggleable surfaces.

    Governs ``@guard`` validators (their tier check is per-call) and any
    future surface that reads the mode at call time. Identity-law decorator
    surfaces already resolved at import are intentionally unaffected — see
    the module docstring's accepted-consequences note.

    Args:
        mode: The mode to activate inside the ``with`` block.

    Yields:
        Control, with the mode (and synchronized guard tier) overridden;
        both are restored on exit, even if the block raises.
    """
    previous_mode = _STATE.mode
    previous_tier = GUARDS.max_tier
    set_validation_mode(mode)
    try:
        yield
    finally:
        _STATE.mode = previous_mode
        GUARDS.max_tier = previous_tier
