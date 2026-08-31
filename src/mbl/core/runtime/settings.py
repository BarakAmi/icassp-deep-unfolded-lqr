"""The single environment-variable ingestion surface (T1.c).

Every ``SC_*`` runtime knob is read through this module, so truthiness
parsing and naming conventions are defined exactly once. This module is
deliberately dependency-free (stdlib only): anything in the project may
import it without pulling in numpy/torch or creating import cycles.

Consumers own their *domain* mapping (e.g. ``core.utils.guards`` maps
``SC_GUARD_LEVEL`` onto a ``GuardTier``); this module owns only the raw
fetch-and-normalize step.

Note: ``core.utils.guards`` and ``core.profiling`` still carry their own
pre-S1 env parsing; their ingestion migrates here when those modules are
next touched (N5 closes fully then). New runtime knobs (``SC_VALIDATION_MODE``)
start here from birth.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_str(name: str) -> str | None:
    """Fetch an environment variable as a stripped string.

    Args:
        name: The environment variable name (e.g. ``"SC_VALIDATION_MODE"``).

    Returns:
        The stripped value, or ``None`` when the variable is unset or blank
        (a blank value is treated as "not configured").
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def env_flag(name: str, *, default: bool = False) -> bool:
    """Fetch an environment variable as a boolean flag.

    Args:
        name: The environment variable name (e.g. ``"SC_PROFILE"``).
        default: The value to return when the variable is unset or blank.

    Returns:
        ``True`` iff the variable is set to a truthy token (``1``, ``true``,
        ``yes``, ``on``; case-insensitive); `default` when unset/blank;
        ``False`` for any other value.
    """
    value = env_str(name)
    if value is None:
        return default
    return value.lower() in _TRUTHY
