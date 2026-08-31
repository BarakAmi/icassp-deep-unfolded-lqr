"""Core-level structural coherence guards.

Cross-component invariants that are *structural* (dimensional agreement, stack
lengths) rather than control-theoretic -- so they live in ``core`` and carry no
dependency on any concrete ``System``/``Cost`` class (they are duck-typed).
Control-theory preconditions (Riccati, controllability) live in
``src/models/guards.py`` instead.

All guards are ``DOMAIN``-tier and therefore skipped when domain guards are
bypassed for performance.
"""

from __future__ import annotations

import numpy as np

from .guards import GuardTier, guard


@guard(GuardTier.DOMAIN)
def ensure_cost_stack_coherence(Q: np.ndarray, R: np.ndarray) -> None:
    """A time-stacked running-state cost ``Q`` must have exactly one more slice
    than the control cost ``R`` (``Q`` covers steps ``0..N``, ``R`` steps
    ``0..N-1``). Time-invariant (2D) matrices broadcast to any horizon and are
    always coherent.

    Args:
        Q: Running-state cost, shape ``(n, n)`` or ``(N+1, n, n)``.
        R: Control cost, shape ``(m, m)`` or ``(N, m, m)``.

    Raises:
        ValueError: If both `Q` and `R` are 3D time-stacks and
            ``Q.shape[0] != R.shape[0] + 1``. Skipped (a no-op) when
            `GuardTier.DOMAIN` is bypassed for performance.
    """
    if Q.ndim == 3 and R.ndim == 3 and Q.shape[0] != R.shape[0] + 1:
        raise ValueError(
            "Time-stacked Q must have exactly one more slice than R "
            f"(got Q length {Q.shape[0]}, R length {R.shape[0]}; "
            f"expected Q length {R.shape[0] + 1})."
        )


@guard(GuardTier.DOMAIN)
def ensure_system_cost_dims_match(system: object, cost: object) -> None:
    """Assert a cost's declared state/control dimensions match the system's.

    Duck-typed: a system without a ``dimensions`` attribute, or a cost that
    doesn't declare ``state_dim``/``control_dim``, is simply skipped -- this
    guard tightens the common concrete case without constraining the abstract
    ``System``/``Cost`` interfaces.

    Args:
        system: A `core.system.system.System`-like object, optionally exposing
            a `StateSpaceModelDimensions`-like ``.dimensions`` attribute.
        cost: A `core.cost.cost.Cost`-like object, optionally exposing
            ``.state_dim``/``.control_dim`` attributes.

    Raises:
        ValueError: If `cost` declares a ``state_dim``/``control_dim`` that
            disagrees with `system.dimensions`. Skipped (a no-op) when either
            side omits the relevant attribute, or when `GuardTier.DOMAIN` is
            bypassed for performance.
    """
    dims = getattr(system, "dimensions", None)
    if dims is None:
        return
    state_dim = getattr(cost, "state_dim", None)
    control_dim = getattr(cost, "control_dim", None)
    if state_dim is not None and state_dim != dims.state_dim:
        raise ValueError(
            f"cost state dimension {state_dim} does not match system state "
            f"dimension {dims.state_dim}."
        )
    if control_dim is not None and control_dim != dims.control_dim:
        raise ValueError(
            f"cost control dimension {control_dim} does not match system control "
            f"dimension {dims.control_dim}."
        )
