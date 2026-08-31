from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from .system.system import System
from .cost.cost import Cost
from .constraint.constraint import Constraint
from .utils.coherence import ensure_system_cost_dims_match


@dataclass(frozen=True)
class OptimalControlProblem:
    """Immutable aggregate of the system, cost, and (optional) constraints
    defining one optimal control problem.

    Attributes
    ----------
     system : System
        The system dynamics of the optimal control problem.
     cost : Cost
        The cost function to be minimized.
     constraints : Sequence[Constraint] | None
        Optional projections onto a feasible control set (e.g. a box bound),
        applied by learnable controllers after each control update.

    Notes
    -----
    This class is intentionally minimal and immutable to encourage functional usage.
    """

    system: System
    cost: Cost
    constraints: Sequence[Constraint] | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards on the aggregate's types and cross-component
        dimensional coherence.

        Raises:
            TypeError: If `system` is not a `System`, `cost` is not a `Cost`,
                `constraints` is neither ``None`` nor a `Sequence`, or any
                element of `constraints` is not callable.
            ValueError: If `cost` declares state/control dimensions that
                disagree with `system.dimensions` (via
                `ensure_system_cost_dims_match`; skipped for abstract
                systems/costs that don't declare dimensions).
        """
        if not isinstance(self.system, System):
            raise TypeError(
                f"system must be a System instance, got {type(self.system).__name__}."
            )
        if not isinstance(self.cost, Cost):
            raise TypeError(
                f"cost must be a Cost instance, got {type(self.cost).__name__}."
            )
        if self.constraints is not None:
            if not isinstance(self.constraints, Sequence):
                raise TypeError(
                    f"constraints must be a sequence of Constraint instances, got {type(self.constraints).__name__}."
                )
            for i, constraint in enumerate(self.constraints):
                if not callable(constraint):
                    raise TypeError(
                        f"constraint at index {i} must be callable, got {type(constraint).__name__}."
                    )

        # Fail fast on a system/cost dimensional mismatch at assembly time,
        # rather than deep inside a rollout. Duck-typed: skipped for abstract
        # systems/costs that don't declare dimensions.
        ensure_system_cost_dims_match(self.system, self.cost)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` composite: aggregates `system`/`cost`/`constraints`.

        No duck-typed fallback for constraints -- every `Constraint` in
        `self.constraints` must implement `get_signature()` (enforced
        structurally by the `Constraint` Protocol itself).

        Returns:
            ``{"system": ..., "cost": ..., "constraints": [...]}``, a nested
            tree (see `core.utils.signing.flatten_signature` for how this is
            turned into collision-proof persisted keys).
        """
        return {
            "system": self.system.get_signature(),
            "cost": self.cost.get_signature(),
            "constraints": [c.get_signature() for c in self.constraints or []],
        }
