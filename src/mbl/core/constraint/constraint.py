from ..system.system import BatchedStateSpaceVector
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    """A projection onto a feasible control set (e.g. a box/norm bound),
    applied by IterativeRefinement/NeuralPolicy after each control update."""

    def __call__(self, u: "BatchedStateSpaceVector") -> "BatchedStateSpaceVector":
        """Return the closest feasible point to `u` in this constraint's set."""
        ...

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: describe this constraint's type/parameters.

        Required (not a soft/duck-typed fallback): every `Constraint` passed
        to an `OptimalControlProblem` must implement this so
        `OptimalControlProblem.get_signature()` can aggregate it directly,
        with no silent partial signatures.
        """
        ...
