"""COCPSolverSpec: the frozen, signable choice of which conic-QP backend
COCP's per-step convex program is solved with, and on which device. Solvers
do not share a `solver_args` vocabulary (SCS's `acceleration_lookback` is
meaningless to MOREAU; MOREAU's `max_iter`/`ipm_settings` are meaningless to
SCS), so this spec carries the two knobs genuinely shared in concept --
convergence tolerance and iteration budget -- and translates them into each
solver's own vocabulary, with `extra_args` as the escape hatch for anything
solver-specific.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: SCS-specific: Anderson acceleration lookback window. 0 disables it -- this
#: project's pre-existing default (Anderson acceleration is occasionally
#: unstable near the box boundary), preserved verbatim under the new spec.
_DIFFCP_ACCELERATION_LOOKBACK = 0

#: The `CvxpyLayer(solver=...)` names this spec supports. "DIFFCP" (not
#: "SCS") is deliberate: it is cvxpylayers' own name for this project's
#: long-standing default path, and `solver=None` normalizes to exactly this
#: string internally -- SCS is diffcp's own choice of cone solver for this
#: problem's cone types (zero/nonneg/SOC, no PSD/exponential), verified
#: empirically (its solver banner and eps_abs/max_iters both confirm SCS).
SUPPORTED_SOLVERS = frozenset({"DIFFCP", "MOREAU"})


@dataclass(frozen=True)
class COCPSolverSpec:
    """Which conic solver backs COCP's one-step QP, and how.

    Attributes:
        name: ``"DIFFCP"`` (this project's long-standing default) or
            ``"MOREAU"``.
        device: The torch device string the solve executes on (``"cpu"``,
            ``"cuda"``, ``"cuda:0"``, ...) -- MOREAU picks its CPU/CUDA
            backend from where the input tensors already live, so this must
            match the controller's own `ComputeContext.device`.
        eps: Convergence tolerance, translated into each solver's own
            tolerance field(s) by `to_solver_args`.
        max_iters: Iteration budget, translated into each solver's own field
            name by `to_solver_args`.
        extra_args: Solver-specific keyword arguments with no shared
            equivalent (e.g. DIFFCP's `acceleration_lookback` override,
            MOREAU's `verbose`/`auto_tune`), merged in verbatim.
    """

    name: str = "DIFFCP"
    device: str = "cpu"
    eps: float = 1e-8
    max_iters: int = 10000
    extra_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail-fast guards on the solver name and the two numeric knobs.

        Raises:
            ValueError: If `name` is not in `SUPPORTED_SOLVERS`, or `eps`/
                `max_iters` is not strictly positive.
        """
        if self.name not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unknown COCP solver {self.name!r}; supported: "
                f"{sorted(SUPPORTED_SOLVERS)}."
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {self.max_iters}.")

    def to_solver_args(self) -> dict[str, Any]:
        """Translate this spec into `CvxpyLayer`'s `solver_args` vocabulary
        for `self.name` -- the fix for the non-portability defect (SCS's
        `acceleration_lookback` silently sent to MOREAU; MOREAU's
        `max_iter`/`ipm_settings` never reaching SCS).

        Returns:
            A `solver_args` dict ready to forward to `CvxpyLayer.__call__`.
        """
        if self.name == "MOREAU":
            return {
                "max_iter": self.max_iters,
                "ipm_settings": {
                    "tol_gap_abs": self.eps,
                    "tol_gap_rel": self.eps,
                    "tol_feas": self.eps,
                },
                **self.extra_args,
            }
        return {
            "acceleration_lookback": _DIFFCP_ACCELERATION_LOOKBACK,
            "eps": self.eps,
            "max_iters": self.max_iters,
            **self.extra_args,
        }

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member (the device-in-signature law: a CUDA-resolved
        run and a CPU-resolved run are distinct cache entries, never
        conflated as interchangeable).

        Returns:
            ``{"type", "name", "device", "eps", "max_iters", "extra_args"}``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "device": self.device,
            "eps": self.eps,
            "max_iters": self.max_iters,
            "extra_args": dict(self.extra_args),
        }
