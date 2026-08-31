"""The convex policy, solved as the box-QP it is rather than as a cone program.

Same controller as `constrained.cocp.COCPController` in what it *computes* -- a
one-step lookahead over a learnable quadratic cost-to-go, under a box on the
control -- and a different object in how it gets there. Eliminating the next
state leaves a strictly convex box-QP in the control alone whose Hessian is
shared by the whole batch and horizon, which `constrained.box_qp` solves exactly
and differentiates analytically.

**This family is additive and reaches nothing.** `COCPController` and its recipes
are untouched; the 71 stored models they produced keep their identifiers. The two
are separate registered families precisely so that neither can move the other.

**One device authority, not two.** `COCPController` carries a `COCPSolverSpec`
whose device is declared *independently* of the run's `ComputeContext`, so a CUDA
study hops to the CPU for every QP solve. Nothing here is outside torch, so this
family follows `ctx` like every other contender and the second authority is gone.

**The linear term is declared, not deleted.** ``q`` can only add a
state-independent bias to the control, and the optimal bias is zero for a problem
symmetric about the origin -- which every study in this project is. Measured, it
moves the cost by 0.001 % at inference and 0.0002 % retrained, and by **+3.9 %**
the moment a drift breaks the symmetry. So `use_linear_term` is a declaration
signed into `ModelID`, defaulting off to match the formulation the COCP paper
publishes for box-constrained LQR, and the capability survives for asymmetric
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from ...core.constraint.box_constraint import BoxConstraint
from ...core.kernels import time_invariant_slice
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ...core.utils.signing import hash_array
from ..base import Config
from ..guards import require_linear_quadratic
from ..registry import register_model
from .box_qp import BoxQPCertificate, box_qp, certify_box_qp, solve_box_qp


@dataclass(frozen=True)
class ExactCOCPConfig(Config):
    """What this controller is, beyond the problem it controls.

    Attributes:
        P_sqrt_init: Initial cost-to-go matrix square root, shape ``(n, n)``,
            so ``P_sqrt.T @ P_sqrt`` is PSD by construction. Typically
            ``sqrtm`` of a Riccati solution.
        q_init: Optional initial linear cost-to-go term, shape ``(n,)``.
            Ignored unless `use_linear_term`; defaults to zero.
        use_linear_term: Whether the policy carries the linear cost-to-go term
            at all (see the module docstring). Part of what the model IS, and
            hashed into its identifier.
        dtype: The working precision of the learnable parameters.
    """

    P_sqrt_init: np.ndarray
    q_init: np.ndarray | None = None
    use_linear_term: bool = False
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        """Fail-fast guard: `P_sqrt_init` must be a square 2D matrix.

        Raises:
            ValueError: If `P_sqrt_init` is not 2D or not square.
        """
        if (
            self.P_sqrt_init.ndim != 2
            or self.P_sqrt_init.shape[0] != self.P_sqrt_init.shape[1]
        ):
            raise ValueError(
                f"P_sqrt_init must be a square matrix, got shape {self.P_sqrt_init.shape}."
            )


@register_model("cocp_exact")
class ExactCOCPController(nn.Module):
    """A one-step convex-QP policy differentiated through its own KKT system.

    Attributes:
        problem: The `OptimalControlProblem` this controller solves.
        constraint: The scalar-``u_max`` `BoxConstraint` the QP enforces.
        config: This controller's `ExactCOCPConfig`.
        P_sqrt: Learnable cost-to-go matrix square root, shape ``(n, n)``.
        q: Learnable linear cost-to-go term, shape ``(n,)``. Present only when
            `ExactCOCPConfig.use_linear_term`.
    """

    #: The plant, held as buffers so it follows the module across devices.
    #: Annotated because ``nn.Module.__getattr__`` is typed ``Tensor | Module``,
    #: which would otherwise leak into every expression they appear in.
    A: Tensor
    B: Tensor
    R: Tensor

    def __init__(
        self,
        problem: OptimalControlProblem,
        constraint: BoxConstraint,
        config: ExactCOCPConfig,
    ) -> None:
        """
        Args:
            problem: The problem to solve; must be linear-quadratic.
            constraint: The box on the control; only a scalar `u_max` is used.
            config: Seeds the cost-to-go and declares the formulation.
        """
        super().__init__()
        self.problem = problem
        self.constraint = constraint
        self.config = config

        system, cost = require_linear_quadratic(problem)
        n = system.dimensions.state_dim
        self.u_max = float(np.asarray(constraint.u_max))

        # torch.tensor always copies, so the parameters never alias (and mutate)
        # the specification's own arrays.
        self.P_sqrt = nn.Parameter(torch.tensor(config.P_sqrt_init, dtype=config.dtype))
        if config.use_linear_term:
            q_init = config.q_init if config.q_init is not None else np.zeros(n)
            self.q: nn.Parameter | None = nn.Parameter(
                torch.tensor(q_init, dtype=config.dtype)
            )
        else:
            self.q = None

        # The plant enters only through these three, and only as buffers: they
        # are data, not parameters, and must follow the module across devices.
        self.register_buffer("A", torch.tensor(system.A_t.array, dtype=config.dtype))
        self.register_buffer("B", torch.tensor(system.B_t.array, dtype=config.dtype))
        self.register_buffer(
            "R",
            torch.tensor(np.asarray(time_invariant_slice(cost.R)), dtype=config.dtype),
        )

    def reduced_program(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """The ``(H, c)`` of the eliminated program at a batch of states.

        ``H`` is shared across the batch and the horizon; only ``c`` moves with
        the state, which is the whole reason this family is cheap.

        Args:
            y: States, shape ``(batch, n)``.

        Returns:
            ``(H, c)`` with shapes ``(m, m)`` and ``(batch, m)``.
        """
        P = self.P_sqrt.T @ self.P_sqrt
        H = 2.0 * (self.R + self.B.T @ P @ self.B)
        c = 2.0 * (y @ (self.A.T @ P @ self.B))
        if self.q is not None:
            c = c + self.q @ self.B
        return H, c

    def get_control_policy(self) -> ControlPolicy:
        """Return the one-step-lookahead QP policy, re-solved at every call.

        Returns:
            A `ControlPolicy`: ``(t, y_t) -> u_t``.
        """

        def policy(t: int, y: Tensor) -> Tensor:
            state = y.to(dtype=self.P_sqrt.dtype)
            H, c = self.reduced_program(state)
            u = box_qp(H, c, self.u_max)
            return u.to(dtype=y.dtype)

        return cast("ControlPolicy", policy)

    def certificate_at(self, y: Tensor) -> BoxQPCertificate:
        """Grade this controller's own solve at a batch of states.

        The monitor the plan asks for: ``cond(H)`` drifts as `P_sqrt` trains, so
        optimality is re-established at every call rather than assumed from a
        measurement taken at initialisation.

        Args:
            y: States, shape ``(batch, n)``.

        Returns:
            The `BoxQPCertificate` of the solve at `y`.
        """
        with torch.no_grad():
            H, c = self.reduced_program(y.to(dtype=self.P_sqrt.dtype))
            return certify_box_qp(H, c, solve_box_qp(H, c, self.u_max).u, self.u_max)

    def as_module(self) -> nn.Module:
        """`TrainableController` seam (T2.b).

        Returns:
            ``self`` -- the parameters live on this module.
        """
        return self

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: what makes this model this model.

        Hashes the *specification-time* seeds, never the live trained values,
        and records `use_linear_term` because a policy that carries a linear
        term is a different policy.

        Returns:
            The signature dict.
        """
        return {
            "type": type(self).__name__,
            "P_sqrt_init_hash": hash_array(self.config.P_sqrt_init),
            "q_init_hash": (
                hash_array(self.config.q_init)
                if self.config.q_init is not None
                else None
            ),
            "use_linear_term": self.config.use_linear_term,
            "dtype": self.config.dtype,
            "constraint": self.constraint.get_signature(),
        }
