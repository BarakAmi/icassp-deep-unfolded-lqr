"""COCP (Convex Optimization Control Policy): a one-step-lookahead controller
that solves a small convex QP at every time step -- minimize the immediate
control cost plus a quadratic cost-to-go under the box constraint on u --
via a differentiable `CvxpyLayer`, ported from the legacy
`src/lqr/cocp_constraint_lqr.py` script.

The method is that of Agrawal, Barratt, Boyd and Stellato, "Learning convex
optimization control policies" (2019), whose reference implementation is at
https://github.com/cvxgrp/cocp under the Apache License 2.0. This module is an
independent implementation of that method against this project's problem and
training abstractions -- the legacy script it was ported from was written from
the paper rather than copied from that repository (confirmed by the author,
2026-08-30), so the citation here is scholarly attribution for the *idea* and
no licence notice is owed. `lower_bound.py` is the file that carries a derived
work, and it carries the Apache-2.0 notice its origin requires.

The cost-to-go is parameterized by a learnable matrix square root `P_sqrt`
(so `P_sqrt.T @ P_sqrt` is always PSD) and a learnable linear term `q`,
directly analogous to the unfolded models' learned Riccati-replacement
matrix -- except here the "unfolding" is a single differentiable convex
solve per time step instead of an unrolled gradient loop.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn

from ...core.constraint.box_constraint import BoxConstraint
from ...core.kernels import time_invariant_slice
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ..guards import require_linear_quadratic
from ...core.utils.signing import hash_array
from ..base import Config
from ..registry import register_model
from .solver_spec import COCPSolverSpec


@dataclass(frozen=True)
class COCPConfig(Config):
    """`P_sqrt_init`/`q_init` seed the learnable cost-to-go -- typically the
    matrix square root of a Riccati solution (`scipy.linalg.sqrtm`), so
    training starts from a sensible unconstrained-optimal policy rather than
    from scratch.

    Attributes:
        P_sqrt_init: Initial cost-to-go matrix square root, shape ``(n, n)``
            (so ``P_sqrt_init.T @ P_sqrt_init`` is the initial PSD cost-to-go).
        q_init: Optional initial linear cost-to-go term, shape ``(n,)``;
            defaults to zero.
        dtype: The torch dtype for `P_sqrt`/`q` and the QP layer's inputs.
        solver: Which conic backend solves the QP, on which device, and with
            what tolerance/iteration budget (`COCPSolverSpec`) -- `P_sqrt`/
            `q` are placed on `solver.device`, so this is also this
            controller's one device authority.
    """

    P_sqrt_init: np.ndarray
    q_init: np.ndarray | None = None
    dtype: torch.dtype = torch.float64
    solver: COCPSolverSpec = field(default_factory=COCPSolverSpec)

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


@register_model("cocp")
class COCPController(nn.Module):
    """A one-step convex-QP policy with a learnable quadratic cost-to-go,
    differentiable end-to-end through the QP's KKT conditions via
    `CvxpyLayer`.

    Assumes a scalar `constraint.u_max` (a single infinity-norm bound shared
    by every control dimension), matching `cp.norm(u, "inf") <= u_max`.

    Attributes:
        problem: The `OptimalControlProblem` this controller solves.
        constraint: The scalar-``u_max`` `BoxConstraint` enforced by the QP.
        config: This controller's `COCPConfig`.
        P_sqrt: Learnable cost-to-go matrix square root, shape ``(n, n)``.
        q: Learnable linear cost-to-go term, shape ``(n,)``.
    """

    def __init__(
        self,
        problem: OptimalControlProblem,
        constraint: BoxConstraint,
        config: COCPConfig,
    ) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` to solve; `problem.system`
                must expose `.dimensions`, `.A_t.array`, `.B_t.array`, and
                `problem.cost.R`.
            constraint: The box constraint on the control input; only
                `constraint.u_max` is used (must be scalar; see class docstring).
            config: Seeds `P_sqrt`/`q` and configures the QP solver.
        """
        super().__init__()
        self.problem = problem
        self.constraint = constraint
        self.config = config

        n = problem.system.dimensions.state_dim
        q_init = config.q_init if config.q_init is not None else np.zeros(n)
        device = torch.device(config.solver.device)

        # torch.tensor(...) (unlike torch.as_tensor) always copies, so the
        # resulting nn.Parameter never silently aliases (and mutates)
        # config.P_sqrt_init/q_init's underlying numpy storage.
        self.P_sqrt = nn.Parameter(
            torch.tensor(config.P_sqrt_init, dtype=config.dtype, device=device)
        )
        self.q = nn.Parameter(torch.tensor(q_init, dtype=config.dtype, device=device))

        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        B = system.B_t.array
        R_2d = cast(np.ndarray, time_invariant_slice(cost.R))
        self._layer = self.build_qp_layer(
            A, B, R_2d, float(np.asarray(constraint.u_max)), config.solver
        )

    @staticmethod
    def build_qp_layer(
        A: np.ndarray,
        B: np.ndarray,
        R: np.ndarray,
        u_max: float,
        solver: COCPSolverSpec | None = None,
    ) -> CvxpyLayer:
        """Public (not just internal) so callers that only have saved
        (A, B, R, u_max, P_sqrt, q) values -- e.g. the dashboard, re-solving
        a persisted COCP model's control for one illustrative state, without
        reconstructing a full COCPController -- can build the same QP layer
        directly.

        Args:
            A: State transition matrix, shape ``(n, n)``.
            B: Control input matrix, shape ``(n, m)``.
            R: Control cost matrix, shape ``(m, m)``.
            u_max: The scalar infinity-norm bound on the control input.
            solver: Which conic backend to build the layer for; defaults to
                `COCPSolverSpec()` (this project's long-standing DIFFCP/SCS
                choice), exactly reproducing this method's pre-existing
                behavior when omitted.

        Returns:
            A `CvxpyLayer` differentiable w.r.t. its ``[P_sqrt, q]`` parameters,
            solving ``min_u  u^T R u + ||P_sqrt @ x_next||^2 + q @ x_next``
            subject to ``x_next = A x + B u`` and ``||u||_inf <= u_max``, with
            parameters ``[x, P_sqrt, q]`` and output variable ``u``.
        """
        n, m = A.shape[0], B.shape[1]
        resolved = solver if solver is not None else COCPSolverSpec()

        x = cp.Parameter((n, 1))
        P_sqrt = cp.Parameter((n, n))
        q = cp.Parameter(n)

        u = cp.Variable((m, 1))
        x_next = cp.Variable((n, 1))

        objective = cp.quad_form(u, R) + cp.sum_squares(P_sqrt @ x_next) + q @ x_next
        constraints = [x_next == A @ x + B @ u, cp.norm(u, "inf") <= u_max]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        return CvxpyLayer(
            problem, parameters=[x, P_sqrt, q], variables=[u], solver=resolved.name
        )

    def get_control_policy(self) -> ControlPolicy:
        """Return the one-step-lookahead QP policy, re-solved at every call.

        Returns:
            A `ControlPolicy`: ``(t, y_t) -> u_t``, solving the QP layer
            (see `build_qp_layer`) with the current `P_sqrt`/`q` at state `y_t`.
        """

        def policy(t: int, y: torch.Tensor) -> torch.Tensor:
            batch_size = y.shape[0]
            # `self.P_sqrt`/`self.q` live on `config.solver.device` -- the
            # RESOLVED SOLVER's own device (MOREAU-cuda is excluded from
            # auto-resolution, so this is CPU in practice), which need not
            # match `y`'s device: the caller's `ComputeContext` (e.g. the
            # engine rollout, running on CUDA) is a SEPARATE device
            # authority. Casting `x` here -- and casting `u` back to `y`'s
            # own device/dtype before returning -- makes this policy behave
            # identically regardless of which device the REST of the
            # experiment runs on: same batch, same samples, same horizon,
            # only a temporary device hop around the QP solve itself. Never
            # skip or shrink anything to route around this: the batch this
            # policy sees is exactly the shared, common-noise batch every
            # other contender sees (Stochastic Fairness doctrine) --
            # unaffected by which device the solve happens to run on.
            x = y.unsqueeze(-1).to(device=self.P_sqrt.device, dtype=self.P_sqrt.dtype)
            P_sqrt = self.P_sqrt.unsqueeze(0).expand(batch_size, -1, -1)
            q = self.q.unsqueeze(0).expand(batch_size, -1)
            (u,) = self._layer(
                x, P_sqrt, q, solver_args=self.config.solver.to_solver_args()
            )
            return cast(torch.Tensor, u.squeeze(-1).to(device=y.device, dtype=y.dtype))

        return cast("ControlPolicy", policy)

    def as_module(self) -> nn.Module:
        """`TrainableController` seam (T2.b): the trainable ``nn.Module``
        through which the engine sees this family's parameters.

        `P_sqrt`/`q` are registered ``nn.Parameter`` attributes of this
        module itself, so -- like `NeuralPolicy` -- the controller is its own
        parameter module.

        Returns:
            ``self``.
        """
        return self

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: type + hashed init cost-to-go seeds + dtype +
        solver spec + this controller's constraint + the controlled problem's
        signature.

        Hashes `config.P_sqrt_init`/`config.q_init` (the specification-time
        seed values), never the live, possibly-trained `self.P_sqrt`/`self.q`
        parameters (those already have a home as `ParameterSnapshotCallback`/
        `ModelCheckpointCallback` artifacts).

        The solved `problem`'s signature is intentionally omitted --
        `ProblemSignatureCallback` already logs it once at the root
        (``problem.*``); embedding it again here would just duplicate every
        key under ``controller.problem.*``.

        Returns:
            ``{"type": "COCPController", "P_sqrt_init_hash":, "q_init_hash":
            or None, "dtype":, "solver":, "constraint": ...}``.
        """
        return {
            "type": type(self).__name__,
            "P_sqrt_init_hash": hash_array(self.config.P_sqrt_init),
            "q_init_hash": (
                hash_array(self.config.q_init)
                if self.config.q_init is not None
                else None
            ),
            "dtype": self.config.dtype,
            "solver": self.config.solver.get_signature(),
            "constraint": self.constraint.get_signature(),
        }
