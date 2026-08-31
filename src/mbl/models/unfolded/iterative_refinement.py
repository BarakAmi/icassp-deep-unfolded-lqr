"""Iterative (deep-unfolded gradient-descent) control refinement strategies:
map an initial control proposal to a refined one via one or more learnable
gradient-descent steps against an LQR-shaped quadratic objective.

`IterativeRefinement`/`GradientDescentRefinement` (the shared, algorithm-
agnostic mechanics) are promoted to `models.iterative.refinement` (Phase
2.5) -- imported here, not redefined, so the loop body exists exactly once
and is shared with `models.analytic.iterative_gd`'s whole-horizon solver."""

from abc import abstractmethod

from typing import Any, cast

import torch

from ..iterative.refinement import GradientDescentRefinement
from ..lqr_gradient import compute_lqr_gradient_matrices


class LQRGDRefinement(GradientDescentRefinement):
    """Specializes `GradientDescentRefinement` to the LQR-shaped quadratic
    objective ``grad = M @ u + C^T @ y``, leaving `pre_iteration_hook` (how
    ``M``/``C`` are obtained) to subclasses."""

    @abstractmethod
    def pre_iteration_hook(
        self, t: int, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this time step's ``(M, y_Ct)`` pair for `get_gradient`.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.

        Returns:
            ``(M, y_Ct)``: `M` of shape ``(control_dim, control_dim)``, `y_Ct`
            of shape ``(batch, control_dim)`` (``y @ C.T`` precomputed).

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    def get_gradient(
        self, u: torch.Tensor, M: torch.Tensor, y_Ct: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient of the quadratic cost function w.r.t. u, which is given by
        grad = M @ u + C^T @ y, where M and C are precomputed matrices for the current time step t.

        Args:
            u: Control input (batch_size, control_dim)
            M: Precomputed matrix (control_dim, control_dim)
            y_Ct: Precomputed vector (batch_size, control_dim)
        Returns:
            grad: Gradient of the cost w.r.t. u (batch_size, control_dim)
        """
        return u @ M + y_Ct


class StepSizeRefinement(LQRGDRefinement):
    """LQR gradient refinement with **static, precomputed** ``M``/``C``
    (only the per-iteration step size is learnable). Expects
    `static_parameters` to hold ``"M_stack"``/``"C_stack"``, each pre-stacked
    over the full horizon at construction time."""

    def pre_iteration_hook(
        self, t: int, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice this time step's precomputed ``M``/``C`` from the static stacks.

        Args:
            t: The current time step (indexes ``M_stack``/``C_stack``).
            y: Current observation, shape ``(batch, observation_dim)``.

        Returns:
            ``(M, y_Ct)`` (see `LQRGDRefinement.pre_iteration_hook`).
        """
        C = self.static_parameters["C_stack"][t]
        M = self.static_parameters["M_stack"][t]
        y_Ct = y @ C.T
        return M, y_Ct


class RiccatiRefinement(LQRGDRefinement):
    """Unlike StepSizeRefinement (whose M/C are static, precomputed once at
    construction), RiccatiRefinement's gradient matrices depend on the
    *learnable* Riccati matrix P, so they must be recomputed whenever P changes.
    They are, however, identical across the inner gradient-descent iterations and
    differ across time steps only by the (static) A_t/B_t/R_t slice -- so the
    whole horizon is computed in **one batched call** at the first time step of a
    rollout and cached, replacing the previous per-time-step recompute. The cache
    is cleared per rollout (`on_rollout_start`) so each forward rebuilds a fresh,
    correctly-connected autograd graph from the current P."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """See `IterativeRefinement.__init__`; `static_parameters` must hold
        ``"A"``/``"B"``/``"R"`` (each time-stacked, shape ``(T, ...)``) and
        `learnable_parameters` must hold ``"riccati_matrix"``/``"step_size"``."""
        super().__init__(*args, **kwargs)
        self._cached_gradient_stacks: tuple[torch.Tensor, torch.Tensor] | None = None

    def on_rollout_start(self) -> None:
        """Invalidate the cached horizon gradient matrices (see class
        docstring), so the next `pre_iteration_hook` call recomputes them from
        the current (possibly updated) ``riccati_matrix``."""
        self._cached_gradient_stacks = None

    def _horizon_gradient_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """M_stack = 2(R + BᵀPB), C_stack = 2(BᵀPA) for every time step at once.
        `compute_lqr_gradient_matrices` broadcasts the single (n,n) P across the
        time-stacked (T,·,·) A/B/R, so this is one batched op, not a loop. The
        factor of 2 turns the cost-to-go matrices into gradient coefficients:
        grad_u = 2(R + BᵀPB) u + 2(BᵀPA) x.

        Returns:
            ``(M_stack, C_stack)``, each shape ``(T, control_dim, ...)``,
            computed on first access per rollout and cached thereafter (see
            `on_rollout_start`).
        """
        if self._cached_gradient_stacks is None:
            riccati_matrix = self.learnable_parameters["riccati_matrix"].get()
            A = self.static_parameters["A"]
            B = self.static_parameters["B"]
            R = self.static_parameters["R"]
            M_stack, C_stack = compute_lqr_gradient_matrices(riccati_matrix, A, B, R)
            self._cached_gradient_stacks = (
                cast(torch.Tensor, 2 * M_stack),
                cast(torch.Tensor, 2 * C_stack),
            )
        return self._cached_gradient_stacks

    def pre_iteration_hook(
        self, t: int, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice this time step's ``M``/``C`` out of the (cached) batched
        horizon computation.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.

        Returns:
            ``(M, y_Ct)`` (see `LQRGDRefinement.pre_iteration_hook`).
        """
        M_stack, C_stack = self._horizon_gradient_matrices()
        y_Ct = y @ C_stack[t].T
        return M_stack[t], y_Ct


class IterationVaryingRiccatiRefinement(RiccatiRefinement):
    """Shared mechanics for a cost-to-go P that varies across the J
    unfolding ITERATIONS (NB05 plan Sec 14: rungs R1.5/R2) -- unlike
    `RiccatiRefinement`'s single matrix, shared identically by every
    iteration. The two leaf classes below (`PerIterationRiccatiRefinement`,
    `ScalarModulatedRiccatiRefinement`) differ only in how
    `_iteration_p_stack` builds this rollout's ``(J, n, n)`` P stack; the
    constructor, the per-rollout cache (`on_rollout_start`), and the
    gradient-matrix broadcasting mechanism are all inherited unchanged from
    `RiccatiRefinement`.

    `GradientDescentRefinement.__call__` computes `pre_iteration_hook(t, y)`
    ONCE per time step and reuses its result across every inner iteration
    `k`, so an iteration-varying P cannot be expressed there; `refine_step`
    IS called once per `k` and is exactly the seam this class exploits:
    `pre_iteration_hook` returns the FULL per-iteration stack for this time
    step, and `refine_step` slices it down to iteration `k`'s matrices
    before delegating to the shared single-iteration update body.
    """

    def _iteration_p_stack(self) -> torch.Tensor:
        """Return this rollout's per-iteration P stack.

        Returns:
            Shape ``(num_iterations, state_dim, state_dim)``.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError

    def _horizon_gradient_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """M_stack = 2(R + BᵀP^(j)B), C_stack = 2(BᵀP^(j)A) for every
        unfolding iteration j AND time step t at once:
        `compute_lqr_gradient_matrices` broadcasts a ``(J, 1, n, n)`` P
        stack against the time-stacked ``(T, ·, ·)`` A/B/R into
        ``(J, T, ·, ·)`` in a single batched call -- verified: slice j of
        the batched result is bit-identical to a single-P call with P^(j).

        Returns:
            ``(M_stack, C_stack)``, each shape ``(J, T, ...)``, computed on
            first access per rollout and cached thereafter (see
            `RiccatiRefinement.on_rollout_start`).
        """
        if self._cached_gradient_stacks is None:
            P_stack = self._iteration_p_stack().unsqueeze(1)  # (J, 1, n, n)
            A = self.static_parameters["A"]
            B = self.static_parameters["B"]
            R = self.static_parameters["R"]
            M_stack, C_stack = compute_lqr_gradient_matrices(P_stack, A, B, R)
            self._cached_gradient_stacks = (
                cast(torch.Tensor, 2 * M_stack),
                cast(torch.Tensor, 2 * C_stack),
            )
        return self._cached_gradient_stacks

    def pre_iteration_hook(
        self, t: int, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice this time step's FULL per-iteration ``M``/``C`` stacks out
        of the cached batched horizon computation -- `refine_step` below
        further slices these down to one iteration's matrices.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.

        Returns:
            ``(M_t, y_Ct)``: `M_t` of shape ``(J, control_dim, control_dim)``,
            `y_Ct` of shape ``(J, batch, control_dim)``.
        """
        M_stack, C_stack = self._horizon_gradient_matrices()
        M_t = M_stack[:, t]
        C_t = C_stack[:, t]
        y_Ct = (C_t @ y.T).mT
        return M_t, y_Ct

    def refine_step(
        self, iteration_index: int, u: torch.Tensor, *args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice `pre_iteration_hook`'s per-iteration stacks down to THIS
        iteration's ``(m, m)``/``(batch, m)`` matrices, then delegate to the
        shared single-iteration update body (`GradientDescentRefinement
        .refine_step`, via the MRO -- `RiccatiRefinement` does not override
        it).

        Args:
            iteration_index: Selects this iteration's slice of the stacks
                `pre_iteration_hook` returned, and this iteration's step
                size (see the parent implementation).
            u: The current control iterate, shape ``(batch, control_dim)``.
            *args: ``(M_t_all, y_Ct_all)`` from `pre_iteration_hook`.

        Returns:
            ``(u_next, grad)`` (see `GradientDescentRefinement.refine_step`).
        """
        M_t_all, y_Ct_all = args
        return super().refine_step(
            iteration_index, u, M_t_all[iteration_index], y_Ct_all[iteration_index]
        )


class PerIterationRiccatiRefinement(IterationVaryingRiccatiRefinement):
    """R2 (NB05 plan Sec 14.1-14.10): J fully independent learned matrices
    P^(0), ..., P^(J-1)
    (`models.unfolded.parameters.PerIterationRiccatiMatrixParameter`, whose
    ``.get()`` already returns the ``(J, n, n)`` stack directly -- no
    combination step needed)."""

    def _iteration_p_stack(self) -> torch.Tensor:
        """See `IterationVaryingRiccatiRefinement._iteration_p_stack`."""
        return cast(torch.Tensor, self.learnable_parameters["riccati_matrix"].get())


class ScalarModulatedRiccatiRefinement(IterationVaryingRiccatiRefinement):
    """R1.5 (NB05 plan Sec 14.9): one shared learned matrix P rescaled per
    iteration by a learned positive scalar, P^(j) = c_j * P
    (`models.unfolded.parameters.RiccatiMatrixParameter` +
    `MatrixModulationParameter`) -- the parameter-matched control for R2,
    nearly n²+J parameters against R2's J*n²."""

    def _iteration_p_stack(self) -> torch.Tensor:
        """See `IterationVaryingRiccatiRefinement._iteration_p_stack`."""
        P = self.learnable_parameters["riccati_matrix"].get()  # (n, n)
        c = self.learnable_parameters["matrix_modulation"].get()  # (J,)
        return cast(torch.Tensor, c[:, None, None] * P[None, :, :])
