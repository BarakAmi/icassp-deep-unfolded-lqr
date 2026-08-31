from typing import Any, cast

import numpy as np

from .cost import Cost

from ..utils import (
    add_batch_dim_to_multiple,
    ensure_equal_ints,
    ensure_same_batch,
    validate_bool,
    ensure_batched_horizon_vector,
)
from ..kernels.quadratic import CostConventions, cumulative_quadratic_cost
from ..utils.coherence import ensure_cost_stack_coherence
from ..utils.descriptors import TimeStackedPD, TimeStackedPSD
from ..utils.signing import hash_array


class QuadraticCost(Cost):
    """Quadratic cost function of the form:
    $$sum_{k=0}^{N-1}{x_k^T Q_k x_k + u_k^T R_k u_k} + x_N^T Q_N x_N$$
    or
    $$sum_{k=0}^{N-1}{x_k^T Q_k x_k + u_k^T R_k u_k}$$
    where $Q_k$ is the running state cost matrix at time $k$ and $R_k$ is the control cost matrix at time $k$; the
    terminal term (if include_terminal_cost is True) always reuses $Q_N$, `Q`'s own slice at the true end of the
    trajectory -- there is no separate terminal cost matrix. If include_terminal_cost is False, the terminal cost
    term is omitted.

    Q and R are exposed through write-once, self-validating descriptors
    (``TimeStackedPSD``/``TimeStackedPD``): they are validated positive
    semi-definite / positive definite on assignment and cannot subsequently be
    mutated into an invalid state.

    Attributes:
        Q (np.ndarray): Running state cost matrix of shape (N+1, n, n) or (n, n).
        R (np.ndarray): Control cost matrix of shape (N, m, m) or (m, m).
        include_terminal_cost (bool): Whether to include the terminal cost term. Defaults to False.
        is_time_averaged (bool): Whether `__call__` returns the per-step running average
            (dividing the cumulative cost at step k by the elapsed step count k+1) or the raw,
            un-normalized cumulative cost. Defaults to True (Phase 1F): this preserves the
            behavior every existing caller already depends on -- `__call__` unconditionally
            divided by the elapsed step count before this flag existed.
        state_dim (int): Inferred from `Q`'s trailing dimension. Not part of
            `get_signature()` -- dimensions are the system's concern
            (`LinearSystem.get_signature()` already reports them); exposed
            here only so callers (e.g. `ensure_system_cost_dims_match`) can
            cross-check this cost against a system's dimensions.
        control_dim (int): Inferred from `R`'s trailing dimension. See `state_dim`.
    """

    Q = TimeStackedPSD()
    R = TimeStackedPD()

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        include_terminal_cost: bool = False,
        is_time_averaged: bool = True,
    ) -> None:
        """
        Args:
            Q: Running state cost, shape ``(n, n)`` (time-invariant) or
                ``(N+1, n, n)`` (time-stacked). Must be positive semi-definite
                (or every slice, if time-stacked).
            R: Control cost, shape ``(m, m)`` or ``(N, m, m)``. Must be
                positive definite (or every slice, if time-stacked).
            include_terminal_cost: Whether `__call__` adds the terminal-state
                cost term (see class docstring).
            is_time_averaged: Whether `__call__` divides the cumulative cost
                at each step k by the elapsed step count k+1 (the per-step
                running average, ``True``) or returns the raw, un-normalized
                cumulative cost (``False``). Defaults to ``True``, matching
                this class's behavior before the flag existed exactly, so no
                existing caller's numbers change unless it opts in to ``False``.

        Raises:
            TypeError: If `Q`/`R` are not ``numpy.ndarray``, or
                `include_terminal_cost`/`is_time_averaged` is not a ``bool``.
            ValueError: If `Q`/`R` are not square, not symmetric, fail
                their PSD/PD requirement (`TimeStackedPSD`/`TimeStackedPD`
                descriptors -- Phase 1A fail-fast guards), or if a
                time-stacked `Q`/`R` pair is horizon-incoherent (`Q` must
                have exactly one more slice than `R`; see
                `ensure_cost_stack_coherence`).
        """
        self.Q = Q  # descriptor validates a PSD matrix or time-stack
        self.R = R  # descriptor validates a PD matrix or time-stack
        self.include_terminal_cost = validate_bool(
            "include_terminal_cost", include_terminal_cost
        )
        self.is_time_averaged = validate_bool("is_time_averaged", is_time_averaged)

        # Fail fast at construction on an incoherent Q/R horizon, rather than
        # deep inside a Monte-Carlo rollout's cost evaluation.
        ensure_cost_stack_coherence(self.Q, self.R)

        # Set eagerly (not deferred to _validate_trajectories/__call__): callers
        # like ensure_system_cost_dims_match run at OptimalControlProblem
        # construction time, before this cost's __call__ has ever executed --
        # a lazily-set attribute would still be absent then, silently
        # disabling that fail-fast cross-check.
        self.state_dim = self.Q.shape[-1]
        self.control_dim = self.R.shape[-1]

    @property
    def conventions(self) -> CostConventions:
        """This cost's declared terminal/averaging pair, as the one bundle
        every kernel entry point (`total_quadratic_cost`,
        `cumulative_quadratic_cost`) accepts — ``conventions=cost.conventions``
        is the canonical way to evaluate trajectories under exactly the
        objective this cost declares.

        Returns:
            The `CostConventions` for this cost.
        """
        return CostConventions(
            include_terminal_cost=self.include_terminal_cost,
            is_time_averaged=self.is_time_averaged,
        )

    def _validate_trajectories(
        self, X: np.ndarray, U: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and fail-fast validate `X`/`U` before cost evaluation.

        Args:
            X: State trajectory, shape ``(batch, N+1, n)`` or ``(N+1, n)``.
            U: Control trajectory, shape ``(batch, N, m)`` or ``(N, m)``.

        Returns:
            ``(X, U)``, each guaranteed batched (3D).

        Raises:
            TypeError: If `X`/`U` are not ``numpy.ndarray``.
            ValueError: If `X`/`U` disagree on batch size, either has the
                wrong feature dimension, or ``X``'s horizon is not exactly one
                more than ``U``'s.
        """
        # Add batch dimension if missing
        X, U = add_batch_dim_to_multiple(X, U)
        # Ensure batch sizes match
        ensure_same_batch(
            "X and U must share the same batch size.",
            X=X,
            U=U,
        )
        # Validate state dimensions and control dimensions
        ensure_batched_horizon_vector("X", X, self.state_dim)
        ensure_batched_horizon_vector("U", U, self.control_dim)
        # Ensure horizon lengths are consistent
        ensure_equal_ints(
            "X horizon must be exactly one more than U horizon.",
            X_horizon_minus_one=X.shape[1] - 1,
            U_horizon=U.shape[1],
        )
        return X, U

    def __call__(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Evaluate the quadratic cost for given state and control trajectories.

        Args:
            X: State trajectory, shape ``(batch, N+1, n)`` or ``(N+1, n)``.
            U: Control trajectory, shape ``(batch, N, m)`` or ``(N, m)``.

        Returns:
            Cumulative cost per step, shape ``(N,)`` -- one value for each
            ``k = 1, ..., N`` (see `_validate_trajectories` and the class
            docstring's formula), normalized by the elapsed step count
            (`self.is_time_averaged` ``True``) or left as the raw cumulative
            sum (``False``).

        Raises:
            TypeError: If `X`/`U` are not ``numpy.ndarray`` (via
                `_validate_trajectories`).
            ValueError: If `X`/`U` disagree on batch size, have the wrong
                feature dimension, or `X`'s horizon is not exactly one more
                than `U`'s (via `_validate_trajectories`).
        """
        X, U = self._validate_trajectories(X, U)

        # The curve mathematics lives once in the dual-backend kernel (T2.d);
        # this class contributes validation, the batch expectation, and the
        # Signable surface.
        J_cum = cumulative_quadratic_cost(
            self.Q,
            self.R,
            X,
            U,
            conventions=self.conventions,
        )
        return cast(np.ndarray, np.mean(J_cum, axis=0))

    def get_signature(self) -> dict[str, Any]:
        """`Signable` override: flags + content hashes of Q/R.

        Hashes the *specification-time* cost matrices only (never a live/
        trained value -- there is none here, `Q`/`R` are write-once).
        `state_dim`/`control_dim` are deliberately omitted: they belong to
        the system, not the cost (`LinearSystem.get_signature()` already
        reports them), so signing them here would only duplicate that key.

        Returns:
            ``{"type": "QuadraticCost", "include_terminal_cost":,
            "is_time_averaged":, "Q_hash":, "R_hash":}``.
        """
        return super().get_signature() | {
            "include_terminal_cost": self.include_terminal_cost,
            "is_time_averaged": self.is_time_averaged,
            "Q_hash": hash_array(self.Q),
            "R_hash": hash_array(self.R),
        }
