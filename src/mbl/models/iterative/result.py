"""`OptimizationResult`: the plain, structured output of an engine-free
iterative solver (e.g. `models.analytic.iterative_gd
.AnalyticalIterativeGDController.solve`) -- generic across whichever
`SweepStrategy`/`GradientDescentRefinement` produced it, so every future
analytical solver in this family returns the same shape."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """The full record of one `.solve()` call.

    Attributes:
        U_history: Per-macro-iteration control-sequence iterates, shape
            ``(K+1, batch, T, m)`` -- row 0 is the control initializer's
            proposal ``U^(0)``, row ``K`` is the converged/final iterate
            ``U^(K)``. Directly consumable by the Phase 2B visualization
            toolkit as ``candidate_U_history`` (its own ``(I, B, T, m)`` form).
        J_history: Per-macro-iteration cost, shape ``(K+1,)``, index-aligned
            with `U_history`: ``J_history[i] == J(U_history[i])`` exactly (no
            append/reconciliation needed -- each sweep's cost is the cost of
            the trajectory it just realized). Directly consumable as the
            toolkit's ``iteration_costs``.
        U_final: ``U_history[-1]``, shape ``(batch, T, m)``.
        J_final: ``float(J_history[-1])``.
        X_final: The final iterate's realized state trajectory, shape
            ``(batch, T+1, n)``.
        iterations_run: ``K``, the number of macro-iterations (sweeps)
            actually executed (``<= max_iters``; less than `max_iters` iff
            `converged`).
        converged: Whether the ``tolerance`` early-stop criterion triggered.
        signature: Provenance tree -- problem/initializer/samplers/step-size/
            sweep-strategy signatures -- making the result independently
            reproducible from its own record.
    """

    U_history: np.ndarray
    J_history: np.ndarray
    U_final: np.ndarray
    J_final: float
    X_final: np.ndarray
    iterations_run: int
    converged: bool
    signature: dict[str, Any]
