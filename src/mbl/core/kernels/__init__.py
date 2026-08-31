"""Tier 1.e / T2.d dual-backend kernel layer (REFACTOR_PLAN v3, Stage S2).

The single home for mathematics that must execute identically under NumPy
and PyTorch: the quadratic objective (T2.d — the root-cause fix for the
C2/M-class cost drift), the canonical time-invariant cost-matrix slice
(T2.e), and the finite-horizon Riccati recursion (T1.e — solvers
untethered from their substrate).

Contractual laws:

* **Solvers never branch on backend.** Backend selection happens in exactly
  one dispatch site (`dispatch.array_namespace`); every kernel is written
  once against that seam and executes natively on whichever substrate its
  inputs live on — NumPy on CPU, torch on CPU or GPU.
* **Bit-stability on the NumPy path.** Each kernel reproduces the exact
  operation sequence of the pre-S2 implementation it consolidates, so the
  S0 golden-master fixtures reproduce bit-for-bit (§7.1).
* **Array authorship stays with `ComputeContext`.** Kernels never create
  arrays on a substrate their inputs do not already occupy; materializing
  inputs onto a substrate is the caller's job via `ComputeContext.asarray`
  (T1.b / T1.f). Mixing backends in one call is a conversion-boundary
  violation and fails loudly at the dispatch site.
"""

from .dispatch import array_namespace
from .quadratic import (
    CostConventions,
    CostReduction,
    cumulative_quadratic_cost,
    quadratic_form,
    time_invariant_slice,
    total_quadratic_cost,
)
from .riccati import compute_lqr_gradient_matrices, riccati_recursion

__all__ = [
    "array_namespace",
    "CostConventions",
    "CostReduction",
    "cumulative_quadratic_cost",
    "quadratic_form",
    "time_invariant_slice",
    "total_quadratic_cost",
    "compute_lqr_gradient_matrices",
    "riccati_recursion",
]
