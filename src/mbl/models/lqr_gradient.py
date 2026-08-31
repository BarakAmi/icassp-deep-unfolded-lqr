"""Deprecated location shim: the shared LQR gradient-matrix math now lives
in the dual-backend kernel layer (``core.kernels.riccati``, T1.e Stage S2)
alongside the Riccati recursion it feeds. This module survives for import
stability (`models.analytic.riccati` and
`models.unfolded.iterative_refinement` historically import it here) and
re-exports the kernel symbol unchanged; import from ``core.kernels``
directly in new code.
"""

from ..core.kernels.riccati import compute_lqr_gradient_matrices

__all__ = ["compute_lqr_gradient_matrices"]
