"""Deprecated location shim: the per-step quadratic form now lives in the
dual-backend kernel layer (``core.kernels.quadratic``, T2.d), where the
pre-S2 NumPy einsum and its private torch twin merged into one
implementation. This module survives one stage for import stability and
re-exports the kernel symbol unchanged; import from ``core.kernels``
directly in new code.
"""

from ..kernels.quadratic import quadratic_form

__all__ = ["quadratic_form"]
