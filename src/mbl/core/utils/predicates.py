"""Pure boolean predicates over values/arrays. Never raise -- ``core.utils.validation``
builds its ``ensure_*``/``validate_*`` (raising) functions on top of these."""

from functools import partial
from typing import TypeGuard

import numpy as np

# Direct submodule import (never the package root); see the matching note in
# core.utils.validation.
from mbl.core.runtime.tolerances import DEFINITENESS_TOL, SYMMETRY_ATOL

from .guards import content_cached


def is_positive_integer(value: object) -> bool:
    """Return whether `value` is an ``int`` strictly greater than zero.

    Args:
        value: The object to check.

    Returns:
        ``True`` iff `value` is an ``int`` and ``value > 0``.
    """
    return isinstance(value, int) and value > 0


def is_numpy_array(value: object) -> TypeGuard[np.ndarray]:
    """Return whether `value` is a ``numpy.ndarray`` (of any shape/dtype).

    Args:
        value: The object to check.

    Returns:
        ``isinstance(value, np.ndarray)``.
    """
    return isinstance(value, np.ndarray)


def is_matrix(value: object) -> TypeGuard[np.ndarray]:
    """Return whether `value` is a 2D ``numpy.ndarray`` of any shape.

    Args:
        value: The object to check.

    Returns:
        ``True`` iff `value` is a ``numpy.ndarray`` with ``ndim == 2``.
    """
    return is_numpy_array(value) and value.ndim == 2


def is_square_matrix(value: object) -> TypeGuard[np.ndarray]:
    """Return whether `value` is a 2D array with equal row and column counts.

    Args:
        value: The object to check.

    Returns:
        ``True`` iff `value` is a matrix (see `is_matrix`) and
        ``value.shape[0] == value.shape[1]``.
    """
    return is_matrix(value) and value.shape[0] == value.shape[1]


def is_symmetric_matrix(
    value: object, atol: float = SYMMETRY_ATOL
) -> TypeGuard[np.ndarray]:
    """Return whether `value` is a square matrix equal to its own transpose.

    Args:
        value: The object to check.
        atol: Absolute tolerance passed to ``np.allclose(value, value.T)``.

    Returns:
        ``True`` iff `value` is square (see `is_square_matrix`) and
        ``np.allclose(value, value.T, atol=atol)``.
    """
    return is_square_matrix(value) and np.allclose(value, value.T, atol=atol)


@content_cached()
def _min_eigenvalue(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix, memoized on the matrix's
    contents so the (expensive) eigendecomposition of a *static* cost/system
    matrix runs exactly once even when several call sites re-check it.

    Args:
        matrix: A symmetric matrix, shape ``(n, n)``.

    Returns:
        The smallest eigenvalue of `matrix` (via ``numpy.linalg.eigvalsh``,
        which assumes -- but does not itself verify -- symmetry).
    """
    return float(np.linalg.eigvalsh(matrix)[0])


def _has_positive_eigenvalues(
    matrix: np.ndarray, strict: bool, tol: float = DEFINITENESS_TOL
) -> bool:
    """Return whether every eigenvalue of a symmetric `matrix` exceeds a
    (strict or non-strict) tolerance threshold.

    Args:
        matrix: A symmetric matrix, shape ``(n, n)``.
        strict: If ``True``, require ``eigenvalue > tol`` (positive definite);
            if ``False``, require ``eigenvalue > -tol`` (positive semi-definite,
            with `tol` as numerical slack around zero).
        tol: The (non-negative) tolerance.

    Returns:
        ``True`` iff every eigenvalue clears the threshold.
    """
    threshold = tol if strict else -tol
    # eigvalsh returns eigenvalues in ascending order, so the minimum alone
    # decides definiteness (min > threshold  <=>  all > threshold).
    return _min_eigenvalue(matrix) > threshold


def _is_definite(value: object, *, strict: bool, tol: float = DEFINITENESS_TOL) -> bool:
    """Return whether `value` is a symmetric matrix with every eigenvalue
    exceeding `tol` (`strict=True`) or `-tol` (`strict=False`).

    Args:
        value: The object to check.
        strict: Selects positive-definite (``True``) vs. positive
            semi-definite (``False``) semantics.
        tol: The eigenvalue tolerance (see `_has_positive_eigenvalues`).

    Returns:
        ``False`` immediately if `value` is not symmetric (eigenvalue-based
        definiteness is only meaningful for symmetric square matrices);
        otherwise the result of `_has_positive_eigenvalues`.
    """
    # Eigenvalue-based definiteness is only valid for symmetric square matrices.
    return is_symmetric_matrix(value) and _has_positive_eigenvalues(
        value, strict=strict, tol=tol
    )


# One definiteness core; the strict/non-strict pair is a partial application.
is_positive_definite = partial(_is_definite, strict=True)
"""Return whether `value` is a symmetric matrix with every eigenvalue > `tol`.

Args:
    value: The object to check.
    tol: Eigenvalue strictness threshold (default ``1e-9``).

Returns:
    ``True`` iff `value` is symmetric and positive definite within `tol`.
"""

is_positive_semi_definite = partial(_is_definite, strict=False)
"""Return whether `value` is a symmetric matrix with every eigenvalue > `-tol`.

Args:
    value: The object to check.
    tol: Eigenvalue non-negativity slack (default ``1e-9``).

Returns:
    ``True`` iff `value` is symmetric and positive semi-definite within `tol`.
"""
