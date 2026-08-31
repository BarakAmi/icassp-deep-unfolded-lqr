"""Fail-fast validators and validator-combinators for the project's core types.

Each ``ensure_*`` function raises immediately on an invalid value and returns
``None``; each ``validate_*`` function does the same check but returns the
(possibly unwrapped) value, so it can be used inline at an assignment site
(e.g. ``self.Q = validate_positive_semi_definite("Q", Q)``). These are the
primitives the Phase 1A fail-fast guards (``core.utils.guards``,
``core.utils.coherence``, ``models.guards``) and the write-once descriptors
(``core.utils.descriptors``) are built from.
"""

from collections.abc import Callable
from typing import cast
from functools import wraps

import numpy as np

# Direct submodule import (never the package root): this module is executed
# during core.utils package init, and importing attributes off src.core.runtime
# could observe that package partially initialized.
from mbl.core.runtime.tolerances import DEFINITENESS_TOL, SYMMETRY_ATOL

from .predicates import (
    is_numpy_array,
    is_positive_definite,
    is_positive_integer,
    is_positive_semi_definite,
    is_square_matrix,
    is_symmetric_matrix,
)


def ensure_callable(name: str, value: object) -> None:
    """Assert `value` is callable.

    Args:
        name: Human-readable identifier used in the error message.
        value: The object to check.

    Raises:
        TypeError: If `value` is not callable.
    """
    if not callable(value):
        raise TypeError(f"{name} must be callable.")


def ensure_positive_integer(value: object, name: str) -> None:
    """Assert `value` is a strictly positive ``int``.

    Args:
        value: The object to check.
        name: Human-readable identifier used in the error message.

    Raises:
        ValueError: If `value` is not a positive integer.
    """
    if not is_positive_integer(value):
        raise ValueError(f"{name} must be a positive integer.")


def ensure_bool(name: str, value: object) -> None:
    """Assert `value` is a ``bool``.

    Args:
        name: Human-readable identifier used in the error message.
        value: The object to check.

    Raises:
        TypeError: If `value` is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def ensure_numpy_ndarray(name: str, arr: object) -> None:
    """Assert `arr` is a ``numpy.ndarray``.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
    """
    if not is_numpy_array(arr):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")


def ensure_ndim(name: str, arr: np.ndarray, expected_ndim: int) -> None:
    """Assert `arr` has exactly `expected_ndim` dimensions.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The array to check.
        expected_ndim: The required number of dimensions.

    Raises:
        ValueError: If `arr.ndim != expected_ndim`.
    """
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be a {expected_ndim}D array, got shape {arr.shape}."
        )


def ensure_last_dim(name: str, arr: np.ndarray, expected_last_dim: int) -> None:
    """Assert `arr`'s last axis has size `expected_last_dim`.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The array to check.
        expected_last_dim: The required size of `arr.shape[-1]`.

    Raises:
        ValueError: If `arr.shape[-1] != expected_last_dim`.
    """
    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_last_dim}, got {arr.shape[-1]}."
        )


def ensure_axis_size(name: str, arr: np.ndarray, axis: int, expected_size: int) -> None:
    """Assert `arr.shape[axis] == expected_size`.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The array to check.
        axis: The axis index to check.
        expected_size: The required size of that axis.

    Raises:
        ValueError: If the axis size does not match.
    """
    if arr.shape[axis] != expected_size:
        raise ValueError(
            f"{name} must have axis {axis} size {expected_size}, got {arr.shape[axis]}."
        )


def ensure_axis_positive(name: str, arr: np.ndarray, axis: int) -> None:
    """Assert `arr.shape[axis] >= 1` (the axis is non-empty).

    Args:
        name: Human-readable identifier used in the error message.
        arr: The array to check.
        axis: The axis index to check.

    Raises:
        ValueError: If the axis has size 0.
    """
    if arr.shape[axis] < 1:
        raise ValueError(
            f"{name} must have axis {axis} size >= 1, got {arr.shape[axis]}."
        )


def ensure_equal_ints(message: str, **values: int) -> None:
    """Assert every keyword value is numerically equal (e.g. matching batch
    sizes or horizon lengths across several named arrays).

    Args:
        message: Error message prefix used if the values disagree.
        **values: Named integers that must all be equal.

    Raises:
        ValueError: If more than one distinct value is present, listing every
            name and its value.
    """
    unique_values = set(values.values())
    if len(unique_values) > 1:
        as_text = ", ".join(f"{k}={v}" for k, v in values.items())
        raise ValueError(f"{message} Got: {as_text}.")


def ensure_same_axis_size(message: str, axis: int, **named_arrays: np.ndarray) -> None:
    """Assert every named array has the same size along `axis`.

    Args:
        message: Error message prefix used if the sizes disagree.
        axis: The axis index compared across all arrays.
        **named_arrays: Named arrays whose `axis` sizes must match.

    Raises:
        ValueError: If the arrays disagree on that axis's size.
    """
    ensure_equal_ints(message, **{k: v.shape[axis] for k, v in named_arrays.items()})


def ensure_same_batch(message: str, **named_arrays: np.ndarray) -> None:
    """Assert every named array shares the same batch size (axis 0).

    Args:
        message: Error message prefix used if the batch sizes disagree.
        **named_arrays: Named arrays of shape ``(batch, ...)``.

    Raises:
        ValueError: If the arrays disagree on axis-0 size.
    """
    ensure_same_axis_size(message, axis=0, **named_arrays)


def ensure_same_horizon(message: str, **named_arrays: np.ndarray) -> None:
    """Assert every named array shares the same horizon length (axis 1).

    Args:
        message: Error message prefix used if the horizon lengths disagree.
        **named_arrays: Named arrays of shape ``(batch, horizon, ...)``.

    Raises:
        ValueError: If the arrays disagree on axis-1 size.
    """
    ensure_same_axis_size(message, axis=1, **named_arrays)


def ensure_array_shape(
    array: np.ndarray, expected_shape: tuple[int, ...], name: str
) -> None:
    """Assert `array.shape == expected_shape` exactly.

    Args:
        array: The array to check.
        expected_shape: The required exact shape.
        name: Human-readable identifier used in the error message.

    Raises:
        ValueError: If the shapes differ.
    """
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, but got {array.shape}."
        )


def ensure_matrix_shape(
    matrix: np.ndarray, expected_shape: tuple[int, int], name: str
) -> None:
    """Assert a 2D `matrix` has exactly `expected_shape`.

    Args:
        matrix: The matrix to check, shape ``(rows, cols)``.
        expected_shape: The required exact ``(rows, cols)``.
        name: Human-readable identifier used in the error message.

    Raises:
        ValueError: If the shapes differ.
    """
    ensure_array_shape(matrix, expected_shape, name)


def ensure_square_matrix(name: str, matrix: object) -> None:
    """Assert `matrix` is 2D with equal row and column counts.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.

    Raises:
        ValueError: If `matrix` is not a square 2D array.
    """
    if not is_square_matrix(matrix):
        raise ValueError(
            f"{name} must be square, got {getattr(matrix, 'shape', None)}."
        )


def ensure_symmetric_matrix(
    name: str, matrix: object, atol: float = SYMMETRY_ATOL
) -> None:
    """Assert `matrix` is square and equals its own transpose within `atol`.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        atol: Absolute tolerance passed to ``np.allclose(matrix, matrix.T)``.

    Raises:
        ValueError: If `matrix` is not symmetric.
    """
    if not is_symmetric_matrix(matrix, atol=atol):
        raise ValueError(f"{name} must be symmetric.")


def validate_numpy_ndarray(name: str, arr: object) -> np.ndarray:
    """Validate and return `arr` as a ``numpy.ndarray``.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.

    Returns:
        `arr`, unchanged.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
    """
    ensure_numpy_ndarray(name, arr)
    return cast(np.ndarray, arr)


def validate_bool(name: str, value: object) -> bool:
    """Validate and return `value` as a ``bool``.

    Args:
        name: Human-readable identifier used in the error message.
        value: The object to check.

    Returns:
        `value`, unchanged.

    Raises:
        TypeError: If `value` is not a ``bool``.
    """
    ensure_bool(name, value)
    return cast(bool, value)


def ensure_batched_vector(name: str, arr: object, expected_last_dim: int) -> None:
    """Assert `arr` is a 2D ``(batch, expected_last_dim)`` array.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.
        expected_last_dim: The required feature dimension.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
        ValueError: If `arr` is not 2D or its last dimension mismatches.
    """
    validate_batched_vector(name, arr, expected_last_dim)


def ensure_batched_horizon_vector(
    name: str, arr: object, expected_last_dim: int
) -> None:
    """Assert `arr` is a 3D ``(batch, horizon, expected_last_dim)`` array.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.
        expected_last_dim: The required feature dimension.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
        ValueError: If `arr` is not 3D or its last dimension mismatches.
    """
    validate_batched_horizon_vector(name, arr, expected_last_dim)


def validate_batched_vector(
    name: str, arr: object, expected_last_dim: int
) -> np.ndarray:
    """Validate and return `arr` as a 2D ``(batch, expected_last_dim)`` array.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.
        expected_last_dim: The required feature dimension.

    Returns:
        `arr`, unchanged.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
        ValueError: If `arr` is not 2D or its last dimension mismatches.
    """
    array = validate_numpy_ndarray(name, arr)
    ensure_ndim(name, array, 2)
    ensure_last_dim(name, array, expected_last_dim)
    return array


def validate_batched_horizon_vector(
    name: str, arr: object, expected_last_dim: int
) -> np.ndarray:
    """Validate and return `arr` as a 3D ``(batch, horizon, expected_last_dim)``
    array.

    Args:
        name: Human-readable identifier used in the error message.
        arr: The object to check.
        expected_last_dim: The required feature dimension.

    Returns:
        `arr`, unchanged.

    Raises:
        TypeError: If `arr` is not a ``numpy.ndarray``.
        ValueError: If `arr` is not 3D or its last dimension mismatches.
    """
    array = validate_numpy_ndarray(name, arr)
    ensure_ndim(name, array, 3)
    ensure_last_dim(name, array, expected_last_dim)
    return array


def validate_matrix(
    name: str, matrix: object, expected_shape: tuple[int, int]
) -> np.ndarray:
    """Validate and return `matrix` as a 2D array of exactly `expected_shape`.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        expected_shape: The required exact ``(rows, cols)``.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not 2D or its shape mismatches.
    """
    array = validate_numpy_ndarray(name, matrix)
    ensure_ndim(name, array, 2)
    ensure_matrix_shape(array, expected_shape, name)
    return array


def validate_square_matrix(name: str, matrix: object) -> np.ndarray:
    """Validate and return `matrix` as a square 2D array.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not 2D or not square.
    """
    array = validate_numpy_ndarray(name, matrix)
    ensure_ndim(name, array, 2)
    ensure_square_matrix(name, array)
    return array


def validate_symmetric_matrix(
    name: str, matrix: object, atol: float = SYMMETRY_ATOL
) -> np.ndarray:
    """Validate and return `matrix` as a square, symmetric 2D array.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        atol: Absolute tolerance for the symmetry check.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not square or not symmetric within `atol`.
    """
    array = validate_square_matrix(name, matrix)
    ensure_symmetric_matrix(name, array, atol=atol)
    return array


def validate_time_varying_matrix(
    name: str,
    matrix: object,
    expected_trailing_shape: tuple[int, int],
) -> np.ndarray:
    """Validate and return `matrix` as a non-empty 3D time-stack of shape
    ``(horizon, *expected_trailing_shape)``.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        expected_trailing_shape: The required ``(rows, cols)`` of every slice.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not 3D, has a zero-length horizon axis, or
            its trailing shape mismatches.
    """
    array = validate_numpy_ndarray(name, matrix)
    ensure_ndim(name, array, 3)
    ensure_axis_positive(name, array, axis=0)
    if array.shape[1:] != expected_trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {expected_trailing_shape}, got {array.shape[1:]}."
        )
    return array


def ensure_positive_definite(
    name: str, matrix: np.ndarray, tol: float = DEFINITENESS_TOL
) -> None:
    """Assert `matrix` is symmetric positive definite.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The (assumed symmetric) matrix to check.
        tol: Eigenvalue strictness threshold (see ``predicates.is_positive_definite``).

    Raises:
        ValueError: If `matrix` is not positive definite.
    """
    if not is_positive_definite(matrix, tol=tol):
        raise ValueError(f"{name} must be positive definite")


def ensure_positive_semi_definite(
    name: str, matrix: np.ndarray, tol: float = DEFINITENESS_TOL
) -> None:
    """Assert `matrix` is symmetric positive semi-definite.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The (assumed symmetric) matrix to check.
        tol: Eigenvalue strictness threshold (see ``predicates.is_positive_semi_definite``).

    Raises:
        ValueError: If `matrix` is not positive semi-definite.
    """
    if not is_positive_semi_definite(matrix, tol=tol):
        raise ValueError(f"{name} must be positive semidefinite")


def _resolve_definiteness_tolerances(
    atol: float | None, tol: float | None
) -> tuple[float, float]:
    """Resolve the (symmetry, eigenvalue) tolerance pair for the definiteness
    validators (C3 fix, T1.d): an explicit `atol` governs *both* checks unless
    `tol` explicitly overrides the eigenvalue side; omitted values fall back
    to the named tolerance policy (`SYMMETRY_ATOL` / `DEFINITENESS_TOL`), so
    default-argument call sites behave exactly as before the fix.

    Args:
        atol: The caller's symmetry tolerance, or ``None`` for the policy
            default.
        tol: The caller's eigenvalue tolerance, or ``None`` to follow `atol`.

    Returns:
        The effective ``(symmetry_atol, eigenvalue_tol)`` pair.
    """
    symmetry_atol = SYMMETRY_ATOL if atol is None else atol
    if tol is not None:
        eigenvalue_tol = tol
    else:
        eigenvalue_tol = DEFINITENESS_TOL if atol is None else atol
    return symmetry_atol, eigenvalue_tol


def validate_positive_definite(
    name: str, matrix: object, atol: float | None = None, tol: float | None = None
) -> np.ndarray:
    """Validate and return `matrix` as a symmetric, positive definite 2D array.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        atol: Absolute tolerance used for both the symmetry and (as `tol`) the
            eigenvalue-positivity checks. Defaults to `SYMMETRY_ATOL` for
            symmetry and `DEFINITENESS_TOL` for the eigenvalue check when
            omitted.
        tol: Explicit eigenvalue-positivity threshold, overriding `atol` for
            the eigenvalue check only.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not square, not symmetric, or not positive
            definite.
    """
    symmetry_atol, eigenvalue_tol = _resolve_definiteness_tolerances(atol, tol)
    array = validate_symmetric_matrix(name, matrix, atol=symmetry_atol)
    ensure_positive_definite(name, array, tol=eigenvalue_tol)
    return array


def validate_positive_semi_definite(
    name: str, matrix: object, atol: float | None = None, tol: float | None = None
) -> np.ndarray:
    """Validate and return `matrix` as a symmetric, positive semi-definite 2D
    array.

    Args:
        name: Human-readable identifier used in the error message.
        matrix: The object to check.
        atol: Absolute tolerance used for both the symmetry and (as `tol`) the
            eigenvalue-nonnegativity checks. Defaults to `SYMMETRY_ATOL` for
            symmetry and `DEFINITENESS_TOL` for the eigenvalue check when
            omitted.
        tol: Explicit eigenvalue-nonnegativity slack, overriding `atol` for
            the eigenvalue check only.

    Returns:
        `matrix`, unchanged.

    Raises:
        TypeError: If `matrix` is not a ``numpy.ndarray``.
        ValueError: If `matrix` is not square, not symmetric, or not positive
            semi-definite.
    """
    symmetry_atol, eigenvalue_tol = _resolve_definiteness_tolerances(atol, tol)
    array = validate_symmetric_matrix(name, matrix, atol=symmetry_atol)
    ensure_positive_semi_definite(name, array, tol=eigenvalue_tol)
    return array


type StackValidator = Callable[..., np.ndarray]


def over_time_stack(per_slice_validator: Callable[..., np.ndarray]) -> StackValidator:
    """Lift a single-matrix validator into one that accepts either a single 2D
    matrix (validated directly) or a 3D time-stack (each slice validated as
    ``{name}[k]``). Collapses the otherwise byte-identical ``*_stack`` twins
    into one shared higher-order primitive.

    Args:
        per_slice_validator: A ``(name, matrix, atol=..., tol=...) -> matrix``
            validator (e.g. `validate_positive_definite`) applied to every 2D
            slice.

    Returns:
        A `StackValidator` with signature
        ``(name, matrix, atol=None, tol=None) -> matrix``.
    """

    def validate_stack(
        name: str, matrix: object, atol: float | None = None, tol: float | None = None
    ) -> np.ndarray:
        """Validate `matrix` as a single 2D matrix or a 3D time-stack of them.

        Args:
            name: Human-readable identifier used in the error message.
            matrix: A 2D matrix, or a 3D array of shape ``(horizon, *matrix_shape)``.
            atol: Absolute tolerance forwarded to `per_slice_validator`.
            tol: Eigenvalue tolerance forwarded to `per_slice_validator`.

        Returns:
            `matrix`, unchanged.

        Raises:
            TypeError: If `matrix` is not a ``numpy.ndarray``.
            ValueError: If `matrix` is neither 2D nor 3D, or any slice fails
                `per_slice_validator`.
        """
        array = validate_numpy_ndarray(name, matrix)
        if array.ndim == 2:
            return per_slice_validator(name, array, atol=atol, tol=tol)
        ensure_ndim(name, array, 3)
        for k, step_matrix in enumerate(array):
            per_slice_validator(f"{name}[{k}]", step_matrix, atol=atol, tol=tol)
        return array

    return validate_stack


validate_positive_definite_stack = over_time_stack(validate_positive_definite)
"""Validate a single matrix (2D) or a time-stacked sequence (3D), each PD."""

validate_positive_semi_definite_stack = over_time_stack(validate_positive_semi_definite)
"""Validate a single matrix (2D) or a time-stacked sequence (3D), each PSD."""


type StateSpaceMap = Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def enforce_state_space_map_dims(
    name: str,
    func: StateSpaceMap,
    *,
    state_dim: int,
    control_dim: int,
    noise_dim: int,
    out_dim: int,
) -> StateSpaceMap:
    """Wrap a state-space map (state-transition or observation map) with
    runtime shape validation of its batched inputs and output.

    The wrapper validates that ``x``, ``u``, and ``noise`` are each 2D
    ``(batch, dim)`` arrays with the declared feature dimensions and a shared
    batch size, then validates the wrapped `func`'s output the same way.
    Under ``python -O`` (``__debug__`` is ``False``) the wrapper is skipped
    entirely and `func` is returned unwrapped, for zero overhead.

    Args:
        name: Human-readable identifier used in error messages (e.g. the map's
            name).
        func: The state-space map to wrap, called as
            ``func(k, x, u, noise) -> out``.
        state_dim: Required feature dimension of `x`.
        control_dim: Required feature dimension of `u`.
        noise_dim: Required feature dimension of `noise`.
        out_dim: Required feature dimension of `func`'s return value.

    Returns:
        `func` unchanged (if ``__debug__`` is ``False``), or a validating
        wrapper around it.

    Raises:
        TypeError: If `func` is not callable (checked eagerly, regardless of
            ``__debug__``).
        ValueError: (from the wrapper, at call time) if any argument or the
            output has the wrong rank, feature dimension, or an inconsistent
            batch size across `x`/`u`/`noise`.
    """
    ensure_callable(name, func)

    if not __debug__:
        return func

    @wraps(func)
    def wrapper(k: int, x: np.ndarray, u: np.ndarray, noise: np.ndarray) -> np.ndarray:
        x = validate_batched_vector("state", x, state_dim)
        u = validate_batched_vector("control", u, control_dim)
        noise = validate_batched_vector("noise", noise, noise_dim)

        ensure_equal_ints(
            f"{name} step {k} batch",
            state=x.shape[0],
            control=u.shape[0],
            noise=noise.shape[0],
        )

        out = func(k, x, u, noise)
        return validate_batched_vector("output", out, out_dim)

    return wrapper
