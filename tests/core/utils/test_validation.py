import numpy as np
import pytest

from mbl.core.utils.validation import (
    ensure_array_shape,
    ensure_axis_positive,
    ensure_axis_size,
    ensure_bool,
    ensure_callable,
    ensure_equal_ints,
    ensure_last_dim,
    ensure_matrix_shape,
    ensure_ndim,
    ensure_numpy_ndarray,
    ensure_positive_definite,
    ensure_positive_integer,
    ensure_positive_semi_definite,
    ensure_same_axis_size,
    ensure_same_batch,
    ensure_same_horizon,
    ensure_square_matrix,
    ensure_symmetric_matrix,
    validate_batched_horizon_vector,
    validate_batched_vector,
    validate_bool,
    validate_matrix,
    validate_numpy_ndarray,
    validate_positive_definite,
    validate_positive_semi_definite,
    validate_square_matrix,
    validate_symmetric_matrix,
    validate_time_varying_matrix,
)


def test_ensure_positive_integer_uses_existing_call_order() -> None:
    ensure_positive_integer(1, "n")
    with pytest.raises(ValueError, match="n"):
        ensure_positive_integer(0, "n")


def test_numpy_array_validation() -> None:
    arr = np.zeros((2, 2))
    ensure_numpy_ndarray("arr", arr)
    assert validate_numpy_ndarray("arr", arr) is arr
    with pytest.raises(TypeError, match="numpy.ndarray"):
        ensure_numpy_ndarray("arr", [1, 2])


def test_bool_and_callable_validation() -> None:
    ensure_bool("flag", True)
    assert validate_bool("flag", False) is False
    with pytest.raises(TypeError, match="boolean"):
        ensure_bool("flag", 1)

    ensure_callable("fn", lambda x: x)
    with pytest.raises(TypeError, match="callable"):
        ensure_callable("fn", 1)


def test_dimension_and_shape_validation() -> None:
    arr = np.zeros((2, 3, 4))
    ensure_ndim("arr", arr, 3)
    ensure_last_dim("arr", arr, 4)
    ensure_axis_size("arr", arr, 1, 3)
    ensure_array_shape(np.zeros((2, 3)), (2, 3), "m")
    ensure_matrix_shape(np.zeros((2, 3)), (2, 3), "m")

    with pytest.raises(ValueError, match="2D"):
        ensure_ndim("arr", arr, 2)
    with pytest.raises(ValueError, match="last dimension"):
        ensure_last_dim("arr", arr, 3)
    with pytest.raises(ValueError, match="axis 1"):
        ensure_axis_size("arr", arr, 1, 2)
    with pytest.raises(ValueError, match="shape"):
        ensure_array_shape(np.zeros((2, 3)), (3, 2), "m")


def test_axis_and_consistency_helpers() -> None:
    with pytest.raises(ValueError, match="Got"):
        ensure_equal_ints("mismatch", a=1, b=2)

    a = np.zeros((2, 3, 1))
    b = np.zeros((2, 3, 2))
    c = np.zeros((4, 3, 1))
    d = np.zeros((2, 5, 1))

    ensure_same_axis_size("axis ok", axis=0, a=a, b=b)
    with pytest.raises(ValueError, match="Got"):
        ensure_same_batch("batch mismatch", a=a, c=c)
    with pytest.raises(ValueError, match="Got"):
        ensure_same_horizon("horizon mismatch", a=a, d=d)
    with pytest.raises(ValueError, match="size >= 1"):
        ensure_axis_positive("empty", np.zeros((0, 2, 2)), 0)


def test_square_and_symmetric_matrix_validation() -> None:
    square = np.eye(2)
    symmetric = np.array([[1.0, 2.0], [2.0, 1.0]])

    ensure_square_matrix("Q", square)
    ensure_symmetric_matrix("Q", symmetric)
    assert validate_square_matrix("Q", square) is square
    assert validate_symmetric_matrix("Q", symmetric) is symmetric

    with pytest.raises(ValueError, match="square"):
        ensure_square_matrix("Q", np.zeros((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        ensure_symmetric_matrix("Q", np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_batched_vector_validators() -> None:
    x = np.zeros((4, 2))
    u = np.zeros((4, 5, 2))

    assert validate_batched_vector("x", x, 2) is x
    assert validate_batched_horizon_vector("u", u, 2) is u

    with pytest.raises(ValueError, match="2D"):
        validate_batched_vector("x", np.zeros((1, 2, 3)), 3)
    with pytest.raises(ValueError, match="3D"):
        validate_batched_horizon_vector("u", np.zeros((3, 2)), 2)


def test_matrix_validators() -> None:
    matrix = np.zeros((2, 3))
    assert validate_matrix("M", matrix, (2, 3)) is matrix

    tvm = np.zeros((3, 2, 4))
    assert validate_time_varying_matrix("A_t", tvm, (2, 4)) is tvm
    with pytest.raises(ValueError, match="trailing shape"):
        validate_time_varying_matrix("A_t", np.zeros((3, 2, 3)), (2, 4))


def test_positive_definite_validators() -> None:
    pd = np.eye(2)
    psd = np.array([[1.0, 0.0], [0.0, 0.0]])

    ensure_positive_definite("R", pd)
    ensure_positive_semi_definite("Q", psd)
    assert validate_positive_definite("R", pd) is pd
    assert validate_positive_semi_definite("Q", psd) is psd

    with pytest.raises(ValueError, match="positive definite"):
        ensure_positive_definite("R", psd)
