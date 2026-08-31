import numpy as np

from mbl.core.utils.predicates import (
    is_matrix,
    is_numpy_array,
    is_positive_definite,
    is_positive_integer,
    is_positive_semi_definite,
    is_square_matrix,
    is_symmetric_matrix,
)


def test_is_positive_integer() -> None:
    assert is_positive_integer(1)
    assert not is_positive_integer(0)
    assert not is_positive_integer(-1)
    assert not is_positive_integer(1.0)


def test_is_numpy_array() -> None:
    assert is_numpy_array(np.zeros((2, 2)))
    assert not is_numpy_array([[1, 2]])


def test_is_matrix() -> None:
    assert is_matrix(np.zeros((2, 3)))
    assert not is_matrix(np.zeros((2, 3, 1)))


def test_is_square_matrix() -> None:
    assert is_square_matrix(np.eye(3))
    assert not is_square_matrix(np.zeros((2, 3)))


def test_is_symmetric_matrix() -> None:
    assert is_symmetric_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_symmetric_matrix(np.array([[1.0, 2.0], [3.0, 1.0]]))


def test_positive_definiteness_predicates() -> None:
    pd = np.eye(3)
    psd = np.array([[1.0, 0.0], [0.0, 0.0]])
    nonsym = np.array([[1.0, 1.0], [0.0, 1.0]])

    assert is_positive_definite(pd)
    assert not is_positive_definite(psd)
    assert is_positive_semi_definite(psd)
    assert not is_positive_definite(nonsym)
    assert not is_positive_semi_definite(nonsym)
