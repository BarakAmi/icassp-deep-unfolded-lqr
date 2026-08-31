import numpy as np
import pytest

from mbl.core.utils.array_ops import add_batch_dim, add_batch_dim_to_multiple, cummean


def test_add_batch_dim_adds_dim_for_2d() -> None:
    x = np.zeros((4, 2))
    out = add_batch_dim(x)
    assert out.shape == (1, 4, 2)


def test_add_batch_dim_keeps_non_2d() -> None:
    x = np.zeros((3, 4, 2))
    out = add_batch_dim(x)
    assert out.shape == x.shape


def test_add_batch_dim_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        add_batch_dim([1, 2, 3])


def test_add_batch_dim_to_multiple() -> None:
    x = np.zeros((4, 2))
    u = np.zeros((3, 1))
    x_out, u_out = add_batch_dim_to_multiple(x, u)
    assert x_out.shape == (1, 4, 2)
    assert u_out.shape == (1, 3, 1)


def test_cummean_axis_zero() -> None:
    x = np.array([1.0, 3.0, 6.0])
    out = cummean(x)
    assert np.allclose(out, np.array([1.0, 2.0, 10.0 / 3.0]))


def test_cummean_axis_one() -> None:
    x = np.array([[1.0, 3.0], [2.0, 4.0]])
    out = cummean(x, axis=1)
    assert np.allclose(out, np.array([[1.0, 2.0], [2.0, 3.0]]))
