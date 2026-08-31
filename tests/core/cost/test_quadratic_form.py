import numpy as np

from mbl.core.cost.quadratic_form import quadratic_form


def test_quadratic_form_time_invariant_2d_matrix() -> None:
    """A 2D matrix is broadcast across every time step."""
    M = np.array([[2.0, 0.0], [0.0, 3.0]])
    arr = np.array([[[1.0, 1.0], [2.0, 0.0]]])  # (batch=1, N=2, d=2)

    out = quadratic_form(M, arr)

    # [1,1]^T M [1,1] = 2 + 3 = 5 ; [2,0]^T M [2,0] = 4*2 = 8
    assert out.shape == (1, 2)
    assert np.allclose(out, [[5.0, 8.0]])


def test_quadratic_form_time_stacked_3d_matrix() -> None:
    """A 3D stack applies a distinct matrix per time step."""
    M = np.stack([np.eye(2), 10.0 * np.eye(2)])  # (N=2, d=2, d=2)
    arr = np.array([[[1.0, 1.0], [1.0, 1.0]]])  # (batch=1, N=2, d=2)

    out = quadratic_form(M, arr)

    # step 0: [1,1] I [1,1] = 2 ; step 1: [1,1] 10I [1,1] = 20
    assert out.shape == (1, 2)
    assert np.allclose(out, [[2.0, 20.0]])


def test_quadratic_form_matches_explicit_einsum_over_batch() -> None:
    rng = np.random.default_rng(0)
    batch, N, d = 4, 3, 2
    M = rng.standard_normal((N, d, d))
    M = np.einsum("tij,tkj->tik", M, M)  # symmetric PSD per slice
    arr = rng.standard_normal((batch, N, d))

    out = quadratic_form(M, arr)

    expected = np.empty((batch, N))
    for b in range(batch):
        for t in range(N):
            expected[b, t] = arr[b, t] @ M[t] @ arr[b, t]
    assert np.allclose(out, expected)
