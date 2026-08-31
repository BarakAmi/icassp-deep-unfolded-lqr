import numpy as np
import torch

from mbl.models.lqr_gradient import compute_lqr_gradient_matrices


def _manual_single_step(P, A, B, R):
    BtP = B.T @ P
    M = R + BtP @ B
    C = BtP @ A
    return M, C


def test_compute_lqr_gradient_matrices_numpy_single_step() -> None:
    rng = np.random.default_rng(0)
    n, m = 3, 2
    P = np.eye(n) * 2.0
    A = rng.normal(size=(n, n))
    B = rng.normal(size=(n, m))
    R = np.eye(m)

    M, C = compute_lqr_gradient_matrices(P, A, B, R)
    expected_M, expected_C = _manual_single_step(P, A, B, R)

    assert M.shape == (m, m)
    assert C.shape == (m, n)
    assert np.allclose(M, expected_M)
    assert np.allclose(C, expected_C)


def test_compute_lqr_gradient_matrices_numpy_batched_matches_per_step() -> None:
    rng = np.random.default_rng(1)
    horizon, n, m = 4, 3, 2
    P = rng.normal(size=(horizon, n, n))
    A = rng.normal(size=(horizon, n, n))
    B = rng.normal(size=(horizon, n, m))
    R = np.stack([np.eye(m)] * horizon)

    M, C = compute_lqr_gradient_matrices(P, A, B, R)
    assert M.shape == (horizon, m, m)
    assert C.shape == (horizon, m, n)

    for k in range(horizon):
        expected_M, expected_C = _manual_single_step(P[k], A[k], B[k], R[k])
        assert np.allclose(M[k], expected_M)
        assert np.allclose(C[k], expected_C)


def test_compute_lqr_gradient_matrices_works_identically_for_torch_tensors() -> None:
    rng = np.random.default_rng(2)
    n, m = 3, 2
    P_np = np.eye(n) * 1.5
    A_np = rng.normal(size=(n, n))
    B_np = rng.normal(size=(n, m))
    R_np = np.eye(m)

    M_np, C_np = compute_lqr_gradient_matrices(P_np, A_np, B_np, R_np)

    P_t, A_t, B_t, R_t = (
        torch.as_tensor(P_np, dtype=torch.float64),
        torch.as_tensor(A_np, dtype=torch.float64),
        torch.as_tensor(B_np, dtype=torch.float64),
        torch.as_tensor(R_np, dtype=torch.float64),
    )
    M_t, C_t = compute_lqr_gradient_matrices(P_t, A_t, B_t, R_t)

    assert np.allclose(M_t.numpy(), M_np)
    assert np.allclose(C_t.numpy(), C_np)
