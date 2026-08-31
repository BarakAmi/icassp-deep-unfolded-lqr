"""Unit + cross-backend equivalence tests for the dual-backend Riccati
kernel (T1.e, Stage S2; REFACTOR_PLAN v3 §7.9 first slice).

Bit-stability on the NumPy path is the golden-master load-bearing property
(`finite_horizon_riccati` now delegates here); backend equivalence is the
substrate-flexibility mandate — parameterized over what the host offers,
so the full suite passes on GPU-less machines.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mbl.core.kernels import compute_lqr_gradient_matrices, riccati_recursion

N_DIM, M_DIM, HORIZON = 4, 2, 12


@pytest.fixture(scope="module")
def lqr() -> dict[str, np.ndarray]:
    """A seeded, marginally-scaled, genuinely time-VARYING float64 instance
    (time-varying A/B so the per-step indexing path is actually exercised)."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(HORIZON, N_DIM, N_DIM))
    A /= np.abs(np.linalg.eigvals(A)).max(axis=1)[:, None, None]
    B = rng.normal(size=(HORIZON, N_DIM, M_DIM))
    Q = np.repeat(np.eye(N_DIM)[None], HORIZON + 1, axis=0)
    R = np.repeat((0.1 * np.eye(M_DIM))[None], HORIZON, axis=0)
    return {"A": A, "B": B, "Q": Q, "R": R}


def _legacy_recursion(A, B, Q, R, horizon):
    """Inline replica of the pre-S2 `finite_horizon_riccati` loop body."""
    P_list: list = [None] * (horizon + 1)
    K_list: list = [None] * horizon
    P = Q[horizon]
    P_list[horizon] = P
    for k in reversed(range(horizon)):
        A_k, B_k = A[k], B[k]
        BtP = B_k.T @ P
        M = R[k] + BtP @ B_k
        C = BtP @ A_k
        AtP = A_k.T @ P
        K = np.linalg.solve(M, C)
        P = Q[k] + AtP @ (A_k - B_k @ K)
        P_list[k], K_list[k] = P, K
    return np.stack(P_list), np.stack(K_list)


class TestNumpyPath:
    def test_bit_identical_to_the_legacy_loop(self, lqr):
        P_arr, K_arr = riccati_recursion(
            lqr["A"], lqr["B"], lqr["Q"], lqr["R"], HORIZON
        )
        P_ref, K_ref = _legacy_recursion(
            lqr["A"], lqr["B"], lqr["Q"], lqr["R"], HORIZON
        )
        assert P_arr.shape == (HORIZON + 1, N_DIM, N_DIM)
        assert K_arr.shape == (HORIZON, M_DIM, N_DIM)
        assert np.array_equal(P_arr, P_ref)
        assert np.array_equal(K_arr, K_ref)

    def test_gradient_matrices_serve_2d_and_3d_on_both_backends(self, lqr):
        P_next = lqr["Q"][1:]
        M_np, C_np = compute_lqr_gradient_matrices(P_next, lqr["A"], lqr["B"], lqr["R"])
        M_t, C_t = compute_lqr_gradient_matrices(
            torch.as_tensor(P_next),
            torch.as_tensor(lqr["A"]),
            torch.as_tensor(lqr["B"]),
            torch.as_tensor(lqr["R"]),
        )
        np.testing.assert_allclose(M_t.numpy(), M_np, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(C_t.numpy(), C_np, atol=1e-12, rtol=1e-12)

        M_2d, C_2d = compute_lqr_gradient_matrices(
            P_next[0], lqr["A"][0], lqr["B"][0], lqr["R"][0]
        )
        np.testing.assert_allclose(M_2d, M_np[0], atol=1e-15, rtol=0)
        np.testing.assert_allclose(C_2d, C_np[0], atol=1e-15, rtol=0)


class TestCrossBackend:
    def test_torch_cpu_agrees_with_numpy(self, lqr):
        """§7.9: same seeded problem, NumPy/CPU vs torch/CPU, float64
        tolerance."""
        P_np, K_np = riccati_recursion(lqr["A"], lqr["B"], lqr["Q"], lqr["R"], HORIZON)
        tensors = {k: torch.as_tensor(v) for k, v in lqr.items()}
        P_t, K_t = riccati_recursion(
            tensors["A"], tensors["B"], tensors["Q"], tensors["R"], HORIZON
        )
        assert isinstance(P_t, torch.Tensor) and P_t.dtype == torch.float64
        np.testing.assert_allclose(P_t.numpy(), P_np, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(K_t.numpy(), K_np, atol=1e-10, rtol=1e-10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="host has no CUDA")
    def test_torch_cuda_agrees_with_numpy(self, lqr):
        P_np, K_np = riccati_recursion(lqr["A"], lqr["B"], lqr["Q"], lqr["R"], HORIZON)
        tensors = {k: torch.as_tensor(v, device="cuda") for k, v in lqr.items()}
        P_t, K_t = riccati_recursion(
            tensors["A"], tensors["B"], tensors["Q"], tensors["R"], HORIZON
        )
        assert P_t.device.type == "cuda"
        np.testing.assert_allclose(P_t.cpu().numpy(), P_np, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(K_t.cpu().numpy(), K_np, atol=1e-9, rtol=1e-9)
