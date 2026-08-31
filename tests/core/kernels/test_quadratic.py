"""Unit + cross-backend equivalence tests for the dual-backend quadratic
kernel (T2.d/T2.e/T1.e, Stage S2; REFACTOR_PLAN v3 §7.9 first slice).

The kernel's two contracts under test:

1. **Bit-stability heritage.** Each entry point reproduces, operation for
   operation, the pre-S2 implementation it consolidated — asserted with
   *exact* equality against inline replicas of the legacy op sequences.
2. **Backend equivalence.** NumPy and torch paths agree at
   precision-appropriate tolerance on every flag combination, on every
   host (CUDA parameterizations skip, never fail, when absent).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mbl.core.kernels import (
    CostConventions,
    CostReduction,
    cumulative_quadratic_cost,
    quadratic_form,
    time_invariant_slice,
    total_quadratic_cost,
)

FLAG_COMBOS = [(False, True), (True, True), (True, False), (False, False)]

BATCH, HORIZON, N_DIM, M_DIM = 5, 7, 3, 2


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    """One seeded float64 problem instance shared by every test."""
    rng = np.random.default_rng(42)
    return {
        "Q": np.repeat((2.0 * np.eye(N_DIM))[None], HORIZON + 1, axis=0),
        "R": np.repeat((0.5 * np.eye(M_DIM))[None], HORIZON, axis=0),
        "X": rng.normal(size=(BATCH, HORIZON + 1, N_DIM)),
        "U": rng.normal(size=(BATCH, HORIZON, M_DIM)),
    }


def _as_torch(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in data.items()}


def _legacy_curve(Q, R, X, U, include_terminal_cost, is_time_averaged):
    """Inline replica of QuadraticCost.__call__'s pre-S2 NumPy body."""
    cost_X = np.einsum("btn,tnk,btk->bt", X, Q, X)
    cost_U = np.einsum("btn,tnk,btk->bt", U, R, U)
    J_cum = np.cumsum(cost_X[:, :-1] + cost_U, axis=1)
    if include_terminal_cost:
        J_cum += cost_X[:, 1:].copy()
    if is_time_averaged:
        J_cum = J_cum / np.arange(1, X.shape[1])
    return J_cum


class TestCumulativeCurve:
    @pytest.mark.parametrize(("terminal", "averaged"), FLAG_COMBOS)
    def test_numpy_path_is_bit_identical_to_the_legacy_sequence(
        self, data, terminal, averaged
    ):
        fresh = cumulative_quadratic_cost(
            data["Q"],
            data["R"],
            data["X"],
            data["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
        )
        legacy = _legacy_curve(
            data["Q"], data["R"], data["X"], data["U"], terminal, averaged
        )
        assert fresh.dtype == legacy.dtype
        assert np.array_equal(fresh, legacy)

    @pytest.mark.parametrize(("terminal", "averaged"), FLAG_COMBOS)
    def test_torch_path_matches_numpy_path(self, data, terminal, averaged):
        """§7.9: cross-backend equivalence at float64 tolerance."""
        numpy_curve = cumulative_quadratic_cost(
            data["Q"],
            data["R"],
            data["X"],
            data["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
        )
        tensors = _as_torch(data)
        torch_curve = cumulative_quadratic_cost(
            tensors["Q"],
            tensors["R"],
            tensors["X"],
            tensors["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
        )
        np.testing.assert_allclose(
            torch_curve.numpy(), numpy_curve, atol=1e-12, rtol=1e-12
        )


class TestTotalCost:
    @pytest.mark.parametrize(("terminal", "averaged"), FLAG_COMBOS)
    def test_per_sample_reproduces_the_legacy_evaluator_sequence(
        self, data, terminal, averaged
    ):
        """The exact torch op order `_build_cost_evaluator` always used —
        the frozen GD-channel `J_history` goldens ride it."""
        tensors = _as_torch(data)
        Q_2d, R_2d = tensors["Q"][0], tensors["R"][0]
        X, U = tensors["X"], tensors["U"]

        fresh = total_quadratic_cost(
            Q_2d,
            R_2d,
            X,
            U,
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
            reduction=CostReduction.PER_SAMPLE,
        )

        legacy = torch.einsum("bti,ij,btj->bt", X[:, :-1], Q_2d, X[:, :-1]).sum(
            dim=1
        ) + torch.einsum("bti,ij,btj->bt", U, R_2d, U).sum(dim=1)
        if terminal:
            legacy = legacy + torch.einsum("bi,ij,bj->b", X[:, -1], Q_2d, X[:, -1])
        if averaged:
            legacy = legacy / U.shape[1]

        assert fresh.shape == (BATCH,)
        assert torch.equal(fresh, legacy)

    def test_batch_mean_reproduces_the_legacy_training_loss_bit_for_bit(self, data):
        """Under the default flags (False, True), BATCH_MEAN is the exact
        pre-S2 RolloutModel op order — the frozen `J_opt` golden rides it."""
        tensors = _as_torch(data)
        Q_2d, R_2d = tensors["Q"][0], tensors["R"][0]
        X, U = tensors["X"], tensors["U"]

        fresh = total_quadratic_cost(
            Q_2d,
            R_2d,
            X,
            U,
            conventions=CostConventions(
                include_terminal_cost=False, is_time_averaged=True
            ),
            reduction=CostReduction.BATCH_MEAN,
        )
        legacy = (
            torch.einsum("bti,ij,btj->bt", X[:, :-1], Q_2d, X[:, :-1]).mean()
            + torch.einsum("bti,ij,btj->bt", U, R_2d, U).mean()
        )
        assert torch.equal(fresh, legacy)

    @pytest.mark.parametrize(("terminal", "averaged"), FLAG_COMBOS)
    def test_the_two_reductions_agree_on_the_objective(self, data, terminal, averaged):
        """PER_SAMPLE.mean() and BATCH_MEAN are the same mathematical
        objective in two named association orders."""
        tensors = _as_torch(data)
        per_sample = total_quadratic_cost(
            tensors["Q"],
            tensors["R"],
            tensors["X"],
            tensors["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
            reduction=CostReduction.PER_SAMPLE,
        )
        batch_mean = total_quadratic_cost(
            tensors["Q"],
            tensors["R"],
            tensors["X"],
            tensors["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
            reduction=CostReduction.BATCH_MEAN,
        )
        torch.testing.assert_close(
            batch_mean, per_sample.mean(), atol=1e-12, rtol=1e-12
        )

    @pytest.mark.parametrize(("terminal", "averaged"), FLAG_COMBOS)
    def test_total_agrees_with_the_curve_final_entry(self, data, terminal, averaged):
        """The scalar convention IS the curve's final cumulative entry (the
        C2 acceptance relation), across both flags and both backends."""
        curve = cumulative_quadratic_cost(
            data["Q"],
            data["R"],
            data["X"],
            data["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
        )
        per_sample = total_quadratic_cost(
            data["Q"],
            data["R"],
            data["X"],
            data["U"],
            conventions=CostConventions(
                include_terminal_cost=terminal, is_time_averaged=averaged
            ),
            reduction=CostReduction.PER_SAMPLE,
        )
        np.testing.assert_allclose(per_sample, curve[:, -1], atol=1e-12, rtol=1e-12)

    def test_flags_genuinely_change_the_value(self, data):
        """Flag-honoring is observable: all four combinations produce four
        distinct totals on a generic instance."""
        values = {
            (terminal, averaged): float(
                total_quadratic_cost(
                    data["Q"],
                    data["R"],
                    data["X"],
                    data["U"],
                    conventions=CostConventions(
                        include_terminal_cost=terminal, is_time_averaged=averaged
                    ),
                ).mean()
            )
            for terminal, averaged in FLAG_COMBOS
        }
        assert len(set(values.values())) == len(FLAG_COMBOS)

    def test_batch_mean_stays_differentiable(self, data):
        """The training-loss path keeps the autograd graph alive."""
        tensors = _as_torch(data)
        U = tensors["U"].clone().requires_grad_(True)
        loss = total_quadratic_cost(
            tensors["Q"],
            tensors["R"],
            tensors["X"],
            U,
            conventions=CostConventions(
                include_terminal_cost=True, is_time_averaged=True
            ),
            reduction=CostReduction.BATCH_MEAN,
        )
        loss.backward()
        assert U.grad is not None
        assert torch.isfinite(U.grad).all()


class TestPrimitives:
    def test_quadratic_form_merges_both_conventions(self, data):
        """2D (broadcast) and 3D (time-stacked) mats agree when the stack is
        constant — on both backends."""
        stacked = quadratic_form(data["Q"], data["X"])
        invariant = quadratic_form(data["Q"][0], data["X"])
        assert np.allclose(stacked, invariant)

        tensors = _as_torch(data)
        stacked_t = quadratic_form(tensors["Q"], tensors["X"])
        np.testing.assert_allclose(stacked_t.numpy(), stacked, atol=1e-12, rtol=0)

    def test_time_invariant_slice(self, data):
        assert time_invariant_slice(data["Q"]) is not data["Q"]
        assert np.array_equal(time_invariant_slice(data["Q"]), data["Q"][0])
        Q_2d = data["Q"][0]
        assert time_invariant_slice(Q_2d) is Q_2d

    def test_mixed_backends_are_refused_at_the_dispatch_site(self, data):
        """T1.f: an implicit NumPy<->torch conversion inside a kernel is a
        loud TypeError, never a silent materialization."""
        with pytest.raises(TypeError, match="boundary"):
            quadratic_form(data["Q"], torch.as_tensor(data["X"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host has no CUDA")
class TestCudaEquivalence:
    """§7.9: the torch/GPU leg, parameterized over host availability —
    skips (never fails) on GPU-less machines."""

    def test_total_cost_matches_cpu(self, data):
        tensors = {k: torch.as_tensor(v, device="cuda") for k, v in data.items()}
        gpu = total_quadratic_cost(
            tensors["Q"],
            tensors["R"],
            tensors["X"],
            tensors["U"],
            conventions=CostConventions(
                include_terminal_cost=True, is_time_averaged=True
            ),
        )
        cpu = total_quadratic_cost(
            data["Q"],
            data["R"],
            data["X"],
            data["U"],
            conventions=CostConventions(
                include_terminal_cost=True, is_time_averaged=True
            ),
        )
        np.testing.assert_allclose(gpu.cpu().numpy(), cpu, atol=1e-10, rtol=1e-10)
