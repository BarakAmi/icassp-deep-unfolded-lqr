"""Phase 1 acceptance tests for `applications.uncertainty.noise`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.1/11): `GAUSSIAN`
reproduces `GaussianBatchSpec` bit-for-bit; every family's realized
covariance matches its declared standardization (Cauchy exempt, asserted
heavy-tailed instead); the `scale_multiplier` axis scales jointly.
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec
from mbl.applications.uncertainty.noise import ExoticBatchSpec, NoiseFamily
from mbl.core.runtime import Backend

_N = 4
_HORIZON = 20
_BATCH = 4096
_SEED = 0
_PROCESS_STD = 0.5
_INITIAL_STD = 1.0


class TestGaussianDegeneracy:
    @pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.TORCH])
    def test_reproduces_gaussian_batch_spec_bit_for_bit(self, backend: Backend) -> None:
        gaussian = GaussianBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=_BATCH,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
        )
        exotic = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=_BATCH,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=NoiseFamily.GAUSSIAN,
        )
        sample_gaussian, _ = gaussian.build(backend, torch_dtype=torch.float64)
        sample_exotic, _ = exotic.build(backend, torch_dtype=torch.float64)
        x0_g, w_g, v_g = sample_gaussian()
        x0_e, w_e, v_e = sample_exotic()
        if backend is Backend.NUMPY:
            np.testing.assert_array_equal(x0_g, x0_e)
            np.testing.assert_array_equal(w_g, w_e)
            np.testing.assert_array_equal(v_g, v_e)
        else:
            torch.testing.assert_close(x0_g, x0_e, rtol=0, atol=0)
            torch.testing.assert_close(w_g, w_e, rtol=0, atol=0)
            torch.testing.assert_close(v_g, v_e, rtol=0, atol=0)


class TestStandardization:
    """Every family's realized standard deviation matches the declared
    standardization rule (variance-matched for Uniform/Laplace), to Monte
    Carlo tolerance at a large batch size. Cauchy is exempted -- see
    `test_cauchy_is_heavy_tailed`."""

    @pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.TORCH])
    @pytest.mark.parametrize("family", [NoiseFamily.UNIFORM, NoiseFamily.LAPLACE])
    def test_realized_std_matches_declared(
        self, backend: Backend, family: NoiseFamily
    ) -> None:
        spec = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=200_000,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=family,
        )
        sample, _ = spec.build(backend, torch_dtype=torch.float64)
        _, w, _ = sample()
        w_np = w if backend is Backend.NUMPY else w.numpy()
        realized_std = float(np.std(w_np))
        assert realized_std == pytest.approx(_PROCESS_STD, rel=0.05)

    @pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.TORCH])
    def test_cauchy_is_heavy_tailed(self, backend: Backend) -> None:
        spec = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=100_000,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=NoiseFamily.CAUCHY,
        )
        sample, _ = spec.build(backend, torch_dtype=torch.float64)
        _, w, _ = sample()
        w_np = w if backend is Backend.NUMPY else w.numpy()
        # Heavy-tailed: sample variance is not a stable/small quantity, but
        # the median absolute value should sit near the declared scale
        # (Cauchy's median absolute deviation is exactly its scale parameter).
        median_abs = float(np.median(np.abs(w_np)))
        assert median_abs == pytest.approx(_PROCESS_STD, rel=0.1)
        # The realized max should be many multiples of the scale -- a finite-
        # variance family at this batch size would not produce this.
        assert float(np.max(np.abs(w_np))) > 20 * _PROCESS_STD

    def test_uniform_is_bounded(self) -> None:
        spec = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=100_000,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=NoiseFamily.UNIFORM,
        )
        sample, _ = spec.build(Backend.NUMPY)
        _, w, _ = sample()
        half_width = _PROCESS_STD * np.sqrt(3.0)
        assert np.max(np.abs(w)) <= half_width + 1e-9


class TestScaleMultiplier:
    def test_scales_process_and_initial_jointly(self) -> None:
        base = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=50_000,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=NoiseFamily.GAUSSIAN,
        )
        scaled = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=50_000,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            initial_state_std=_INITIAL_STD,
            family=NoiseFamily.GAUSSIAN,
            scale_multiplier=2.0,
        )
        sample_base, _ = base.build(Backend.NUMPY)
        sample_scaled, _ = scaled.build(Backend.NUMPY)
        x0_b, w_b, _ = sample_base()
        x0_s, w_s, _ = sample_scaled()
        assert np.std(w_s) == pytest.approx(2.0 * np.std(w_b), rel=0.05)
        assert np.std(x0_s) == pytest.approx(2.0 * np.std(x0_b), rel=0.05)

    def test_rejects_nonpositive_multiplier(self) -> None:
        with pytest.raises(ValueError, match="scale_multiplier"):
            ExoticBatchSpec(
                state_dim=_N,
                horizon=_HORIZON,
                batch_size=8,
                seed=_SEED,
                process_noise_std=_PROCESS_STD,
                scale_multiplier=0.0,
            )


class TestSignature:
    def test_get_signature_covers_fields(self) -> None:
        spec = ExoticBatchSpec(
            state_dim=_N,
            horizon=_HORIZON,
            batch_size=8,
            seed=_SEED,
            process_noise_std=_PROCESS_STD,
            family=NoiseFamily.LAPLACE,
        )
        signature = spec.get_signature()
        assert signature["family"] == NoiseFamily.LAPLACE
        assert signature["state_dim"] == _N
