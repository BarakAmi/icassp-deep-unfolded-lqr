"""Phase 1 acceptance tests for `applications.ood.noise`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.4/3.1): the degeneracy
law (`NoiseFamily.GAUSSIAN` must reproduce `GaussianBatchSpec` bit-for-bit),
each exotic family's realized sample statistics matching its declared
standardization (Cauchy exempt -- asserted heavy-tailed instead, since it
has no finite variance to match), and the scale-multiplier's role as Axis S.
"""

import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec
from mbl.applications.ood.noise import (
    ExoticBatchSpec,
    NoiseFamily,
    family_scale_parameter,
)
from mbl.core.runtime import Backend

_LARGE_BATCH = 200_000


def test_gaussian_family_reproduces_gaussian_batch_spec_bit_for_bit() -> None:
    exotic = ExoticBatchSpec(
        state_dim=3,
        horizon=5,
        batch_size=8,
        seed=11,
        process_noise_std=0.7,
        initial_state_std=1.3,
        family=NoiseFamily.GAUSSIAN,
    )
    nominal = GaussianBatchSpec(
        state_dim=3,
        horizon=5,
        batch_size=8,
        seed=11,
        process_noise_std=0.7,
        initial_state_std=1.3,
    )
    exotic_sampler, _ = exotic.build(Backend.TORCH, torch_dtype=torch.float64)
    nominal_sampler, _ = nominal.build(Backend.TORCH, torch_dtype=torch.float64)

    x0_e, w_e, v_e = exotic_sampler()
    x0_n, w_n, v_n = nominal_sampler()

    assert torch.equal(x0_e, x0_n)
    assert torch.equal(w_e, w_n)
    assert torch.equal(v_e, v_n)


def test_gaussian_family_with_scale_multiplier_matches_a_rescaled_gaussian_spec() -> (
    None
):
    exotic = ExoticBatchSpec(
        state_dim=2,
        horizon=4,
        batch_size=8,
        seed=3,
        process_noise_std=0.5,
        initial_state_std=1.0,
        family=NoiseFamily.GAUSSIAN,
        scale_multiplier=2.0,
    )
    nominal = GaussianBatchSpec(
        state_dim=2,
        horizon=4,
        batch_size=8,
        seed=3,
        process_noise_std=1.0,
        initial_state_std=2.0,
    )
    exotic_sampler, _ = exotic.build(Backend.TORCH, torch_dtype=torch.float64)
    nominal_sampler, _ = nominal.build(Backend.TORCH, torch_dtype=torch.float64)

    x0_e, w_e, _ = exotic_sampler()
    x0_n, w_n, _ = nominal_sampler()
    assert torch.equal(x0_e, x0_n)
    assert torch.equal(w_e, w_n)


@pytest.mark.parametrize(
    "family", [NoiseFamily.UNIFORM, NoiseFamily.LAPLACE, NoiseFamily.STUDENT_T]
)
def test_finite_variance_families_match_the_declared_std(family: NoiseFamily) -> None:
    std = 0.6
    spec = ExoticBatchSpec(
        state_dim=1,
        horizon=1,
        batch_size=_LARGE_BATCH,
        seed=5,
        process_noise_std=std,
        family=family,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    _, w, _ = sampler()
    realized_std = float(w.std())
    assert realized_std == pytest.approx(std, rel=0.02)


def test_cauchy_is_heavy_tailed_not_matched_to_a_finite_std() -> None:
    # Cauchy has no finite variance; instead of asserting a std match,
    # assert the realized sample is heavy-tailed: a much larger fraction of
    # draws exceed 5 nominal-sigma than a Gaussian would produce.
    std = 0.4
    spec = ExoticBatchSpec(
        state_dim=1,
        horizon=1,
        batch_size=_LARGE_BATCH,
        seed=5,
        process_noise_std=std,
        family=NoiseFamily.CAUCHY,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    _, w, _ = sampler()
    exceed_5_sigma = float((w.abs() > 5 * std).float().mean())
    # A Gaussian would put ~5.7e-7 of its mass beyond 5 sigma; Cauchy puts
    # far more (~1/(pi*5) ~= 0.06 beyond 5 scale-widths one-sided, doubled).
    assert exceed_5_sigma > 0.01


def test_reproducible_given_the_same_seed() -> None:
    spec = ExoticBatchSpec(
        state_dim=2,
        horizon=3,
        batch_size=16,
        seed=42,
        process_noise_std=0.3,
        family=NoiseFamily.LAPLACE,
    )
    sampler_a, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    sampler_b, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    x0_a, w_a, _ = sampler_a()
    x0_b, w_b, _ = sampler_b()
    assert torch.equal(x0_a, x0_b)
    assert torch.equal(w_a, w_b)


def test_non_torch_backend_raises_for_exotic_families() -> None:
    spec = ExoticBatchSpec(
        state_dim=2,
        horizon=3,
        batch_size=4,
        seed=0,
        process_noise_std=0.3,
        family=NoiseFamily.CAUCHY,
    )
    with pytest.raises(NotImplementedError):
        spec.build(Backend.NUMPY)


def test_student_t_df_at_or_below_two_raises() -> None:
    with pytest.raises(ValueError):
        ExoticBatchSpec(
            state_dim=1,
            horizon=1,
            batch_size=4,
            seed=0,
            process_noise_std=0.3,
            family=NoiseFamily.STUDENT_T,
            student_t_df=2.0,
        )
    with pytest.raises(ValueError):
        family_scale_parameter(NoiseFamily.STUDENT_T, std=1.0, df=1.5)


def test_get_signature_reports_family_as_plain_string() -> None:
    spec = ExoticBatchSpec(
        state_dim=2,
        horizon=3,
        batch_size=4,
        seed=0,
        process_noise_std=0.3,
        family=NoiseFamily.LAPLACE,
    )
    signature = spec.get_signature()
    assert signature["family"] == "laplace"
    assert isinstance(signature["family"], str)
