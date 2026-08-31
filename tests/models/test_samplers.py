import numpy as np
import pytest
import torch

from mbl.models.samplers import (
    ConstantDistribution,
    NumpyGaussianDistribution,
    RandomSampler,
    TorchGaussianDistribution,
    TorchZeroDistribution,
    ZeroDistribution,
    gaussian_sampler,
)


# --- NumpyGaussianDistribution: behavior-preservation (bit-for-bit) ----------


def test_numpy_gaussian_distribution_matches_direct_rng_normal_call_bit_for_bit() -> (
    None
):
    """The wrap must be numerically invisible: identical seed -> identical
    bytes as calling rng.normal(...) directly (today's call sites)."""
    seed, shape = 42, (4, 3, 2)

    direct_rng = np.random.default_rng(seed)
    expected = direct_rng.normal(loc=0.0, scale=0.5, size=shape)

    dist = NumpyGaussianDistribution(std=0.5, seed=seed)
    actual = dist(*shape)

    assert np.array_equal(actual, expected)


def test_numpy_gaussian_distribution_shared_rng_advances_one_stream() -> None:
    """Two Distribution instances sharing one rng (as the batch-sampler
    refactor requires) must draw sequentially from the SAME stream, exactly
    like today's single-rng closures -- not two independent, correlated streams."""
    seed = 7
    shared_rng = np.random.default_rng(seed)
    initial_state_dist = NumpyGaussianDistribution(std=1.0, seed=seed, rng=shared_rng)
    process_noise_dist = NumpyGaussianDistribution(std=0.5, seed=seed, rng=shared_rng)

    x0 = initial_state_dist(3, 2)
    w = process_noise_dist(3, 4, 2)

    reference_rng = np.random.default_rng(seed)
    expected_x0 = reference_rng.normal(loc=0.0, scale=1.0, size=(3, 2))
    expected_w = reference_rng.normal(loc=0.0, scale=0.5, size=(3, 4, 2))

    assert np.array_equal(x0, expected_x0)
    assert np.array_equal(w, expected_w)


def test_numpy_gaussian_distribution_requires_exactly_one_of_std_or_covariance() -> (
    None
):
    with pytest.raises(ValueError, match="Exactly one"):
        NumpyGaussianDistribution(seed=0)
    with pytest.raises(ValueError, match="Exactly one"):
        NumpyGaussianDistribution(std=1.0, covariance=np.eye(2), seed=0)


def test_numpy_gaussian_distribution_full_covariance_path() -> None:
    seed = 0
    mean = np.zeros(2)
    covariance = np.eye(2) * 0.25
    dist = NumpyGaussianDistribution(mean=mean, covariance=covariance, seed=seed)
    sample = dist(10_000, 2)
    assert sample.shape == (10_000, 2)
    assert abs(sample.std() - 0.5) < 0.02


# --- NumpyGaussianDistribution: get_signature --------------------------------


def test_numpy_gaussian_distribution_signature_isotropic_case() -> None:
    signature = NumpyGaussianDistribution(std=0.5, seed=1).get_signature()
    assert signature == {"type": "Gaussian", "seed": 1, "mean": 0.0, "std": 0.5}


def test_numpy_gaussian_distribution_signature_covariance_case_hashes_matrices() -> (
    None
):
    signature = NumpyGaussianDistribution(
        mean=np.zeros(2), covariance=np.eye(2), seed=1
    ).get_signature()
    assert signature["type"] == "Gaussian"
    assert signature["mean_hash"].startswith("sha256:")
    assert signature["covariance_hash"].startswith("sha256:")


def test_numpy_gaussian_distribution_signatures_differ_for_different_std() -> None:
    sig1 = NumpyGaussianDistribution(std=0.5, seed=1).get_signature()
    sig2 = NumpyGaussianDistribution(std=0.7, seed=1).get_signature()
    assert sig1 != sig2


def test_numpy_gaussian_distribution_identical_params_identical_signature() -> None:
    sig1 = NumpyGaussianDistribution(std=0.5, seed=1).get_signature()
    sig2 = NumpyGaussianDistribution(std=0.5, seed=1).get_signature()
    assert sig1 == sig2


# --- TorchGaussianDistribution: behavior-preservation ------------------------


def test_torch_gaussian_distribution_matches_direct_torch_randn_call_bit_for_bit() -> (
    None
):
    seed, shape = 42, (4, 3, 2)
    dtype = torch.float64

    generator = torch.Generator().manual_seed(seed)
    expected = torch.randn(shape, generator=generator, dtype=dtype) * 0.5

    dist = TorchGaussianDistribution(std=0.5, seed=seed, dtype=dtype)
    actual = dist(*shape)

    assert torch.equal(actual, expected)


def test_torch_gaussian_distribution_shared_generator_advances_one_stream() -> None:
    seed, dtype = 7, torch.float64
    shared_generator = torch.Generator().manual_seed(seed)
    initial_state_dist = TorchGaussianDistribution(
        std=1.0, seed=seed, dtype=dtype, generator=shared_generator
    )
    process_noise_dist = TorchGaussianDistribution(
        std=0.5, seed=seed, dtype=dtype, generator=shared_generator
    )

    x0 = initial_state_dist(3, 2)
    w = process_noise_dist(3, 4, 2)

    reference_generator = torch.Generator().manual_seed(seed)
    expected_x0 = torch.randn((3, 2), generator=reference_generator, dtype=dtype) * 1.0
    expected_w = (
        torch.randn((3, 4, 2), generator=reference_generator, dtype=dtype) * 0.5
    )

    assert torch.equal(x0, expected_x0)
    assert torch.equal(w, expected_w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_torch_gaussian_distribution_on_cuda_matches_cpu_bit_for_bit() -> None:
    """Residency must never change numerics: a CUDA-placed draw, moved back
    to CPU, must equal the CPU-only draw exactly -- the RNG always runs on
    the (CPU) generator's own device; `device` only relocates the result."""
    seed, shape, dtype = 42, (4, 3, 2), torch.float64

    cpu_dist = TorchGaussianDistribution(std=0.5, seed=seed, dtype=dtype)
    cpu_sample = cpu_dist(*shape)

    cuda_dist = TorchGaussianDistribution(
        std=0.5, seed=seed, dtype=dtype, device=torch.device("cuda")
    )
    cuda_sample = cuda_dist(*shape)

    assert cuda_sample.device.type == "cuda"
    assert torch.equal(cuda_sample.cpu(), cpu_sample)


def test_torch_gaussian_distribution_signature_isotropic_case() -> None:
    signature = TorchGaussianDistribution(
        std=0.5, seed=1, dtype=torch.float64
    ).get_signature()
    assert signature == {"type": "Gaussian", "seed": 1, "mean": 0.0, "std": 0.5}


# --- ZeroDistribution ---------------------------------------------------------


def test_zero_distribution_returns_zeros_and_signs_as_zero() -> None:
    dist = ZeroDistribution()
    sample = dist(3, 4, 2)
    assert np.array_equal(sample, np.zeros((3, 4, 2)))
    assert dist.get_signature() == {"type": "Zero"}


def test_torch_zero_distribution_returns_zeros_and_signs_as_zero() -> None:
    dist = TorchZeroDistribution(dtype=torch.float64)
    sample = dist(3, 4, 2)
    assert torch.equal(sample, torch.zeros(3, 4, 2, dtype=torch.float64))
    assert dist.get_signature() == {"type": "Zero"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_torch_zero_distribution_honors_the_requested_device() -> None:
    dist = TorchZeroDistribution(dtype=torch.float64, device=torch.device("cuda"))
    sample = dist(3, 4, 2)
    assert sample.device.type == "cuda"


# --- ConstantDistribution (Phase 2.5: a fixed, non-zero deterministic point) -


def test_constant_distribution_broadcasts_a_scalar_to_the_requested_shape() -> None:
    dist = ConstantDistribution(1.5)
    sample = dist(3, 4)
    assert np.array_equal(sample, np.full((3, 4), 1.5))


def test_constant_distribution_broadcasts_a_per_dimension_vector() -> None:
    dist = ConstantDistribution(np.array([1.0, -0.5, 0.75]))
    sample = dist(2, 3)
    assert np.array_equal(sample, np.tile([1.0, -0.5, 0.75], (2, 1)))


def test_constant_distribution_returns_an_independent_copy_per_call() -> None:
    dist = ConstantDistribution(1.0)
    a = dist(2, 2)
    a[:] = 99.0
    b = dist(2, 2)
    assert np.array_equal(b, np.full((2, 2), 1.0))


def test_constant_distribution_get_signature_hashes_the_value() -> None:
    dist = ConstantDistribution(np.array([1.0, 2.0]))
    signature = dist.get_signature()
    assert signature["type"] == "Constant"
    assert signature["value_hash"].startswith("sha256:")


def test_constant_distribution_identical_values_have_identical_signatures() -> None:
    assert (
        ConstantDistribution(2.0).get_signature()
        == ConstantDistribution(2.0).get_signature()
    )


# --- RandomSampler retrofit ---------------------------------------------------


def test_random_sampler_get_signature_before_binding_reports_sampler_name_only() -> (
    None
):
    sampler = RandomSampler("process_noise", gaussian_sampler)
    assert sampler.get_signature() == {
        "type": "RandomSampler",
        "sampler": "gaussian_sampler",
    }


def test_random_sampler_get_signature_reflects_bound_kwargs() -> None:
    sampler = RandomSampler("process_noise", gaussian_sampler)
    sampler.unify_sampler(device=torch.device("cpu"), dtype=torch.float64)

    signature = sampler.get_signature()

    assert signature["sampler"] == "gaussian_sampler"
    assert signature["dtype"] == torch.float64


def test_random_sampler_get_signature_hashes_tensor_valued_bound_kwargs() -> None:
    sampler = RandomSampler("process_noise", gaussian_sampler)
    mu = torch.zeros(2, dtype=torch.float64)
    sampler.unify_sampler(mu=mu, device=torch.device("cpu"), dtype=torch.float64)

    signature = sampler.get_signature()
    assert signature["mu"].startswith("sha256:")  # hashed, not embedded raw


def test_random_sampler_still_callable_after_unify_sampler_unchanged_contract() -> None:
    """Backward-compatibility regression: RandomSampler's __call__ contract
    (used by SamplerInitializer) must be untouched by the get_signature retrofit."""
    sampler = RandomSampler("u0", gaussian_sampler)
    sampler.unify_sampler(device=torch.device("cpu"), dtype=torch.float64)
    out = sampler(5, 3)
    assert out.shape == (5, 3)
    assert out.dtype == torch.float64
