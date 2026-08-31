"""Exotic (non-Gaussian) training/evaluation noise families
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.4/5.1): `ExoticBatchSpec`
mirrors `applications.factories.GaussianBatchSpec`'s `build`/`get_signature`
shape exactly, so it drops into every existing call site
(`ModelRecipe._make_sampler`, `EvaluationProtocol.build_batches`) without any
edit to either. Its ``family=NoiseFamily.GAUSSIAN`` case *delegates* to a
`GaussianBatchSpec` instance rather than re-implementing it, so the
degeneracy law (bit-for-bit reproduction at a given seed) holds by
construction, not by a parallel implementation that could drift.
"""

import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch

from ..factories import GaussianBatchSpec
from ...core.runtime import Backend
from ...engine.strategy import BatchSampler
from ...models.samplers import Distribution, TorchZeroDistribution, ZeroDistribution


class NoiseFamily(StrEnum):
    """The training/evaluation noise laws `ExoticBatchSpec` can draw from."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    LAPLACE = "laplace"
    CAUCHY = "cauchy"


def _uniform_half_width(std: float) -> float:
    """Variance-matched half-width of ``Uniform(-a, a)``: ``Var = a^2/3``."""
    return std * math.sqrt(3.0)


def _laplace_scale(std: float) -> float:
    """Variance-matched scale of ``Laplace(0, b)``: ``Var = 2b^2``."""
    return std / math.sqrt(2.0)


def _cauchy_scale(std: float) -> float:
    """Cauchy has no variance to match; `std` is used directly as the unit
    scale parameter gamma (NB07 plan Sec 4.4's declared standardization)."""
    return std


class NumpyUniformDistribution:
    """Zero-mean ``Uniform(-half_width, half_width)``, NumPy-backed."""

    def __init__(
        self, *, half_width: float, seed: int, rng: np.random.Generator | None = None
    ) -> None:
        self.half_width = half_width
        self._seed = seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self._rng.uniform(-self.half_width, self.half_width, size=shape)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Uniform", "seed": self._seed, "half_width": self.half_width}


class NumpyLaplaceDistribution:
    """Zero-mean ``Laplace(0, scale)``, NumPy-backed."""

    def __init__(
        self, *, scale: float, seed: int, rng: np.random.Generator | None = None
    ) -> None:
        self.scale = scale
        self._seed = seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self._rng.laplace(loc=0.0, scale=self.scale, size=shape)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Laplace", "seed": self._seed, "scale": self.scale}


class NumpyCauchyDistribution:
    """Zero-median ``Cauchy(0, scale)``, NumPy-backed. Has no finite mean or
    variance -- callers must use median/IQR statistics, never the sample
    mean, on data drawn from this distribution."""

    def __init__(
        self, *, scale: float, seed: int, rng: np.random.Generator | None = None
    ) -> None:
        self.scale = scale
        self._seed = seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self.scale * self._rng.standard_cauchy(size=shape)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Cauchy", "seed": self._seed, "scale": self.scale}


class TorchUniformDistribution:
    """Torch counterpart of `NumpyUniformDistribution`: drawn via
    `torch.rand` on an explicit `torch.Generator` (the same
    generator-controlled pattern `TorchGaussianDistribution` uses), so the
    bit-for-bit-identical-for-a-given-seed contract holds regardless of
    residency."""

    def __init__(
        self,
        *,
        half_width: float,
        seed: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.half_width = half_width
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._seed = seed
        self._generator = generator or torch.Generator().manual_seed(seed)

    def __call__(self, *shape: int) -> torch.Tensor:
        unit = torch.rand(
            shape,
            generator=self._generator,
            dtype=self.dtype,
            device=self._generator.device,
        )
        sample = (unit * 2 - 1) * self.half_width
        return sample.to(self.device)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Uniform", "seed": self._seed, "half_width": self.half_width}


class TorchLaplaceDistribution:
    """Torch counterpart of `NumpyLaplaceDistribution`, via the inverse-CDF
    transform of a `torch.rand` draw (so it stays generator-controlled)."""

    def __init__(
        self,
        *,
        scale: float,
        seed: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.scale = scale
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._seed = seed
        self._generator = generator or torch.Generator().manual_seed(seed)

    def __call__(self, *shape: int) -> torch.Tensor:
        centered = (
            torch.rand(
                shape,
                generator=self._generator,
                dtype=self.dtype,
                device=self._generator.device,
            )
            - 0.5
        )
        sample = -self.scale * torch.sign(centered) * torch.log1p(-2 * centered.abs())
        return sample.to(self.device)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Laplace", "seed": self._seed, "scale": self.scale}


class TorchCauchyDistribution:
    """Torch counterpart of `NumpyCauchyDistribution`, via the inverse-CDF
    transform ``scale * tan(pi * (U - 1/2))`` of a `torch.rand` draw. Has no
    finite mean or variance -- see `NumpyCauchyDistribution`."""

    def __init__(
        self,
        *,
        scale: float,
        seed: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.scale = scale
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._seed = seed
        self._generator = generator or torch.Generator().manual_seed(seed)

    def __call__(self, *shape: int) -> torch.Tensor:
        unit = torch.rand(
            shape,
            generator=self._generator,
            dtype=self.dtype,
            device=self._generator.device,
        )
        sample = self.scale * torch.tan(torch.pi * (unit - 0.5))
        return sample.to(self.device)

    def get_signature(self) -> dict[str, Any]:
        return {"type": "Cauchy", "seed": self._seed, "scale": self.scale}


@dataclass(frozen=True)
class ExoticBatchSpec:
    """One specification for seeded, possibly non-Gaussian training/
    evaluation batches -- structurally interchangeable with
    `GaussianBatchSpec` (same `build`/`get_signature` shape) wherever a
    `BatchSpec` is expected.

    Attributes:
        state_dim: State (and noise) dimension ``n``.
        horizon: Number of noise steps per trajectory.
        batch_size: Trajectories per batch.
        seed: Seed of the shared random stream.
        process_noise_std: Standard deviation (or, for `NoiseFamily.CAUCHY`,
            the unit scale parameter -- see `_cauchy_scale`) of the process
            noise, before `scale_multiplier`.
        initial_state_std: Same convention, for the initial state.
        family: Which noise law to draw from; `NoiseFamily.GAUSSIAN`
            reproduces `GaussianBatchSpec` bit-for-bit.
        scale_multiplier: Axis-S severity: scales both `process_noise_std`
            and `initial_state_std` jointly (NB07 plan Sec 4.4).
    """

    state_dim: int
    horizon: int
    batch_size: int
    seed: int
    process_noise_std: float
    initial_state_std: float = 1.0
    family: NoiseFamily = NoiseFamily.GAUSSIAN
    scale_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.scale_multiplier <= 0:
            raise ValueError(
                f"scale_multiplier must be positive, got {self.scale_multiplier}."
            )

    def build(
        self,
        backend: Backend,
        *,
        torch_dtype: torch.dtype = torch.float64,
        torch_device: torch.device | None = None,
        batch_size: int | None = None,
        horizon: int | None = None,
    ) -> tuple[BatchSampler, dict[str, Distribution]]:
        """Build the batch sampler for `backend` (see
        `GaussianBatchSpec.build` for the argument contract, mirrored
        exactly).

        Returns:
            ``(sampler, distributions)``.
        """
        scaled_process_std = self.process_noise_std * self.scale_multiplier
        scaled_initial_std = self.initial_state_std * self.scale_multiplier

        if self.family == NoiseFamily.GAUSSIAN:
            delegate = GaussianBatchSpec(
                state_dim=self.state_dim,
                horizon=self.horizon,
                batch_size=self.batch_size,
                seed=self.seed,
                process_noise_std=scaled_process_std,
                initial_state_std=scaled_initial_std,
            )
            return delegate.build(
                backend,
                torch_dtype=torch_dtype,
                torch_device=torch_device,
                batch_size=batch_size,
                horizon=horizon,
            )

        effective_batch = batch_size or self.batch_size
        effective_horizon = horizon or self.horizon
        if backend is Backend.NUMPY:
            distributions = self._numpy_distributions(
                scaled_initial_std, scaled_process_std
            )
        else:
            distributions = self._torch_distributions(
                scaled_initial_std, scaled_process_std, torch_dtype, torch_device
            )

        n = self.state_dim
        initial_state = distributions["initial_state"]
        process_noise = distributions["process_noise"]
        measurement_noise = distributions["measurement_noise"]

        def sample() -> tuple[Any, Any, Any]:
            x0 = initial_state(effective_batch, n)
            w = process_noise(effective_batch, effective_horizon, n)
            v = measurement_noise(effective_batch, effective_horizon, n)
            return x0, w, v

        return sample, distributions

    def _numpy_distributions(
        self, initial_std: float, process_std: float
    ) -> dict[str, Distribution]:
        rng = np.random.default_rng(self.seed)
        initial_state, process_noise = self._numpy_family(initial_std, process_std, rng)
        return {
            "initial_state": initial_state,
            "process_noise": process_noise,
            "measurement_noise": ZeroDistribution(),
        }

    def _numpy_family(
        self, initial_std: float, process_std: float, rng: np.random.Generator
    ) -> tuple[Distribution, Distribution]:
        if self.family == NoiseFamily.UNIFORM:
            return (
                NumpyUniformDistribution(
                    half_width=_uniform_half_width(initial_std), seed=self.seed, rng=rng
                ),
                NumpyUniformDistribution(
                    half_width=_uniform_half_width(process_std), seed=self.seed, rng=rng
                ),
            )
        if self.family == NoiseFamily.LAPLACE:
            return (
                NumpyLaplaceDistribution(
                    scale=_laplace_scale(initial_std), seed=self.seed, rng=rng
                ),
                NumpyLaplaceDistribution(
                    scale=_laplace_scale(process_std), seed=self.seed, rng=rng
                ),
            )
        if self.family == NoiseFamily.CAUCHY:
            return (
                NumpyCauchyDistribution(
                    scale=_cauchy_scale(initial_std), seed=self.seed, rng=rng
                ),
                NumpyCauchyDistribution(
                    scale=_cauchy_scale(process_std), seed=self.seed, rng=rng
                ),
            )
        raise ValueError(f"Unknown NoiseFamily {self.family!r}.")

    def _torch_distributions(
        self,
        initial_std: float,
        process_std: float,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> dict[str, Distribution]:
        generator = torch.Generator().manual_seed(self.seed)
        initial_state, process_noise = self._torch_family(
            initial_std, process_std, dtype, device, generator
        )
        return {
            "initial_state": initial_state,
            "process_noise": process_noise,
            "measurement_noise": TorchZeroDistribution(dtype=dtype, device=device),
        }

    def _torch_family(
        self,
        initial_std: float,
        process_std: float,
        dtype: torch.dtype,
        device: torch.device | None,
        generator: torch.Generator,
    ) -> tuple[Distribution, Distribution]:
        if self.family == NoiseFamily.UNIFORM:
            return (
                TorchUniformDistribution(
                    half_width=_uniform_half_width(initial_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
                TorchUniformDistribution(
                    half_width=_uniform_half_width(process_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
            )
        if self.family == NoiseFamily.LAPLACE:
            return (
                TorchLaplaceDistribution(
                    scale=_laplace_scale(initial_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
                TorchLaplaceDistribution(
                    scale=_laplace_scale(process_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
            )
        if self.family == NoiseFamily.CAUCHY:
            return (
                TorchCauchyDistribution(
                    scale=_cauchy_scale(initial_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
                TorchCauchyDistribution(
                    scale=_cauchy_scale(process_std),
                    seed=self.seed,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ),
            )
        raise ValueError(f"Unknown NoiseFamily {self.family!r}.")

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full sampling specification.

        Returns:
            ``{"type": "ExoticBatchSpec", **fields}``.
        """
        return {"type": type(self).__name__, **dataclasses.asdict(self)}
