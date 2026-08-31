"""Exotic (non-Gaussian) evaluation-noise families for zero-shot OOD
generalization (docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.4/3.1):
`ExoticBatchSpec` mirrors `applications.factories.GaussianBatchSpec`'s own
``build``/``get_signature`` contract so it drops into the same call sites,
with one behavioral law:

**The `GAUSSIAN` family reproduces `GaussianBatchSpec` bit-for-bit** (the
degeneracy law) -- it is implemented by literally delegating to
`GaussianBatchSpec`, never by re-deriving the same arithmetic a second time
(numpy's and torch's Gaussian generators are different PRNG algorithms
entirely, so an independent reimplementation could never match bit-for-bit
even with "the same seed"). The other four families (`UNIFORM`, `LAPLACE`,
`STUDENT_T`, `CAUCHY`) have no such pre-existing contract to match, and are
drawn via `numpy.random.Generator` (which ships correct, well-tested
`uniform`/`laplace`/`standard_t`/`standard_cauchy` samplers) then converted
to a torch tensor at the boundary -- the `Distribution` Protocol's return
type is ``np.ndarray | torch.Tensor`` either way, and every consumer
(`OptimalControlProblem.system.run`) needs a torch tensor on the torch
substrate this project's evaluation batches always use.

Standardization convention (deliberately not neutral -- stated here and
repeated on every figure that uses it): every finite-variance family is
**variance-matched** to the nominal Gaussian's `std` (`family_scale_parameter`);
`CAUCHY` has no variance to match, so its own scale parameter is set equal to
`std` as a nominal label only, not a true standard deviation.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch

from ..factories import GaussianBatchSpec
from ...core.runtime import Backend
from ...engine.strategy import BatchSampler
from ...models.samplers import Distribution, TorchZeroDistribution

#: Student-t's variance is only finite for df > 2 -- a genuine mathematical
#: precondition of `family_scale_parameter`'s variance-matching formula, not
#: a "can't happen" input (the notebook's own Control Panel exposes `df`).
_STUDENT_T_MIN_DF = 2.0


class NoiseFamily(StrEnum):
    """Which noise distribution `ExoticBatchSpec` draws (NB06 plan Sec 2.4)."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    LAPLACE = "laplace"
    STUDENT_T = "student_t"
    CAUCHY = "cauchy"


def family_scale_parameter(family: NoiseFamily, std: float, df: float) -> float:
    """This family's own native scale parameter, standardized against the
    nominal Gaussian's `std` (module docstring's convention).

    Args:
        family: Which family to compute the parameter for.
        std: The nominal Gaussian standard deviation every family is
            standardized against.
        df: `STUDENT_T`'s degrees of freedom; ignored for every other family.

    Returns:
        The family's native parameter: `std` itself for `GAUSSIAN`; the
        half-width ``a`` of ``Uniform(-a, a)`` for `UNIFORM` (variance
        ``a^2/3``); the scale ``b`` of `LAPLACE` (variance ``2*b^2``); the
        multiplier ``c`` such that ``c * StudentT(df)`` has variance
        ``std^2`` for `STUDENT_T` (variance ``df/(df-2)`` unscaled); `std`
        itself (a nominal label, not a true std) for `CAUCHY`.

    Raises:
        ValueError: If `family` is `STUDENT_T` and `df <= 2` (the variance
            used for matching is undefined), or `family` is not a known
            member.
    """
    if family is NoiseFamily.GAUSSIAN:
        return std
    if family is NoiseFamily.UNIFORM:
        return std * math.sqrt(3.0)
    if family is NoiseFamily.LAPLACE:
        return std / math.sqrt(2.0)
    if family is NoiseFamily.STUDENT_T:
        if df <= _STUDENT_T_MIN_DF:
            raise ValueError(
                f"Student-t variance-matching requires df > {_STUDENT_T_MIN_DF}, got {df}."
            )
        return std * math.sqrt((df - 2.0) / df)
    if family is NoiseFamily.CAUCHY:
        return std
    raise ValueError(f"Unknown noise family: {family!r}.")


class _ExoticNumpyDistribution:
    """A non-`GAUSSIAN` family, drawn via a shared `numpy.random.Generator`
    and converted to a torch tensor at `__call__` (module docstring)."""

    def __init__(
        self,
        *,
        family: NoiseFamily,
        scale: float,
        df: float,
        rng: np.random.Generator,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> None:
        self._family = family
        self._scale = scale
        self._df = df
        self._rng = rng
        self._dtype = dtype
        self._device = device or torch.device("cpu")

    def __call__(self, *shape: int) -> torch.Tensor:
        """See `models.samplers.Distribution.__call__`."""
        array = self._draw(shape)
        return torch.as_tensor(array, dtype=self._dtype, device=self._device)

    def _draw(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._family is NoiseFamily.UNIFORM:
            return self._rng.uniform(-self._scale, self._scale, size=shape)
        if self._family is NoiseFamily.LAPLACE:
            return self._rng.laplace(loc=0.0, scale=self._scale, size=shape)
        if self._family is NoiseFamily.STUDENT_T:
            return self._scale * self._rng.standard_t(self._df, size=shape)
        if self._family is NoiseFamily.CAUCHY:
            return self._scale * self._rng.standard_cauchy(size=shape)
        raise ValueError(f"{self._family!r} is not a numpy-drawn exotic family.")

    def get_signature(self) -> dict[str, Any]:
        """See `models.samplers.Distribution.get_signature`.

        Returns:
            ``{"type": "Exotic", "family":, "scale":, "df": (STUDENT_T
            only, else None)}``.
        """
        return {
            "type": "Exotic",
            "family": self._family.value,
            "scale": self._scale,
            "df": self._df if self._family is NoiseFamily.STUDENT_T else None,
        }


@dataclass(frozen=True)
class ExoticBatchSpec:
    """`GaussianBatchSpec`-compatible evaluation-batch specification over any
    `NoiseFamily`, with a per-axis scale multiplier (NB06 Axis S folds in
    here) applied to both the process noise and the initial-state spread.

    Attributes:
        state_dim: State (and noise) dimension ``n``.
        horizon: Number of noise steps per trajectory.
        batch_size: Trajectories per batch.
        seed: Seed of the shared random stream.
        process_noise_std: Nominal process-noise standard deviation (before
            `scale_multiplier`).
        initial_state_std: Nominal initial-state standard deviation (before
            `scale_multiplier`).
        family: Which noise distribution to draw; defaults to `GAUSSIAN`
            (the degeneracy anchor).
        scale_multiplier: Multiplies both `process_noise_std` and
            `initial_state_std` (NB06 Axis S); ``1.0`` leaves them nominal.
        student_t_df: `STUDENT_T`'s degrees of freedom; ignored otherwise.
    """

    state_dim: int
    horizon: int
    batch_size: int
    seed: int
    process_noise_std: float
    initial_state_std: float = 1.0
    family: NoiseFamily = NoiseFamily.GAUSSIAN
    scale_multiplier: float = 1.0
    student_t_df: float = 3.0

    def __post_init__(self) -> None:
        """Fail-fast on a Student-t degrees-of-freedom that makes
        `family_scale_parameter`'s variance-matching undefined.

        Raises:
            ValueError: If `family` is `STUDENT_T` and `student_t_df <= 2`.
        """
        if (
            self.family is NoiseFamily.STUDENT_T
            and self.student_t_df <= _STUDENT_T_MIN_DF
        ):
            raise ValueError(
                "ExoticBatchSpec(family=STUDENT_T) requires student_t_df > "
                f"{_STUDENT_T_MIN_DF}, got {self.student_t_df}."
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
        """Build the batch sampler for `backend`. See
        `applications.factories.GaussianBatchSpec.build` for the shared
        contract this mirrors.

        Args:
            backend: Which array backend the batches are authored on;
                exotic (non-`GAUSSIAN`) families support `Backend.TORCH`
                only (every consumer in this project rolls out on the torch
                substrate; see module docstring).
            torch_dtype: dtype of the returned tensors.
            torch_device: Residency of the returned tensors; defaults to CPU.
            batch_size: Optional override of `self.batch_size`.
            horizon: Optional override of `self.horizon`.

        Returns:
            ``(sampler, distributions)`` -- see `GaussianBatchSpec.build`.

        Raises:
            NotImplementedError: If `family` is not `GAUSSIAN` and `backend`
                is not `Backend.TORCH`.
        """
        scaled_process_std = self.process_noise_std * self.scale_multiplier
        scaled_initial_std = self.initial_state_std * self.scale_multiplier

        if self.family is NoiseFamily.GAUSSIAN:
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

        if backend is not Backend.TORCH:
            raise NotImplementedError(
                f"Exotic noise family {self.family!r} is implemented for "
                "Backend.TORCH only."
            )
        effective_batch = batch_size or self.batch_size
        effective_horizon = horizon or self.horizon
        rng = np.random.default_rng(self.seed)
        initial_scale = family_scale_parameter(
            self.family, scaled_initial_std, self.student_t_df
        )
        process_scale = family_scale_parameter(
            self.family, scaled_process_std, self.student_t_df
        )
        distributions: dict[str, Distribution] = {
            "initial_state": _ExoticNumpyDistribution(
                family=self.family,
                scale=initial_scale,
                df=self.student_t_df,
                rng=rng,
                dtype=torch_dtype,
                device=torch_device,
            ),
            "process_noise": _ExoticNumpyDistribution(
                family=self.family,
                scale=process_scale,
                df=self.student_t_df,
                rng=rng,
                dtype=torch_dtype,
                device=torch_device,
            ),
            "measurement_noise": TorchZeroDistribution(
                dtype=torch_dtype, device=torch_device
            ),
        }
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

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full sampling specification.

        Returns:
            ``{"type": "ExoticBatchSpec", **fields}`` (``family`` reports as
            its plain string value).
        """
        return {
            "type": type(self).__name__,
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "process_noise_std": self.process_noise_std,
            "initial_state_std": self.initial_state_std,
            "family": self.family.value,
            "scale_multiplier": self.scale_multiplier,
            "student_t_df": self.student_t_df,
        }
