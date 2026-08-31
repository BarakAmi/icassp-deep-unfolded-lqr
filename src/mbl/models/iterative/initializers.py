"""Strategies for proposing the starting control u₀ that an `IterativeRefinement`
then refines -- shared, algorithm-agnostic vocabulary (Phase 2.5): consumed by
both `models.unfolded.UnfoldedController` (per-time-step refinement) and
`models.analytic.iterative_gd.AnalyticalIterativeGDController` (whole-horizon
macro-sweep refinement)."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import torch

from ..samplers import Distribution, RandomSampler, TorchGaussianDistribution


class ControlInitializer(ABC):
    """Produces the starting point u₀ for the iterative refinement at each time step."""

    @abstractmethod
    def __call__(
        self, t: int, y: torch.Tensor, prev_u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the initial control u_0 for time step t, given the current state y and previous control prev_u.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.
            prev_u: The previous time step's (already-refined) control, shape
                ``(batch, control_dim)``, or ``None`` at ``t == 0``.

        Returns:
            The initial control u_0, shape ``(batch, control_dim)``.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        ...

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: this initializer's own type (and any subclass-
        specific parameters, via an override) -- lets a solve's provenance
        record (e.g. `models.iterative.OptimizationResult.signature`) capture
        which initialization strategy produced a given run.

        Returns:
            ``{"type": <class name>}`` by default.
        """
        return {"type": type(self).__name__}


class ConstantInitializer(ControlInitializer):
    """Always initializes u_0 to the same constant value, ignoring `y`/`prev_u`."""

    def __init__(
        self,
        constant: float,
        control_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Args:
            constant: The constant value to fill every control entry with.
            control_dim: The control vector dimension.
            dtype: The output tensor's dtype.
            device: The output tensor's device.
        """
        self.constant = constant
        self.control_dim = control_dim
        self.dtype = dtype
        self.device = device

    def __call__(
        self, t: int, y: torch.Tensor, prev_u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            t: Unused (present to satisfy `ControlInitializer`).
            y: Used only for its batch size.
            prev_u: Unused (present to satisfy `ControlInitializer`).

        Returns:
            A tensor of shape ``(y.shape[0], control_dim)`` filled with
            `self.constant`.
        """
        return torch.full(
            (y.shape[0], self.control_dim),
            self.constant,
            dtype=self.dtype,
            device=self.device,
        )

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "ConstantInitializer", "constant":, "control_dim":,
            "dtype":}``.
        """
        return {
            "type": type(self).__name__,
            "constant": self.constant,
            "control_dim": self.control_dim,
            "dtype": str(self.dtype),
        }


class WarmStartInitializer(ControlInitializer):
    """Reuses prev_u at t > 0; delegates to a fallback at t = 0."""

    def __init__(self, fallback: ControlInitializer) -> None:
        """
        Args:
            fallback: The initializer used at ``t == 0`` (where `prev_u` is
                ``None``).
        """
        self.fallback = fallback

    def __call__(
        self, t: int, y: torch.Tensor, prev_u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            t: The current time step (forwarded to `fallback` at ``t == 0``).
            y: Current observation (forwarded to `fallback` at ``t == 0``).
            prev_u: The previous step's control, reused directly if not ``None``.

        Returns:
            `prev_u` if given; otherwise `self.fallback(t, y, None)`.
        """
        return prev_u if prev_u is not None else self.fallback(t, y, None)

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "WarmStartInitializer", "fallback": <fallback's
            signature>}``.
        """
        return {"type": type(self).__name__, "fallback": self.fallback.get_signature()}


class SamplerInitializer(ControlInitializer):
    """Initializes u_0 by drawing from a `RandomSampler` (e.g. Gaussian noise)."""

    def __init__(
        self,
        sampler: RandomSampler | Distribution,
        control_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        Args:
            sampler: The sampler to draw u_0 from, called as
                ``sampler(batch_size, control_dim)`` -- any callable
                signal generator with a ``get_signature`` (`RandomSampler`,
                or a seeded `Distribution` such as `TorchGaussianDistribution`).
            control_dim: The control vector dimension.
            dtype: The output tensor's dtype.
            device: The output tensor's device.
        """
        self.sampler = sampler
        self.control_dim, self.dtype, self.device = control_dim, dtype, device

    def __call__(
        self, t: int, y: torch.Tensor, prev_u: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            t: Unused (present to satisfy `ControlInitializer`).
            y: Used only for its batch size.
            prev_u: Unused (present to satisfy `ControlInitializer`).

        Returns:
            A sampled tensor of shape ``(y.shape[0], control_dim)``.
        """
        return torch.as_tensor(
            self.sampler(y.shape[0], self.control_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "SamplerInitializer", "sampler": <sampler's
            signature>, "control_dim":}``.
        """
        return {
            "type": type(self).__name__,
            "sampler": self.sampler.get_signature(),
            "control_dim": self.control_dim,
        }


class ControlInitMethod(StrEnum):
    """How an iterative controller seeds ``u^(0)`` before its refinement
    iterations -- the user-selectable initialization knob (relevant to every
    iterative/gradient-based controller). `build_control_initializer` maps each
    member to a concrete `ControlInitializer`."""

    #: Constant zero -- ``u^(0) = 0`` at every time step (the classical, and
    #: historically the only, deep-unfolding cold start).
    COLD = "cold"
    #: Reuse the previous time step's already-refined control as this step's
    #: starting point (`WarmStartInitializer`); falls back to the cold start at
    #: ``t == 0``, where there is no previous control.
    WARM = "warm"
    #: A seeded, device-native Gaussian draw ``u^(0) ~ N(0, random_std^2 I)``
    #: (`SamplerInitializer` over `TorchGaussianDistribution`) -- reproducible
    #: from `seed` so a run stays cacheable and every contender is compared on
    #: the same realization (Stochastic Fairness & Determinism doctrine).
    RANDOMIZED = "randomized"


def build_control_initializer(
    method: ControlInitMethod,
    *,
    control_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    random_std: float = 1.0,
    seed: int | None = None,
) -> ControlInitializer:
    """Construct the concrete `ControlInitializer` a `ControlInitMethod`
    names -- the single mapping site, so a controller-build path selects an
    initialization strategy by enum rather than hard-coding one class.

    Args:
        method: The initialization strategy to build.
        control_dim: The control vector dimension ``m``.
        dtype: The initial control's dtype.
        device: The initial control's device.
        random_std: Standard deviation of the Gaussian draw for
            `ControlInitMethod.RANDOMIZED`; ignored otherwise.
        seed: Seed for the reproducible, device-native generator used by
            `ControlInitMethod.RANDOMIZED`; ignored otherwise. **`None` means
            inherit the ambient global torch RNG**, which is how a training
            replicate reaches this draw (Annex 01 §2.3.3): the producer seeds
            that stream per replicate, so an unpinned initializer redraws with
            it while a declared seed pins it across every replicate. The
            inherited value is drawn once, here, so the distribution below
            still owns a reproducible generator and still signs the seed it
            actually used.

    Returns:
        The constructed `ControlInitializer`.

    Raises:
        ValueError: If `method` is not a known `ControlInitMethod`.
    """
    cold = ConstantInitializer(0.0, control_dim, dtype, device)
    if method is ControlInitMethod.COLD:
        return cold
    if method is ControlInitMethod.WARM:
        return WarmStartInitializer(cold)
    if method is ControlInitMethod.RANDOMIZED:
        return SamplerInitializer(
            TorchGaussianDistribution(
                std=random_std,
                seed=_inherited_seed() if seed is None else seed,
                dtype=dtype,
                device=device,
            ),
            control_dim,
            dtype,
            device,
        )
    raise ValueError(f"Unknown control-init method {method!r}.")


def _inherited_seed() -> int:
    """One seed drawn from the ambient global torch stream.

    Drawn rather than passed through as `None`, because `TorchGaussianDistribution`
    owns a dedicated `torch.Generator` -- the property that makes a draw
    bit-identical for a given seed regardless of device. Reading the ambient
    stream once and handing over the value keeps that contract intact while
    still inheriting whatever the caller seeded the global RNG with.
    """
    return int(torch.randint(0, 2**62, (1,)).item())
