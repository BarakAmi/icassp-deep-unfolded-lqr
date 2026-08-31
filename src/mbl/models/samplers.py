"""Noise samplers for Monte-Carlo rollouts (process/measurement noise), and a
`functools.partial`-based helper to fix a sampler's non-shape parameters
(device/dtype/distribution params) so it reduces to a plain shape-only callable.

Also defines `Distribution` (Phase 1D): a named, parameterized signal
generator that both draws samples AND describes itself via `get_signature()`
-- the single-source-of-truth object replacing any parallel "shadow" spec.
`NumpyGaussianDistribution`/`TorchGaussianDistribution`/`ZeroDistribution` wrap
today's exact sampling arithmetic (bit-for-bit identical output for a given
seed) so refactoring a call site to use them changes nothing numerically.
"""

from typing import Any, Optional, Protocol, Callable, cast, runtime_checkable
from functools import partial

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from ..core.utils.signing import hash_array


type Sampler[*T] = Callable[
    [tuple[int, ...], torch.device, torch.dtype, *T], torch.Tensor
]
"""A noise-generating function: ``(*shape, device, dtype, *extra) -> Tensor``."""

type UnifiedSampler = Callable[[tuple[int, ...]], torch.Tensor]
"""A `Sampler` with every parameter except `shape` already bound (see `unify_sampler`)."""


def zero_sampler(*shape: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate zero noise of shape (batch_size, horizon, state_dim).

    Args:
        *shape: The desired output shape, e.g. ``(batch_size, horizon, state_dim)``.
        device: The device to allocate the tensor on.
        dtype: The tensor's dtype.

    Returns:
        A zero tensor of shape `shape`.
    """
    return torch.zeros(shape, device=device, dtype=dtype)


def gaussian_sampler(
    *shape: int,
    device: torch.device,
    dtype: torch.dtype,
    W: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate noise of shape (batch_size, horizon, state_dim).

    Args:
        *shape: The desired output shape, e.g. ``(batch_size, horizon, state_dim)``;
            the last entry is the Gaussian's dimension.
        device: The device to allocate the tensor on.
        dtype: The tensor's dtype.
        W: Optional covariance matrix, shape ``(shape[-1], shape[-1])``;
            defaults to the identity.
        mu: Optional mean vector, shape ``(shape[-1],)``; defaults to zero.

    Returns:
        A tensor of shape `shape`, drawn i.i.d. per leading index from
        ``MultivariateNormal(mu, W)``.
    """
    mu = mu if mu is not None else torch.zeros(shape[-1], dtype=dtype, device=device)
    W = W if W is not None else torch.eye(shape[-1], dtype=dtype, device=device)
    noise = MultivariateNormal(loc=mu, covariance_matrix=W)
    return noise.sample(shape[:-1])


def unify_sampler(sampler: Sampler, **kwargs: Any) -> UnifiedSampler:
    """Return a sampler with all parameters except shape fixed.

    Args:
        sampler: A `Sampler` (e.g. `zero_sampler`, `gaussian_sampler`).
        **kwargs: The sampler's non-shape parameters (``device``, ``dtype``,
            and any sampler-specific extras like ``W``/``mu``) to bind.

    Returns:
        A `UnifiedSampler`: ``(*shape) -> Tensor``.
    """
    return cast(UnifiedSampler, partial(sampler, **kwargs))


def _signature_value(value: Any) -> Any:
    """Hash array/tensor values for signature purposes, pass everything else
    through unchanged (mirrors the same rule used by every `get_signature()`
    in `core`/`models`: never embed a raw matrix, only its content hash)."""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return hash_array(value)
    return value


class RandomSampler:
    """Callable wrapper binding a `Sampler`'s non-shape parameters once, so it
    can be invoked as ``sampler(*shape) -> Tensor`` thereafter."""

    def __init__(self, signal_name: str, random_sampler_func: Sampler) -> None:
        """
        Args:
            signal_name: Human-readable identifier for the noise signal this
                sampler generates (e.g. ``"process_noise"``).
            random_sampler_func: The underlying `Sampler` to wrap.
        """
        self.signal_name = signal_name
        self.random_sampler_func = random_sampler_func
        self.sampler_name = self.random_sampler_func.__name__
        self._bound_kwargs: dict[str, Any] = {}

    def unify_sampler(self, **kwargs: Any) -> None:
        """Rebind `self.random_sampler_func` with `kwargs` fixed (see `unify_sampler`).

        Args:
            **kwargs: The sampler's non-shape parameters to bind.
        """
        self.random_sampler_func = unify_sampler(  # type: ignore[assignment]  # deliberate self-rebinding (legacy M9 family, kept for golden reproduction)
            self.random_sampler_func, **kwargs
        )
        self._bound_kwargs.update(kwargs)  # retained for get_signature (see below)

    def __call__(self, *shape: int) -> torch.Tensor:
        """Generate noise of shape `shape` using the bound sampler.

        Args:
            *shape: The desired output shape.

        Returns:
            The sampled tensor, shape `shape`.
        """
        return self.random_sampler_func(*shape)  # type: ignore[arg-type]  # bound via unify_sampler

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the underlying sampler function's name plus
        whatever non-shape parameters were bound via `unify_sampler`.

        Array/tensor-valued bound kwargs (e.g. `gaussian_sampler`'s `W`/`mu`)
        are content-hashed, never embedded raw.

        Returns:
            ``{"type": "RandomSampler", "sampler": <function name>,
            **bound kwargs (array-valued ones hashed)}``.
        """
        return {
            "type": type(self).__name__,
            "sampler": self.sampler_name,
            **{k: _signature_value(v) for k, v in self._bound_kwargs.items()},
        }


@runtime_checkable
class Distribution(Protocol):
    """A named, parameterized signal generator that both draws samples AND
    describes itself -- the single object mandated by Phase 1D, eliminating
    any possibility of a shadow/parallel spec drifting from what is actually
    sampled."""

    def __call__(self, *shape: int) -> "np.ndarray | torch.Tensor": ...
    def get_signature(self) -> dict[str, Any]: ...


class NumpyGaussianDistribution:
    """Isotropic (scalar `std`) or full-covariance Gaussian, NumPy-backed.

    Constructed with an already-instantiated `np.random.Generator` (so
    repeated `__call__`s correctly advance one stream, exactly as the
    call sites this replaces do) plus the *original* `seed` retained
    separately purely for signing -- the generator's mutating internal state
    is not itself a meaningful, reproducible signature field, but the seed
    used to construct it is.
    """

    def __init__(
        self,
        *,
        mean: float | np.ndarray = 0.0,
        std: float | None = None,
        covariance: np.ndarray | None = None,
        seed: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Args:
            mean: Scalar or per-dimension mean.
            std: Isotropic standard deviation; mutually exclusive with `covariance`.
            covariance: Full covariance matrix, shape ``(dim, dim)``; mutually
                exclusive with `std`.
            seed: The seed identifying this distribution's stream (signed, not
                mutated).
            rng: An already-constructed generator to share across multiple
                `Distribution` instances drawing from the *same* stream (e.g.
                `initial_state` then `process_noise` sequentially, as today's
                closures do); defaults to a fresh ``np.random.default_rng(seed)``.

        Raises:
            ValueError: If neither or both of `std`/`covariance` are given.
        """
        if (std is None) == (covariance is None):
            raise ValueError("Exactly one of `std` or `covariance` must be given.")
        self.mean = mean
        self.std = std
        self.covariance = covariance
        self._seed = seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        """
        Args:
            *shape: The desired output shape.

        Returns:
            A sample of shape `shape` (isotropic: `rng.normal`; full
            covariance: `rng.multivariate_normal` over the leading axes).
        """
        if self.std is not None:
            return self._rng.normal(loc=self.mean, scale=self.std, size=shape)
        assert self.covariance is not None  # __init__ enforced std XOR covariance
        return self._rng.multivariate_normal(
            np.broadcast_to(np.asarray(self.mean), self.covariance.shape[-1:]),
            self.covariance,
            size=shape[:-1],
        )

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "Gaussian", "seed":, "mean":, "std":}`` (isotropic) or
            ``{"type": "Gaussian", "seed":, "mean_hash":, "covariance_hash":}``
            (full covariance) -- the scalar case is reported directly (more
            useful in a human-read ``metadata.json`` than a hash of a
            never-materialized ``std^2 * I`` matrix); the matrix case is hashed.
        """
        base: dict[str, Any] = {"type": "Gaussian", "seed": self._seed}
        if self.std is not None:
            return {**base, "mean": self.mean, "std": self.std}
        assert self.covariance is not None  # __init__ enforced std XOR covariance
        return {
            **base,
            "mean_hash": hash_array(np.asarray(self.mean)),
            "covariance_hash": hash_array(self.covariance),
        }


class TorchGaussianDistribution:
    """Same contract as `NumpyGaussianDistribution`, torch-backed
    (`torch.Generator` + `torch.randn`/`MultivariateNormal`), for the torch
    batch-sampler paths."""

    def __init__(
        self,
        *,
        std: float | None = None,
        covariance: torch.Tensor | None = None,
        seed: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        The distribution is zero-mean by construction (matching every
        consumer in the tree; `self.mean` stays as a fixed attribute so the
        persisted signature shape is unchanged).

        Args:
            std: Isotropic standard deviation; mutually exclusive with `covariance`.
            covariance: Full covariance matrix, shape ``(dim, dim)``; mutually
                exclusive with `std`.
            seed: The seed identifying this distribution's stream (signed, not
                mutated).
            dtype: The sampled tensor's dtype.
            device: The sampled tensor's device; defaults to CPU.
            generator: An already-constructed `torch.Generator` to share
                across multiple `Distribution` instances drawing from the
                *same* stream; defaults to a fresh
                ``torch.Generator().manual_seed(seed)``.

        Raises:
            ValueError: If neither or both of `std`/`covariance` are given.
        """
        if (std is None) == (covariance is None):
            raise ValueError("Exactly one of `std` or `covariance` must be given.")
        self.mean = 0.0
        self.std = std
        self.covariance = covariance
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._seed = seed
        self._generator = generator or torch.Generator().manual_seed(seed)

    def __call__(self, *shape: int) -> torch.Tensor:
        """
        Args:
            *shape: The desired output shape.

        Returns:
            A sample of shape `shape` (isotropic: ``torch.randn(...) * std +
            mean``; full covariance: `MultivariateNormal` over the leading axes).
        """
        if self.std is not None:
            # Drawn on self._generator's own device (CPU unless the caller
            # supplied a device-matched generator) and moved to self.device
            # afterward: torch's RNG algorithm differs per device, so
            # generating directly on a non-CPU device would silently break
            # the documented bit-for-bit-identical-for-a-given-seed contract
            # (module docstring) the moment self.device != the generator's
            # device -- a residency change must never also be a numerics
            # change.
            noise = torch.randn(
                shape,
                generator=self._generator,
                dtype=self.dtype,
                device=self._generator.device,
            ).to(self.device)
            return noise * self.std + self.mean
        assert self.covariance is not None  # __init__ enforced std XOR covariance
        dim = self.covariance.shape[-1]
        mu = torch.zeros(dim, dtype=self.dtype, device=self.device)
        dist = MultivariateNormal(loc=mu, covariance_matrix=self.covariance)
        return dist.sample(shape[:-1])

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            Same shape as `NumpyGaussianDistribution.get_signature`.
        """
        base: dict[str, Any] = {"type": "Gaussian", "seed": self._seed}
        if self.std is not None:
            return {**base, "mean": self.mean, "std": self.std}
        assert self.covariance is not None  # __init__ enforced std XOR covariance
        dim = self.covariance.shape[-1]
        zero_mean = torch.zeros(dim, dtype=self.dtype, device=self.device)
        return {
            **base,
            "mean_hash": hash_array(zero_mean),
            "covariance_hash": hash_array(self.covariance),
        }


class ConstantDistribution:
    """A fixed, non-zero-in-general 'distribution': every draw broadcasts the
    same `value` to the requested shape -- covers a deterministic initial-
    state/noise signal (e.g. `models.analytic.iterative_gd
    .AnalyticalIterativeGDController.solve`'s reproducible convergence
    check), which `ZeroDistribution` cannot express for a non-zero value."""

    def __init__(self, value: float | np.ndarray) -> None:
        """
        Args:
            value: The scalar or per-dimension value every draw broadcasts.
        """
        self.value = value

    def __call__(self, *shape: int) -> np.ndarray:
        """
        Args:
            *shape: The desired output shape.

        Returns:
            `self.value` broadcast to `shape`.
        """
        return np.broadcast_to(np.asarray(self.value, dtype=float), shape).copy()

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "Constant", "value_hash":}``.
        """
        return {
            "type": "Constant",
            "value_hash": hash_array(np.asarray(self.value, dtype=float)),
        }


class ZeroDistribution:
    """Degenerate (Dirac-delta-at-zero) 'noise' -- covers a placeholder
    measurement-noise signal that is exactly zero, which is not Gaussian and
    must not be misrepresented as one."""

    def __call__(self, *shape: int) -> np.ndarray:
        """
        Args:
            *shape: The desired output shape.

        Returns:
            ``np.zeros(shape)``.
        """
        return np.zeros(shape)

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "Zero"}``.
        """
        return {"type": "Zero"}


class TorchZeroDistribution:
    """Torch-tensor-returning counterpart to `ZeroDistribution`, for the
    torch batch-sampler paths."""

    def __init__(
        self, *, dtype: torch.dtype, device: torch.device | None = None
    ) -> None:
        """
        Args:
            dtype: The returned tensor's dtype.
            device: The returned tensor's device; defaults to CPU.
        """
        self.dtype = dtype
        self.device = device or torch.device("cpu")

    def __call__(self, *shape: int) -> torch.Tensor:
        """
        Args:
            *shape: The desired output shape.

        Returns:
            ``torch.zeros(shape, dtype=self.dtype, device=self.device)``.
        """
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def get_signature(self) -> dict[str, Any]:
        """
        Returns:
            ``{"type": "Zero"}``.
        """
        return {"type": "Zero"}
