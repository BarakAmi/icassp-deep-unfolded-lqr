"""Infinity-norm ("box") constraint on control inputs: u_min <= u_i <= u_max
for every component i, independent of the others -- the simplest
actuator-limit constraint, e.g. a saturating actuator."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
import torch

from ..system.system import BatchedStateSpaceVector
from ..utils.signing import hash_array

#: Per-dimension bounds up to this many entries stay human-readable/diffable
#: in a persisted signature (reported as a plain list); larger arrays are
#: content-hashed instead. control_dim is usually small (a handful of
#: actuators), so this keeps the common case inline while still guaranteeing
#: a well-defined, compact signature for unusually high-dimensional control
#: spaces.
_INLINE_ARRAY_MAX_SIZE = 16


@dataclass(frozen=True)
class BoxConstraint:
    """Satisfies the `Constraint` protocol.

    Args:
        u_max: The upper bound, either a scalar (same limit for every
            control dimension) or an array of per-dimension bounds.
        u_min: Optional lower bound, either a scalar or an array of
            per-dimension bounds. If not provided, defaults to ``-u_max``
            (in which case every entry of `u_max` must be strictly
            positive); if provided explicitly, every entry must be strictly
            less than the corresponding entry of `u_max`.
    """

    u_max: ArrayLike
    u_min: ArrayLike | None = None

    def __post_init__(self) -> None:
        """Fail-fast guard: `u_max` must be strictly positive when `u_min` is
        not given (so ``-u_max`` is a valid lower bound); if `u_min` is
        given, it must be strictly less than `u_max` entrywise.

        Raises:
            ValueError: If `u_max` is not strictly positive (when `u_min` is
                omitted), or `u_min` is not strictly less than `u_max`.
        """
        # Normalize both bounds to ndarrays once, at construction; every
        # later read goes through `_max_bound`/`_min_bound`, which encode
        # this invariant for the type checker.
        u_max = np.asarray(self.u_max)
        object.__setattr__(self, "u_max", u_max)
        if self.u_min is None:
            if np.any(u_max <= 0):
                raise ValueError(
                    "u_max must be strictly positive when u_min isn't "
                    f"provided, got u_max={u_max}."
                )
            object.__setattr__(self, "u_min", -u_max)
            return

        u_min = np.asarray(self.u_min)
        object.__setattr__(self, "u_min", u_min)
        if np.any(u_min >= u_max):
            raise ValueError(
                "u_min must be strictly less than u_max, got "
                f"u_min={u_min}, u_max={u_max}."
            )

    @property
    def _max_bound(self) -> np.ndarray:
        """`u_max` as the ndarray `__post_init__` normalized it to."""
        return cast(np.ndarray, self.u_max)

    @property
    def _min_bound(self) -> np.ndarray:
        """`u_min` as the ndarray `__post_init__` normalized it to."""
        return cast(np.ndarray, self.u_min)

    def __call__(self, u: BatchedStateSpaceVector) -> BatchedStateSpaceVector:
        """Clip `u` elementwise into ``[u_min, u_max]``. Works for both a
        numpy.ndarray and a torch.Tensor.

        Args:
            u: Control input, shape ``(batch, control_dim)`` (or broadcastable
                against `u_max`/`u_min`); ``numpy.ndarray`` or ``torch.Tensor``.

        Returns:
            `u` clipped elementwise, same type as `u`.
        """
        if isinstance(u, torch.Tensor):
            max_bound = torch.as_tensor(self._max_bound, dtype=u.dtype, device=u.device)
            min_bound = torch.as_tensor(self._min_bound, dtype=u.dtype, device=u.device)
            return torch.clamp(u, min=min_bound, max=max_bound)
        return np.clip(u, self._min_bound, self._max_bound)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: `u_max`/`u_min` reported directly (as a scalar
        or plain list) when small enough to stay human-readable and diffable
        in persisted metadata, content-hashed instead once the per-dimension
        bound array exceeds `_INLINE_ARRAY_MAX_SIZE` entries.

        Returns:
            ``{"type": "BoxConstraint", "u_max": ..., "u_min": ...}``, each
            either a ``float``/``list[float]`` (small bounds) or a
            ``"sha256:..."`` content-hash string (large bounds).
        """
        return {
            "type": type(self).__name__,
            "u_max": self._bound_signature(self._max_bound),
            "u_min": self._bound_signature(self._min_bound),
        }

    @staticmethod
    def _bound_signature(bound: np.ndarray) -> float | list[float] | str:
        """Inline (scalar or list) for a small bound; content-hashed once its
        size exceeds `_INLINE_ARRAY_MAX_SIZE`."""
        if bound.ndim == 0:
            return float(bound)
        if bound.size <= _INLINE_ARRAY_MAX_SIZE:
            return cast("list[float]", bound.tolist())
        return hash_array(bound)
