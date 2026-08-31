"""The fixed, non-learned step-size geometry shared by every analytical
iterative-refinement solver: `StepSizeSchedule` stores a single tensor at one
of four broadcasting "levels" (scalar, per-component, per-component-per-
iteration, or fully time-varying) and slices out the right broadcastable
shape for a given gradient-descent iteration.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ...core.utils import ensure_positive_integer


@runtime_checkable
class StepSizeProvider(Protocol):
    """Structural protocol: yields the iteration-``i`` step-size tensor.
    Satisfied by both the fixed `StepSizeSchedule` (analytical, this module)
    and the learnable `models.unfolded.parameters.StepSizeParameter` (whose
    ``.for_iteration(i)`` slices its own ``(num_iterations, action_dim)``
    tensor) -- the shared abstraction that harmonizes every iterative-
    refinement consumer (open-loop, unfolded, and the analytical solver)
    without forcing them under one class hierarchy.
    """

    def for_iteration(self, i: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class StepSizeSchedule:
    """A fixed (``requires_grad=False``), user-defined step-size tensor at one
    of four broadcasting levels, storage always **iteration-leading,
    component-trailing** (``num_iterations`` first, `control_dim` last,
    `horizon` in between when present):

    | Level | ``raw`` shape        | `for_iteration(i)` returns          |
    |-------|-----------------------|--------------------------------------|
    | 0D    | ``()``                | ``(1, 1)``                           |
    | 1D    | ``(m,)``               | ``(1, m)``                           |
    | 2D    | ``(M, m)``             | ``(1, m)`` == ``raw[i]``             |
    | 3D    | ``(M, T, m)``          | ``(T, m)`` == ``raw[i]``             |

    where ``M = num_iterations``, ``T = horizon``, ``m = control_dim``. The
    returned tensor broadcasts directly against a gradient of shape
    ``(batch, T, m)``.

    Attributes:
        raw: The step-size tensor, rank 0-3 (see table above);
            ``requires_grad`` must be ``False`` -- this is the analytical,
            non-learned counterpart to a learnable step-size parameter.
        num_iterations: ``M``, the number of gradient-descent iterations this
            schedule is defined over.
        horizon: ``T``, the control horizon.
        control_dim: ``m``, the control vector dimension.
    """

    raw: torch.Tensor
    num_iterations: int
    horizon: int
    control_dim: int

    def __post_init__(self) -> None:
        """Fail-fast guards: positive dimensions, a non-learned `raw`, and
        `raw`'s rank/shape validated against ``(num_iterations, horizon,
        control_dim)`` per the level it declares.

        Raises:
            ValueError: If `num_iterations`/`horizon`/`control_dim` is not a
                positive ``int``, `raw` requires grad, `raw` is not 0D-3D, or
                `raw`'s shape disagrees with the declared level's expected
                shape.
        """
        ensure_positive_integer(self.num_iterations, "num_iterations")
        ensure_positive_integer(self.horizon, "horizon")
        ensure_positive_integer(self.control_dim, "control_dim")
        if self.raw.requires_grad:
            raise ValueError(
                "StepSizeSchedule.raw must not require grad -- it is the "
                "fixed, analytical, non-learned step-size geometry."
            )
        self._validate_shape()

    def _validate_shape(self) -> None:
        """See `__post_init__`."""
        shape = tuple(self.raw.shape)
        ndim = self.raw.ndim
        if ndim == 0:
            return
        expected: tuple[int, ...]
        if ndim == 1:
            expected = (self.control_dim,)
        elif ndim == 2:
            expected = (self.num_iterations, self.control_dim)
        elif ndim == 3:
            expected = (self.num_iterations, self.horizon, self.control_dim)
        else:
            raise ValueError(f"StepSizeSchedule.raw must be 0D-3D, got {ndim}D.")
        if shape != expected:
            raise ValueError(
                f"StepSizeSchedule.raw of rank {ndim} must have shape "
                f"{expected}, got {shape}."
            )

    def for_iteration(self, i: int) -> torch.Tensor:
        """Return the step-size tensor for gradient-descent iteration `i`,
        broadcastable against a gradient of shape ``(batch, T, m)``.

        Args:
            i: The gradient-descent iteration index (only consulted when
                `raw` is 2D or 3D; the 0D/1D levels are iteration-invariant).

        Returns:
            ``(1, 1)`` (0D), ``(1, m)`` (1D or 2D), or ``(T, m)`` (3D) -- see
            the class docstring's table.
        """
        ndim = self.raw.ndim
        if ndim == 0:
            return self.raw.reshape(1, 1)
        if ndim == 1:
            return self.raw.reshape(1, -1)
        if ndim == 2:
            return self.raw[i].reshape(1, -1)
        return self.raw[i]
