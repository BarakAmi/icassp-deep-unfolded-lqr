"""Shared value objects and array-normalization helpers for the agnostic
visualization toolkit (Phase 2B). The toolkit is algorithm/problem-agnostic
by construction: everything it renders is a function of exactly two injected
seams --

    baseline_U:           the reference control sequence (Riccati, a
                           converged GD solution, an SDP lower bound, ...)
    candidate_U_history:  per-iteration control iterates (e.g. from
                           `engine.callbacks.ControlSequenceHistoryCallback`)
    cost_oracle:          a `CostOracle` scoring full-horizon control batches

-- never a concrete controller, problem, or the string "Riccati"/"GD". This
module defines those seams' types plus the small immutable specs describing
what to project/select.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, cast

import numpy as np
import torch

from ...core.utils import to_numpy


@runtime_checkable
class CostOracle(Protocol):
    """Maps a *batch* of full-horizon control sequences to one scalar cost
    per sample -- the sole channel through which problem physics enters the
    landscape assets (Assets 3-6).

    Must preserve the batch axis. This said `RolloutModel.forward`'s own cost
    "reduces over *both* batch and time to a single scalar" and, since Annex 06
    §4.3's correction of 2026-08-04, it does not -- the rollout returns a
    `(B,)` cost and the batch reduction is the caller's. The requirement is
    unchanged and now easier to meet; `oracles.make_rollout_cost_oracle`
    remains the reference adapter, and it must still average over *time* only.
    """

    def __call__(self, U: torch.Tensor) -> torch.Tensor:  # (B, T, m) -> (B,)
        ...


@dataclass(frozen=True)
class SliceSpec:
    """Which 2D/3D slice of control-space to project the cost onto: the
    timestep whose control is varied, which of its components vary, and the
    coordinate grid for each varied component. Every other ``(t, j)`` entry
    of the projected backdrop stays fixed at its reference value.

    Attributes:
        timestep: ``t*``, the horizon index whose control ``u_t`` is varied.
        components: which coordinate indices of ``u_t`` vary -- length 1 for
            a 1D line slice, length 2 for a 2D contour/surface slice, length
            3 for a 3D scatter slice.
        ranges: one 1D coordinate array per entry in `components`, same
            length/order.
    """

    timestep: int
    components: tuple[int, ...]
    ranges: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Fail-fast guards on the slice's own internal consistency (the
        control/timestep dimensions it must additionally agree with --
        `reference_U`'s horizon/control_dim -- are checked by
        `grids.GridProjector.project`, which is where that context exists).

        Raises:
            ValueError: If `timestep` is negative, `components` doesn't have
                length 1, 2, or 3, `components` has duplicate indices, or
                `ranges` doesn't have one entry per `components` entry.
        """
        if self.timestep < 0:
            raise ValueError(f"timestep must be non-negative, got {self.timestep}.")
        if len(self.components) not in (1, 2, 3):
            raise ValueError(
                "components must have length 1 (line), 2 (contour/surface), "
                f"or 3 (scatter), got {len(self.components)}."
            )
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"components must be distinct, got {self.components}.")
        if len(self.ranges) != len(self.components):
            raise ValueError(
                f"ranges must have one entry per component "
                f"({len(self.components)}), got {len(self.ranges)}."
            )


@dataclass(frozen=True)
class CostField:
    """Output of `grids.GridProjector.project`: the evaluated cost slice plus
    enough provenance for a renderer to draw it without recomputing anything.

    Attributes:
        grids: meshgrid arrays (as from `numpy.meshgrid`), one per varied
            component, each shape ``(G, G)`` (2D) or ``(G, G, G)`` (3D).
        cost_grid: cost at each grid node, same shape as each `grids` entry.
        spec: the `SliceSpec` this field was projected from.
        reference_U: the fixed backdrop control sequence the slice was cut
            from, shape ``(T, m)``.
    """

    grids: tuple[np.ndarray, ...]
    cost_grid: np.ndarray
    spec: SliceSpec
    reference_U: np.ndarray


@dataclass(frozen=True)
class IterationSelection:
    """A sparse set of iteration indices to mark/annotate on a static plot
    (or drive an animation's frame count) -- Asset 6's subsampling engine
    (`overlay.select_iterations` builds these).

    Attributes:
        indices: the selected iteration indices, sorted ascending, always
            including the first and last iteration of the range they were
            selected from.
        mode: how `indices` was chosen -- ``"log"``, ``"linear"``, or
            ``"explicit"``.
    """

    indices: np.ndarray
    mode: str


def normalize_control(
    name: str, array: np.ndarray | torch.Tensor, *, sample_index: int = 0
) -> np.ndarray:
    """Normalize a single (possibly batched) control sequence down to its
    unbatched ``(T, m)`` form -- the same reduction convention
    `viz.plots.trajectories._as_unbatched_2d` uses for state/control/
    observation trajectories.

    Args:
        name: identifies `array` in error messages.
        array: shape ``(T, m)`` or ``(B, T, m)``.
        sample_index: which batch element to select when `array` carries a
            leading batch axis.

    Returns:
        `array` as a ``(T, m)`` numpy array.

    Raises:
        ValueError: If `array` is not 2D or 3D.
    """
    array = to_numpy(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return cast(np.ndarray, array[sample_index])
    raise ValueError(f"{name} must be (T, m) or (B, T, m), got shape {array.shape}.")


def normalize_history(
    name: str, array: np.ndarray | torch.Tensor, *, sample_index: int = 0
) -> np.ndarray:
    """Normalize a (possibly batched) per-iteration control history down to
    its unbatched ``(I, T, m)`` form -- e.g. as persisted by
    `engine.callbacks.ControlSequenceHistoryCallback` (``(I, B, T, m)``).

    Args:
        name: identifies `array` in error messages.
        array: shape ``(I, T, m)`` or ``(I, B, T, m)``.
        sample_index: which batch element to select when `array` carries a
            batch axis (the second one, after the leading iteration axis).

    Returns:
        `array` as an ``(I, T, m)`` numpy array.

    Raises:
        ValueError: If `array` is not 3D or 4D.
    """
    array = to_numpy(array)
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return array[:, sample_index]
    raise ValueError(
        f"{name} must be (I, T, m) or (I, B, T, m), got shape {array.shape}."
    )


def ensure_slice_matches_control(spec: SliceSpec, reference_U: np.ndarray) -> None:
    """Cross-check a `SliceSpec` against the ``(T, m)`` control it will be
    projected from -- deferred here (rather than into `SliceSpec.__post_init__`)
    since `SliceSpec` itself carries no horizon/control_dim of its own.

    Args:
        spec: the slice to validate.
        reference_U: the ``(T, m)`` backdrop `spec` will be projected from.

    Raises:
        ValueError: If `spec.timestep` is out of ``[0, T)``, or any of
            `spec.components` is out of ``[0, m)``.
    """
    horizon, control_dim = reference_U.shape
    if not (0 <= spec.timestep < horizon):
        raise ValueError(f"spec.timestep ({spec.timestep}) must be in [0, {horizon}).")
    for component in spec.components:
        if not (0 <= component < control_dim):
            raise ValueError(
                f"spec.components {spec.components} must all be in "
                f"[0, {control_dim}), got {component}."
            )
