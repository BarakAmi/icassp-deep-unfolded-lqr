"""Depth-parameterized trajectory diagnostics (NB03 Phase 3, requirement 4):
how an unfolded contender's full time-indexed rollout at unfolding depth
``K`` compares to the Riccati baseline, faceted (static) or animated
(frame = depth) across a chosen set of depths. Distinct from
`trajectories.plot_trajectory_comparison` (a named-series, single-depth
snapshot): this module's series axis is UNFOLDING DEPTH, not an arbitrary
label, and every x-axis here is the time horizon ``k`` -- never epochs
(epochs are `training_curves.plot_training_curves`'s axis alone).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from ..landscape.animators import FrameAnimator
from ..landscape.signals import compute_aggregate_relative_error
from ..landscape.types import normalize_control
from ..style import BASELINE_COLOR, DEFAULT_EPSILON, styled_figure

#: Ordinal colormap for "unfolding depth K" -- one hue family in light ->
#: dark steps (mirrors `viz.landscape.signals._ITERATION_CMAP`'s "position
#: in the optimization" idiom, applied to depth instead of iteration), so a
#: shallow K reads visibly lighter than a deep K without a colorbar.
_DEPTH_CMAP = "BuPu"
_DEPTH_CMAP_RANGE = (0.35, 0.95)

#: The animated candidate's line color -- distinct from every static-figure
#: depth color (`_depth_colors` never emits pure "tab:purple" at its
#: extremes) so the moving line reads unambiguously as "the current frame's
#: depth," not as one more depth in a ramp.
_ANIMATED_CANDIDATE_COLOR = "tab:purple"

#: The box-constraint reference lines' color/style (NB04 plan Sec 3.5/5) --
#: a neutral gray, deliberately outside `_DEPTH_CMAP`/`BASELINE_COLOR`/
#: `_ANIMATED_CANDIDATE_COLOR` so "this is the constraint, not a contender"
#: reads unambiguously at a glance.
_BOX_BOUND_COLOR = "dimgray"
_BOX_BOUND_LINESTYLE = ":"
_BOX_BOUND_LINEWIDTH = 1.3


def _depth_colors(depths: Sequence[int]) -> list:
    """One color per entry of `depths`, light (shallowest) to dark
    (deepest) along `_DEPTH_CMAP` -- the same color identifies a given depth
    across every subplot (each control component and the shared error
    panel alike)."""
    cmap = plt.get_cmap(_DEPTH_CMAP)
    if len(depths) == 1:
        return [cmap(_DEPTH_CMAP_RANGE[1])]
    lo, hi = _DEPTH_CMAP_RANGE
    return [cmap(t) for t in np.linspace(lo, hi, len(depths))]


@dataclass(frozen=True)
class DepthTrajectoryStyle:
    """Presentation knobs shared by `plot_trajectory_vs_depth_with_error`
    and `DepthTrajectoryAnimator`.

    Attributes:
        sample_index: Batch element to select if an input is batched.
        variable_name: Symbol used for the y-axis labels, e.g. ``"u"``.
        dim_labels: Optional per-component labels; ``None`` derives
            ``"{variable_name}_{j}"``-style defaults.
        xlabel: Shared x-axis label -- always the time horizon ``k``.
        baseline_label: Legend label for the depth-invariant baseline this
            candidate is compared against -- ``"Riccati"`` by default (NB03's
            usage: the exact, unconstrained optimum), but freely overridable
            (e.g. NB04's Truncated-Riccati or COCP baselines, neither of
            which IS Riccati) -- both the default title (Sec below) and the
            baseline's own legend entry are derived from this field, so the
            two can never name two different references.
        contender_name: Display name of the plotted contender, used in the
            default title (e.g. ``"Unfolded-alpha"``); ``None`` falls back
            to a generic "Candidate".
        epsilon: Relative-error denominator floor.
        box_bounds: Optional infinity-norm control bound(s) to draw as
            horizontal reference lines in every SIGNAL row (never the
            relative-error row) -- ``None`` (default) draws nothing, so
            every unconstrained (NB03) call site is unaffected. Either a
            single ``(u_min, u_max)`` pair shared by every control
            component, or one pair per component
            (``Sequence[tuple[float, float]]``, length ``control_dim``) for
            an anisotropic box (NB04 plan Sec 3.5).
    """

    sample_index: int = 0
    variable_name: str = "u"
    dim_labels: Sequence[str] | None = None
    xlabel: str = "Time step k"
    baseline_label: str = "Riccati"
    contender_name: str | None = None
    epsilon: float = DEFAULT_EPSILON
    box_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None


def _prepare(
    baseline: np.ndarray,
    depth_trajectories: Mapping[int, np.ndarray],
    *,
    style: DepthTrajectoryStyle,
) -> tuple[np.ndarray, dict[int, np.ndarray], int, int]:
    """Normalize `baseline`/every `depth_trajectories` entry to unbatched
    ``(T, m)`` and fail-fast validate every depth's trajectory shares
    `baseline`'s ``(T, m)`` -- shared by the static renderer and the
    animator below.

    Returns:
        ``(baseline, trajectories, horizon, control_dim)``.

    Raises:
        ValueError: If `depth_trajectories` is empty, any entry's ``(T, m)``
            disagrees with `baseline`'s, or `style.dim_labels`'s length
            disagrees with ``m``.
    """
    if not depth_trajectories:
        raise ValueError("depth_trajectories must contain at least one depth.")
    base = normalize_control("baseline", baseline, sample_index=style.sample_index)
    horizon, control_dim = base.shape
    trajectories: dict[int, np.ndarray] = {}
    for depth, array in depth_trajectories.items():
        candidate = normalize_control(
            f"depth_trajectories[{depth}]", array, sample_index=style.sample_index
        )
        if candidate.shape != base.shape:
            raise ValueError(
                f"depth_trajectories[{depth}]'s (T, m) = {candidate.shape} "
                f"must match baseline's (T, m) = {base.shape}."
            )
        trajectories[depth] = candidate
    if style.dim_labels is not None and len(style.dim_labels) != control_dim:
        raise ValueError(
            f"dim_labels must have length {control_dim}, got {len(style.dim_labels)}."
        )
    return base, trajectories, horizon, control_dim


def _dim_label(j: int, style: DepthTrajectoryStyle) -> str:
    """The y-axis label for control component `j` -- `style.dim_labels[j]`
    if given, else ``"{variable_name}_{j}"``."""
    if style.dim_labels is not None:
        return style.dim_labels[j]
    return f"${style.variable_name}_{{{j}}}$"


def _resolve_box_bounds(
    j: int, style: DepthTrajectoryStyle
) -> tuple[float, float] | None:
    """Component `j`'s ``(u_min, u_max)`` pair from `style.box_bounds`, or
    `None` if unset -- resolves the shared-pair-vs-per-component-sequence
    ambiguity (see `DepthTrajectoryStyle.box_bounds`) once, shared by the
    static plotter and the animator."""
    bounds = style.box_bounds
    if bounds is None:
        return None
    first = bounds[0]
    if isinstance(first, int | float):
        return cast(tuple[float, float], bounds)
    return cast(Sequence[tuple[float, float]], bounds)[j]


def _draw_box_bounds(ax: Axes, bounds: tuple[float, float] | None) -> None:
    """Draw `bounds` as two horizontal reference lines (one shared legend
    entry, "u_max bound") -- a no-op when `bounds` is `None`, so an
    unconstrained call site's figure is unaffected."""
    if bounds is None:
        return
    low, high = bounds
    ax.axhline(
        high,
        color=_BOX_BOUND_COLOR,
        linestyle=_BOX_BOUND_LINESTYLE,
        linewidth=_BOX_BOUND_LINEWIDTH,
        label="u_max bound",
        zorder=5,
    )
    ax.axhline(
        low,
        color=_BOX_BOUND_COLOR,
        linestyle=_BOX_BOUND_LINESTYLE,
        linewidth=_BOX_BOUND_LINEWIDTH,
        zorder=5,
    )


def _ylim_including_bounds(
    values: np.ndarray, bounds: tuple[float, float] | None
) -> tuple[float, float] | None:
    """``(lo, hi)`` y-limits spanning `values` and, if given, `bounds`, with
    a 10% margin, so a drawn constraint line is never clipped off-axis
    (NB04 plan Sec 3.5). `None` when `bounds` is `None`, deferring to
    matplotlib's own autoscale exactly as before this feature existed."""
    if bounds is None:
        return None
    lo = min(float(values.min()), bounds[0])
    hi = max(float(values.max()), bounds[1])
    pad = 0.1 * (hi - lo + 1e-9)
    return lo - pad, hi + pad


def _depth_label(depth: int, costs: Mapping[int, float] | None) -> str:
    """Legend label for one swept depth -- ``"K={depth}"``, with its total
    cost ``J`` appended when `costs` carries an entry for it."""
    if costs is not None and depth in costs:
        return f"K={depth} (J={costs[depth]:.4f})"
    return f"K={depth}"


def plot_trajectory_vs_depth_with_error(
    baseline: np.ndarray,
    depth_trajectories: Mapping[int, np.ndarray],
    *,
    costs: Mapping[int, float] | None = None,
    baseline_cost: float | None = None,
    style: DepthTrajectoryStyle = DepthTrajectoryStyle(),
    title: str | None = None,
) -> Figure:
    """The static ``m + 1``-subplot depth-convergence grid: the first `m`
    rows overlay ``u_k`` vs. time step ``k`` for every depth in
    `depth_trajectories` against the Riccati baseline; the ``(m + 1)``-th
    row plots `viz.landscape.signals.compute_aggregate_relative_error`
    (candidate vs. baseline) per depth, vs. ``k`` -- reusing that function
    rather than re-deriving relative error (DRY).

    Args:
        baseline: reference (Riccati) control, shape ``(T, m)`` or
            ``(B, T, m)``.
        depth_trajectories: unfolding depth ``K`` -> control trajectory,
            shape ``(T, m)`` or ``(B, T, m)`` each.
        costs: optional depth ``K`` -> total expected cost ``J``, appended
            to that depth's legend entry.
        baseline_cost: optional total expected cost ``J`` of `baseline`,
            appended to its legend entry.
        style: slicing/labeling knobs (see `DepthTrajectoryStyle`).
        title: figure title; defaults to naming `style.contender_name`
            against the Riccati baseline (the swept depths already label
            each curve in the legend, so they are not repeated here).

    Returns:
        The Figure: ``m`` signal rows + one shared relative-error row.

    Raises:
        ValueError: If `depth_trajectories` is empty, any entry's shape
            disagrees with `baseline`'s, or `style.dim_labels` disagrees
            with the control dimension (see `_prepare`).
    """
    base, trajectories, horizon, control_dim = _prepare(
        baseline, depth_trajectories, style=style
    )
    depths = sorted(trajectories)
    colors = _depth_colors(depths)
    t = np.arange(horizon)
    baseline_label = style.baseline_label
    if baseline_cost is not None:
        baseline_label = f"{baseline_label} (J={baseline_cost:.4f})"

    with styled_figure():
        fig, axes = plt.subplots(
            nrows=control_dim + 1,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(4.0, 2.2 * (control_dim + 1))),
        )
        for j in range(control_dim):
            ax = axes[j, 0]
            ax.plot(
                t,
                base[:, j],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=2.0,
                label=baseline_label,
                zorder=10,
            )
            for depth, color in zip(depths, colors):
                ax.plot(
                    t,
                    trajectories[depth][:, j],
                    color=color,
                    linewidth=1.6,
                    label=_depth_label(depth, costs),
                )
            bounds = _resolve_box_bounds(j, style)
            _draw_box_bounds(ax, bounds)
            ax.set_ylabel(_dim_label(j, style))
            ylim = _ylim_including_bounds(
                np.concatenate([base[:, j], *(trajectories[d][:, j] for d in depths)]),
                bounds,
            )
            if ylim is not None:
                ax.set_ylim(ylim)

        error_ax = axes[-1, 0]
        for depth, color in zip(depths, colors):
            error = compute_aggregate_relative_error(
                base, trajectories[depth], epsilon=style.epsilon
            )
            error_ax.plot(
                t, error, color=color, linewidth=1.4, label=_depth_label(depth, costs)
            )
        error_ax.set_yscale("log")
        error_ax.set_ylabel("Relative Error (%)")
        error_ax.set_xlabel(style.xlabel)

        # No horizontal `rect=` reservation (NB03 overhaul Phase 4c): a
        # fixed-fraction right margin squeezed the axes to make room for a
        # legend whose real width varies with the number of swept depths;
        # `bbox_inches="tight"` at save time (`viz.style.io
        # .save_vector_figure`) crops to the actual legend width instead,
        # closing the dead-space gap this left behind. The top fraction is
        # kept -- unlike the legend, `fig.suptitle`'s headroom is NOT
        # cropped by `bbox_inches="tight"` (it already sits inside the
        # figure bbox), so it still needs an explicit reservation.
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0.0,
        )
        contender = style.contender_name or "Candidate"
        fig.suptitle(
            title
            or f"{contender} Control Trajectories (${style.variable_name}$) "
            f"vs. {style.baseline_label}"
        )
    return fig


class DepthTrajectoryAnimator(FrameAnimator):
    """Animated counterpart of `plot_trajectory_vs_depth_with_error`: the
    identical ``m + 1`` layout, but each frame is one unfolding depth ``K``
    (ascending) -- redrawing that depth's ``u_k``-vs-``k`` curves and its
    relative-error row against the ALWAYS-PRESENT Riccati baseline, so the
    reader watches the candidate trajectory morph into alignment with
    Riccati as ``K`` grows.
    """

    def __init__(
        self,
        baseline: np.ndarray,
        depth_trajectories: Mapping[int, np.ndarray],
        *,
        interval_ms: int = 200,
        style: DepthTrajectoryStyle = DepthTrajectoryStyle(),
        costs: Mapping[int, float] | None = None,
        baseline_cost: float | None = None,
    ) -> None:
        """
        Args:
            baseline: reference (Riccati) control, shape ``(T, m)`` or
                ``(B, T, m)``.
            depth_trajectories: unfolding depth ``K`` -> control trajectory,
                shape ``(T, m)`` or ``(B, T, m)`` each.
            interval_ms: milliseconds between frames.
            style: slicing/labeling knobs (see `DepthTrajectoryStyle`).
            costs: optional depth ``K`` -> total expected cost ``J``,
                shown in the legend as the animated depth advances.
            baseline_cost: optional total expected cost ``J`` of
                `baseline`, appended to its (fixed) legend entry.

        Raises:
            ValueError: See `plot_trajectory_vs_depth_with_error`.
        """
        super().__init__(interval_ms=interval_ms, blit=True)
        self.baseline, self.trajectories, self.horizon, self.control_dim = _prepare(
            baseline, depth_trajectories, style=style
        )
        self.depths = sorted(self.trajectories)
        self.style = style
        self.costs = costs
        self.baseline_cost = baseline_cost
        self._signal_lines: list[Line2D] = []
        self._error_line: Line2D | None = None
        self._title: Text | None = None
        self._candidate_legend_text: Text | None = None

    @property
    def num_frames(self) -> int:
        return len(self.depths)

    def setup(self) -> tuple[Figure, Axes]:
        style = self.style
        candidate_name = style.contender_name or "Unfolded"
        baseline_label = style.baseline_label
        if self.baseline_cost is not None:
            baseline_label = f"{baseline_label} (J={self.baseline_cost:.4f})"
        t = np.arange(self.horizon)
        fig, axes = plt.subplots(
            nrows=self.control_dim + 1,
            ncols=1,
            sharex=True,
            squeeze=False,
            # Wider than the static figure's default (12in vs. 10in): the
            # legend here carries a per-frame depth + cost annotation (see
            # `update`) and, unlike the static save path, an animation's
            # canvas size is fixed at `setup()` time -- there is no
            # `bbox_inches="tight"` crop at save time to recover space from.
            figsize=(12, max(4.0, 2.2 * (self.control_dim + 1))),
        )
        self._signal_lines = []
        for j in range(self.control_dim):
            ax = axes[j, 0]
            ax.plot(
                t,
                self.baseline[:, j],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=2.0,
                label=baseline_label,
                zorder=10,
            )
            (line,) = ax.plot(
                [],
                [],
                color=_ANIMATED_CANDIDATE_COLOR,
                linewidth=2.2,
                label=candidate_name,
            )
            self._signal_lines.append(line)
            bounds = _resolve_box_bounds(j, style)
            _draw_box_bounds(ax, bounds)
            ax.set_ylabel(_dim_label(j, style))
            all_y = np.concatenate(
                [
                    self.baseline[:, j],
                    *(traj[:, j] for traj in self.trajectories.values()),
                ]
            )
            ylim = _ylim_including_bounds(all_y, bounds)
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                pad = 0.1 * (all_y.max() - all_y.min() + 1e-9)
                ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        error_ax = axes[-1, 0]
        (self._error_line,) = error_ax.plot(
            [], [], color=_ANIMATED_CANDIDATE_COLOR, linewidth=1.8
        )
        error_ax.set_yscale("log")
        error_ax.set_ylabel("Relative Error (%)")
        error_ax.set_xlabel(style.xlabel)
        all_errors = np.stack(
            [
                compute_aggregate_relative_error(
                    self.baseline, traj, epsilon=style.epsilon
                )
                for traj in self.trajectories.values()
            ]
        )
        error_ax.set_ylim(
            max(float(all_errors.min()) * 0.5, 1e-6), float(all_errors.max()) * 2.0
        )

        # Both the title and the per-depth legend entry are AXES-CHILD text
        # (`transform=axes[0, 0].transAxes`, `clip_on=False`), never
        # figure-level `fig.suptitle`/`fig.legend` (NB03 overhaul Phase 4c):
        # `blit=True` (see `__init__`) silently drops any figure-level text
        # artist positioned outside every axes' own bbox from the SAVED
        # frames (confirmed empirically -- the identical failure mode
        # `LandscapeAnimator`/`LineAnimator` hit and fixed the same way).
        # `fig.subplots_adjust` reserves the room `tight_layout` can't,
        # since animation frames are written directly (no `bbox_inches=
        # "tight"` cropping at save time, unlike the static figure above).
        legend = axes[0, 0].legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9
        )
        self._candidate_legend_text = legend.get_texts()[1]
        self._title = axes[0, 0].text(
            0.5,
            1.12,
            "",
            transform=axes[0, 0].transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
        fig.subplots_adjust(right=0.7, top=0.85)
        self.update(0)
        return fig, axes[0, 0]

    def update(self, frame_index: int) -> list[Artist]:
        depth = self.depths[frame_index]
        trajectory = self.trajectories[depth]
        t = np.arange(self.horizon)
        for j, line in enumerate(self._signal_lines):
            line.set_data(t, trajectory[:, j])
        assert (
            self._error_line is not None
            and self._title is not None
            and self._candidate_legend_text is not None
        )  # set in setup()
        error = compute_aggregate_relative_error(
            self.baseline, trajectory, epsilon=self.style.epsilon
        )
        self._error_line.set_data(t, error)
        candidate_name = self.style.contender_name or "Unfolded"
        self._title.set_text(
            f"{candidate_name} vs. {self.style.baseline_label} — "
            f"Unfolding Depth K = {depth}"
        )
        self._candidate_legend_text.set_text(
            f"{candidate_name} ({_depth_label(depth, self.costs)})"
        )
        return [
            *self._signal_lines,
            self._error_line,
            self._title,
            self._candidate_legend_text,
        ]
