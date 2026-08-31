"""Animated counterparts of Assets 1, 2, and 3-5: `FrameAnimator` (the
generic `FuncAnimation` driver -- owns the loop, the blit policy, and the
save path) plus three concrete animators. Nothing in this module knows
about controls, costs, gradient descent, or Riccati -- only matplotlib
Artists, frame indices, and the same `CostOracle`/array seams the static
renderers use.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

from ..style.io import save_animation
from .box_region import (
    draw_box_region_1d,
    draw_box_region_2d,
    draw_box_region_3d_shadow,
    draw_box_region_3d_wireframe,
    resolve_component_bounds,
)
from .grids import GridProjector
from .overlay import TrajectoryOverlay
from ..style import (
    BASELINE_COLOR,
    CANDIDATE_COLOR,
    COCP_COLOR,
    COCP_MARKER_ALPHA,
    DEFAULT_EPSILON,
    MASKED_SHADE,
    OPTIMUM_COLOR,
    SCATTER_CMAP,
    place_measured_legend,
)
from .signals import SignalStyle, compute_relative_error
from .types import (
    CostField,
    CostOracle,
    IterationSelection,
    SliceSpec,
    normalize_control,
    normalize_history,
)
from ..style import style_3d_axes, styled_figure


class FrameAnimator(ABC):
    """Generic `FuncAnimation` driver: subclasses declare their static
    backdrop (`setup`) and how one frame mutates the dynamic artists
    (`update`); this base owns the loop, the blit policy, and the save path.
    """

    def __init__(self, *, interval_ms: int = 200, blit: bool = True) -> None:
        """
        Args:
            interval_ms: milliseconds between frames.
            blit: whether `FuncAnimation` may blit (redraw only the artists
                `update` returns, rather than the whole figure) -- only
                correct when `update` mutates existing artists' *data*
                in-place rather than adding/removing artists (e.g. rebuilding
                a contour's collections); subclasses set this appropriately.
        """
        self.interval_ms = interval_ms
        self.blit = blit
        self._fig: Figure | None = None
        self._ax: Axes | None = None

    @abstractmethod
    def setup(self) -> tuple[Figure, Axes]:
        """Draw the static backdrop once and create the empty dynamic
        artists; called exactly once, before the first frame.

        Returns:
            ``(figure, axes)``.
        """
        ...

    @abstractmethod
    def update(self, frame_index: int) -> list[Artist]:
        """Mutate the dynamic artists for `frame_index`.

        Args:
            frame_index: the frame to render, ``0 <= frame_index <
                self.num_frames``.

        Returns:
            The mutated artists (used by `FuncAnimation` when `blit=True`).
        """
        ...

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Total number of frames this animation spans."""
        ...

    def build(self) -> FuncAnimation:
        """Construct (but do not save/show) the underlying `FuncAnimation`,
        with the static backdrop drawn under `viz.style.theme
        .styled_figure`'s rcParams.

        Returns:
            The constructed `FuncAnimation`.
        """
        with styled_figure():
            self._fig, self._ax = self.setup()
            return FuncAnimation(
                self._fig,
                self.update,
                frames=self.num_frames,
                interval=self.interval_ms,
                blit=self.blit,
                repeat=True,
            )

    def save(self, path: Path | str, *, fps: int = 10, writer: Any = None) -> Path:
        """Build (if needed) and persist this animation.

        `styled_figure`'s rcParams stay active for the whole call (not just
        `setup`): a per-frame-redrawn backdrop (e.g. `LandscapeAnimator`'s
        ``frame_axis="timestep"`` mode) creates *new* artists during
        `Animation.save`'s own frame loop, which must also see our style,
        not whatever the global rcParams happen to be at save time.

        Args:
            path: destination path (see `encode.save_animation`).
            fps: frames per second.
            writer: optional explicit matplotlib `Writer` instance.

        Returns:
            The resolved path the animation was written to.
        """
        with styled_figure():
            anim = self.build()
            saved = save_animation(anim, path, fps=fps, writer=writer)
            # THE DOUBLE-FRAME FIX: `self.build()` -> `self.setup()` leaves
            # `self._fig` open, and `%matplotlib inline` auto-displays every
            # STILL-OPEN figure at the end of a notebook cell -- without this,
            # every animation left its own setup figure open, so the cell that
            # saved+displayed the GIF ALSO auto-displayed one extra static
            # frame (whichever frame `anim.save`'s own loop left the artists
            # on) as a second, spurious inline image. Closing it here is safe
            # regardless of caller: the encoded GIF/MP4 bytes are already
            # written to `path` by this point, so the figure object itself is
            # no longer needed.
            if self._fig is not None:
                plt.close(self._fig)
            return saved


class SignalAnimator(FrameAnimator):
    """Asset 1 (animated): backdrop = baseline ``u(t)`` per component
    subplot; each frame ``i`` redraws the candidate ``U^(i)(t)``."""

    def __init__(
        self,
        baseline_U: np.ndarray,
        candidate_history: np.ndarray,
        *,
        interval_ms: int = 200,
        style: SignalStyle = SignalStyle(),
    ) -> None:
        """
        Args:
            baseline_U: reference control, shape ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            interval_ms: milliseconds between frames.
            style: slicing/labeling/title knobs (see `SignalStyle`).

        Raises:
            ValueError: If `candidate_history`'s ``(T, m)`` disagrees with
                `baseline_U`'s.
        """
        super().__init__(interval_ms=interval_ms, blit=True)
        self.baseline = normalize_control(
            "baseline_U", baseline_U, sample_index=style.sample_index
        )
        self.history = normalize_history(
            "candidate_history", candidate_history, sample_index=style.sample_index
        )
        if self.history.shape[1:] != self.baseline.shape:
            raise ValueError(
                f"candidate_history's per-iteration (T, m) = "
                f"{self.history.shape[1:]} must match baseline_U's (T, m) = "
                f"{self.baseline.shape}."
            )
        self.dim_labels = style.dim_labels
        self.xlabel = style.xlabel
        self.title = style.title
        self._lines: list = []

    @property
    def num_frames(self) -> int:
        return int(self.history.shape[0])

    def setup(self) -> tuple[Figure, Axes]:
        horizon, control_dim = self.baseline.shape
        t = np.arange(horizon)
        fig, axes = plt.subplots(
            nrows=control_dim,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(3.0, 2.5 * control_dim)),
        )
        self._lines = []
        for j in range(control_dim):
            ax = axes[j, 0]
            ax.plot(
                t,
                self.baseline[:, j],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=2.0,
                label="Baseline",
            )
            (line,) = ax.plot(
                [], [], color=CANDIDATE_COLOR, linewidth=2.2, label="Candidate"
            )
            self._lines.append(line)
            ax.set_ylabel(self.dim_labels[j] if self.dim_labels else f"$u_{{{j}}}$")
            ax.set_xlim(0, max(horizon - 1, 1))
            y_all = np.concatenate([self.baseline[:, j], self.history[:, :, j].ravel()])
            pad = 0.1 * (y_all.max() - y_all.min() + 1e-9)
            ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
        axes[-1, 0].set_xlabel(self.xlabel)
        fig.suptitle(self.title or "Control Signals (Animated)")
        # `tight_layout()` FIRST, solving the multi-row grid with no legend
        # yet drawn (an in-axes legend needed no extra margin, but this one
        # now sits outside) -- THEN `place_measured_legend` (NB04 COCP/viz
        # refinement plan Sec 2.3, folding in this animator for consistency
        # with every other landscape asset's external-legend convention)
        # reserves the additional right-hand margin on top of it via
        # `subplots_adjust(right=...)` alone, which only overrides that one
        # margin, leaving `tight_layout`'s other computed values in place.
        fig.tight_layout()
        place_measured_legend(fig, axes[0, 0])
        return fig, axes[0, 0]

    def update(self, frame_index: int) -> list[Artist]:
        t = np.arange(self.baseline.shape[0])
        for j, line in enumerate(self._lines):
            line.set_data(t, self.history[frame_index, :, j])
        return list(self._lines)


class RelativeErrorAnimator(FrameAnimator):
    """Asset 2 (animated): each frame ``i`` recomputes the per-component
    relative error for ``U^(i)``."""

    def __init__(
        self,
        baseline_U: np.ndarray,
        candidate_history: np.ndarray,
        *,
        epsilon: float = DEFAULT_EPSILON,
        interval_ms: int = 200,
        style: SignalStyle = SignalStyle(),
    ) -> None:
        """
        Args:
            baseline_U: reference control, shape ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            epsilon: relative-error denominator floor.
            interval_ms: milliseconds between frames.
            style: slicing/labeling/title knobs (see `SignalStyle`).

        Raises:
            ValueError: If `candidate_history`'s ``(T, m)`` disagrees with
                `baseline_U`'s.
        """
        super().__init__(interval_ms=interval_ms, blit=True)
        self.baseline = normalize_control(
            "baseline_U", baseline_U, sample_index=style.sample_index
        )
        self.history = normalize_history(
            "candidate_history", candidate_history, sample_index=style.sample_index
        )
        if self.history.shape[1:] != self.baseline.shape:
            raise ValueError(
                f"candidate_history's per-iteration (T, m) = "
                f"{self.history.shape[1:]} must match baseline_U's (T, m) = "
                f"{self.baseline.shape}."
            )
        self.epsilon = epsilon
        self.masked = np.abs(self.baseline) < epsilon
        self.dim_labels = style.dim_labels
        self.xlabel = style.xlabel
        self.title = style.title
        self._lines: list = []

    @property
    def num_frames(self) -> int:
        return int(self.history.shape[0])

    def setup(self) -> tuple[Figure, Axes]:
        horizon, control_dim = self.baseline.shape
        fig, axes = plt.subplots(
            nrows=control_dim,
            ncols=1,
            sharex=True,
            squeeze=False,
            figsize=(10, max(3.0, 2.5 * control_dim)),
        )
        self._lines = []
        for j in range(control_dim):
            ax = axes[j, 0]
            for t_idx in np.flatnonzero(self.masked[:, j]):
                ax.axvspan(t_idx - 0.5, t_idx + 0.5, color=MASKED_SHADE, zorder=0)
            (line,) = ax.plot([], [], color=CANDIDATE_COLOR, linewidth=2.0)
            self._lines.append(line)
            ax.set_yscale("log")
            ax.set_xlim(0, max(horizon - 1, 1))

            all_errors = compute_relative_error(
                self.baseline[:, j], self.history[:, :, j], epsilon=self.epsilon
            )
            valid = (
                all_errors[:, ~self.masked[:, j]]
                if (~self.masked[:, j]).any()
                else all_errors
            )
            if valid.size:
                lo, hi = float(valid.min()), float(valid.max())
                ax.set_ylim(max(lo * 0.5, 1e-12), max(hi * 2.0, lo * 0.5 * 10))
            ax.set_ylabel(
                f"Rel. error {self.dim_labels[j] if self.dim_labels else f'$u_{{{j}}}$'} (%)"
            )
        axes[-1, 0].set_xlabel(self.xlabel)
        fig.suptitle(self.title or "Relative Error (Animated)")
        fig.tight_layout()
        return fig, axes[0, 0]

    def update(self, frame_index: int) -> list[Artist]:
        t = np.arange(self.baseline.shape[0])
        for j, line in enumerate(self._lines):
            error = compute_relative_error(
                self.baseline[:, j],
                self.history[frame_index, :, j],
                epsilon=self.epsilon,
            )
            error = np.where(self.masked[:, j], np.nan, error)
            line.set_data(t, error)
        return list(self._lines)


@dataclass(frozen=True)
class LandscapeAnimationOptions:
    """Every knob of a `LandscapeAnimator` beyond its data inputs, as one
    frozen value object.

    Attributes:
        frame_axis: ``"iteration"`` (static backdrop, default) or
            ``"timestep"`` (recomputed backdrop, ``kind="contour"`` only).
        iteration_costs: optional ``J^(i)`` per iteration; required for
            ``kind="surface"`` (its moving point's z-height has no other
            source), optional for ``"contour"``/``"scatter"``.
        optimum_value: optional precomputed scalar cost at the reference
            point -- draws the optimum marker for ``kind="surface"``;
            ignored for ``"contour"``/``"scatter"``.
        selection: which iterations to draw the full path through
            (``frame_axis="iteration"`` reveals it progressively); unused
            for ``frame_axis="timestep"``.
        kind: ``"contour"``, ``"surface"``, or ``"scatter"``.
        interval_ms: milliseconds between frames.
        dtype: dtype grid batches are cast to before calling the oracle.
        device: device grid batches are cast to before calling the oracle.
        box_bounds: optional feasible-box bound(s) to highlight on the
            animated backdrop (NB04 COCP/viz refinement plan Sec 2.1) --
            same shape/semantics as `overlay.TrajectoryOverlay.box_bounds`;
            ``None`` (default) draws nothing. Only applies to
            ``frame_axis="iteration"`` (the static-backdrop mode this
            project actually renders); the legacy ``frame_axis="timestep"``
            path is unaffected.
        cocp_point: optional fixed reference control (e.g. COCP's realized
            decision at the frozen state) to mark on the animated backdrop,
            same role as `overlay.TrajectoryOverlay.cocp_point` -- ``None``
            (default) draws nothing.
        cocp_value: the backdrop's cost at `cocp_point`; required alongside
            it (mirrors `TrajectoryOverlay.cocp_value`).
        optimum_label: legend label for the optimum marker -- threaded
            through from the SAME variable the static renderer's own
            `optimum_label` uses (`notebook._render_unfolding_landscape_asset`),
            so the static and animated assets of one figure can never
            disagree (reference-bounds plan Sec 5.2: this used to be a
            hardcoded ``"Optimum"`` literal here, independent of the static
            side's ``"Riccati optimum"``). Defaults to ``"Optimum"``,
            reproducing prior behavior for any caller that doesn't override.
    """

    frame_axis: str = "iteration"
    iteration_costs: np.ndarray | None = None
    optimum_value: float | None = None
    selection: IterationSelection | None = None
    kind: str = "contour"
    interval_ms: int = 200
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")
    box_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    cocp_point: np.ndarray | None = None
    cocp_value: float | None = None
    optimum_label: str = "Optimum"


class LandscapeAnimator(FrameAnimator):
    """Assets 3/4/5 (animated): pluggable landscape backend (``kind``:
    ``"contour"`` | ``"surface"`` | ``"scatter"``) plus a `TrajectoryOverlay`,
    in one of two frame axes:

    - ``frame_axis="iteration"`` (default, the Decision-D1-approved mode):
      the landscape backdrop is **static** -- one grid evaluation total --
      and each frame reveals one more step of the candidate's path at
      ``spec.timestep``.
    - ``frame_axis="timestep"``: the landscape backdrop is **recomputed**
      per frame ``t`` (one grid evaluation per timestep, cached); the
      overlay is the candidate's own realized ``u_t`` at that timestep --
      the legacy ``MultiTimeStepContourAnimator`` behavior, generalized.
      Supported for ``kind="contour"`` only (see `__init__`).

    Blitting is only enabled for ``frame_axis="iteration"`` and
    ``kind="contour"``: 3D (`mpl_toolkits.mplot3d`) axes do not reliably
    support blitting, and ``frame_axis="timestep"`` rebuilds contour
    collections every frame (matplotlib cannot blit a collection swap --
    the same reason the legacy animator used ``blit=False``).
    """

    def __init__(
        self,
        reference_U: np.ndarray,
        candidate_history: np.ndarray,
        cost_oracle: CostOracle,
        spec: SliceSpec,
        *,
        projector: GridProjector | None = None,
        options: LandscapeAnimationOptions = LandscapeAnimationOptions(),
    ) -> None:
        """
        Args:
            reference_U: the fixed backdrop control sequence, shape
                ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            cost_oracle: scores a control batch (see `types.CostOracle`).
            spec: which timestep/components/ranges to project.
            projector: the `GridProjector` to use; defaults to one with the
                standard chunking.
            options: every animation knob -- frame axis, kind, per-iteration
                costs, optimum marker, iteration selection, frame interval,
                and grid-batch dtype/device (see
                `LandscapeAnimationOptions`).

        Raises:
            ValueError: If ``options.frame_axis``/``options.kind`` is not a
                recognized value, ``frame_axis="timestep"`` is combined with
                a 3D `kind`, or ``kind="surface"`` is requested without
                `iteration_costs`.
        """
        frame_axis, kind = options.frame_axis, options.kind
        iteration_costs = options.iteration_costs
        optimum_value, selection = options.optimum_value, options.selection
        interval_ms = options.interval_ms
        dtype, device = options.dtype, options.device
        box_bounds = options.box_bounds
        cocp_point, cocp_value = options.cocp_point, options.cocp_value
        optimum_label = options.optimum_label
        if frame_axis not in ("iteration", "timestep"):
            raise ValueError(
                f"frame_axis must be 'iteration' or 'timestep', got {frame_axis!r}."
            )
        if kind not in ("contour", "surface", "scatter"):
            raise ValueError(
                f"kind must be 'contour', 'surface', or 'scatter', got {kind!r}."
            )
        if frame_axis == "timestep" and kind != "contour":
            raise ValueError(
                "frame_axis='timestep' (a recomputed-per-frame backdrop) is "
                "only supported for kind='contour' -- a 3D surface/scatter "
                "backdrop is too expensive to rebuild every frame; use "
                "frame_axis='iteration' (the Decision-D1-approved default: "
                "a static backdrop the candidate descends over) instead."
            )
        if kind == "surface" and iteration_costs is None:
            raise ValueError(
                "kind='surface' requires iteration_costs (the trajectory's "
                "true per-iteration cost) to place the moving point's "
                "z-height; pass the persisted MetricsHistoryCallback loss "
                "column, or use kind='contour'/'scatter' instead."
            )

        effective_blit = frame_axis == "iteration" and kind == "contour"
        super().__init__(interval_ms=interval_ms, blit=effective_blit)

        self.reference_U = normalize_control("reference_U", reference_U)
        self.candidate_history = normalize_history(
            "candidate_history", candidate_history
        )
        self.cost_oracle = cost_oracle
        self.spec = spec
        self.projector = projector or GridProjector()
        self.frame_axis = frame_axis
        self.kind = kind
        self.dtype, self.device = dtype, device
        self.box_bounds = box_bounds
        self.optimum_label = optimum_label
        self.overlay = TrajectoryOverlay(
            self.candidate_history,
            selection=selection,
            iteration_costs=iteration_costs,
            baseline_U=self.reference_U,
            optimum_value=optimum_value,
            cocp_point=cocp_point,
            cocp_value=cocp_value,
        )
        self._dynamic_artists: list = []
        self._field_cache: dict = {}

    @property
    def num_frames(self) -> int:
        if self.frame_axis == "iteration":
            return int(self.candidate_history.shape[0])
        return int(self.reference_U.shape[0])

    def setup(self) -> tuple[Figure, Axes]:
        if self.frame_axis == "iteration":
            return self._setup_iteration_mode()
        return self._setup_timestep_mode()

    def update(self, frame_index: int) -> list[Artist]:
        if self.frame_axis == "iteration":
            return self._update_iteration_mode(frame_index)
        return self._update_timestep_mode(frame_index)

    # -- iteration mode: static backdrop, progressively-revealed path -----

    def _setup_iteration_mode(self) -> tuple[Figure, Axes]:
        field = self.projector.project(
            self.reference_U,
            self.cost_oracle,
            self.spec,
            dtype=self.dtype,
            device=self.device,
        )
        optimum = self.overlay.optimum_point(self.spec)
        a, b = self.spec.components[0], self.spec.components[1]

        if self.kind == "contour":
            # Wider than the (10, 6) themed default (second review pass):
            # the status text/legend to the right must fit ENTIRELY inside
            # the figure canvas -- unlike a static PNG (saved with
            # `bbox_inches="tight"`, which auto-expands the canvas around
            # everything), a saved animation frame has a FIXED pixel size,
            # so anything positioned past the canvas edge is silently
            # clipped rather than merely cropped-around. Confirmed directly
            # by measuring the POPULATED (not the empty placeholder) status
            # text's `get_window_extent()` against `fig.bbox`: it overflowed
            # the right edge by up to ~70px at the themed-default width.
            fig, ax = plt.subplots(figsize=(13.0, 6.0))
            u1_grid, u2_grid = field.grids
            box_bounds = resolve_component_bounds(self.box_bounds, 2)
            if box_bounds is not None:
                draw_box_region_2d(ax, box_bounds[0], box_bounds[1])
            # `contour` (line contours), not `contourf` (NB03 overhaul Phase
            # 4b): matches the static `ContourLandscapeRenderer`, and drops
            # the colorbar a filled contour would otherwise need.
            contour_fill = ax.contour(
                u1_grid, u2_grid, field.cost_grid, levels=30, cmap="viridis", alpha=0.7
            )
            ax.clabel(contour_fill, inline=True, fmt="%.1f", fontsize=8)
            if optimum is not None:
                # Coordinate/cost-enriched (NB04 COCP/viz refinement plan
                # Sec 2.2, item 1(b)) and fully opaque (dropped the previous
                # alpha=0.8, matching every other renderer's optimum star)
                # -- previously bare "Optimum", the one enrichment gap this
                # animated legend still had versus its static counterpart.
                ax.scatter(
                    optimum[0],
                    optimum[1],
                    color=OPTIMUM_COLOR,
                    marker="*",
                    s=200,
                    edgecolor="black",
                    zorder=10,
                    label=self.overlay.optimum_legend_label(
                        self.spec, self.optimum_label
                    ),
                )
            cocp = self.overlay.cocp_point_at(self.spec)
            if cocp is not None:
                # The violet-diamond COCP marker (NB04 COCP/viz refinement
                # plan Sec 2.2) -- fixed for the whole animation, same as
                # the optimum star above, since it is a controller's single
                # realized decision at the frozen state, not something that
                # advances per frame.
                ax.scatter(
                    cocp[0],
                    cocp[1],
                    color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                    marker="D",
                    s=150,
                    edgecolor="black",
                    zorder=10,
                    label=self.overlay.cocp_legend_label(self.spec, "COCP control"),
                )
            (path_line,) = ax.plot(
                [],
                [],
                color=CANDIDATE_COLOR,
                alpha=0.8,
                linewidth=2.0,
                zorder=6,
                label="Candidate path",
            )
            (point_marker,) = ax.plot(
                [],
                [],
                color=CANDIDATE_COLOR,
                marker="o",
                markeredgecolor="black",
                markersize=8,
                zorder=8,
            )
            # Component-only axis labels (NB03 overhaul Phase 4b): the
            # timestep belongs in the title (already there, below), not
            # doubled into each axis subscript.
            ax.set_xlabel(f"$u_{{{a}}}$")
            ax.set_ylabel(f"$u_{{{b}}}$")
            ax.set_title(f"Cost Landscape at $t={self.spec.timestep}$ (Animated)")
            # The updating status text lives OUTSIDE the axes' data area --
            # an AXES child (`ax.text(..., transform=ax.transAxes)`), never
            # a bare `fig.text(...)`: confirmed by direct testing of the
            # actual `Animation.save` path (not just a live `update()`
            # call): `blit=True` (this kind's `effective_blit`) restores/
            # redraws per-AXES regions, so a figure-level text artist
            # outside every axes' own bbox is silently never blitted into
            # the saved frames, rendering as permanently blank despite
            # holding the correct content in memory. `clip_on=False` is
            # required since its eventual axes-fraction x > 1.0 would
            # otherwise be clipped to the axes' own data bounds. Its actual
            # position is set by `place_measured_legend` below, directly
            # beneath the legend's OWN measured position -- structurally
            # ruling out the previous `x=0.5` (axes-fraction, i.e. the
            # MIDDLE of the plot) placement bug, which merely happened not
            # to collide with the legend at the one figure size it was
            # tuned against.
            # Seeded with the REAL frame-0 content (never an empty
            # placeholder) before `place_measured_legend` measures it below:
            # an empty string measures far narrower than the actual status
            # text `_update_iteration_mode` sets on every real frame, which
            # under-reserved the right margin and left the populated text
            # clipped at the saved GIF's own canvas edge (confirmed directly
            # in a rendered frame).
            initial_text = self.overlay.animated_status_text(
                0, self.overlay.path_at(self.spec)[0]
            )
            text = ax.text(
                0,
                0,
                initial_text,
                transform=ax.transAxes,
                fontsize=10,
                zorder=9,
                clip_on=False,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
            )
            fig.subplots_adjust(left=0.06)  # a snug left margin, not the ~0.125 default
            place_measured_legend(
                fig,
                ax,
                status_text=text,
                legend_kwargs={"loc": "upper left", "fontsize": 9},
            )
            self._dynamic_artists = [path_line, point_marker, text]
            return fig, ax

        # Wider than the (10, 6) themed default (second review pass): this
        # 3D animation carries a colorbar AND (NB04 COCP/viz refinement plan
        # Sec 2.3: previously absent here entirely) a legend, sharing the
        # same right-hand margin as the status text -- (15.0, 6.5) is sized
        # to what a 3-entry (path, optimum, COCP) legend plus a colorbar
        # plus the status text actually need, snug gaps included (matches
        # the static `SurfaceLandscapeRenderer`/`ScatterLandscapeRenderer`'s
        # own comparable figure width).
        fig = plt.figure(figsize=(15.0, 6.5))
        ax = fig.add_subplot(projection="3d")
        optimum_label = self.overlay.optimum_legend_label(self.spec, self.optimum_label)
        cocp = self.overlay.cocp_point_at(self.spec)
        cocp_label = self.overlay.cocp_legend_label(self.spec, "COCP control")
        if self.kind == "surface":
            u1_grid, u2_grid = field.grids
            surface = ax.plot_surface(
                u1_grid,
                u2_grid,
                field.cost_grid,
                cmap="viridis",
                alpha=0.7,
                linewidth=0,
                antialiased=True,
            )
            # pad=0.05 (third review pass, down from 0.12): the larger pad
            # left a big dead gap between the axes and the colorbar,
            # confirmed directly in a rendered GIF.
            fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Cost")
            ax.set_zlabel(
                "Cost"
            )  # not a control component -- no timestep/index to strip
            box_bounds = resolve_component_bounds(self.box_bounds, 2)
            if box_bounds is not None:
                draw_box_region_3d_shadow(
                    ax, box_bounds[0], box_bounds[1], z=float(field.cost_grid.min())
                )
            # Static (never updated per-frame, unlike path_line/text below):
            # the optimum's z-height generally doesn't fall on a discretized
            # grid node, so it can only be placed given an explicit
            # `overlay.optimum_value` (see `TrajectoryOverlay`'s docstring).
            # Labeled (NB04 COCP/viz refinement plan Sec 2.2, item 1(b)) --
            # previously bare, since this 3D branch never had a legend at
            # all before this plan.
            if optimum is not None and self.overlay.optimum_value is not None:
                ax.scatter(
                    *optimum,
                    self.overlay.optimum_value,
                    color=OPTIMUM_COLOR,
                    marker="*",
                    s=200,
                    edgecolor="black",
                    zorder=10,
                    label=optimum_label,
                )
            if cocp is not None and self.overlay.cocp_value is not None:
                ax.scatter(
                    cocp[0],
                    cocp[1],
                    self.overlay.cocp_value,
                    color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                    marker="D",
                    s=150,
                    edgecolor="black",
                    zorder=10,
                    label=cocp_label,
                )
        else:  # scatter
            # THE OCCLUSION FIX (NB03 overhaul): without this, mplot3d
            # depth-sorts each ARTIST by its own mean camera-space depth,
            # ignoring the `zorder=` values set below and on `path_line`/
            # `point_marker`/the optimum star -- confirmed directly in a
            # rendered GIF frame: the optimum star never appeared despite
            # being drawn with `zorder=10`. Disabling matplotlib's own depth
            # computation makes it respect `zorder` literally, matching the
            # identical fix in `landscapes.ScatterLandscapeRenderer`.
            ax.computed_zorder = False
            grid_i, grid_j, grid_k = field.grids
            c = self.spec.components[2]
            scatter_alpha = 0.12
            scatter = ax.scatter(
                grid_i.ravel(),
                grid_j.ravel(),
                grid_k.ravel(),
                c=field.cost_grid.ravel(),
                cmap=SCATTER_CMAP,
                alpha=scatter_alpha,
                s=8,
                linewidths=0,
            )
            # pad=0.05 (tighter than 0.12 -- third review pass: the previous
            # padding left a large dead gap between the axes and the
            # colorbar, confirmed directly in a rendered GIF).
            cbar = fig.colorbar(
                scatter, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Cost"
            )
            # THE COLORBAR-INTENSITY FIX (NB03 overhaul, THIRD review pass --
            # reverts the second pass's "undim" attempt, see
            # `landscapes.ScatterLandscapeRenderer`'s identical fix for the
            # full rationale): a single colorbar swatch can only show ONE
            # alpha, and matching it to the exact alpha each plotted point
            # actually uses is the one choice that stays truthful regardless
            # of screen density, zoom, or figure size -- unlike "leave it
            # fully opaque because dense points visually stack," which broke
            # again the moment the figure was widened (fewer points overlap
            # per screen pixel), confirmed directly in a rendered GIF.
            assert cbar.solids is not None  # populated by fig.colorbar above
            cbar.solids.set_alpha(scatter_alpha)
            # Component-only z-label (NB03 overhaul Phase 4b): the timestep
            # belongs in the title (already there, below).
            ax.set_zlabel(f"$u_{{{c}}}$")
            box_bounds = resolve_component_bounds(self.box_bounds, 3)
            if box_bounds is not None:
                draw_box_region_3d_wireframe(
                    ax, box_bounds[0], box_bounds[1], box_bounds[2]
                )
            # Unlike the surface branch above, `optimum`/`cocp` here ALREADY
            # carry all 3 varied coordinates (a 3-component SliceSpec) --
            # there is no separate z-height to append from `optimum_value`/
            # `cocp_value` (that would pass it as a spurious 4th positional
            # arg, misread as `ax.scatter`'s `s` marker-size parameter).
            if optimum is not None:
                ax.scatter(
                    *optimum,
                    color=OPTIMUM_COLOR,
                    marker="*",
                    s=200,
                    edgecolor="black",
                    zorder=10,
                    label=optimum_label,
                )
            if cocp is not None:
                ax.scatter(
                    *cocp,
                    color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                    marker="D",
                    s=150,
                    edgecolor="black",
                    zorder=10,
                    label=cocp_label,
                )

        (path_line,) = ax.plot(
            [], [], color=CANDIDATE_COLOR, alpha=0.8, linewidth=2.0, zorder=6
        )
        (point_marker,) = ax.plot(
            [],
            [],
            color=CANDIDATE_COLOR,
            marker="o",
            markeredgecolor="black",
            markersize=8,
            zorder=8,
        )
        # The updating status text lives OUTSIDE the axes -- an AXES CHILD
        # (`ax.text2D(..., transform=ax.transAxes)`, `Axes3D`'s 2D-overlay
        # counterpart of the plain 2D `Axes.text`), never a bare
        # `fig.text(...)` (NB04 COCP/viz refinement plan Sec 2.3): 3D
        # animations never blit (`effective_blit` is False for every
        # non-contour `kind`), so this was never a correctness bug the way
        # the contour branch's was, but making every landscape asset use the
        # SAME mechanism is simpler than special-casing one. Its actual
        # position is set by `place_measured_legend` below.
        # Seeded with the REAL frame-0 content, not an empty placeholder --
        # see the contour branch's identical fix above for why: an empty
        # string measures far narrower than the actual populated text,
        # under-reserving the margin `place_measured_legend` computes below.
        initial_text = self.overlay.animated_status_text(
            0, self.overlay.path_at(self.spec)[0]
        )
        text = ax.text2D(
            0,
            0,
            initial_text,
            transform=ax.transAxes,
            fontsize=10,
            clip_on=False,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        # Component-only axis labels (NB03 overhaul Phase 4b): the timestep
        # belongs in the title (already there, below), not doubled into each
        # axis subscript.
        ax.set_xlabel(f"$u_{{{a}}}$")
        ax.set_ylabel(f"$u_{{{b}}}$")
        ax.set_title(f"Cost Landscape (3D) at $t={self.spec.timestep}$ (Animated)")
        # Micro-Prompt 5: the scatter animation gets the same "deep bowl"
        # camera/label-padding treatment as the surface (previously
        # surface-only) -- a rotated default camera made Asset 5's sparse
        # cloud read as even sparser/less structured than intended. Fixed
        # BEFORE the legend below, matching `plot_loss_landscape_3d`'s own
        # "camera before annotation" ordering.
        if self.kind in ("surface", "scatter"):
            style_3d_axes(ax)
        # `place_measured_legend` (NB04 COCP/viz refinement plan Sec 2.3)
        # measures against every axes on this figure (the colorbar's own
        # included), so it corrects for whatever this starting anchor
        # doesn't already clear -- replacing the previous bare
        # `fig.subplots_adjust(right=0.53)` (no legend existed to place).
        place_measured_legend(
            fig,
            ax,
            status_text=text,
            legend_kwargs={"loc": "upper left", "bbox_to_anchor": (1.01, 0.95)},
        )
        self._dynamic_artists = [path_line, point_marker, text]
        return fig, ax

    def _update_iteration_mode(self, i: int) -> list[Artist]:
        full_path = self.overlay.path_at(self.spec)  # (I, ncomp)
        path_so_far = full_path[: i + 1]
        # The dynamic "Iteration i | J=... | u=..." status text is
        # centralized on the overlay (NB03 overhaul Phase 4b) so the 2D
        # contour and 3D surface/scatter branches below never drift apart.
        progress_text = self.overlay.animated_status_text(i, path_so_far[-1])

        if self.kind == "contour":
            path_line, point_marker, text = self._dynamic_artists
            path_line.set_data(path_so_far[:, 0], path_so_far[:, 1])
            point_marker.set_data([path_so_far[-1, 0]], [path_so_far[-1, 1]])
            text.set_text(progress_text)
            return [path_line, point_marker, text]

        path_line, point_marker, text = self._dynamic_artists
        if self.kind == "surface":
            assert self.overlay.iteration_costs is not None  # required for surface
            z = self.overlay.iteration_costs[: i + 1]
        else:  # scatter: the 3rd varied component IS the z-position
            z = path_so_far[:, 2]
        path_line.set_data_3d(path_so_far[:, 0], path_so_far[:, 1], z)
        point_marker.set_data_3d([path_so_far[-1, 0]], [path_so_far[-1, 1]], [z[-1]])
        text.set_text(progress_text)
        return [path_line, point_marker, text]

    # -- timestep mode: recomputed-per-frame backdrop (contour only) ------

    def _setup_timestep_mode(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        self._ax = ax
        (point_marker,) = ax.plot(
            [],
            [],
            color=CANDIDATE_COLOR,
            marker="o",
            markersize=12,
            zorder=8,
            label="Realized control",
        )
        text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )
        a, b = self.spec.components
        ax.set_xlabel(f"$u_{{\\cdot,{a}}}$")
        ax.set_ylabel(f"$u_{{\\cdot,{b}}}$")
        ax.set_title("Cost Landscape Across Time Steps (Animated)")
        self._dynamic_artists = [point_marker, text]
        self._update_timestep_mode(0)
        return fig, ax

    def _field_for_timestep(self, t_index: int) -> CostField:
        field = self._field_cache.get(t_index)
        if field is None:
            spec_t = SliceSpec(
                timestep=t_index,
                components=self.spec.components,
                ranges=self.spec.ranges,
            )
            field = self.projector.project(
                self.reference_U,
                self.cost_oracle,
                spec_t,
                dtype=self.dtype,
                device=self.device,
            )
            self._field_cache[t_index] = field
        return field

    def _update_timestep_mode(self, t_index: int) -> list[Artist]:
        field = self._field_for_timestep(t_index)
        ax = self._ax
        assert ax is not None  # setup() ran before any update
        for collection in list(ax.collections):
            collection.remove()
        u1_grid, u2_grid = field.grids
        ax.contourf(
            u1_grid, u2_grid, field.cost_grid, levels=30, cmap="viridis", alpha=0.9
        )

        # The realized candidate control at this timestep, from the final
        # (converged) iterate -- the legacy MultiTimeStepContourAnimator's
        # "final GD point at each time step" semantics, generalized.
        realized = self.candidate_history[-1, t_index, list(self.spec.components)]
        point_marker, text = self._dynamic_artists
        point_marker.set_data([realized[0]], [realized[1]])
        text.set_text(f"t={t_index}")
        ax.set_xlim(u1_grid.min(), u1_grid.max())
        ax.set_ylim(u2_grid.min(), u2_grid.max())
        return [point_marker, text]


class LineAnimator(FrameAnimator):
    """Asset 3a (animated, m=1): backdrop = the static cost-vs-parameter
    curve, drawn once; each frame reveals one more step of the candidate's
    path -- the m=1 counterpart of `LandscapeAnimator`'s
    ``frame_axis="iteration"``/``kind="contour"`` mode, but as its own small
    sibling class rather than a 4th `kind` shoehorned into
    `LandscapeAnimator` (whose `_setup_iteration_mode`/`_update_iteration_mode`
    already hardcode 2-vs-3-component unpacking throughout, not a per-kind
    hook a 4th case could slot into cheaply). There is exactly one way to
    draw a 1D cost curve, so unlike `LandscapeAnimator` this animator has no
    `kind`/`frame_axis` knobs at all.
    """

    def __init__(
        self,
        reference_U: np.ndarray,
        candidate_history: np.ndarray,
        cost_oracle: CostOracle,
        spec: SliceSpec,
        *,
        projector: GridProjector | None = None,
        options: LandscapeAnimationOptions = LandscapeAnimationOptions(),
    ) -> None:
        """
        Args:
            reference_U: the fixed backdrop control sequence, shape
                ``(T, m)`` or ``(B, T, m)``.
            candidate_history: per-iteration control iterates, shape
                ``(I, T, m)`` or ``(I, B, T, m)``.
            cost_oracle: scores a control batch (see `types.CostOracle`).
            spec: which timestep/component/range to project -- must have
                exactly 1 component.
            projector: the `GridProjector` to use; defaults to one with the
                standard chunking.
            options: `options.iteration_costs` is REQUIRED (the y-axis is
                cost, not a second control component -- see
                `overlay.TrajectoryOverlay.draw`'s 1-component branch);
                `options.kind`/`options.frame_axis` are ignored (this
                animator has exactly one mode).

        Raises:
            ValueError: If `spec` doesn't have exactly 1 component, or
                `options.iteration_costs` is ``None``.
        """
        if len(spec.components) != 1:
            raise ValueError(
                f"LineAnimator requires a 1-component SliceSpec, got "
                f"{len(spec.components)}."
            )
        if options.iteration_costs is None:
            raise ValueError(
                "LineAnimator requires options.iteration_costs (the y-axis "
                "is cost, not a second control component)."
            )
        super().__init__(interval_ms=options.interval_ms, blit=True)
        self.reference_U = normalize_control("reference_U", reference_U)
        self.candidate_history = normalize_history(
            "candidate_history", candidate_history
        )
        self.cost_oracle = cost_oracle
        self.spec = spec
        self.projector = projector or GridProjector()
        self.dtype, self.device = options.dtype, options.device
        self.box_bounds = options.box_bounds
        self.optimum_label = options.optimum_label
        self.overlay = TrajectoryOverlay(
            self.candidate_history,
            selection=options.selection,
            iteration_costs=options.iteration_costs,
            baseline_U=self.reference_U,
            optimum_value=options.optimum_value,
            cocp_point=options.cocp_point,
            cocp_value=options.cocp_value,
        )
        self._dynamic_artists: list = []

    @property
    def num_frames(self) -> int:
        return int(self.candidate_history.shape[0])

    def setup(self) -> tuple[Figure, Axes]:
        field = self.projector.project(
            self.reference_U,
            self.cost_oracle,
            self.spec,
            dtype=self.dtype,
            device=self.device,
        )
        (u_grid,) = field.grids
        (a,) = self.spec.components
        optimum = self.overlay.optimum_point(self.spec)

        # Wider than the (10, 6) themed default (second review pass): the
        # status text/legend to the right must fit ENTIRELY inside the
        # figure canvas -- unlike a static PNG (saved with
        # `bbox_inches="tight"`, which auto-expands the canvas around
        # everything), a saved animation frame has a FIXED pixel size, so
        # anything positioned past the canvas edge is silently clipped
        # rather than merely cropped-around. Confirmed directly by measuring
        # the POPULATED (not the empty placeholder) status text's
        # `get_window_extent()` against `fig.bbox`: it overflowed the right
        # edge at the themed-default width.
        fig, ax = plt.subplots(figsize=(13.0, 6.0))
        box_bounds = resolve_component_bounds(self.box_bounds, 1)
        if box_bounds is not None:
            draw_box_region_1d(ax, box_bounds[0])
        ax.plot(
            u_grid,
            field.cost_grid,
            color="tab:blue",
            linewidth=2.0,
            zorder=1,
            label="Cost landscape",
        )
        if optimum is not None and self.overlay.optimum_value is not None:
            # Coordinate/cost-enriched (NB04 COCP/viz refinement plan Sec
            # 2.2, item 1(b)) -- previously bare "Optimum".
            ax.scatter(
                optimum[0],
                self.overlay.optimum_value,
                color=OPTIMUM_COLOR,
                marker="*",
                s=200,
                edgecolor="black",
                zorder=10,
                label=self.overlay.optimum_legend_label(self.spec, self.optimum_label),
            )
        cocp = self.overlay.cocp_point_at(self.spec)
        if cocp is not None and self.overlay.cocp_value is not None:
            # The violet-diamond COCP marker (NB04 COCP/viz refinement plan
            # Sec 2.2) -- fixed for the whole animation, same as the optimum
            # star above.
            ax.scatter(
                cocp[0],
                self.overlay.cocp_value,
                color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                marker="D",
                s=150,
                edgecolor="black",
                zorder=10,
                label=self.overlay.cocp_legend_label(self.spec, "COCP control"),
            )
        (path_line,) = ax.plot(
            [], [], color=CANDIDATE_COLOR, alpha=0.8, linewidth=2.0, zorder=6
        )
        (point_marker,) = ax.plot(
            [],
            [],
            color=CANDIDATE_COLOR,
            marker="o",
            markeredgecolor="black",
            markersize=8,
            zorder=8,
        )
        # THE MINIMUM-CLIPPING FIX (NB03 overhaul Phase 4b): the previous
        # `ax.set_ylim(y_lo, ...)` placed the LOWER bound at the grid's exact
        # minimum -- precisely where the optimum star marker sits -- with
        # zero clearance, so its own marker radius rendered partially cut off
        # by the bottom axis spine. A small padding fraction below `y_lo`
        # (mirroring the existing headroom fraction above `y_hi`) keeps the
        # optimum, and the curve's own minimum, fully inside the axes.
        y_lo, y_hi = float(field.cost_grid.min()), float(field.cost_grid.max())
        span = (y_hi - y_lo) or 1.0
        bottom_pad = 0.08 * span
        ax.set_ylim(y_lo - bottom_pad, y_hi)
        ax.set_xlabel(f"$u_{{{a}}}$")  # component-only (timestep is in the title)
        ax.set_ylabel("Cost")
        ax.set_title(f"Cost Landscape at $t={self.spec.timestep}$ (Animated)")
        # The updating status text lives OUTSIDE the axes' data area -- an
        # AXES child (`ax.text(..., transform=ax.transAxes)`), never a bare
        # `fig.text(...)` -- see `LandscapeAnimator._setup_iteration_mode`'s
        # "THE BLIT-SAVE FIX" note: this animator also blits (`blit=True`),
        # and a figure-level text outside every axes' own bbox is silently
        # never blitted into the saved frames. Its actual position is set by
        # `place_measured_legend` below. Seeded with the REAL frame-0
        # content, not an empty placeholder -- see `LandscapeAnimator`'s
        # identical fix for why an empty string under-measures.
        initial_text = self.overlay.animated_status_text(
            0, self.overlay.path_at(self.spec)[0]
        )
        text = ax.text(
            0,
            0,
            initial_text,
            transform=ax.transAxes,
            fontsize=10,
            zorder=9,
            clip_on=False,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        fig.subplots_adjust(left=0.06)  # a snug left margin, not the ~0.125 default
        # `place_measured_legend` (NB04 COCP/viz refinement plan Sec 2.3)
        # places the legend, pins `text` directly beneath its measured
        # position, and reserves the right margin from that measurement --
        # replacing the previous fixed `right=0.72` + hand-anchored legend.
        place_measured_legend(
            fig,
            ax,
            status_text=text,
            legend_kwargs={"loc": "upper left", "fontsize": 9},
        )
        self._dynamic_artists = [path_line, point_marker, text]
        return fig, ax

    def update(self, frame_index: int) -> list[Artist]:
        full_path = self.overlay.path_at(self.spec)  # (I, 1)
        assert self.overlay.iteration_costs is not None  # required, see __init__
        costs_so_far = self.overlay.iteration_costs[: frame_index + 1]
        path_so_far = full_path[: frame_index + 1, 0]
        progress_text = self.overlay.animated_status_text(
            frame_index, full_path[frame_index]
        )

        path_line, point_marker, text = self._dynamic_artists
        path_line.set_data(path_so_far, costs_so_far)
        point_marker.set_data([path_so_far[-1]], [costs_so_far[-1]])
        text.set_text(progress_text)
        return [path_line, point_marker, text]
