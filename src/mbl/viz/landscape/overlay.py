"""`TrajectoryOverlay`: the shared static + animated engine for Asset 6 --
sparse subsampling, annotation, and the reserved-color optimum marker,
reused by every landscape renderer (Assets 3-5, `landscapes.py`) and the
animated `animators.LandscapeAnimator`. The overlay is a cross-cutting
capability, not a standalone plot (SRP): it owns only how the candidate
trajectory is subsampled/drawn, never the landscape backdrop itself.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba

from ..style import CANDIDATE_COLOR, COCP_COLOR, COCP_MARKER_ALPHA, OPTIMUM_COLOR
from .types import IterationSelection, SliceSpec
from ...core.utils import ensure_positive_integer

#: Default number of sparse markers a TrajectoryOverlay annotates when no
#: explicit IterationSelection is given -- dense enough to trace the descent,
#: sparse enough to avoid the visual crowding the mandate warns against.
_DEFAULT_SPARSE_COUNT = 5

#: Light-to-dark shading ramp for `TrajectoryOverlay.point_shades`, built from
#: `CANDIDATE_COLOR` itself (a pale tint of it -> its own full saturation) --
#: deliberately NOT the backdrop's own cost colormap (`viridis`/`plasma`), so
#: the per-point convergence markers never get confused with the landscape
#: they're overlaid on, while still reading as "the candidate"'s own family
#: of colors. Early iterations render pale, the converged iterate renders at
#: full `CANDIDATE_COLOR` saturation -- a second, free channel (besides
#: position) for "how far along is this point".
_POINT_SHADE_CMAP = LinearSegmentedColormap.from_list(
    "candidate_point_shades", ["#dcf3ea", CANDIDATE_COLOR]
)


def _format_point(point: np.ndarray) -> str:
    """``"[x.xxx, x.xxx]"`` -- the coordinate formatting shared by
    `TrajectoryOverlay.optimum_legend_label`/`.path_legend_label` (Micro-
    Prompt 4c)."""
    return "[" + ", ".join(f"{v:.3f}" for v in point) + "]"


def select_iterations(
    num_iterations: int, k: int, *, mode: str = "log"
) -> IterationSelection:
    """Select a sparse subset of iteration indices from
    ``range(num_iterations)`` for static-plot annotation (Asset 6) --
    log-spaced by default (dense early, where gradient descent moves
    fastest; sparse late), always including ``0`` and
    ``num_iterations - 1``.

    Args:
        num_iterations: total number of iterations available (``I``).
        k: the (maximum) number of indices to select.
        mode: ``"log"`` (log-spaced, default) or ``"linear"`` (evenly
            spaced). For a caller-supplied explicit set of indices, build an
            `IterationSelection` directly instead of calling this function.

    Returns:
        The selected `IterationSelection`, sorted ascending and
        deduplicated.

    Raises:
        ValueError: If `num_iterations`/`k` is not a positive ``int``, or
            `mode` is not ``"log"`` or ``"linear"``.
    """
    ensure_positive_integer(num_iterations, "num_iterations")
    ensure_positive_integer(k, "k")
    k = min(k, num_iterations)

    if mode == "log":
        # geomspace(1, num_iterations, num=k) starts at 1 (-> index 0) and
        # ends at num_iterations (-> index num_iterations - 1), so both
        # endpoints fall out of the log-spacing itself.
        raw = np.round(np.geomspace(1, num_iterations, num=k)).astype(int) - 1
    elif mode == "linear":
        raw = np.round(np.linspace(0, num_iterations - 1, num=k)).astype(int)
    else:
        raise ValueError(f"mode must be 'log' or 'linear', got {mode!r}.")

    indices = np.unique(np.clip(raw, 0, num_iterations - 1))
    return IterationSelection(indices=indices, mode=mode)


@dataclass
class TrajectoryOverlay:
    """Sparse-subsampled, annotated candidate trajectory overlaid on a cost
    landscape (Asset 6).

    Attributes:
        candidate_history: per-iteration control iterates, shape
            ``(I, T, m)`` (use `types.normalize_history` first for a batched
            input).
        selection: which iteration indices to mark/annotate (see
            `select_iterations`); defaults to a log-spaced sparse subset of
            all ``I`` iterations if not given.
        iteration_costs: optional ``J^(i)`` per iteration, shape ``(I,)`` --
            reused directly from e.g. `engine.callbacks
            .MetricsHistoryCallback`'s persisted history (never
            recomputed); annotations fall back to index-only labels when
            omitted.
        baseline_U: optional ``(T, m)`` reference control -- if given, its
            spec-projected point is drawn as the reserved "optimum" marker.
        optimum_value: optional precomputed scalar cost AT `baseline_U`'s
            spec-projected point -- needed only by a 3D surface renderer
            (`landscapes.SurfaceLandscapeRenderer`) to place the optimum
            marker's z-height, since that point generally does not fall
            exactly on a discretized grid node the way `iteration_costs`
            already supplies a z-height for every candidate iterate.
        box_bounds: optional feasible-box bound(s) to highlight on the
            landscape backdrop (NB04 plan Sec 3.4) -- either a single
            ``(low, high)`` pair, broadcast to every one of `SliceSpec
            .components`' varied components (the common ISOTROPIC case,
            e.g. a scalar ``u_max`` applied uniformly, matching
            `core.constraint.box_constraint.BoxConstraint`'s own default),
            or one pair per varied component for an anisotropic box.
            Resolved per-renderer via `box_region.resolve_component_bounds`.
            Lives here (on the already-required overlay bundle) rather than
            as a new parameter on every renderer's own ``render()`` --
            `Bounds1D`/`Bounds2D`/`Bounds3D` renderers all under this
            project's PLR0913=6 signature-budget gate. ``None`` (default)
            draws nothing, so every unconstrained (NB03) call site is
            unaffected.
        cocp_point: optional second reference marker -- a controller's
            REALIZED control decision at the frozen state this landscape
            slice freezes (e.g. COCP's one-step QP solution; NB04 COCP/viz
            refinement plan Sec 2.2/3.4), as opposed to `baseline_U`'s
            *unconstrained* Riccati optimum. Unlike `baseline_U` (a full
            ``(T, m)`` trajectory sliced per-timestep via `optimum_point`),
            this is already the single frozen instant's full control vector,
            shape ``(m_full,)`` -- projected onto `SliceSpec.components` via
            `cocp_point_at`. ``None`` (default) draws nothing, so this
            bundle stays optional exactly like `box_bounds`.
        cocp_value: the SAME backdrop's cost at `cocp_point` (i.e. evaluated
            by the caller against whichever cost-to-go model `optimum_value`
            was also scored against, e.g. `models.analytic.riccati
            .evaluate_local_cost_to_go`), making the two markers directly
            comparable on one bowl. Required alongside `cocp_point` for a
            1-component slice (its y-axis IS cost); optional for 2-/3-
            component slices, where it only enriches the legend.
    """

    candidate_history: np.ndarray
    selection: IterationSelection | None = None
    iteration_costs: np.ndarray | None = None
    baseline_U: np.ndarray | None = None
    optimum_value: float | None = None
    box_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    cocp_point: np.ndarray | None = None
    cocp_value: float | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards, and the default sparse `selection` (Part 5.2:
        static plots must heavily subsample by default).

        Raises:
            ValueError: If `iteration_costs` is given and its length
                disagrees with `candidate_history`'s iteration count.
        """
        num_iterations = self.candidate_history.shape[0]
        if self.iteration_costs is not None and len(self.iteration_costs) != (
            num_iterations
        ):
            raise ValueError(
                f"iteration_costs must have length {num_iterations} (one "
                f"per candidate_history iteration), got "
                f"{len(self.iteration_costs)}."
            )
        if self.selection is None:
            self.selection = select_iterations(
                num_iterations, k=min(_DEFAULT_SPARSE_COUNT, num_iterations)
            )

    def path_at(self, spec: SliceSpec) -> np.ndarray:
        """The full (all ``I`` iterations) trajectory path projected onto
        `spec`'s timestep/components -- the thin connecting line beneath the
        sparse markers.

        Args:
            spec: which timestep/components to project onto.

        Returns:
            Shape ``(I, len(spec.components))``.
        """
        return self.candidate_history[:, spec.timestep, :][:, list(spec.components)]

    def selected_path(self, spec: SliceSpec) -> tuple[np.ndarray, np.ndarray]:
        """The sparsely-selected subset of `path_at`.

        Args:
            spec: which timestep/components to project onto.

        Returns:
            ``(indices, points)``: `indices` shape ``(k',)``, `points`
                shape ``(k', len(spec.components))``.
        """
        assert self.selection is not None  # __post_init__ defaults it
        indices = self.selection.indices
        return indices, self.path_at(spec)[indices]

    def point_shades(self, n: int) -> list[str]:
        """`n` hex colors, light-to-dark along `_POINT_SHADE_CMAP`, one per
        sparse convergence marker -- so a reader can tell which point is
        which directly from the legend's marker colors, without cross-
        referencing index labels (NB03 overhaul: the per-point markers
        `draw`/`landscapes.SurfaceLandscapeRenderer` render were previously
        all the same flat `CANDIDATE_COLOR`).

        Args:
            n: number of markers to shade (typically ``len(self.selection
                .indices)``).

        Returns:
            ``n`` hex color strings, ordered earliest-iteration-first
            (palest) to latest (full `CANDIDATE_COLOR` saturation).
        """
        if n <= 1:
            return [CANDIDATE_COLOR] * n
        return [to_hex(_POINT_SHADE_CMAP(t)) for t in np.linspace(0.0, 1.0, n)]

    def cost_at(self, iteration: int) -> float | None:
        """``J^(i)`` for annotation, or ``None`` if `iteration_costs` was
        never supplied.

        Args:
            iteration: the iteration index to look up.
        """
        if self.iteration_costs is None:
            return None
        return float(self.iteration_costs[iteration])

    def optimum_point(self, spec: SliceSpec) -> np.ndarray | None:
        """`baseline_U`'s value at `spec`'s timestep/components, or ``None``
        if no `baseline_U` was supplied.

        Args:
            spec: which timestep/components to project onto.
        """
        if self.baseline_U is None:
            return None
        return self.baseline_U[spec.timestep, list(spec.components)]

    def cocp_point_at(self, spec: SliceSpec) -> np.ndarray | None:
        """`cocp_point`'s value at `spec`'s components, or ``None`` if no
        `cocp_point` was supplied. Unlike `optimum_point`, there is no
        `spec.timestep` to slice by: `cocp_point` is already the single
        frozen instant's full control vector (NB04 COCP/viz refinement plan
        Sec 2.2), so only the varied components are projected out.

        Args:
            spec: which components to project onto.
        """
        if self.cocp_point is None:
            return None
        return self.cocp_point[list(spec.components)]

    def annotation_text(self, iteration: int) -> str:
        """The ``"i=<iteration>[, J=<cost>]"`` label a marker for `iteration`
        is annotated with -- shared by every renderer that labels individual
        iterates (2D `draw`, and `landscapes.SurfaceLandscapeRenderer`'s 3D
        text annotations), so the format never drifts between them."""
        cost = self.cost_at(iteration)
        return f"i={iteration}" if cost is None else f"i={iteration}, J={cost:.4g}"

    def marker_legend_label(self, iteration: int) -> str:
        """The ``"i=<iteration>[, J=<cost:.3f>]"`` DEDICATED LEGEND entry for
        one selected iteration's marker (NB03 overhaul Phase 4b): every
        static landscape renderer (`landscapes.py`'s Line/Contour/Surface/
        Scatter) labels each sparse marker with this string and lets
        matplotlib's own legend collect it, rather than an inline on-plot
        text annotation next to the marker -- at this toolkit's usual font
        size a dense inline annotation reads as tiny and unreadable and
        crowds the plotted data, whereas a single external legend (see
        `viz.style.theme.place_legend_outside`, ``ncol=1`` so entries read
        top-to-bottom) stays legible regardless of how many iterations are
        selected. Fixed 3-decimal cost precision (unlike `annotation_text`'s
        `.4g`) since a legend column of these entries reads best with every
        row aligned to the same number of decimal places.

        Args:
            iteration: the iteration index to build a label for.

        Returns:
            ``"i=<iteration>"`` if no cost is available (`iteration_costs`
            was never supplied), else the two-line ``"i=<iteration>\\nJ=
            <cost:.3f>"`` (name on the first line, its numeric value on the
            second -- keeps a many-row legend column of these scannable).
        """
        cost = self.cost_at(iteration)
        return f"i={iteration}" if cost is None else f"i={iteration}\nJ={cost:.3f}"

    def animated_status_text(self, iteration: int, point: np.ndarray) -> str:
        """The one-row status text every landscape animation's updating
        tracker displays OUTSIDE the axes: the CHANGING iteration index, its
        cost, and its control-component values. Degrades gracefully to a
        bare iteration/coordinate line when no cost is known.

        Previously a second row additionally reported the optimal cost and
        control-component values -- dropped (reference-bounds plan Sec 5.3)
        because `optimum_legend_label` already puts the identical
        ``(u*=..., J*=...)`` figures on the legend's own optimum entry, so
        the row only ever duplicated information already on screen.

        Args:
            iteration: the frame's iteration index.
            point: this iteration's spec-projected control coordinates
                (e.g. `path_at(spec)[iteration]`).

        Returns:
            The one-line status text.
        """
        cost = self.cost_at(iteration)
        if cost is None:
            return f"Iteration {iteration} | u = {_format_point(point)}"
        return f"Iteration {iteration} | J = {cost:.4f} | u = {_format_point(point)}"

    def optimum_legend_label(self, spec: SliceSpec, base_label: str) -> str:
        """`base_label` enriched with the baseline optimum's exact
        spec-projected coordinates and cost (Micro-Prompt 4c), on its own
        second legend line: ``"<base_label>\\n(u*=[x.xxx, x.xxx], J*=x.xxx)"``
        -- shared by both
        the 2D contour (`landscapes.ContourLandscapeRenderer`) and 3D surface
        (`landscapes.SurfaceLandscapeRenderer`) legends, so the tracking-data
        format never drifts between them.

        Falls back to the bare `base_label` when either `baseline_U` or
        `optimum_value` is missing (mirrors `optimum_point`'s own
        None-when-absent contract) -- there is then no coordinate/cost to
        report.
        """
        point = self.optimum_point(spec)
        if point is None or self.optimum_value is None:
            return base_label
        return f"{base_label}\n(u*={_format_point(point)}, J*={self.optimum_value:.3f})"

    def cocp_legend_label(self, spec: SliceSpec, base_label: str) -> str:
        """`base_label` enriched with `cocp_point`'s exact spec-projected
        coordinates and `cocp_value` (NB04 COCP/viz refinement plan Sec
        2.2), on its own second legend line:
        ``"<base_label>\\n(u=[x.xxx, x.xxx], J=x.xxx)"`` -- the COCP-marker
        counterpart of `optimum_legend_label`, sharing the same format via
        `_format_point` so the two never drift apart.

        Falls back to the bare `base_label` when either `cocp_point` or
        `cocp_value` is missing (mirrors `cocp_point_at`'s own
        None-when-absent contract).
        """
        point = self.cocp_point_at(spec)
        if point is None or self.cocp_value is None:
            return base_label
        return f"{base_label}\n(u={_format_point(point)}, J={self.cocp_value:.3f})"

    def path_legend_label(self, spec: SliceSpec, base_label: str) -> str:
        """`base_label` enriched with the candidate path's FINAL iterate's
        spec-projected coordinates and cost (Micro-Prompt 4c), on its own
        second legend line: ``"<base_label>\\n(u_final=[x.xxx, x.xxx],
        J_final=x.xxx)"`` -- the path counterpart of `optimum_legend_label`,
        shared the same way
        across the 2D contour and 3D surface renderers.

        Falls back to the bare `base_label` when `iteration_costs` was never
        supplied (the final cost is then unavailable).
        """
        final_point = self.path_at(spec)[-1]
        if self.iteration_costs is None:
            return base_label
        final_cost = float(self.iteration_costs[-1])
        return (
            f"{base_label}\n(u_final={_format_point(final_point)}, "
            f"J_final={final_cost:.3f})"
        )

    def draw(
        self,
        ax: Any,
        spec: SliceSpec,
        *,
        annotate: bool = True,
        path_label: str = "Candidate path",
        optimum_label: str = "Optimum",
        cocp_label: str = "COCP control",
    ) -> None:
        """Draw the full thin path, the sparse markers, the optimum marker
        (if `baseline_U` is set), and the COCP marker (if `cocp_point` is
        set) onto `ax` -- a 2D or 3D matplotlib `Axes` matching
        ``len(spec.components)``.

        Each sparse marker is its OWN labeled artist (`marker_legend_label`),
        so it becomes its own top-to-bottom entry in `ax`'s legend (NB03
        overhaul Phase 4b) -- callers still draw the legend itself (e.g. via
        `viz.style.theme.place_legend_outside`, ``ncol=1``), this only adds
        the per-marker LABELED artists for it to collect; no inline on-plot
        text is drawn.

        Args:
            ax: the target Axes (plain 2D for a 1- or 2-component spec,
                ``projection="3d"`` for a 3-component spec).
            spec: which timestep/components to project onto.
            annotate: whether to label the sparse markers at all (their own
                dedicated legend entries) -- ``False`` draws the markers
                unlabeled (no legend entry) when a caller wants the bare
                path/optimum only.
            path_label: legend label for the candidate path -- generic by
                default so this agnostic toolkit never hardcodes an
                algorithm's name; callers pass something more specific, e.g.
                "Gradient descent Path", when they know it.
            optimum_label: legend label for the optimum marker.
            cocp_label: legend label for the COCP marker (NB04 COCP/viz
                refinement plan Sec 2.2).

        Raises:
            ValueError: If `spec` has exactly 1 component and
                `self.iteration_costs` is ``None`` -- a 1-component slice's
                y-axis IS cost (there is no second control component to
                plot against), so it has no other source for the overlay's
                y-coordinate.
        """
        dims = len(spec.components)
        full_path = self.path_at(spec)
        indices, points = self.selected_path(spec)
        optimum = self.optimum_point(spec)
        cocp = self.cocp_point_at(spec)

        if dims == 1:
            if self.iteration_costs is None:
                raise ValueError(
                    "TrajectoryOverlay.draw for a 1-component SliceSpec "
                    "requires iteration_costs: the y-axis is Cost, not a "
                    "second control component, so there is no other source "
                    "for the overlay's y-coordinate."
                )
            full_costs = self.iteration_costs
            point_costs = full_costs[indices]
            ax.plot(
                full_path[:, 0],
                full_costs,
                color=CANDIDATE_COLOR,
                alpha=0.6,
                linewidth=1.5,
                marker="o",
                markersize=3,
                zorder=6,
                label=path_label,
            )
            shades = self.point_shades(len(indices))
            if annotate:
                for iteration, point, cost, shade in zip(
                    indices, points, point_costs, shades
                ):
                    ax.scatter(
                        point[0],
                        cost,
                        color=shade,
                        edgecolor="black",
                        s=30,
                        zorder=8,
                        label=self.marker_legend_label(iteration),
                    )
            else:
                ax.scatter(
                    points[:, 0],
                    point_costs,
                    color=shades,
                    edgecolor="black",
                    s=30,
                    zorder=8,
                )
            if optimum is not None and self.optimum_value is not None:
                ax.scatter(
                    optimum[0],
                    self.optimum_value,
                    color=OPTIMUM_COLOR,
                    marker="*",
                    s=200,
                    edgecolor="black",
                    zorder=10,
                    label=optimum_label,
                )
            if cocp is not None and self.cocp_value is not None:
                ax.scatter(
                    cocp[0],
                    self.cocp_value,
                    color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                    marker="D",
                    s=150,
                    edgecolor="black",
                    zorder=10,
                    label=cocp_label,
                )
        elif dims == 2:
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                color=CANDIDATE_COLOR,
                alpha=0.5,
                linewidth=1.5,
                zorder=6,
                label=path_label,
            )
            shades = self.point_shades(len(indices))
            if annotate:
                for iteration, point, shade in zip(indices, points, shades):
                    ax.scatter(
                        point[0],
                        point[1],
                        color=shade,
                        edgecolor="black",
                        s=30,
                        zorder=8,
                        label=self.marker_legend_label(iteration),
                    )
            else:
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    color=shades,
                    edgecolor="black",
                    s=30,
                    zorder=8,
                )
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
        else:
            # Bold and fully opaque (vs. the 2D branch's thin, semi-
            # transparent line): against a sparse, highly transparent
            # scatter cloud (`landscapes.ScatterLandscapeRenderer`, Asset 5)
            # a faint path would be nearly invisible, whereas the 2D
            # contour's solid fill needs a lighter path to stay legible.
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                full_path[:, 2],
                color=CANDIDATE_COLOR,
                alpha=1.0,
                linewidth=2.5,
                zorder=6,
                label=path_label,
            )
            shades = self.point_shades(len(indices))
            if annotate:
                for iteration, point, shade in zip(indices, points, shades):
                    ax.scatter(
                        point[0],
                        point[1],
                        point[2],
                        color=shade,
                        edgecolor="black",
                        s=60,
                        zorder=8,
                        label=self.marker_legend_label(iteration),
                    )
            else:
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color=shades,
                    edgecolor="black",
                    s=60,
                    zorder=8,
                )
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
