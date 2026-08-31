"""Asset 3a (line), Asset 3 (contour), Asset 4 (surface), and Asset 5
(scatter) static landscape renderers: each wraps a `CostField` (from
`grids.GridProjector`) plus a `TrajectoryOverlay` (Asset 6) and renders it --
none recompute the cost. ``m > 3`` (more control dimensions than a single
landscape slice can show) is out of scope here: use the existing L1
`viz.plots.convergence.plot_high_dimensional_convergence` (a
per-dimension heatmap + distance-to-converged-norm view) directly on the
raw trajectory instead.
"""

from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from .box_region import (
    draw_box_region_1d,
    draw_box_region_2d,
    draw_box_region_3d_shadow,
    draw_box_region_3d_wireframe,
    resolve_component_bounds,
)
from .overlay import TrajectoryOverlay
from .types import CostField
from ..plots.loss_landscape import (
    LossLandscapeLabels,
    LossLandscapeOverlay,
    plot_loss_landscape_3d,
)
from ..style import (
    COCP_COLOR,
    COCP_MARKER_ALPHA,
    OPTIMUM_COLOR,
    SCATTER_CMAP,
    place_measured_legend,
    style_3d_axes,
    styled_figure,
)


def _place_wide_legend_outside(fig: Figure, ax: Any) -> None:
    """External, single-column legend for a renderer whose entries can be
    long (coordinate-enriched path/optimum labels, plus one row per selected
    iteration) -- NB03 overhaul Phase 4b. Widening the figure itself (the
    legend's absolute inches stays ~constant while the canvas grows) is what
    gives the axes room to keep a sane proportion next to a long, many-row
    legend; the actual placement/margin-reservation is
    `viz.style.theme.place_measured_legend` (NB04 COCP/viz refinement plan
    Sec 2.3) -- measured against this call's real rendered content, never a
    fixed `rect` fraction (a fixed fraction previously either clipped the
    legend or collapsed the axes, depending on how it was tuned).

    Args:
        fig: the legend's owning Figure.
        ax: the Axes whose handles/labels populate the legend.
    """
    fig.set_size_inches(14.0, 6.5)
    place_measured_legend(fig, ax, legend_kwargs={"ncol": 1, "fontsize": 9})


def _ensure_component_count(
    field: CostField, expected: int, renderer_name: str
) -> None:
    """Fail-fast guard shared by every renderer below.

    Args:
        field: the `CostField` to check.
        expected: the number of varied components this renderer requires.
        renderer_name: identifies the caller in the error message.

    Raises:
        ValueError: If `field.spec` doesn't have exactly `expected` varied
            components.
    """
    actual = len(field.spec.components)
    if actual != expected:
        raise ValueError(
            f"{renderer_name} requires a {expected}-component SliceSpec, got {actual}."
        )


class LineLandscapeRenderer:
    """Asset 3a (static, m=1): a 2D cost-vs-parameter curve, with the
    candidate trajectory overlaid (Asset 6) as points along that same curve
    (x = parameter value, y = cost -- unlike the 2-/3-component renderers,
    here the overlay's y-coordinate IS the cost, not a second control
    component). Requires a 1-component `SliceSpec`."""

    def render(
        self,
        field: CostField,
        overlay: TrajectoryOverlay,
        *,
        path_label: str = "Candidate path",
        optimum_label: str = "Optimum",
        title: str | None = None,
    ) -> Figure:
        """
        Args:
            field: the projected 1D cost slice (`grids.GridProjector.project`).
            overlay: the candidate trajectory to overlay; its
                `iteration_costs` is REQUIRED (see
                `overlay.TrajectoryOverlay.draw`'s 1-component branch).
                Its optional `overlay.box_bounds`, if set, is shaded as a
                translucent vertical feasible band (NB04 plan Sec 3.4).
            path_label: legend label for the candidate path.
            optimum_label: legend label for the optimum marker.
            title: figure title.

        Returns:
            The Figure.

        Raises:
            ValueError: If `field.spec` doesn't have exactly 1 component, or
                `overlay.iteration_costs` is ``None``.
        """
        _ensure_component_count(field, 1, "LineLandscapeRenderer")
        (u_grid,) = field.grids
        (a,) = field.spec.components

        path_label = overlay.path_legend_label(field.spec, path_label)
        optimum_label = overlay.optimum_legend_label(field.spec, optimum_label)
        # No `cocp_label` parameter (PLR0913=6: this method is already at
        # the cap) -- COCP's marker label isn't customized per call site
        # anywhere in this project, so the base string is fixed here, same
        # enrichment pattern as `optimum_label` above.
        cocp_label = overlay.cocp_legend_label(field.spec, "COCP control")
        box_bounds = resolve_component_bounds(overlay.box_bounds, 1)

        with styled_figure():
            fig, ax = plt.subplots()
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
            overlay.draw(
                ax,
                field.spec,
                path_label=path_label,
                optimum_label=optimum_label,
                cocp_label=cocp_label,
            )
            # Component-only axis label (NB03 overhaul Phase 4b): the
            # timestep this slice was frozen at belongs in the TITLE (already
            # there, below), not doubled into the axis subscript.
            ax.set_xlabel(f"$u_{{{a}}}$")
            ax.set_ylabel("Cost")
            ax.set_title(title or f"Cost Landscape at $t={field.spec.timestep}$")
            # External, single-column legend (NB03 overhaul Phase 4b): every
            # selected iteration is now its OWN labeled marker
            # (`overlay.draw`), so this legend can grow to many, potentially
            # long rows -- see `_place_wide_legend_outside`.
            _place_wide_legend_outside(fig, ax)
        return fig


class ContourLandscapeRenderer:
    """Asset 3 (static): 2D contour of the cost landscape slice, with the
    candidate trajectory overlaid (Asset 6). Requires a 2-component
    `SliceSpec` (works for any ``control_dim >= 2``)."""

    def render(
        self,
        field: CostField,
        overlay: TrajectoryOverlay,
        *,
        levels: int = 20,
        path_label: str = "Candidate path",
        optimum_label: str = "Optimum",
        title: str | None = None,
    ) -> Figure:
        """
        Args:
            field: the projected 2D cost slice (`grids.GridProjector.project`).
            overlay: the candidate trajectory to overlay. Its optional
                `overlay.box_bounds`, if set, is drawn as a translucent
                outlined feasible rectangle (NB04 plan Sec 3.4).
            levels: number of contour fill levels.
            path_label: legend label for the candidate path (see
                `overlay.TrajectoryOverlay.draw`).
            optimum_label: legend label for the optimum marker.
            title: figure title.

        Returns:
            The Figure.

        Raises:
            ValueError: If `field.spec` doesn't have exactly 2 components.
        """
        _ensure_component_count(field, 2, "ContourLandscapeRenderer")
        u1_grid, u2_grid = field.grids
        a, b = field.spec.components

        # Micro-Prompt 4c: the legend documents the exact tracked-point
        # coordinates/costs, not just a bare name -- enrichment lives here
        # (the backend), never in the calling notebook.
        path_label = overlay.path_legend_label(field.spec, path_label)
        optimum_label = overlay.optimum_legend_label(field.spec, optimum_label)
        # No `cocp_label` parameter (PLR0913=6: this method is already at
        # the cap) -- see LineLandscapeRenderer's identical note.
        cocp_label = overlay.cocp_legend_label(field.spec, "COCP control")
        box_bounds = resolve_component_bounds(overlay.box_bounds, 2)

        with styled_figure():
            fig, ax = plt.subplots()
            if box_bounds is not None:
                draw_box_region_2d(ax, box_bounds[0], box_bounds[1])
            contour_fill = ax.contour(
                u1_grid,
                u2_grid,
                field.cost_grid,
                levels=levels,
                cmap="viridis",
                alpha=0.7,
            )
            ax.clabel(contour_fill, inline=True, fmt="%.1f", fontsize=8)
            # fig.colorbar(contour_fill, ax=ax, label="Cost")
            overlay.draw(
                ax,
                field.spec,
                path_label=path_label,
                optimum_label=optimum_label,
                cocp_label=cocp_label,
            )
            # Component-only axis labels (NB03 overhaul Phase 4b): the
            # timestep this slice was frozen at belongs in the TITLE (already
            # there, below), not doubled into each axis subscript.
            ax.set_xlabel(f"$u_{{{a}}}$")
            ax.set_ylabel(f"$u_{{{b}}}$")
            ax.set_title(title or f"Cost Landscape at $t={field.spec.timestep}$")
            # External, single-column legend (NB03 overhaul Phase 4b): every
            # selected iteration is now its OWN labeled marker
            # (`overlay.draw`), so this legend can grow to many, potentially
            # long rows -- see `_place_wide_legend_outside`.
            _place_wide_legend_outside(fig, ax)
        return fig


class SurfaceLandscapeRenderer:
    """Asset 4/4b (static): 3D surface of the cost landscape slice, reusing
    `viz.plots.loss_landscape.plot_loss_landscape_3d` verbatim as the
    surface renderer (including its shared "deep bowl" 3D styling --
    colorbar, axis label padding, box aspect, camera angle -- see
    `viz.style.theme.style_3d_axes`). Requires a 2-component
    `SliceSpec`.

    The candidate trajectory is drawn on the surface using `overlay
    .iteration_costs` as its z-height -- when that array is itself the SAME
    scalar field the grid was built from (evaluated at each historical
    iterate, rather than e.g. a full-horizon rollout cost unrelated to this
    one-instant slice), the path visibly cascades down the surface's walls
    rather than floating above/below it. Its sparse markers are each their
    OWN labeled artist (`overlay.marker_legend_label`), so every selected
    iteration reads as its own dedicated, top-to-bottom legend entry rather
    than an on-plot text annotation. Without `overlay.iteration_costs`, the
    bare surface is rendered with no trajectory overlay (and therefore no
    markers either). Likewise, the optimum marker is only drawn (at `overlay
    .optimum_value`'s height) when both `overlay.baseline_U` and `overlay
    .optimum_value` are given -- unlike `iteration_costs`, that single point
    generally doesn't fall on a discretized grid node, so its z-height
    cannot be read off `field.cost_grid` and must be supplied directly.
    """

    def render(
        self,
        field: CostField,
        overlay: TrajectoryOverlay,
        *,
        trajectory_label: str = "Optimization Path",
        optimum_label: str = "Optimum",
        title: str | None = None,
    ) -> Figure:
        """
        Args:
            field: the projected 2D cost slice.
            overlay: the candidate trajectory to overlay (requires
                `overlay.iteration_costs` for the trajectory to be drawn,
                `overlay.baseline_U` + `overlay.optimum_value` for the
                optimum marker, `overlay.cocp_point` + `overlay.cocp_value`
                for the COCP marker). Its optional `overlay.box_bounds`, if
                set, is shadowed onto the surface's floor as a translucent
                flat patch at the grid's own minimum cost (NB04 plan Sec 3.4).
            trajectory_label: legend label for the candidate path.
            optimum_label: legend label for the optimum marker.
            title: figure title.

        Returns:
            The Figure.

        Raises:
            ValueError: If `field.spec` doesn't have exactly 2 components.
        """
        _ensure_component_count(field, 2, "SurfaceLandscapeRenderer")
        u1_grid, u2_grid = field.grids
        a, b = field.spec.components
        box_bounds = resolve_component_bounds(overlay.box_bounds, 2)

        trajectory, trajectory_losses, indices = None, None, None
        if overlay.iteration_costs is not None:
            indices, points = overlay.selected_path(field.spec)
            trajectory = points
            trajectory_losses = overlay.iteration_costs[indices]
            # Micro-Prompt 4c: enriched here (the backend) so the legend
            # documents the exact final coordinates/cost, never in the
            # calling notebook.
            trajectory_label = overlay.path_legend_label(field.spec, trajectory_label)

        fig = plot_loss_landscape_3d(
            u1_grid,
            u2_grid,
            field.cost_grid,
            labels=LossLandscapeLabels(
                # Component-only axis labels (NB03 overhaul Phase 4b): the
                # timestep belongs in the title (already there, below), not
                # doubled into each axis subscript.
                param_1_label=f"$u_{{{a}}}$",
                param_2_label=f"$u_{{{b}}}$",
                loss_label="Cost",
                title=title or f"Cost Landscape Surface at $t={field.spec.timestep}$",
            ),
            overlay=LossLandscapeOverlay(
                trajectory=trajectory,
                trajectory_losses=trajectory_losses,
                trajectory_label=trajectory_label,
                # `trajectory_annotations` deliberately omitted: this
                # renderer draws its OWN per-marker LABELED artists below
                # instead of `plot_loss_landscape_3d`'s generic on-plot
                # collision-avoiding text (that shared function's own
                # behavior stays untouched for its other consumers, e.g. the
                # dashboard).
                #
                # `optimal_point`/`optimal_value`/`optimum_label`
                # DELIBERATELY OMITTED (NB04 COCP/viz refinement plan Sec
                # 2.2) -- that function's own hardcoded star style
                # (`color="red", s=80`, no alpha) is shared with the
                # dashboard's unrelated parameter-tuning plots and has no
                # room for a SECOND (COCP) marker; both the Riccati optimum
                # and COCP markers are instead drawn manually below, each
                # its own labeled artist in this project's own
                # `OPTIMUM_COLOR`/`COCP_COLOR` styling, exactly like
                # `box_bounds`'s shadow already is.
            ),
        )
        ax = cast("Axes3D", fig.axes[0])

        if box_bounds is not None:
            draw_box_region_3d_shadow(
                ax, box_bounds[0], box_bounds[1], z=float(field.cost_grid.min())
            )

        optimal_point, optimal_value = None, None
        if overlay.baseline_U is not None and overlay.optimum_value is not None:
            optimal_point = overlay.optimum_point(field.spec)
            assert optimal_point is not None  # overlay.baseline_U checked above
            optimal_value = overlay.optimum_value
            optimum_label = overlay.optimum_legend_label(field.spec, optimum_label)
            ax.scatter(
                optimal_point[0],
                optimal_point[1],
                optimal_value,
                color=OPTIMUM_COLOR,
                marker="*",
                s=200,
                edgecolor="black",
                zorder=10,
                label=optimum_label,
            )

        cocp_point = None
        if overlay.cocp_point is not None and overlay.cocp_value is not None:
            cocp_point = overlay.cocp_point_at(field.spec)
            assert cocp_point is not None  # overlay.cocp_point checked above
            cocp_label = overlay.cocp_legend_label(field.spec, "COCP control")
            ax.scatter(
                cocp_point[0],
                cocp_point[1],
                overlay.cocp_value,
                color=to_rgba(COCP_COLOR, COCP_MARKER_ALPHA),
                marker="D",
                s=150,
                edgecolor="black",
                zorder=10,
                label=cocp_label,
            )

        # NB03 overhaul Phase 4b: each selected iteration becomes its OWN
        # labeled marker (`marker_legend_label`), so it reads as its own
        # dedicated, top-to-bottom legend entry -- replacing the on-plot text
        # this renderer used to delegate to `plot_loss_landscape_3d` for.
        if trajectory is not None:
            assert indices is not None and trajectory_losses is not None
            shades = overlay.point_shades(len(indices))
            for iteration, point, cost, shade in zip(
                indices, trajectory, trajectory_losses, shades
            ):
                ax.scatter(
                    point[0],
                    point[1],
                    cost,
                    color=shade,
                    edgecolor="black",
                    s=40,
                    zorder=9,
                    label=overlay.marker_legend_label(iteration),
                )

        # A legend is warranted whenever ANY labeled artist beyond the bare
        # surface exists -- the trajectory branch above, but also a marker
        # alone (e.g. an unfolded contender with no persisted iteration
        # history, or a bare Riccati-vs-COCP comparison). `place_measured_
        # legend` (NB04 COCP/viz refinement plan Sec 2.3) REPLACES whatever
        # `plot_loss_landscape_3d` may have drawn internally (its own
        # `ax.legend()` fires only when ITS OWN `trajectory` param was
        # given) with one collecting every labeled artist now on the axes,
        # and reserves the right-hand margin from the ACTUAL measured
        # content -- never a second `fig.tight_layout()` call, which the
        # class docstring's own history shows collapses this 3D axes to a
        # sliver when re-run for a differently-sized legend.
        if (
            trajectory is not None
            or optimal_point is not None
            or cocp_point is not None
        ):
            place_measured_legend(
                fig,
                ax,
                legend_kwargs={
                    "bbox_to_anchor": (0.55, 0.5),
                    "bbox_transform": fig.transFigure,
                    "prop": {"size": 8.5},
                },
            )
        return fig


class ScatterLandscapeRenderer:
    """Asset 5 (static, Micro-Prompt 5): 3D scatter of the cost landscape
    slice (grid nodes as points, cost as color), with the candidate
    trajectory overlaid as a bold, solid 3D polyline (Asset 6). Requires a
    3-component `SliceSpec` (``control_dim >= 3``) -- the position itself is
    the full varied-component tuple, so unlike Asset 4 no separate cost
    value is needed to place the overlay.

    THE OPAQUE-BLOCK FIX: a dense grid rendered at any single alpha still
    fully occludes its own interior once enough points overplot on screen --
    transparency alone cannot fix that. The caller controls actual sparsity
    by how coarse a `field` it projects (`GridProjector.project`'s own
    `SliceSpec.ranges` resolution -- see `viz.adapters.notebook
    .render_local_cost_landscape_scatter_3d`'s default grid, sized just large
    enough (Micro-Prompt 5b) to read as a volumetric cloud rather than
    isolated columns); this renderer's own `alpha` default is tuned as HEAVY
    transparency for that density, so the interior basin stays visible
    through the outer shell instead of the grid collapsing into an opaque
    block.

    THE PERSPECTIVE FIX (Micro-Prompt 5b): unlike the 2-component surface/
    contour assets, this asset's own trajectory legend is anchored below a
    genuinely cubic (not "deep bowl") 3D domain -- `style_3d_axes`'s
    surface-tuned default camera (`viz.style.theme
    .DEFAULT_3D_ELEV/DEFAULT_3D_AZIM`) left the polyline riding behind the
    scatter cloud from some seeds' geometry. This renderer therefore
    overrides the camera to ``elev=30, azim=-45``, which keeps the candidate
    polyline in front of the cloud and clear of the externally anchored
    legend.
    """

    #: Micro-Prompt 5b: overrides `viz.style.theme.style_3d_axes`'s
    #: surface-tuned default camera (see the class docstring's PERSPECTIVE
    #: FIX) -- this asset's own polyline/legend geometry, not Assets 4/4b's.
    _CAMERA_ELEV = 30.0
    _CAMERA_AZIM = -45.0

    def render(
        self,
        field: CostField,
        overlay: TrajectoryOverlay,
        *,
        alpha: float = 0.18,
        path_label: str = "Candidate path",
        optimum_label: str = "Optimum",
        title: str | None = None,
    ) -> Figure:
        """
        Args:
            field: the projected 3D cost slice.
            overlay: the candidate trajectory to overlay. Its optional
                `overlay.box_bounds`, if set, is outlined as a translucent
                wireframe feasible cube (NB04 plan Sec 3.4).
            alpha: grid-node marker transparency (lower exposes the interior
                basin through the outer shell); the default is deliberately
                HEAVY transparency (Micro-Prompt 5b) so a denser grid still
                reads as a translucent volumetric cloud rather than an
                opaque block.
            path_label: legend label for the candidate path.
            optimum_label: legend label for the optimum marker.
            title: figure title.

        Returns:
            The Figure.

        Raises:
            ValueError: If `field.spec` doesn't have exactly 3 components.
        """
        _ensure_component_count(field, 3, "ScatterLandscapeRenderer")
        grid_i, grid_j, grid_k = field.grids
        a, b, c = field.spec.components
        box_bounds = resolve_component_bounds(overlay.box_bounds, 3)

        # Micro-Prompt 4c: the legend documents the exact tracked-point
        # coordinates/costs, not just a bare name -- same enrichment as
        # Assets 3/4.
        path_label = overlay.path_legend_label(field.spec, path_label)
        optimum_label = overlay.optimum_legend_label(field.spec, optimum_label)
        # No `cocp_label` parameter (PLR0913=6: this method is already at
        # the cap) -- see LineLandscapeRenderer's identical note.
        cocp_label = overlay.cocp_legend_label(field.spec, "COCP control")

        with styled_figure():
            # Wider than the (10, 6) themed default (NB03 overhaul, second
            # review pass): this figure carries BOTH a colorbar AND a side-
            # anchored legend -- measured directly against `fig.bbox`, the
            # 3D axes collapsed to under 20% of the default canvas width
            # trying to share it with both. THIRD review pass: 15in over-
            # corrected -- the legend/colorbar's content doesn't scale with
            # figure width, so any excess becomes a dead margin past the
            # legend's own right edge (measured ~410px of unused canvas at
            # 15in); 7.5in is sized empirically (by directly measuring
            # `get_window_extent()`/`get_tightbbox()` of the axes, colorbar,
            # and legend together) to what the content actually needs, snug
            # gaps included -- matches `plot_loss_landscape_3d`'s identical
            # fit for the m=2 surface counterpart.
            fig = plt.figure(figsize=(15.0, 7.0))
            ax = fig.add_subplot(projection="3d")
            # THE OCCLUSION FIX (NB03 overhaul): mplot3d otherwise depth-sorts
            # each ARTIST (not each point) by its own mean camera-space
            # depth, ignoring the `zorder=` values `overlay.draw` already
            # sets -- confirmed directly: the candidate path/markers/optimum
            # star rendered fully hidden behind this dense grid cloud despite
            # their higher `zorder`. Disabling matplotlib's own depth
            # computation makes it respect `zorder` literally instead, so
            # everything drawn below stays visibly on top of the cloud.
            ax.computed_zorder = False
            scatter = ax.scatter(
                grid_i.ravel(),
                grid_j.ravel(),
                grid_k.ravel(),
                c=field.cost_grid.ravel(),
                cmap=SCATTER_CMAP,
                alpha=alpha,
                s=8,
                linewidths=0,
            )
            # pad=0.05 (tighter than the surface's 0.12 -- third review
            # pass: the previous padding left a large dead gap between the
            # axes and the colorbar, confirmed directly in a rendered PNG).
            cbar = fig.colorbar(
                scatter, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Cost"
            )
            # THE COLORBAR-INTENSITY FIX (NB03 overhaul, THIRD review pass --
            # reverts the second pass's "undim" attempt): a single colorbar
            # swatch can only show ONE alpha, and the only choice that is
            # ALWAYS truthful regardless of screen density, zoom, or figure
            # size is to match it to the exact alpha each plotted point
            # actually uses. The second pass instead left the colorbar fully
            # opaque on the theory that dense overlapping points visually
            # stack to full saturation -- true only at the specific figure
            # size/grid density that reasoning was checked against; widening
            # the figure afterward (more screen pixels per point) reduced
            # that overlap, and the fully-opaque colorbar started reading
            # visibly BRIGHTER/more saturated than the actual, still-
            # translucent cloud -- confirmed directly in a rendered PNG.
            # Matching the swatch alpha to the scatter's own is the one
            # setting that can't drift out of sync with unrelated layout
            # changes again.
            assert cbar.solids is not None  # populated by fig.colorbar above
            cbar.solids.set_alpha(alpha)
            if box_bounds is not None:
                draw_box_region_3d_wireframe(
                    ax, box_bounds[0], box_bounds[1], box_bounds[2]
                )
            # Each selected iteration is now its OWN labeled marker
            # (`overlay.draw`'s `annotate=True` default), giving it a
            # dedicated, top-to-bottom legend entry -- superseding this
            # renderer's own former on-plot collision-avoiding text
            # (`place_trajectory_annotations`), removed below.
            overlay.draw(
                ax,
                field.spec,
                path_label=path_label,
                optimum_label=optimum_label,
                cocp_label=cocp_label,
            )
            # Component-only axis labels (NB03 overhaul Phase 4b): the
            # timestep belongs in the title (already there, below), not
            # doubled into each axis subscript.
            ax.set_xlabel(f"$u_{{{a}}}$")
            ax.set_ylabel(f"$u_{{{b}}}$")
            ax.set_zlabel(f"$u_{{{c}}}$")
            ax.set_title(title or f"Cost Landscape (3D) at $t={field.spec.timestep}$")
            # Fixes the camera (box aspect, elevation, azimuth) before the
            # legend is finalized -- overrides `style_3d_axes`'s own
            # surface-tuned default camera with this asset's own (see the
            # class docstring's PERSPECTIVE FIX) so the polyline stays clear
            # of the externally anchored legend.
            style_3d_axes(ax, elev=self._CAMERA_ELEV, azim=self._CAMERA_AZIM)
            # Shrunk font, anchored to the SIDE (not below -- NB03 overhaul:
            # a bottom anchor forced `tight_layout`'s `rect` to reserve a
            # large bottom margin, squashing the axes), ncol=1 (top-to-bottom,
            # since the per-marker entries above can make this a many-row
            # legend) -- matches `_place_wide_legend_outside`'s convention so
            # every landscape renderer's legend behaves the same way.
            # FIGURE-fraction starting anchor (`bbox_transform=fig
            # .transFigure`), not axes-fraction: an axes-fraction anchor
            # COUPLES the legend's position to the axes' own (variable)
            # width. `place_measured_legend` (NB04 COCP/viz refinement plan
            # Sec 2.3) measures the ACTUAL rendered position against every
            # axes on this figure (the colorbar's own included) and corrects
            # this starting anchor if it doesn't already clear them, then
            # reserves the right-hand margin from that measurement via
            # `subplots_adjust` -- replacing the previous fixed
            # `fig.tight_layout(rect=(0.02, 0.0, 0.52, 1.0))` call.
            place_measured_legend(
                fig,
                ax,
                legend_kwargs={
                    "bbox_to_anchor": (0.55, 0.5),
                    "bbox_transform": fig.transFigure,
                    "prop": {"size": 8.5},
                },
            )
        return fig
