"""Notebook report adapters (viz layer L3, REFACTOR_PLAN v3, T4.d): the thin
orchestration each research-notebook cell calls once a workbench function has
produced its data. Every function here composes THREE ingredients and nothing
else: a pure renderer (`viz.plots`/`viz.landscape`), the dual-format
`FigureSink` (T4.e -- every figure persists WITH its data sidecar), and
inline display. No solving, no training, no metric computation -- that is
`src.workbench`'s job; and conversely no other layer displays or persists.

Importing this module registers every notebook-facing renderer with the
sidecar registry, so `viz.adapters.sink.re_render` can restyle any figure
persisted here from its sidecar alone.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from matplotlib.figure import Figure

from ...core.optimal_control_problem import OptimalControlProblem
from ...core.utils import to_numpy
from ...experiments import ContenderResult
from ...models.guards import require_linear_quadratic
from ...models.analytic.riccati import (
    RiccatiController,
    LocalCostToGoModel,
    evaluate_local_cost_to_go,
)
from ...workbench.alignment_sweep import LTVAlignmentSweepResult
from ...workbench.analysis import (
    OneStepCostModel,
    build_benchmark_table,
    build_convergence_margin_table,
    compute_gradient_descent_path,
    validate_convergence,
)
from ...workbench.depth_sweep import DepthSweepResult
from ...workbench.experiments import M3ScatterStudy
from ...workbench.ood_sweep import CurveWithBand, OODSweepResult
from ...workbench.replay import (
    UnfoldingLandscapeInputs,
    cocp_reference_point,
    load_training_history,
)
from ...workbench.robust_training_sweep import CurveWithBand as RobustCurveWithBand
from ..landscape import (
    ContourLandscapeRenderer,
    CostField,
    CostOracle,
    GridProjector,
    LandscapeAnimationOptions,
    LandscapeAnimator,
    LineAnimator,
    LineLandscapeRenderer,
    RelativeErrorAnimator,
    ScatterLandscapeRenderer,
    SignalAnimator,
    SignalStyle,
    SliceSpec,
    SurfaceLandscapeRenderer,
    TrajectoryOverlay,
    compute_bowl_ranges,
    make_local_cost_to_go_oracle,
    make_rollout_cost_oracle,
    normalize_control,
    normalize_history,
    plot_sparse_signals_and_errors,
)
from ..plots import (
    BandCurveStyle,
    BenchmarkRecord,
    ConvergenceRecord,
    CostComparisonStyle,
    CostVsDepthStyle,
    DepthAblationStyle,
    DepthTrajectoryAnimator,
    DepthTrajectoryStyle,
    InteractionHeatmapStyle,
    LabeledScatterStyle,
    LearningDynamicsStyle,
    MemoryBreakdown,
    OODCostConstraintStyle,
    PhasePortraitStyle,
    RegimeComparisonStyle,
    TrainingCurveStyle,
    plot_band_curves,
    plot_cartesian_performance_tradeoff,
    plot_computational_benchmarks,
    plot_convergence_curves,
    plot_cost_vs_unfolding_depth,
    plot_empirical_vs_theoretical_cost,
    plot_learning_dynamics_with_constraint,
    plot_matrix_evolution,
    plot_ood_cost_and_constraint,
    plot_ood_cost_vs_compute,
    plot_ood_depth_ablation,
    plot_ood_interaction_heatmap,
    plot_ood_mandate_scatter,
    plot_ood_phase_portrait,
    plot_regime_comparison,
    plot_trajectory_comparison,
    plot_trajectory_vs_depth_with_error,
    plot_training_curves,
)
from ..style import ReferenceLineStyle
from .sink import FigureSink, register_kwarg_renderer, register_sidecar_renderer

logger = logging.getLogger(__name__)

__all__ = [
    "render_empirical_vs_theoretical_cost",
    "render_matrix_evolution_layouts",
    "report_convergence_study",
    "render_sparse_signal_diagnostics",
    "render_signal_animations",
    "render_local_cost_landscapes",
    "render_local_cost_landscape_scatter_3d",
    "report_m3_scatter_study",
    "report_benchmarks",
    "render_cartesian_performance_tradeoff",
    "render_learning_curves",
    "render_depth_learning_curves_by_contender",
    "render_cost_vs_iterations",
    "render_alignment_curve",
    "render_regime_comparison",
    "render_trajectory_comparison",
    "UnfoldingLandscapeSpec",
    "render_unfolding_landscape_animations",
    "render_unfolding_learning_curves",
    "DepthTrajectoryReportSpec",
    "render_control_trajectories_vs_riccati",
    "render_ood_cost_and_constraint",
    "render_ood_cost_vs_compute",
    "render_ood_depth_ablation",
    "render_ood_interaction_heatmap",
    "render_ood_mandate_scatter",
    "render_ood_phase_portrait",
    "render_band_curves",
    "render_learning_dynamics_with_constraint",
]


# --- Empirical vs. theoretical cost (notebook 01) ----------------------------


def render_empirical_vs_theoretical_cost(
    curves: Mapping[str, np.ndarray],
    *,
    sink: FigureSink,
    name: str,
    title: str,
    ylabel: str | None = None,
) -> Figure:
    """One cost-convention comparison figure (empirical Monte Carlo vs.
    closed-form theoretical, with the stacked relative-error panel),
    persisted as `name`.pdf + data sidecar and displayed inline.

    Args:
        curves: one entry of `workbench.analysis.evaluate_cost_conventions`'s
            output: ``{"empirical_cost": ..., "theoretical_cost": ...}``.
        sink: the run's `FigureSink`.
        name: the artifact stem (e.g.
            ``"empirical_vs_theoretical_cost_terminal_averaged"``).
        title: figure title.
        ylabel: optional y-axis override (the raw-cumulative conventions
            pass ``"Cumulative cost"``).

    Returns:
        The Figure (already persisted/displayed/closed by the sink).
    """
    config: dict[str, Any] = {"title": title}
    if ylabel is not None:
        config["ylabel"] = ylabel
    fig = plot_empirical_vs_theoretical_cost(
        **curves, style=CostComparisonStyle(**config)
    )
    sink.save(
        fig,
        name,
        renderer="empirical_vs_theoretical_cost",
        data=curves,
        config=config,
    )
    return fig


#: The three layouts `render_matrix_evolution_layouts` renders per matrix
#: stack, with their human-readable title suffixes.
MATRIX_EVOLUTION_LAYOUTS = {"overlay": "Overlay", "rows": "By Row", "cols": "By Column"}


def render_matrix_evolution_layouts(
    matrices: np.ndarray,
    *,
    sink: FigureSink,
    matrix_symbol: str,
    name_prefix: str,
    title_prefix: str,
    layouts: Mapping[str, str] = MATRIX_EVOLUTION_LAYOUTS,
) -> None:
    """The full matrix-evolution suite for one matrix stack (e.g. the
    Riccati ``P_t`` or feedback gain ``K_t``): one figure per layout in
    `layouts`, each persisted as ``{name_prefix}_{layout}`` + sidecar and
    displayed inline.

    Args:
        matrices: the ``(T, m, n)`` matrix stack.
        sink: the run's `FigureSink`.
        matrix_symbol: the entry symbol used in subplot labels (e.g. "P").
        name_prefix: artifact stem prefix (e.g. ``"riccati_P_evolution"``).
        title_prefix: title prefix (e.g. ``r"Riccati $P_t$"``); each figure's
            title is ``f"{title_prefix} — {layout_label}"``.
        layouts: layout key -> human-readable title suffix.
    """
    matrices = to_numpy(matrices)
    for layout, layout_label in layouts.items():
        config = {
            "matrix_symbol": matrix_symbol,
            "layout": layout,
            "title": f"{title_prefix} — {layout_label}",
        }
        fig = plot_matrix_evolution(matrices, **config)
        sink.save(
            fig,
            f"{name_prefix}_{layout}",
            renderer="matrix_evolution",
            data={"matrices": matrices},
            config=config,
        )


# --- Global Convergence Diagnostics (notebook 02, Sections 5-6) --------------


@dataclass(frozen=True)
class ConvergenceReportSpec:
    """How one Global Convergence Diagnostics study is reported: the
    figure's title/artifact stem, the margin table's bands, and the
    acceptance threshold.

    Attributes:
        title: The convergence figure's title.
        name: The figure's artifact stem (e.g. ``"convergence_initializers"``).
        margin_percentages: The requested margins ``p``, forwarded to
            `build_convergence_margin_table`.
        convergence_tolerance: Relative-error acceptance threshold each
            strategy's final cost must clear against ``J_opt``.
    """

    title: str
    name: str
    margin_percentages: Sequence[float]
    convergence_tolerance: float = 1e-3


def report_convergence_study(
    records: Mapping[str, ConvergenceRecord],
    J_opt: float,
    *,
    sink: FigureSink,
    spec: ConvergenceReportSpec,
) -> None:
    """The full report for one Global Convergence Diagnostics study (a
    workbench `run_initializer_study`/`run_topology_study` result): validates
    every strategy reached the optimum, renders + persists the annotated
    convergence figure (with its data sidecar), and displays the
    "Convergence Margins" table.

    Args:
        records: label -> solved record (`OptimizationResult`s).
        J_opt: the theoretical (Riccati) optimal expected cost.
        sink: the run's `FigureSink`.
        spec: the report's naming/margins/threshold (see
            `ConvergenceReportSpec`).
    """
    title = spec.title
    validate_convergence(records, J_opt, spec.convergence_tolerance)
    fig = plot_convergence_curves(records, J_opt, title=title)
    sink.save(
        fig,
        spec.name,
        renderer="convergence_curves",
        data={
            "curves": {
                label: {
                    "J_history": to_numpy(record.J_history),
                    "J_final": float(record.J_final),
                }
                for label, record in records.items()
            },
            "J_opt": float(J_opt),
        },
        config={"title": title},
    )
    display(Markdown("**Convergence Margins**"))
    display(build_convergence_margin_table(records, J_opt, spec.margin_percentages))


# --- Recovered-signal diagnostics (notebook 02, Section 7) -------------------


def render_sparse_signal_diagnostics(
    baseline_U: np.ndarray,
    candidate_history: np.ndarray,
    selected_iterations: Sequence[int],
    *,
    sink: FigureSink,
    name: str = "signals_sparse",
    style: SignalStyle = SignalStyle(),
) -> None:
    """Renders the ``m + 1``-subplot recovered-signal diagnostic
    (`landscape.signals.plot_sparse_signals_and_errors`) for a small,
    explicitly chosen set of iterations, persists it (PDF + sidecar) under
    `name`, and displays it inline -- the single call a notebook cell needs
    once it has chosen `selected_iterations`.

    Args:
        baseline_U: reference (Riccati-optimal) control, shape ``(T, m)`` or
            ``(B, T, m)``.
        candidate_history: per-iteration control iterates, shape ``(I, T,
            m)`` or ``(I, B, T, m)``.
        selected_iterations: the exact iteration indices to overlay, e.g.
            ``[0, 5, 20, 100]``.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: slicing/labeling/title knobs (see `SignalStyle`).
    """
    data = {
        "baseline_U": to_numpy(baseline_U),
        "candidate_history": to_numpy(candidate_history),
        "selected_iterations": [int(i) for i in selected_iterations],
    }
    config = {
        "sample_index": int(style.sample_index),
        "dim_labels": (
            list(style.dim_labels) if style.dim_labels is not None else None
        ),
        "title": style.title,
    }
    fig = plot_sparse_signals_and_errors(
        to_numpy(baseline_U),
        to_numpy(candidate_history),
        [int(i) for i in selected_iterations],
        style=style,
    )
    sink.save(fig, name, renderer="sparse_signal_diagnostics", data=data, config=config)


@dataclass(frozen=True)
class SignalAnimationSpec:
    """Naming and encoding knobs for the paired signal / relative-error
    animations `render_signal_animations` persists.

    Attributes:
        signals_name: Artifact stem for the signal-convergence GIF.
        signals_title: That animation's title.
        error_name: Artifact stem for the relative-error GIF.
        error_title: That animation's title.
        interval_ms: Milliseconds between animation frames.
        fps: Frames per second the GIFs are encoded at.
    """

    signals_name: str = "signals_animated"
    signals_title: str = "Control Signal Convergence (Animated)"
    error_name: str = "relative_error_animated"
    error_title: str = "Relative Error Convergence (Animated)"
    interval_ms: int = 120
    fps: int = 12


def render_signal_animations(
    baseline_U: np.ndarray,
    candidate_history: np.ndarray,
    *,
    sink: FigureSink,
    style: SignalStyle = SignalStyle(),
    spec: SignalAnimationSpec = SignalAnimationSpec(),
) -> None:
    """The animated counterparts of the recovered-signal diagnostics: the
    per-component signal animation and the relative-error animation, each
    persisted as a GIF plus its raw-data sidecar and displayed inline.

    Args:
        baseline_U, candidate_history: see `render_sparse_signal_diagnostics`.
        sink: the run's `FigureSink`.
        style: slicing/labeling knobs shared by both animations (each
            animation's title comes from `spec`, overriding ``style.title``).
        spec: the two animations' names/titles and encoding knobs.
    """
    data = {
        "baseline_U": to_numpy(baseline_U),
        "candidate_history": to_numpy(candidate_history),
    }
    shared = {
        "dim_labels": (
            list(style.dim_labels) if style.dim_labels is not None else None
        ),
        "sample_index": int(style.sample_index),
        "interval_ms": spec.interval_ms,
    }
    sink.save_animation(
        spec.signals_name,
        lambda path: SignalAnimator(
            baseline_U,
            candidate_history,
            interval_ms=spec.interval_ms,
            style=replace(style, title=spec.signals_title),
        ).save(path, fps=spec.fps),
        renderer="signal_animation",
        data=data,
        config={**shared, "title": spec.signals_title, "fps": spec.fps},
    )
    sink.save_animation(
        spec.error_name,
        lambda path: RelativeErrorAnimator(
            baseline_U,
            candidate_history,
            interval_ms=spec.interval_ms,
            style=replace(style, title=spec.error_title),
        ).save(path, fps=spec.fps),
        renderer="relative_error_animation",
        data=data,
        config={**shared, "title": spec.error_title, "fps": spec.fps},
    )


# --- Local cost landscapes (notebook 02, Sections 8-9) -----------------------


_LANDSCAPE_ASSET_DEFAULTS = {
    "contour": {
        "save_name": "landscape_contour_static",
        "gif_name": "landscape_contour_animated",
        "title": "Local Cost Landscape at Instant $t^\\star={t_star}$",
    },
    "surface": {
        "save_name": "landscape_surface_static",
        "gif_name": "landscape_surface_animated",
        "title": "Local Cost Surface at Instant $t^\\star={t_star}$",
    },
}


def _landscape_sidecar_data(
    field: CostField, overlay: TrajectoryOverlay
) -> dict[str, Any]:
    """The serializable face of a projected landscape: the exact grids,
    cost field, slice spec, and overlay arrays a `re_render` needs -- the
    frozen renderer input, verbatim (T4.e's "definitional, not parallel")."""
    return {
        "grids": {str(i): grid for i, grid in enumerate(field.grids)},
        "cost_grid": field.cost_grid,
        "reference_U": field.reference_U,
        "spec": {
            "timestep": int(field.spec.timestep),
            "components": [int(c) for c in field.spec.components],
            "ranges": {str(i): r for i, r in enumerate(field.spec.ranges)},
        },
        "overlay": {
            "candidate_history": overlay.candidate_history,
            "iteration_costs": overlay.iteration_costs,
            "baseline_U": overlay.baseline_U,
            "optimum_value": overlay.optimum_value,
            "box_bounds": overlay.box_bounds,
            "cocp_point": overlay.cocp_point,
            "cocp_value": overlay.cocp_value,
        },
    }


def _landscape_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Rebuild the frozen `CostField`/`TrajectoryOverlay` a landscape sidecar
    carries and re-dispatch to the pure renderer its ``kind`` names."""
    spec_data = data["spec"]
    spec = SliceSpec(
        timestep=int(spec_data["timestep"]),
        components=tuple(int(c) for c in spec_data["components"]),
        ranges=tuple(
            np.asarray(spec_data["ranges"][k])
            for k in sorted(spec_data["ranges"], key=int)
        ),
    )
    field = CostField(
        grids=tuple(
            np.asarray(data["grids"][k]) for k in sorted(data["grids"], key=int)
        ),
        cost_grid=np.asarray(data["cost_grid"]),
        spec=spec,
        reference_U=np.asarray(data["reference_U"]),
    )
    overlay_data = data["overlay"]

    def _optional_array(value: Any) -> np.ndarray | None:
        return None if value is None else np.asarray(value)

    optimum_value = overlay_data.get("optimum_value")
    cocp_value = overlay_data.get("cocp_value")
    overlay = TrajectoryOverlay(
        np.asarray(overlay_data["candidate_history"]),
        iteration_costs=_optional_array(overlay_data.get("iteration_costs")),
        baseline_U=_optional_array(overlay_data.get("baseline_U")),
        optimum_value=None if optimum_value is None else float(optimum_value),
        box_bounds=overlay_data.get("box_bounds"),
        cocp_point=_optional_array(overlay_data.get("cocp_point")),
        cocp_value=None if cocp_value is None else float(cocp_value),
    )

    kind = config["kind"]
    title = config.get("title")
    path_label = config.get("path_label", "Candidate path")
    optimum_label = config.get("optimum_label", "Optimum")
    if kind == "line":
        return LineLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    if kind == "contour":
        return ContourLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    if kind == "surface":
        return SurfaceLandscapeRenderer().render(
            field,
            overlay,
            trajectory_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    if kind == "scatter":
        return ScatterLandscapeRenderer().render(
            field,
            overlay,
            alpha=float(config.get("alpha", 0.18)),
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    raise ValueError(f"Unknown landscape kind {kind!r}.")


@dataclass(frozen=True)
class LandscapeRenderSpec:
    """Every knob of a `render_local_cost_landscapes` call beyond its data
    inputs: the frozen instant, batch slice, noise level, landscape kind,
    grid geometry, animation encoding, and naming/labeling.

    Attributes:
        t_star: The frozen instant ``t*`` the landscape is sliced at.
        sample_index: Which batch element to freeze the state at and slice
            the path from.
        process_noise_std: The problem's (isotropic) process noise standard
            deviation -- ``Sigma_w = process_noise_std**2 * I``.
        kind: ``"contour"`` (Asset 3) or ``"surface"`` (Asset 4/4b).
        grid_resolution: Grid points per varied control component.
        bowl_radius_scale: Safety-margin multiplier for the symmetric "deep
            bowl" grid radius (see `compute_bowl_ranges`).
        interval_ms: Milliseconds between animation frames.
        fps: Frames per second the GIF is encoded at.
        save_name/gif_name: Artifact stems; default to `kind`-specific names.
        title: Static figure title; defaults to a `kind`-specific title
            naming ``t_star``.
        path_label: Legend label for the candidate path.
        optimum_label: Legend label for the Riccati optimum marker.
    """

    t_star: int
    sample_index: int
    process_noise_std: float
    kind: str = "contour"
    grid_resolution: int = 120
    bowl_radius_scale: float = 1.4
    interval_ms: int = 120
    fps: int = 12
    save_name: str | None = None
    gif_name: str | None = None
    title: str | None = None
    path_label: str = "Gradient descent Path"
    optimum_label: str = "Optimum by Riccati"


def render_local_cost_landscapes(
    riccati: RiccatiController,
    X_opt: torch.Tensor | np.ndarray,
    U_opt: np.ndarray,
    candidate_history: np.ndarray,
    *,
    sink: FigureSink,
    spec: LandscapeRenderSpec,
) -> None:
    """Renders the local-cost-landscape diagnostic (static PDF + data sidecar
    + animated GIF, ``kind="contour"`` for Asset 3 or ``kind="surface"`` for
    Asset 4/4b) at a frozen instant `t_star` -- the single call a notebook
    cell needs once it has chosen `t_star`, so no meshgrid math, Matplotlib
    configuration, or animation-building boilerplate remains in the notebook
    itself.

    THE ALIGNMENT FIX: the landscape is the EXACT closed-form one-step
    Bellman cost-to-go (`models.analytic.riccati.evaluate_local_cost_to_go`,
    Eq. 6 of the notebook's own derivation) at the state ``x_star =
    X_opt[sample_index, t_star]`` realized along the optimal trajectory --
    never a Monte Carlo rollout of the full control sequence. Because this
    quadratic's minimizer over ``u`` is provably ``-K_{t_star} x_star`` (the
    Riccati gain applied to that SAME frozen state), the rendered bowl's
    numerical minimum and the analytically known optimum coincide exactly.
    The candidate path's z-height (for ``kind="surface"``) is this identical
    closed-form function evaluated at each historical iterate, so the path
    visibly cascades down the surface's own walls.

    THE SYMMETRIC BOWL FIX (Micro-Prompt 4d): the grid is centered EXACTLY
    at that same closed-form optimum ``u_star``
    (`landscape.framing.compute_bowl_ranges`), radiating outward by an equal
    radius on every axis, so the optimum sits at the domain's geometric
    center with identical curvature enclosed on every side.

    Args:
        riccati: the solved `RiccatiController`; supplies ``P_arr``/``K_arr``
            AND the problem's system/cost (via ``riccati.problem``).
        X_opt: the Riccati-optimal state trajectory, shape ``(B, T+1, n)``.
        U_opt: the Riccati-optimal control, shape ``(T, m)`` or ``(B, T, m)``.
        candidate_history: the GD solver's per-iteration control iterates,
            shape ``(I, T, m)`` or ``(I, B, T, m)``.
        sink: the run's `FigureSink`.
        spec: every rendering knob (see `LandscapeRenderSpec`).

    Raises:
        ValueError: If ``spec.kind`` is not ``"contour"`` or ``"surface"``.
    """
    system, cost = require_linear_quadratic(riccati.problem)
    t_star, sample_index = spec.t_star, spec.sample_index
    kind, title = spec.kind, spec.title
    path_label, optimum_label = spec.path_label, spec.optimum_label
    interval_ms, fps = spec.interval_ms, spec.fps
    if kind not in _LANDSCAPE_ASSET_DEFAULTS:
        raise ValueError(f"kind must be 'contour' or 'surface', got {kind!r}.")
    defaults = _LANDSCAPE_ASSET_DEFAULTS[kind]

    baseline = normalize_control("U_opt", U_opt, sample_index=sample_index)
    history = normalize_history(
        "candidate_history", candidate_history, sample_index=sample_index
    )
    control_dim = baseline.shape[1]
    x_star = to_numpy(X_opt)[sample_index, t_star]

    local_model = LocalCostToGoModel(
        A=system.A_t[t_star],
        B=system.B_t[t_star],
        Q=cost.Q[t_star] if cost.Q.ndim == 3 else cost.Q,
        R=cost.R[t_star] if cost.R.ndim == 3 else cost.R,
        P_next=riccati.P_arr[t_star + 1],
        process_noise_cov=spec.process_noise_std**2 * np.eye(x_star.shape[0]),
    )

    oracle = make_local_cost_to_go_oracle(x_star, model=local_model, timestep=t_star)

    path_u = history[:, t_star, :]  # (I, m)
    iteration_costs = evaluate_local_cost_to_go(x_star, path_u, local_model)
    u_star = -riccati.K_arr[t_star] @ x_star
    optimum_value = float(
        evaluate_local_cost_to_go(x_star, u_star[None, :], local_model)[0]
    )

    ranges = compute_bowl_ranges(
        path_u, u_star, spec.grid_resolution, radius_scale=spec.bowl_radius_scale
    )
    slice_spec = SliceSpec(
        timestep=t_star, components=tuple(range(control_dim)), ranges=ranges
    )
    field = GridProjector().project(baseline, oracle, slice_spec)
    overlay = TrajectoryOverlay(
        history,
        iteration_costs=iteration_costs,
        baseline_U=baseline,
        optimum_value=optimum_value,
    )

    resolved_title = (title or defaults["title"]).format(t_star=t_star)
    config = {
        "kind": kind,
        "title": resolved_title,
        "path_label": path_label,
        "optimum_label": optimum_label,
    }
    if kind == "contour":
        fig = ContourLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=resolved_title,
        )
    else:
        fig = SurfaceLandscapeRenderer().render(
            field,
            overlay,
            trajectory_label=path_label,
            optimum_label=optimum_label,
            title=resolved_title,
        )
    sidecar_data = _landscape_sidecar_data(field, overlay)
    sink.save(
        fig,
        spec.save_name or defaults["save_name"],
        renderer="local_cost_landscape",
        data=sidecar_data,
        config=config,
    )

    sink.save_animation(
        spec.gif_name or defaults["gif_name"],
        lambda path: LandscapeAnimator(
            baseline,
            history,
            oracle,
            slice_spec,
            options=LandscapeAnimationOptions(
                iteration_costs=iteration_costs,
                optimum_value=optimum_value,
                kind=kind,
                frame_axis="iteration",
                interval_ms=interval_ms,
                optimum_label=optimum_label,
            ),
        ).save(path, fps=fps),
        renderer="landscape_animation",
        data=sidecar_data,
        config={**config, "interval_ms": interval_ms, "fps": fps},
    )


#: Micro-Prompt 5b: a much coarser grid than Assets 3/4's default (``120``
#: per component, over 2 components -> 14,400 points) -- a 3D grid's point
#: count grows as ``O(G**3)``, so the same ``G`` here would evaluate/render
#: hundreds of thousands of points and read as an opaque, unreadable block
#: regardless of marker transparency. ``22`` yields ``22**3 = 10,648`` points
#: -- dense enough, combined with the renderer's own heavy transparency
#: default, to read as a volumetric heat cloud rather than sparse, isolated
#: columns.
DEFAULT_SCATTER_3D_GRID_RESOLUTION = 22


@dataclass(frozen=True)
class GDDescent:
    """A solved gradient descent's renderable artifacts: the baseline it
    chases, its per-iteration iterates, and their per-iteration costs.

    Attributes:
        U_opt: The Riccati-optimal control, shape ``(T, m)`` or ``(B, T, m)``.
        candidate_history: Per-iteration control iterates, shape
            ``(I, T, m)`` or ``(I, B, T, m)``.
        iteration_costs: ``J^(i)`` per iteration (e.g. ``record.J_history``).
    """

    U_opt: np.ndarray
    candidate_history: np.ndarray
    iteration_costs: np.ndarray


@dataclass(frozen=True)
class Scatter3DRenderSpec:
    """Every knob of a `render_local_cost_landscape_scatter_3d` call beyond
    its data inputs.

    Attributes:
        t_star: The frozen instant ``t*`` the landscape is sliced at.
        sample_index: Which batch element of the descent artifacts to slice
            the path from.
        grid_resolution: Grid points per varied component -- kept small (see
            `DEFAULT_SCATTER_3D_GRID_RESOLUTION`) since a 3D grid's point
            count is ``grid_resolution ** 3``.
        bowl_radius_scale/scatter_alpha/interval_ms/fps: rendering knobs
            (see `LandscapeRenderSpec` / `ScatterLandscapeRenderer.render`).
        save_name/gif_name: Artifact stems.
        title: Static figure title; defaults to naming ``t_star``.
        path_label/optimum_label: Legend labels.
        dtype: dtype grid batches are cast to before calling the cost
            oracle.
    """

    t_star: int
    sample_index: int
    grid_resolution: int = DEFAULT_SCATTER_3D_GRID_RESOLUTION
    bowl_radius_scale: float = 1.4
    scatter_alpha: float = 0.18
    interval_ms: int = 400
    fps: int = 2
    save_name: str = "landscape_scatter_static"
    gif_name: str = "landscape_scatter_animated"
    title: str | None = None
    path_label: str = "Gradient descent Path"
    optimum_label: str = "Optimum by Riccati"
    dtype: torch.dtype = torch.float64


def render_local_cost_landscape_scatter_3d(
    problem: OptimalControlProblem,
    reference_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    descent: GDDescent,
    *,
    sink: FigureSink,
    spec: Scatter3DRenderSpec,
) -> None:
    """Asset 5 (Section 9, ``control_dim == 3``): the 3D-scatter counterpart
    of `render_local_cost_landscapes` -- one call renders + persists the
    static PDF, its data sidecar, and the animated GIF.

    Unlike `render_local_cost_landscapes` (an exact closed-form local Bellman
    cost-to-go), this uses a Monte Carlo rollout cost oracle
    (`landscape.oracles.make_rollout_cost_oracle`) evaluated at a single
    fixed `reference_batch` sample: the secondary ``m=3`` instance has no
    analogous closed-form per-instant slice through a full-horizon rollout.

    Args:
        problem: the secondary (``m=3``) instance's `OptimalControlProblem`.
        reference_batch: ``(x0, w, v)``, each batch-1 -- the fixed rollout
            point the grid batch is broadcast against.
        descent: the solved GD's renderable artifacts (see `GDDescent`).
        sink: this instance's own `FigureSink`.
        spec: every rendering knob (see `Scatter3DRenderSpec`).
    """
    t_star, sample_index = spec.t_star, spec.sample_index
    title, scatter_alpha = spec.title, spec.scatter_alpha
    path_label, optimum_label = spec.path_label, spec.optimum_label
    interval_ms, fps, dtype = spec.interval_ms, spec.fps, spec.dtype
    iteration_costs = descent.iteration_costs
    baseline = normalize_control("U_opt", descent.U_opt, sample_index=sample_index)
    history = normalize_history(
        "candidate_history", descent.candidate_history, sample_index=sample_index
    )
    control_dim = baseline.shape[1]

    oracle = make_rollout_cost_oracle(problem, reference_batch, dtype=dtype)
    path_u = history[:, t_star, :]
    u_star = baseline[t_star]
    ranges = compute_bowl_ranges(
        path_u, u_star, spec.grid_resolution, radius_scale=spec.bowl_radius_scale
    )
    slice_spec = SliceSpec(
        timestep=t_star, components=tuple(range(control_dim)), ranges=ranges
    )
    field = GridProjector().project(baseline, oracle, slice_spec, dtype=dtype)
    overlay = TrajectoryOverlay(
        history,
        iteration_costs=to_numpy(iteration_costs),
        baseline_U=baseline,
        optimum_value=descent.iteration_costs[-1],
    )

    resolved_title = title or (
        f"Local Cost Landscape (Three Control Dimensions) at Instant $t^\\star={t_star}$"
    )
    config = {
        "kind": "scatter",
        "title": resolved_title,
        "path_label": path_label,
        "optimum_label": optimum_label,
        "alpha": scatter_alpha,
    }
    fig = ScatterLandscapeRenderer().render(
        field,
        overlay,
        alpha=scatter_alpha,
        path_label=path_label,
        optimum_label=optimum_label,
        title=resolved_title,
    )
    sidecar_data = _landscape_sidecar_data(field, overlay)
    sink.save(
        fig,
        spec.save_name,
        renderer="local_cost_landscape",
        data=sidecar_data,
        config=config,
    )

    sink.save_animation(
        spec.gif_name,
        lambda path: LandscapeAnimator(
            baseline,
            history,
            oracle,
            slice_spec,
            options=LandscapeAnimationOptions(
                iteration_costs=to_numpy(iteration_costs),
                kind="scatter",
                frame_axis="iteration",
                interval_ms=interval_ms,
                optimum_label=optimum_label,
            ),
        ).save(path, fps=fps),
        renderer="landscape_animation",
        data=sidecar_data,
        config={**config, "interval_ms": interval_ms, "fps": fps},
    )


def report_m3_scatter_study(
    study: M3ScatterStudy,
    *,
    sink: FigureSink | None = None,
    grid_resolution: int = DEFAULT_SCATTER_3D_GRID_RESOLUTION,
    scatter_alpha: float = 0.18,
) -> None:
    """The full Section 9 report for a workbench `run_m3_scatter_study`
    result: displays the instance's convergence diagnostics and renders +
    persists Asset 5 into the instance's own run directory.

    Args:
        study: the solved `M3ScatterStudy`.
        sink: optional `FigureSink` override; defaults to one over the
            study's own run-figures directory.
        grid_resolution, scatter_alpha: forwarded to
            `render_local_cost_landscape_scatter_3d`.
    """
    lqr = study.lqr
    diagnostics = pd.DataFrame(
        {
            r"$J_{\mathrm{opt}}$": [lqr.J_opt],
            r"$J^{(M)}$": [study.record.J_final],
            "Relative error": [study.relative_error],
            r"$t^\star$": [study.t_star],
        },
        index=["Secondary instance (m=3)"],
    )
    display(diagnostics)

    render_local_cost_landscape_scatter_3d(
        lqr.problem,
        (lqr.x0_t[0:1], lqr.w_t[0:1], lqr.v_t[0:1]),
        GDDescent(
            U_opt=lqr.U_opt,
            candidate_history=study.record.U_history,
            iteration_costs=study.record.J_history,
        ),
        sink=sink if sink is not None else FigureSink(lqr.figures_dir),
        spec=Scatter3DRenderSpec(
            t_star=study.t_star,
            sample_index=0,
            grid_resolution=grid_resolution,
            scatter_alpha=scatter_alpha,
        ),
    )


# --- Computational benchmarking (notebook 02, Section 10) --------------------


def _benchmarks_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Rebuild the frozen `BenchmarkRecord`s a benchmark sidecar carries and
    re-dispatch to the pure renderer."""
    return plot_computational_benchmarks(
        _records_from_benchmark_sidecar(data), **config
    )


def report_benchmarks(
    records: Sequence[BenchmarkRecord],
    *,
    sink: FigureSink,
    name: str = "computational_benchmarks",
    title: str = "Computational Benchmarking",
) -> pd.DataFrame:
    """The full Section 10 report for a workbench `run_benchmark_suite`
    result: displays the phase-split benchmark table and renders + persists
    the 2x2 offline/online x time/memory barplot grid (Asset 7) with its
    data sidecar.

    Args:
        records: the measured `BenchmarkRecord`s.
        sink: the run's `FigureSink`.
        name: the figure's artifact stem.
        title: figure suptitle.

    Returns:
        The displayed benchmark table (see
        `workbench.analysis.build_benchmark_table`).
    """
    table = build_benchmark_table(records)
    display(table)

    fig = plot_computational_benchmarks(records, title=title)
    sink.save(
        fig,
        name,
        renderer="computational_benchmarks",
        data={
            "records": [
                {
                    "label": r.label,
                    "offline_s": float(r.offline_s),
                    "online_s": float(r.online_s),
                    "offline_numpy_mb": r.offline_memory.numpy_mb,
                    "offline_torch_mb": r.offline_memory.torch_mb,
                    "online_numpy_mb": r.online_memory.numpy_mb,
                    "online_torch_mb": r.online_memory.torch_mb,
                }
                for r in records
            ]
        },
        config={"title": title},
    )
    return table


def _records_from_benchmark_sidecar(data: Mapping[str, Any]) -> list[BenchmarkRecord]:
    """Rebuild the frozen `BenchmarkRecord`s a benchmark sidecar carries (the
    shared reconstruction both benchmark sidecar renderers use)."""
    return [
        BenchmarkRecord(
            label=entry["label"],
            offline_s=float(entry["offline_s"]),
            online_s=float(entry["online_s"]),
            offline_memory=MemoryBreakdown(
                numpy_mb=entry["offline_numpy_mb"], torch_mb=entry["offline_torch_mb"]
            ),
            online_memory=MemoryBreakdown(
                numpy_mb=entry["online_numpy_mb"], torch_mb=entry["online_torch_mb"]
            ),
        )
        for entry in data["records"]
    ]


def _cartesian_tradeoff_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Re-render the Cartesian performance-tradeoff barplot from its sidecar."""
    return plot_cartesian_performance_tradeoff(
        _records_from_benchmark_sidecar(data), **config
    )


def render_cartesian_performance_tradeoff(
    records: Sequence[BenchmarkRecord],
    *,
    sink: FigureSink,
    name: str = "cartesian_performance_tradeoff",
    title: str = "The Deep-Unfolding Trade-off: Offline vs. Online x Time vs. Space",
) -> None:
    """NB03 Phase B (directive 4): render + persist the 2x2 Cartesian
    performance barplot -- Offline vs. Online crossed with Time vs. Space --
    from per-contender `BenchmarkRecord`s, with its data sidecar. The
    online-time panel is in microseconds per control step (see
    `viz.plots.plot_cartesian_performance_tradeoff`).

    Args:
        records: one phase-split `BenchmarkRecord` per contender (offline from
            cache/metadata, online from the frozen-policy micro-batch).
        sink: the run's `FigureSink`.
        name: the figure's artifact stem.
        title: figure suptitle.
    """
    fig = plot_cartesian_performance_tradeoff(records, title=title)
    sink.save(
        fig,
        name,
        renderer="cartesian_performance_tradeoff",
        data={
            "records": [
                {
                    "label": r.label,
                    "offline_s": float(r.offline_s),
                    "online_s": float(r.online_s),
                    "offline_numpy_mb": r.offline_memory.numpy_mb,
                    "offline_torch_mb": r.offline_memory.torch_mb,
                    "online_numpy_mb": r.online_memory.numpy_mb,
                    "online_torch_mb": r.online_memory.torch_mb,
                }
                for r in records
            ]
        },
        config={"title": title},
    )


# --- Deep-Unfolded LQR Benchmarking (notebook 03) ----------------------------


def render_learning_curves(
    histories: Mapping[str, np.ndarray],
    *,
    baselines: Mapping[str, float] | None = None,
    sink: FigureSink,
    name: str = "learning_curves",
    style: TrainingCurveStyle = TrainingCurveStyle(),
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
) -> None:
    """NB03's headline figure (blueprint v2 SS5): per-epoch training loss for
    every trainable contender, with non-iterative baselines (Theo-Riccati,
    Sim-Riccati) overlaid as flat reference lines -- the figure that
    visually answers whether layer-wise training closes the gap to the
    Riccati optimum.

    Args:
        histories: label -> per-epoch loss curve, e.g.
            ``{label: workbench.replay.load_training_history(result)["loss"]
            .to_numpy() for label, result in ... if history is not None}``
            -- callers filter out `None` returns themselves (an analytic
            contender has no curve to plot), so every entry here is genuine.
        baselines: label -> constant cost (e.g. Theo-Riccati's closed-form
            value, Sim-Riccati's rollout ``eval_expected_cost``), drawn as
            dashed horizontal reference lines.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: title/scale/marker/line/legend-placement knobs (see
            `viz.plots.training_curves.TrainingCurveStyle`); defaults to a
            log-scaled loss axis (loss curves typically span orders of
            magnitude early in training) with per-curve markers.
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides for `baselines`' entries, so removing one from
            `baselines` never silently reassigns the remaining lines'
            colors/linestyles (NB04 reference-bounds plan Sec 2.2).

    Returns:
        ``None`` -- the sink persists AND displays the figure exactly once
        (Phase C, directive 1: returning the Figure would make the notebook
        cell echo it as an ``Out[]`` on top of the sink's inline display, the
        Jupyter double-plot bug).
    """
    curves = {label: to_numpy(values) for label, values in histories.items()}
    reference_lines = dict(baselines) if baselines else None
    fig = plot_training_curves(
        curves,
        reference_lines=reference_lines,
        reference_line_styles=reference_line_styles,
        style=style,
    )
    data: dict[str, Any] = {"curves": curves}
    if reference_lines is not None:
        data["reference_lines"] = reference_lines
    sink.save(
        fig,
        name,
        renderer="training_curves",
        data=data,
        config=asdict(style),
    )


def render_cost_vs_iterations(
    sweep: DepthSweepResult,
    *,
    sink: FigureSink,
    name: str = "cost_vs_unfolding_depth",
    style: CostVsDepthStyle = CostVsDepthStyle(),
    extra_reference_lines: Mapping[str, float] | None = None,
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
) -> None:
    """The "Cost vs. Unfolding Depth" figure (blueprint v2 SS5): every
    K-dependent contender's final cost against the swept unfolding depth,
    with every K-independent baseline overlaid as a flat reference line --
    the single call a notebook cell needs once `workbench
    .run_unfolding_depth_sweep_study` has produced `sweep`.

    Args:
        sweep: The depth sweep's aggregated result (`curves` = K-dependent
            contenders, `reference_lines` = the depth-invariant Sim-Riccati
            baseline already present among NB03's own contenders).
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: title/labeling/legend-placement knobs (see
            `viz.plots.cost_analysis.CostVsDepthStyle`).
        extra_reference_lines: additional K-independent baselines to merge
            in alongside `sweep.reference_lines` -- e.g. the closed-form
            Theo-Riccati cost, which never runs through `run_experiment`
            (see `workbench.depth_sweep`'s own docstring) and so is computed
            by the caller and merged in here, not inside the sweep itself.
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides for the merged reference lines, so a notebook that
            selectively drops entries from `extra_reference_lines` (or
            `sweep.reference_lines`) never silently reassigns the remaining
            lines' colors/linestyles (NB04 reference-bounds plan Sec 2.2).

    Returns:
        ``None`` -- persisted AND displayed once by the sink (Phase C,
        directive 1: no returned Figure for the cell to double-render).
    """
    reference_lines = dict(sweep.reference_lines)
    if extra_reference_lines:
        reference_lines.update(extra_reference_lines)
    fig = plot_cost_vs_unfolding_depth(
        sweep.k_values,
        sweep.curves,
        reference_lines=reference_lines,
        reference_line_styles=reference_line_styles,
        style=style,
    )
    sink.save(
        fig,
        name,
        renderer="cost_vs_unfolding_depth",
        data={
            "k_values": list(sweep.k_values),
            "curves": {label: list(values) for label, values in sweep.curves.items()},
            "reference_lines": reference_lines,
        },
        config=asdict(style),
    )


def render_alignment_curve(
    sweep: LTVAlignmentSweepResult,
    *,
    sink: FigureSink,
    name: str = "alignment_curve",
    style: CostVsDepthStyle = CostVsDepthStyle(
        xlabel="Distinct dynamics slices D",
        title="Contender Cost vs. Time-Variation (Alignment Curve)",
    ),
    extra_reference_lines: Mapping[str, float] | None = None,
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
) -> None:
    """The "Alignment Curve" figure (NB05 plan Sec 0.3/7/12): every
    contender's cost against the REALIZED distinct-slice count ``D`` of
    `workbench.alignment_sweep.run_ltv_alignment_sweep_study`'s own swept
    points -- reuses `plot_cost_vs_unfolding_depth` directly (identical
    shape of data: one scalar x-axis, one line per contender, optional flat
    reference lines), since D-vs-cost and K-vs-cost are the SAME chart
    grammar over a different swept quantity. `sweep.costs` covers EVERY
    contender here (unlike a depth sweep, there is no depth-invariant
    baseline to instead route through `reference_lines` -- see
    `LTVAlignmentSweepResult`'s own docstring), so `extra_reference_lines`
    is reserved for the two SDP-analogue floors at a chosen anchor point
    (e.g. the NB04/D=1 point's own `LTVFloors`), not for any contender.

    Args:
        sweep: The alignment sweep's aggregated result.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: title/labeling/legend-placement knobs (see
            `viz.plots.cost_analysis.CostVsDepthStyle`); defaults to this
            figure's own axis labels/title (NOT NB03/NB04's depth-axis
            defaults).
        extra_reference_lines: e.g. ``{"J_LQR floor (D=1)": ..., "J_box
            floor (D=1)": ...}`` -- floors are point-dependent on this axis
            (see `LTVAlignmentSweepResult.floors`), so the caller picks
            which point's floor(s) to draw as a flat reference, rather than
            this adapter guessing one.
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides for `extra_reference_lines` (see
            `render_cost_vs_iterations`'s identical parameter).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_cost_vs_unfolding_depth(
        sweep.distinct_slice_counts,
        sweep.costs,
        reference_lines=extra_reference_lines,
        reference_line_styles=reference_line_styles,
        style=style,
    )
    sink.save(
        fig,
        name,
        renderer="alignment_curve",
        data={
            "d_values": list(sweep.distinct_slice_counts),
            "curves": {label: list(values) for label, values in sweep.costs.items()},
            "reference_lines": dict(extra_reference_lines or {}),
        },
        config=asdict(style),
    )


def render_regime_comparison(
    costs: Mapping[str, Mapping[str, float]],
    *,
    sink: FigureSink,
    name: str = "regime_comparison",
    regime_order: Sequence[str] | None = None,
    style: RegimeComparisonStyle = RegimeComparisonStyle(),
) -> None:
    """The cross-regime comparison figure (NB05 plan Sec 7/11b): a grouped
    bar chart, one group per contender, one bar per LTV regime within each
    group, at a fixed/matched distinct-slice count ``D`` -- the direct
    "reuse (periodic) vs. dwell time (block-constant) vs. no structure
    (fully varying)" comparison the alignment curve alone cannot show (that
    figure fixes ONE regime and sweeps `D`; this one fixes `D` and compares
    regimes).

    Args:
        costs: contender label -> {regime label -> cost}, e.g. built by
            filtering several `workbench.alignment_sweep
            .LTVAlignmentSweepResult`s (one per regime) down to their
            matching-``D`` point and reshaping into this nested form.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        regime_order: Fixed bar order/coloring within each group (see
            `viz.plots.regime_comparison.plot_regime_comparison`).
        style: labeling/legend knobs (see
            `viz.plots.regime_comparison.RegimeComparisonStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_regime_comparison(costs, regime_order=regime_order, style=style)
    sink.save(
        fig,
        name,
        renderer="regime_comparison",
        data={
            "costs": {label: dict(per_regime) for label, per_regime in costs.items()},
            "regime_order": list(regime_order) if regime_order is not None else None,
        },
        config=asdict(style),
    )


def render_trajectory_comparison(
    series: Mapping[str, np.ndarray],
    *,
    sink: FigureSink,
    name: str = "control_trajectory_comparison",
    variable_name: str = "u",
    dim_labels: Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    """The trained-vs-baseline control-signal comparison for NB03 (blueprint
    v2 SS5): one subplot per control dimension, overlaying every named
    trajectory -- e.g. a trained contender's rollout against Sim-Riccati's.
    Unlike NB02's `render_sparse_signal_diagnostics` (built for an
    ITERATION-indexed history of macro-iterative solves), this compares
    single trajectories directly, matching what a trained deep-unfolded
    controller's own rollout actually is.

    Args:
        series: label -> control trajectory, shape ``(horizon, m)`` or
            ``(batch, horizon, m)`` -- e.g. each contender's persisted
            ``trajectory_controls`` artifact
            (`workbench.replay.load_contender_artifacts`).
        sink: the run's `FigureSink`.
        name: the artifact stem.
        variable_name: the y-axis symbol (``"u"`` for controls).
        dim_labels: optional per-dimension labels.
        title: figure title; defaults to the renderer's own.

    Returns:
        ``None`` -- persisted AND displayed once by the sink (Phase C,
        directive 1: no returned Figure for the cell to double-render).
    """
    config: dict[str, Any] = {"variable_name": variable_name}
    if dim_labels is not None:
        config["dim_labels"] = list(dim_labels)
    if title is not None:
        config["title"] = title
    normalized = {label: to_numpy(array) for label, array in series.items()}
    fig = plot_trajectory_comparison(normalized, **config)
    sink.save(
        fig,
        name,
        renderer="trajectory_comparison",
        data={"series": normalized},
        config=config,
    )


#: Default static-figure title (naming `spec.t_star`) per landscape asset
#: `kind` -- `render_unfolding_landscape_animations`'s per-kind dispatch
#: table, mirroring `_LANDSCAPE_ASSET_DEFAULTS`'s ``"title"`` entries.
_UNFOLDING_LANDSCAPE_TITLES = {
    "line": "Unfolded Inner-Loop Cost Landscape at $t^\\star={t_star}$",
    "contour": "Unfolded Inner-Loop Cost Landscape at $t^\\star={t_star}$",
    "surface": "Unfolded Inner-Loop Cost Landscape Surface at $t^\\star={t_star}$",
    "scatter": "Unfolded Inner-Loop Cost Landscape (3D) at $t^\\star={t_star}$",
}


@dataclass(frozen=True)
class UnfoldingLandscapeSpec:
    """Every knob of a `render_unfolding_landscape_animations` call beyond
    its data inputs (PLR0913: keeps the function under the 6-argument cap).

    Attributes:
        t_star: The frozen horizon instant the landscape/replay are
            evaluated at (display only -- `UnfoldingLandscapeInputs.x_star`
            already carries the state realized AT this instant; see
            `workbench.replay.unfolding_landscape_inputs`).
        process_noise_std: The problem's (isotropic) process noise standard
            deviation -- ``Sigma_w = process_noise_std**2 * I`` -- for the
            TRUE Riccati backdrop's local cost-to-go model.
        grid_resolution: Grid points per varied control component.
        bowl_radius_scale: Safety-margin multiplier for the symmetric "deep
            bowl" grid radius (see `landscape.framing.compute_bowl_ranges`).
        interval_ms: Milliseconds between animation frames.
        fps: Frames per second the GIF(s) are encoded at -- kept LOW
            (default 2, Phase C directive 3) so a reviewer can trace the
            inner-loop iterate's progression frame by frame.
        name_prefix: Artifact stem prefix; every persisted figure/animation
            is named ``f"{name_prefix}_{kind}[_animated]"``.
        title: Static figure title; defaults to a kind-specific title
            naming `t_star` (see `_UNFOLDING_LANDSCAPE_TITLES`).
        box_bounds: Optional infinity-norm control bound(s), highlighted as
            a dashed feasible-region outline on every landscape asset
            (`landscape.overlay.TrajectoryOverlay.box_bounds`, dimension-
            adaptively drawn per `landscape.box_region`) -- ``None``
            (default) draws nothing, so NB03's figures are byte-unchanged.
            Either a single ``(low, high)`` pair (broadcast to every
            control component, the common isotropic case) or one pair per
            component. Reaches BOTH the static figure and its animated GIF
            counterpart (`landscape.animators.LandscapeAnimator`/
            `LineAnimator`, threaded via `LandscapeAnimationOptions
            .box_bounds` -- NB04 COCP/viz refinement plan Sec 2.1; NB04's
            original plan Sec 3.4 scoped this to the static figure only,
            since the animators had no path for it at the time).
        cocp_result: Optional COCP contender's result (e.g.
            ``report.results["cocp"]``), scoring its realized control at
            this landscape's frozen state on the SAME Riccati bowl the
            optimum star is scored against (`workbench.replay
            .cocp_reference_point`) and drawing it as a second marker, on
            every asset, static and animated (NB04 COCP/viz refinement plan
            Sec 2.2/3.4). ``None`` (default) draws no COCP marker, so an
            NB03/pre-COCP call site is unaffected.
    """

    t_star: int
    process_noise_std: float
    grid_resolution: int = 60
    bowl_radius_scale: float = 1.4
    interval_ms: int = 400
    fps: int = 2
    name_prefix: str = "unfolding_landscape"
    title: str | None = None
    box_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    cocp_result: ContenderResult | None = None


def _render_unfolding_landscape_asset(
    kind: str,
    field: CostField,
    overlay: TrajectoryOverlay,
    oracle: CostOracle,
    *,
    sink: FigureSink,
    spec: UnfoldingLandscapeSpec,
) -> None:
    """Render + persist ONE landscape asset `kind` (static PDF + sidecar,
    animated GIF + sidecar) -- the per-kind body
    `render_unfolding_landscape_animations` calls once for m=1/m=3, twice
    (``"contour"`` + ``"surface"``) for m=2.

    Args:
        kind: ``"line"``, ``"contour"``, ``"surface"``, or ``"scatter"``.
        field: the projected cost slice (`landscape.grids.GridProjector
            .project`), shared by every `kind` at a given `m` (the SAME
            field renders as both a contour and a surface for m=2).
        overlay: the candidate (unfolded-controller-replayed) trajectory to
            overlay; `field.spec` is reused directly as every animator's
            `SliceSpec` (`field.spec IS the exact spec `field` was
            projected from).
        oracle: the TRUE Riccati local cost-to-go oracle the animation
            recomputes nothing from (a static backdrop -- `frame_axis`
            defaults to ``"iteration"``) but still needs to construct the
            `LandscapeAnimator`/`LineAnimator`.
        sink: the run's `FigureSink`.
        spec: every rendering knob (see `UnfoldingLandscapeSpec`).
    """
    title = (spec.title or _UNFOLDING_LANDSCAPE_TITLES[kind]).format(t_star=spec.t_star)
    path_label, optimum_label = "Unfolded inner-loop path", "Riccati optimum"

    if kind == "line":
        fig = LineLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    elif kind == "contour":
        fig = ContourLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    elif kind == "surface":
        fig = SurfaceLandscapeRenderer().render(
            field,
            overlay,
            trajectory_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )
    else:  # scatter
        fig = ScatterLandscapeRenderer().render(
            field,
            overlay,
            path_label=path_label,
            optimum_label=optimum_label,
            title=title,
        )

    sidecar_data = _landscape_sidecar_data(field, overlay)
    config = {
        "kind": kind,
        "title": title,
        "path_label": path_label,
        "optimum_label": optimum_label,
    }
    sink.save(
        fig,
        f"{spec.name_prefix}_{kind}",
        renderer="local_cost_landscape",
        data=sidecar_data,
        config=config,
    )

    assert overlay.baseline_U is not None  # always set by the caller below
    animation_options = LandscapeAnimationOptions(
        iteration_costs=overlay.iteration_costs,
        optimum_value=overlay.optimum_value,
        kind=kind,
        frame_axis="iteration",
        interval_ms=spec.interval_ms,
        # NB04 COCP/viz refinement plan Sec 2.1/2.2: reuses the SAME
        # overlay the static renderer above just drew from, so the
        # animated GIF's constraint outline/COCP marker are never a
        # separate (and potentially inconsistent) computation.
        box_bounds=overlay.box_bounds,
        cocp_point=overlay.cocp_point,
        cocp_value=overlay.cocp_value,
        # The SAME `optimum_label` local variable the static renderer above
        # was just given (reference-bounds plan Sec 5.2) -- previously
        # dropped here, so the animated legend fell back to the animator's
        # own hardcoded "Optimum" while the static figure already said
        # "Riccati optimum"; static and animated can no longer disagree.
        optimum_label=optimum_label,
    )
    animator: LandscapeAnimator | LineAnimator
    if kind == "line":
        animator = LineAnimator(
            overlay.baseline_U,
            overlay.candidate_history,
            oracle,
            field.spec,
            options=animation_options,
        )
    else:
        animator = LandscapeAnimator(
            overlay.baseline_U,
            overlay.candidate_history,
            oracle,
            field.spec,
            options=animation_options,
        )
    sink.save_animation(
        f"{spec.name_prefix}_{kind}_animated",
        lambda path: animator.save(path, fps=spec.fps),
        renderer="landscape_animation",
        data=sidecar_data,
        config={**config, "interval_ms": spec.interval_ms, "fps": spec.fps},
    )


def render_unfolding_landscape_animations(
    riccati: RiccatiController,
    inputs: UnfoldingLandscapeInputs,
    *,
    sink: FigureSink,
    spec: UnfoldingLandscapeSpec,
) -> None:
    """NB03 Phase 3 requirement 1: the m-DEPENDENT landscape/animation
    router. Dispatches purely on ``m = inputs.B.shape[1]`` -- no caller
    special-cases a specific dimension anywhere in this function or its
    callers, so changing a notebook's ``CONTROL_DIM`` and re-running changes
    `m` here, and only here:

    * ``m == 1``: a 2D line plot + animation (`LineLandscapeRenderer` /
      `LineAnimator`) -- parameter value on the x-axis, cost on the y-axis.
    * ``m == 2``: a 2D contour AND a 3D surface (both static + animated) --
      `ContourLandscapeRenderer`/`SurfaceLandscapeRenderer`, exactly the
      Asset 3/4 machinery `render_local_cost_landscapes` already uses for
      notebook 02.
    * ``m == 3``: a 3D scatter (static + animated) -- `ScatterLandscapeRenderer`,
      the Asset 5 machinery `render_local_cost_landscape_scatter_3d` already
      uses.
    * ``m > 3``: no plots; logs a message at INFO and returns.

    THE BACKDROP is the TRUE Riccati local cost-to-go bowl (Eq. 6, the exact
    same closed-form construction `render_local_cost_landscapes` uses:
    `LocalCostToGoModel` seeded from `riccati.P_arr[t_star + 1]`). THE
    OVERLAID PATH is `inputs`' OWN trained inner-loop replay -- "the actual
    historical values captured by the recently fixed
    `engine.callbacks.ParameterSnapshotCallback`" (`inputs.step_size`,
    `inputs.P_own`), reconstructed via the existing
    `workbench.analysis.compute_gradient_descent_path` (never
    re-derived here). Because `inputs.P_own` may differ from the backdrop's
    TRUE P (e.g. `unfolded_alpha_p`'s learned matrix), the figure directly
    visualizes whether the trained reparameterization still walks to the
    TRUE bowl's minimum -- not merely a re-plot of Riccati's own path.

    Args:
        riccati: the solved `RiccatiController` supplying the TRUE backdrop
            (`P_arr`/`K_arr`) and the problem's system/cost (via
            ``riccati.problem``).
        inputs: this contender's persisted parameters (see
            `workbench.replay.unfolding_landscape_inputs`).
        sink: the run's `FigureSink`.
        spec: every rendering knob (see `UnfoldingLandscapeSpec`).
    """
    m = inputs.B.shape[1]
    if m > 3:
        logger.info(
            "render_unfolding_landscape_animations: control_dim=%d exceeds "
            "the m<=3 supported landscape range; skipping (name_prefix=%r).",
            m,
            spec.name_prefix,
        )
        return

    _, cost = require_linear_quadratic(riccati.problem)
    t_star, x_star = spec.t_star, inputs.x_star
    n = x_star.shape[0]

    local_model = LocalCostToGoModel(
        A=inputs.A,
        B=inputs.B,
        Q=cost.Q[t_star] if cost.Q.ndim == 3 else cost.Q,
        R=inputs.R,
        P_next=riccati.P_arr[t_star + 1],
        process_noise_cov=spec.process_noise_std**2 * np.eye(n),
    )
    # timestep=0, not t_star: `baseline`/`history` below are collapsed to a
    # single frozen instant (T=1), so the oracle's row index must match
    # `slice_spec.timestep` (0), not the problem's own horizon index --
    # `local_model` already encodes t_star via `P_next=riccati.P_arr[t_star+1]`.
    oracle = make_local_cost_to_go_oracle(x_star, model=local_model, timestep=0)

    own_model = OneStepCostModel(
        A=inputs.A, B=inputs.B, R=inputs.R, P_next=inputs.P_own
    )
    # The problem's OWN constraint (never re-derived from spec.box_bounds,
    # which only drives the visual shading below): the replayed path must
    # respect exactly what the real controller's inner loop projects
    # through, so it can never wander somewhere the deployed policy could
    # never actually reach (NB04 plan Sec 3.4/7). `None` for an
    # unconstrained (NB03) problem, so that replay is unaffected.
    _replay_constraint = (
        riccati.problem.constraints[0] if riccati.problem.constraints else None
    )
    trajectory = compute_gradient_descent_path(
        x_star,
        model=own_model,
        step_sizes=inputs.step_size,
        constraint=_replay_constraint,
    )  # (J+1, m)
    history = trajectory[:, None, :]  # (J+1, 1, m): T=1, a single frozen instant

    u_star = -riccati.K_arr[t_star] @ x_star
    baseline = u_star[None, :]  # (1, m)
    iteration_costs = evaluate_local_cost_to_go(x_star, trajectory, local_model)
    optimum_value = float(
        evaluate_local_cost_to_go(x_star, u_star[None, :], local_model)[0]
    )

    # COCP's realized control at the SAME frozen x_star, scored on this
    # SAME local_model bowl -- so its marker is directly comparable to the
    # Riccati optimum's (NB04 COCP/viz refinement plan Sec 2.2/3.4). `None`
    # (the default) when no COCP contender is given, so an NB03/pre-COCP
    # call site draws no second marker.
    cocp_point, cocp_value = None, None
    if spec.cocp_result is not None:
        cocp_reference = cocp_reference_point(
            spec.cocp_result, x_star, local_model=local_model
        )
        cocp_point, cocp_value = cocp_reference.point, cocp_reference.value

    ranges = compute_bowl_ranges(
        trajectory, u_star, spec.grid_resolution, radius_scale=spec.bowl_radius_scale
    )
    slice_spec = SliceSpec(timestep=0, components=tuple(range(m)), ranges=ranges)
    field = GridProjector().project(baseline, oracle, slice_spec)
    overlay = TrajectoryOverlay(
        history,
        iteration_costs=iteration_costs,
        baseline_U=baseline,
        optimum_value=optimum_value,
        box_bounds=spec.box_bounds,
        cocp_point=cocp_point,
        cocp_value=cocp_value,
    )

    kinds = {1: ("line",), 2: ("contour", "surface"), 3: ("scatter",)}[m]
    for kind in kinds:
        _render_unfolding_landscape_asset(
            kind, field, overlay, oracle, sink=sink, spec=spec
        )


def render_unfolding_learning_curves(
    sweep: DepthSweepResult,
    depths: Sequence[int],
    *,
    sink: FigureSink,
    extra_reference_lines: Mapping[str, float] | None = None,
    trainable_labels: Sequence[str] = ("unfolded_alpha", "unfolded_alpha_p"),
    name_prefix: str = "learning_curves_depth",
) -> None:
    """NB03 Phase 3 requirement 2: the layer-wise learning curves, one
    figure per swept unfolding depth. For each ``k`` in `depths`, reuses
    the EXISTING `render_learning_curves` verbatim (never a second
    `plot_training_curves` call site) with that depth's own per-epoch
    histories (`workbench.replay.load_training_history`, disposition-
    agnostic) and its own depth-specific baselines -- Sim-Riccati's
    ``eval_expected_cost`` at that depth's report, plus any
    depth-INVARIANT `extra_reference_lines` (e.g. the closed-form
    Theo-Riccati cost, merged in identically at every depth).

    Args:
        sweep: The depth sweep's aggregated result
            (`workbench.run_unfolding_depth_sweep_study`).
        depths: Which of `sweep.k_values` to render one figure for (e.g. a
            curated subset like ``[2, 4, 5, 6]``).
        sink: the run's `FigureSink`.
        extra_reference_lines: depth-invariant baselines merged into every
            figure (e.g. ``{"Theo-Riccati": THEO_RICCATI_COST}``).
        trainable_labels: contender labels to plot a curve for, filtered to
            whichever ones actually have a genuine training curve at that
            depth (an analytical contender has none -- see
            `load_training_history`).
        name_prefix: artifact stem prefix; each depth's figure is named
            ``f"{name_prefix}_{k}"``.
    """
    for k in depths:
        report = sweep.reports[k]
        histories = {
            label: history["loss"].to_numpy()
            for label in trainable_labels
            if label in report.results
            and (history := load_training_history(report.results[label])) is not None
        }
        if not histories:
            continue
        baselines: dict[str, float] = dict(extra_reference_lines or {})
        if "sim_riccati" in report.results:
            baselines["Sim-Riccati"] = report.metric(
                "sim_riccati", "eval_expected_cost"
            )
        render_learning_curves(
            histories,
            baselines=baselines,
            sink=sink,
            name=f"{name_prefix}_{k}",
            style=TrainingCurveStyle(
                title=f"Learning Curves at Unfolding Depth $J={k}$"
            ),
        )


#: NB03's two trainable contenders mapped to their display names -- the
#: default `render_depth_learning_curves_by_contender` produces one figure for
#: each (a mapping, so the pretty label and the artifact key stay together and
#: the signature stays under the 6-argument budget).
_NB03_TRAINABLE_DISPLAY_NAMES: Mapping[str, str] = {
    "unfolded_alpha": "unfolded_alpha",
    "unfolded_alpha_p": "unfolded_alpha_p",
}


def render_depth_learning_curves_by_contender(  # noqa: PLR0913 -- a deliberate
    # 7th param (`style`), now an 8th (`reference_line_styles`), so the
    # marker/linestyle/legend-placement/reference-styling knobs of NB03's
    # headline learning-curve figures stay notebook-accessible without
    # collapsing `baselines`/`contenders`/`name_prefix` into a catch-all (each
    # already has its own distinct, independently-defaulted meaning).
    sweep: DepthSweepResult,
    depths: Sequence[int],
    *,
    sink: FigureSink,
    baselines: Mapping[str, float] | None = None,
    contenders: Mapping[str, str] | None = None,
    name_prefix: str = "learning_curves_by_contender",
    style: TrainingCurveStyle | None = None,
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
) -> None:
    """NB03 Phase C (directive 2): one learning-curve figure PER CONTENDER,
    each overlaying that contender's FAMILY of per-epoch loss curves across
    every swept unfolding depth ``J`` in `depths` (Cost vs. Epochs, one line
    per depth) against the flat, depth-invariant Riccati `baselines`
    (Theoretical + Simulated). This is the transpose of
    `render_unfolding_learning_curves` (which draws one figure per depth with
    both contenders overlaid): here each figure isolates a single contender so
    its depth-dependence reads cleanly instead of two families tangling on one
    axis.

    Reuses `render_learning_curves` (the single `plot_training_curves` call
    site) per contender -- the family of depth curves is passed as its
    ``histories`` mapping (``"$J=k$" -> loss curve``), so nothing new is
    plotted from scratch, and the loss axis stays on the log scale that call
    defaults to.

    Args:
        sweep: the depth sweep's aggregated result
            (`workbench.run_unfolding_depth_sweep_study`).
        depths: which of `sweep.k_values` to draw one curve each for.
        sink: the run's `FigureSink`.
        baselines: depth-invariant reference lines drawn flat on every figure
            (e.g. ``{"Theo-Riccati": ..., "Sim-Riccati": ...}``).
        contenders: label -> display name for each contender to produce a
            figure for (defaults to NB03's two trainable contenders); a
            contender with no genuine training curve at any depth is skipped.
        name_prefix: artifact stem prefix; each contender's figure is named
            ``f"{name_prefix}_{label}"``.
        style: marker/line/legend-placement knobs (see
            `viz.plots.training_curves.TrainingCurveStyle`), shared by every
            contender's figure; each figure's own ``title`` is always this
            call's own (per-contender), overriding whatever `style.title` is
            set to. ``None`` uses `TrainingCurveStyle`'s defaults.
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides for `baselines`' entries, shared by every contender's
            figure, so a notebook that selectively enables/disables entries
            in `baselines` never silently reassigns the remaining lines'
            colors/linestyles (NB04 reference-bounds plan Sec 2.2/2.3).
    """
    contenders = contenders if contenders is not None else _NB03_TRAINABLE_DISPLAY_NAMES
    base_style = style if style is not None else TrainingCurveStyle()
    for label, display_name in contenders.items():
        curves: dict[str, np.ndarray] = {}
        for k in depths:
            report = sweep.reports[k]
            if (
                label in report.results
                and (history := load_training_history(report.results[label]))
                is not None
            ):
                curves[f"$J={k}$"] = history["loss"].to_numpy()
        if not curves:
            continue
        render_learning_curves(
            curves,
            baselines=baselines,
            sink=sink,
            name=f"{name_prefix}_{label}",
            style=replace(
                base_style,
                title=f"Learning Curves across Unfolding Depth — {display_name}",
            ),
            reference_line_styles=reference_line_styles,
        )


@dataclass(frozen=True)
class DepthTrajectoryReportSpec:
    """Every knob of a `render_control_trajectories_vs_riccati` call beyond
    its data inputs.

    Attributes:
        variable_name: the y-axis symbol (``"u"`` for controls).
        dim_labels: optional per-dimension labels.
        contender_name: display name of the plotted contender, used in
            the static title and the animation's title/legend.
        name_prefix: artifact stem prefix; the static figure and animation
            are named ``f"{name_prefix}_static"``/``f"{name_prefix}_animated"``.
        title: static figure title; defaults to the renderer's own.
        interval_ms: milliseconds between animation frames.
        fps: frames per second the GIF is encoded at -- kept LOW (default
            2) so the depth-to-depth morph across a handful of frames stays
            human-readable, unlike a dense per-iteration animation.
        box_bounds: Optional infinity-norm control bound(s) to draw as
            horizontal reference lines in every signal row -- ``None``
            (default) draws nothing, so NB03's figures are byte-unchanged.
            Either a single ``(u_min, u_max)`` pair shared by every control
            component, or one pair per component for an anisotropic box
            (NB04 plan Sec 3.5; see `DepthTrajectoryStyle.box_bounds`).
        baseline_label: Legend label for `baseline` -- defaults to
            ``"Riccati"`` (NB03's usage: the exact, unconstrained optimum).
            A box-constrained caller comparing against a SATURATED baseline
            (NB04's Truncated-Riccati, not the raw Riccati gain) should
            override this so the legend never claims optimality the plotted
            curve doesn't have.
    """

    variable_name: str = "u"
    dim_labels: Sequence[str] | None = None
    contender_name: str | None = None
    name_prefix: str = "trajectory_vs_depth"
    title: str | None = None
    interval_ms: int = 500
    fps: int = 2
    box_bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    baseline_label: str = "Riccati"


def render_control_trajectories_vs_riccati(
    baseline: np.ndarray,
    depth_trajectories: Mapping[int, np.ndarray],
    *,
    sink: FigureSink,
    spec: DepthTrajectoryReportSpec,
    costs: Mapping[int, float] | None = None,
    baseline_cost: float | None = None,
) -> None:
    """NB03 Phase 3 requirement 4: the depth-parameterized ``m + 1``
    trajectory grid -- additive to (never a replacement of) the single-depth
    `render_trajectory_comparison`. Persists BOTH the static figure
    (`viz.plots.plot_trajectory_vs_depth_with_error`: one line per depth in
    `depth_trajectories`, overlaid against the Riccati `baseline`, plus a
    shared relative-error row) and its animated counterpart
    (`viz.plots.DepthTrajectoryAnimator`: the identical layout, one frame
    per depth, showing the candidate trajectory morph into alignment with
    Riccati as depth grows) via `sink`.

    Args:
        baseline: the Riccati reference control, shape ``(T, m)`` or
            ``(B, T, m)`` -- e.g. a depth-invariant contender's persisted
            ``trajectory_controls`` artifact (`workbench.replay
            .load_contender_artifacts`).
        depth_trajectories: unfolding depth ``K`` -> this contender's
            persisted ``trajectory_controls`` at that depth, same shape
            convention as `baseline`.
        sink: the run's `FigureSink`.
        spec: every naming/labeling/encoding knob (see
            `DepthTrajectoryReportSpec`).
        costs: optional depth ``K`` -> total expected cost ``J``, shown
            next to each depth's legend entry (e.g. `report.metric`).
        baseline_cost: optional total expected cost ``J`` of `baseline`,
            shown next to the Riccati legend entry.
    """
    style = DepthTrajectoryStyle(
        variable_name=spec.variable_name,
        dim_labels=spec.dim_labels,
        contender_name=spec.contender_name,
        box_bounds=spec.box_bounds,
        baseline_label=spec.baseline_label,
    )
    baseline_np = to_numpy(baseline)
    trajectories_np = {k: to_numpy(v) for k, v in depth_trajectories.items()}

    fig = plot_trajectory_vs_depth_with_error(
        baseline_np,
        trajectories_np,
        costs=costs,
        baseline_cost=baseline_cost,
        style=style,
        title=spec.title,
    )
    data: dict[str, Any] = {
        "baseline": baseline_np,
        "depth_trajectories": {str(k): v for k, v in trajectories_np.items()},
        "costs": {str(k): v for k, v in costs.items()} if costs is not None else None,
        "baseline_cost": baseline_cost,
    }
    config: dict[str, Any] = {
        "variable_name": spec.variable_name,
        "dim_labels": list(spec.dim_labels) if spec.dim_labels is not None else None,
        "contender_name": spec.contender_name,
        "title": spec.title,
        "box_bounds": spec.box_bounds,
        "baseline_label": spec.baseline_label,
    }
    sink.save(
        fig,
        f"{spec.name_prefix}_static",
        renderer="trajectory_vs_depth",
        data=data,
        config=config,
    )

    animator = DepthTrajectoryAnimator(
        baseline_np,
        trajectories_np,
        interval_ms=spec.interval_ms,
        style=style,
        costs=costs,
        baseline_cost=baseline_cost,
    )
    sink.save_animation(
        f"{spec.name_prefix}_animated",
        lambda path: animator.save(path, fps=spec.fps),
        renderer="trajectory_vs_depth_animation",
        data=data,
        config={**config, "interval_ms": spec.interval_ms, "fps": spec.fps},
    )


# --- Sidecar renderer registration (import side effect, adapters-only) -------


def _convergence_curves_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Rebuild the record mapping a convergence sidecar carries (as plain
    `ConvergenceRecord`-shaped rows) and re-dispatch to the pure renderer."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ConvergenceCurve:
        J_history: np.ndarray
        J_final: float

    records = {
        label: ConvergenceCurve(
            J_history=np.asarray(entry["J_history"]),
            J_final=float(entry["J_final"]),
        )
        for label, entry in data["curves"].items()
    }
    return plot_convergence_curves(records, float(data["J_opt"]), **config)


def _empirical_vs_theoretical_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `CostComparisonStyle`
    (its fields map onto the config keys verbatim) and re-dispatch."""
    return plot_empirical_vs_theoretical_cost(
        np.asarray(data["empirical_cost"]),
        np.asarray(data["theoretical_cost"]),
        style=CostComparisonStyle(**config),
    )


def _sparse_signals_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `SignalStyle` and
    re-dispatch to the pure renderer."""
    style_kwargs = dict(config)
    dim_labels = style_kwargs.get("dim_labels")
    if dim_labels is not None:
        style_kwargs["dim_labels"] = list(dim_labels)
    return plot_sparse_signals_and_errors(
        np.asarray(data["baseline_U"]),
        np.asarray(data["candidate_history"]),
        [int(i) for i in data["selected_iterations"]],
        style=SignalStyle(**style_kwargs),
    )


register_sidecar_renderer(
    "empirical_vs_theoretical_cost", _empirical_vs_theoretical_from_sidecar
)
register_kwarg_renderer("matrix_evolution", plot_matrix_evolution)
register_sidecar_renderer("sparse_signal_diagnostics", _sparse_signals_from_sidecar)
register_sidecar_renderer("convergence_curves", _convergence_curves_from_sidecar)
register_sidecar_renderer("computational_benchmarks", _benchmarks_from_sidecar)
register_sidecar_renderer(
    "cartesian_performance_tradeoff", _cartesian_tradeoff_from_sidecar
)
register_sidecar_renderer("local_cost_landscape", _landscape_from_sidecar)


def _training_curves_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `TrainingCurveStyle` and
    re-dispatch to the pure renderer (the sidecar itself stays flat/
    human-readable; only the nested `style` parameter is a Phase 4 addition
    on the renderer's own signature)."""
    return plot_training_curves(**data, style=TrainingCurveStyle(**config))


def _cost_vs_unfolding_depth_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `CostVsDepthStyle` and
    re-dispatch to the pure renderer."""
    return plot_cost_vs_unfolding_depth(**data, style=CostVsDepthStyle(**config))


register_sidecar_renderer("training_curves", _training_curves_from_sidecar)
register_sidecar_renderer(
    "cost_vs_unfolding_depth", _cost_vs_unfolding_depth_from_sidecar
)


def _alignment_curve_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `CostVsDepthStyle` and
    re-dispatch to the pure renderer -- a DISTINCT renderer tag from
    `cost_vs_unfolding_depth` (Sec 7's alignment curve is genuinely a
    different figure, over a different swept quantity) even though both
    happen to call the same underlying `plot_cost_vs_unfolding_depth`, so
    the persisted sidecar's ``"d_values"`` field name stays honest about
    what it holds rather than reusing the K-sweep's ``"k_values"`` name."""
    return plot_cost_vs_unfolding_depth(
        data["d_values"],
        data["curves"],
        reference_lines=data["reference_lines"],
        style=CostVsDepthStyle(**config),
    )


def _regime_comparison_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `RegimeComparisonStyle`
    and re-dispatch to the pure renderer."""
    return plot_regime_comparison(
        data["costs"],
        regime_order=data["regime_order"],
        style=RegimeComparisonStyle(**config),
    )


register_sidecar_renderer("alignment_curve", _alignment_curve_from_sidecar)
register_sidecar_renderer("regime_comparison", _regime_comparison_from_sidecar)
register_kwarg_renderer("trajectory_comparison", plot_trajectory_comparison)


def _trajectory_vs_depth_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the sidecar's string-keyed ``depth_trajectories`` mapping
    back into ``int -> array`` (sidecar payload keys are always ``str`` --
    see `sink._flatten`) and the flat `config` into a `DepthTrajectoryStyle`,
    then re-dispatch to the pure renderer."""
    depth_trajectories = {
        int(depth): np.asarray(array)
        for depth, array in data["depth_trajectories"].items()
    }
    dim_labels = config.get("dim_labels")
    style = DepthTrajectoryStyle(
        variable_name=config.get("variable_name", "u"),
        dim_labels=list(dim_labels) if dim_labels is not None else None,
        contender_name=config.get("contender_name"),
        box_bounds=config.get("box_bounds"),
        baseline_label=config.get("baseline_label", "Riccati"),
    )
    raw_costs = data.get("costs")
    costs = {int(k): float(v) for k, v in raw_costs.items()} if raw_costs else None
    baseline_cost = data.get("baseline_cost")
    return plot_trajectory_vs_depth_with_error(
        np.asarray(data["baseline"]),
        depth_trajectories,
        costs=costs,
        baseline_cost=float(baseline_cost) if baseline_cost is not None else None,
        style=style,
        title=config.get("title"),
    )


register_sidecar_renderer("trajectory_vs_depth", _trajectory_vs_depth_from_sidecar)


# --- Zero-shot OOD generalization (notebook 06) ------------------------------


def _curve_with_band_payload(band: CurveWithBand) -> dict[str, Any]:
    """`CurveWithBand` as a JSON-safe sidecar payload -- `x` may hold
    `NoiseFamily` (a `StrEnum`, already JSON-safe) alongside plain
    floats/ints, so no conversion is needed beyond `list(...)`."""
    return {
        "x": list(band.x),
        "median": list(band.median),
        "q25": list(band.q25),
        "q75": list(band.q75),
    }


def render_ood_cost_and_constraint(
    result: OODSweepResult,
    *,
    sink: FigureSink,
    use_gap: bool = True,
    name: str = "ood_cost_and_constraint",
    style: OODCostConstraintStyle = OODCostConstraintStyle(),
    reference_lines: Mapping[str, float] | None = None,
) -> None:
    """NB06's workhorse figure (plan Sec 5.1/5.2): every contender's
    median-cost-or-gap band on top, its saturation-rate band directly below,
    both against `result`'s own perturbation-level x-axis -- one call per
    swept axis.

    Args:
        result: One axis's `workbench.ood_sweep.OODSweepResult` (e.g. from
            `run_horizon_ood_sweep`/`run_scale_ood_sweep`/
            `run_dynamics_ood_sweep`/`run_noise_family_ood_sweep`).
        sink: the run's `FigureSink`.
        use_gap: If ``True`` (the default), the top panel plots
            `result.suboptimality_gap_band` (relative to the per-point SDP
            floor, skipping any point whose floor is undefined); if
            ``False``, plots the raw `result.cost_band` instead (e.g. for
            `OODAxis.NOISE_FAMILY`'s Cauchy points, which have no floor at
            all -- the raw-cost fallback the plan's Sec 2.4 anticipates).
        name: the artifact stem.
        style: title/labeling/scale knobs (see
            `viz.plots.ood.OODCostConstraintStyle`).
        reference_lines: optional label -> constant value drawn as a dashed
            reference line on the cost/gap panel (e.g. the nominal, K=0
            reference point).

    Returns:
        ``None`` -- persisted AND displayed once by the sink (no returned
        Figure for the cell to double-render).
    """
    cost_bands = {
        label: (
            result.suboptimality_gap_band(label) if use_gap else result.cost_band(label)
        )
        for label in result.points
    }
    saturation_bands = {label: result.saturation_band(label) for label in result.points}
    fig = plot_ood_cost_and_constraint(
        cost_bands, saturation_bands, reference_lines=reference_lines, style=style
    )
    sink.save(
        fig,
        name,
        renderer="ood_cost_and_constraint",
        data={
            "axis": result.axis.value,
            "cost_bands": {
                label: _curve_with_band_payload(band)
                for label, band in cost_bands.items()
            },
            "saturation_bands": {
                label: _curve_with_band_payload(band)
                for label, band in saturation_bands.items()
            },
            "reference_lines": dict(reference_lines) if reference_lines else {},
        },
        config=asdict(style),
    )


def render_ood_depth_ablation(
    results_by_depth: Mapping[int, OODSweepResult],
    contenders: Sequence[str],
    *,
    sink: FigureSink,
    use_gap: bool = True,
    name: str = "ood_depth_ablation",
    style: DepthAblationStyle = DepthAblationStyle(),
) -> None:
    """The "over-thinking" ablation figure (NB06 plan Sec 5.4): one subplot
    per contender in `contenders`, each overlaying one band per depth in
    `results_by_depth`.

    Args:
        results_by_depth: depth K -> that depth's `OODSweepResult` (from
            `workbench.ood_sweep.run_ood_depth_ablation`).
        contenders: which contender labels to draw one panel each for
            (typically the two unfolded contenders -- the only ones whose
            synthesis genuinely varies with depth).
        sink: the run's `FigureSink`.
        use_gap: See `render_ood_cost_and_constraint`.
        name: the artifact stem.
        style: title/labeling/scale knobs (see
            `viz.plots.ood.DepthAblationStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    bands_by_contender: dict[str, dict[int, CurveWithBand]] = {}
    for label in contenders:
        per_depth: dict[int, CurveWithBand] = {}
        for depth, result in results_by_depth.items():
            if label not in result.points:
                continue
            per_depth[depth] = (
                result.suboptimality_gap_band(label)
                if use_gap
                else result.cost_band(label)
            )
        if per_depth:
            bands_by_contender[label] = per_depth

    fig = plot_ood_depth_ablation(bands_by_contender, style=style)
    sink.save(
        fig,
        name,
        renderer="ood_depth_ablation",
        data={
            "bands_by_contender": {
                label: {
                    str(depth): _curve_with_band_payload(band)
                    for depth, band in per_depth.items()
                }
                for label, per_depth in bands_by_contender.items()
            }
        },
        config=asdict(style),
    )


def render_ood_phase_portrait(
    trajectories: Mapping[str, np.ndarray],
    *,
    sink: FigureSink,
    name: str = "ood_phase_portrait",
    style: PhasePortraitStyle = PhasePortraitStyle(),
) -> None:
    """The single static OOD trajectory snapshot (NB06 plan Sec 5.5): every
    named trajectory projected onto the same two leading principal
    directions, plus a log-scale state-norm-vs-time inset.

    Args:
        trajectories: label -> states, shape ``(T+1, n)`` -- one batch
            element of a captured OOD rollout (e.g.
            `experiments.zero_shot.evaluate_under_shift`'s
            ``capture_trajectories=True`` output, sliced to one sample), the
            SAME initial state and noise realization across every label
            (the fairness law) at one severe perturbation level.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: labeling knobs (see `viz.plots.ood.PhasePortraitStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_ood_phase_portrait(trajectories, style=style)
    sink.save(
        fig,
        name,
        renderer="ood_phase_portrait",
        data={
            "trajectories": {
                label: to_numpy(states) for label, states in trajectories.items()
            }
        },
        config=asdict(style),
    )


def render_ood_cost_vs_compute(
    latency_us_by_label: Mapping[str, float],
    degradation_by_label: Mapping[str, float],
    *,
    sink: FigureSink,
    name: str = "ood_cost_vs_compute",
    style: LabeledScatterStyle | None = None,
) -> None:
    """The cost-vs-compute scatter (NB06 plan Sec 13.5): does OOD
    robustness cost extra online compute? One point per contender.

    Args:
        latency_us_by_label: label -> per-step online inference latency in
            microseconds (e.g.
            `workbench.benchmarks.OnlineInferenceRecord.online_latency_us`).
        degradation_by_label: label -> relative cost degradation at the
            severest swept level (`workbench.ood_sweep
            .build_ood_degradation_table`'s own column), same keys as
            `latency_us_by_label`.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: labeling/scale knobs (see `viz.plots.ood.LabeledScatterStyle`);
            ``None`` uses `plot_ood_cost_vs_compute`'s own default.

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    kwargs = {"style": style} if style is not None else {}
    fig = plot_ood_cost_vs_compute(latency_us_by_label, degradation_by_label, **kwargs)
    sink.save(
        fig,
        name,
        renderer="ood_cost_vs_compute",
        data={
            "latency_us_by_label": dict(latency_us_by_label),
            "degradation_by_label": dict(degradation_by_label),
        },
        config=asdict(style) if style is not None else {},
    )


def render_ood_mandate_scatter(
    saturation_by_label: Mapping[str, float],
    degradation_by_label: Mapping[str, float],
    *,
    sink: FigureSink,
    name: str = "ood_mandate_scatter",
    style: LabeledScatterStyle | None = None,
) -> None:
    """The mandate scatter (NB06 plan Sec 13.9): nominal saturation rate
    against OOD degradation, one point per contender -- the blueprint's own
    question ("do contenders that lean on the constraint generalize worse?")
    in a single panel.

    Args:
        saturation_by_label: label -> nominal (in-distribution) saturation
            rate, in ``[0, 1]``.
        degradation_by_label: label -> relative cost degradation at the
            severest swept level, same keys as `saturation_by_label`.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: labeling/scale knobs; ``None`` uses
            `plot_ood_mandate_scatter`'s own default.

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    kwargs = {"style": style} if style is not None else {}
    fig = plot_ood_mandate_scatter(saturation_by_label, degradation_by_label, **kwargs)
    sink.save(
        fig,
        name,
        renderer="ood_mandate_scatter",
        data={
            "saturation_by_label": dict(saturation_by_label),
            "degradation_by_label": dict(degradation_by_label),
        },
        config=asdict(style) if style is not None else {},
    )


def render_ood_interaction_heatmap(
    scale_levels: tuple[float, ...],
    bound_levels: tuple[float, ...],
    grid: np.ndarray,
    *,
    sink: FigureSink,
    name: str = "ood_interaction_heatmap",
    style: InteractionHeatmapStyle = InteractionHeatmapStyle(),
) -> None:
    """The scale x bound interaction heatmap (NB06 plan Sec 13.7): read
    directly from `workbench.ood_sweep.OODSweepResult.interaction_grid`.

    Args:
        scale_levels: Row coordinates (ascending).
        bound_levels: Column coordinates (ascending).
        grid: Shape ``(len(scale_levels), len(bound_levels))``.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: labeling/colormap knobs (see
            `viz.plots.ood.InteractionHeatmapStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_ood_interaction_heatmap(scale_levels, bound_levels, grid, style=style)
    sink.save(
        fig,
        name,
        renderer="ood_interaction_heatmap",
        data={
            "scale_levels": list(scale_levels),
            "bound_levels": list(bound_levels),
            "grid": to_numpy(grid),
        },
        config=asdict(style),
    )


# --- Robust training under uncertainty (notebook 07) -------------------------


def _robust_curve_payload(band: RobustCurveWithBand) -> dict[str, Any]:
    """`workbench.robust_training_sweep.CurveWithBand` as a JSON-safe sidecar
    payload -- NB07's own copy of `_curve_with_band_payload` above: that
    helper is typed against `workbench.ood_sweep`'s OWN, unrelated
    `CurveWithBand` (notebook 06), never shared with this one. `x` may hold
    `NoiseFamily` (a `StrEnum`, already JSON-safe) alongside plain
    floats/ints, so no conversion is needed beyond `list(...)`."""
    return {
        "x": list(band.x),
        "median": list(band.median),
        "q25": list(band.q25),
        "q75": list(band.q75),
    }


def _robust_bands_from_payload(
    payload: Mapping[str, Mapping[str, Any]],
) -> dict[str, RobustCurveWithBand]:
    """The inverse of `_robust_curve_payload`, applied to a whole
    label -> payload mapping."""
    return {
        label: RobustCurveWithBand(
            x=tuple(band["x"]),
            median=tuple(band["median"]),
            q25=tuple(band["q25"]),
            q75=tuple(band["q75"]),
        )
        for label, band in payload.items()
    }


def render_band_curves(
    bands: Mapping[str, RobustCurveWithBand],
    *,
    sink: FigureSink,
    name: str,
    reference_lines: Mapping[str, float] | None = None,
    style: BandCurveStyle = BandCurveStyle(),
) -> None:
    """NB07's generic single-panel band figure (plan Sec 6.2/6.3/6.4): one
    median+IQR curve per contender (or, for the spectral-analysis figures,
    per eigenvalue rank) against a shared x-axis -- levels, trajectory
    counts, or epoch indices. Covers every NB07 figure except the per-epoch
    learning-dynamics one (Sec 6.1, `render_learning_dynamics_with_constraint`
    below), which pairs a cost panel with a constraint panel.

    Args:
        bands: label -> `workbench.robust_training_sweep.CurveWithBand`,
            e.g. from `RobustTrainingResult.cost_band`/`.isotropy_band`/
            `.spectral_norm_band`/`.condition_number_band`/
            `.principal_angle_band`, or one entry per eigenvalue rank from
            `.eigenvalue_bands`.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        reference_lines: optional label -> constant value, drawn as a
            dashed horizontal reference line (e.g. the nominally-trained
            baseline, or an SDP floor).
        style: labeling/scale knobs (see
            `viz.plots.robust_training.BandCurveStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_band_curves(bands, reference_lines=reference_lines, style=style)
    sink.save(
        fig,
        name,
        renderer="robust_band_curves",
        data={
            "bands": {
                label: _robust_curve_payload(band) for label, band in bands.items()
            },
            "reference_lines": dict(reference_lines) if reference_lines else {},
        },
        config=asdict(style),
    )


def _band_curves_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `BandCurveStyle` and each
    persisted band back into a `RobustCurveWithBand`, then re-dispatch to
    the pure renderer."""
    bands = _robust_bands_from_payload(data["bands"])
    reference_lines = data.get("reference_lines") or None
    return plot_band_curves(
        bands, reference_lines=reference_lines, style=BandCurveStyle(**config)
    )


register_sidecar_renderer("robust_band_curves", _band_curves_from_sidecar)


def render_learning_dynamics_with_constraint(
    cost_bands: Mapping[str, RobustCurveWithBand],
    constraint_bands: Mapping[str, RobustCurveWithBand],
    *,
    sink: FigureSink,
    name: str,
    style: LearningDynamicsStyle = LearningDynamicsStyle(),
) -> None:
    """NB07's learning-dynamics workhorse (plan Sec 6.1): per-epoch training
    loss (median + IQR band across seeds) on top, the paired per-epoch
    saturation-rate band directly below on the same (epoch) x-axis -- the
    plan's Sec 4.6 rule that no cost figure ships without its constraint
    panel.

    Args:
        cost_bands: label -> `workbench.robust_training_sweep.CurveWithBand`
            (e.g. `RobustTrainingResult.learning_curve_band` per contender,
            at one severity level).
        constraint_bands: label -> `CurveWithBand`, same (epoch) x-axis as
            `cost_bands`.
        sink: the run's `FigureSink`.
        name: the artifact stem.
        style: labeling/scale knobs (see
            `viz.plots.robust_training.LearningDynamicsStyle`).

    Returns:
        ``None`` -- persisted AND displayed once by the sink.
    """
    fig = plot_learning_dynamics_with_constraint(
        cost_bands, constraint_bands, style=style
    )
    sink.save(
        fig,
        name,
        renderer="robust_learning_dynamics",
        data={
            "cost_bands": {
                label: _robust_curve_payload(band) for label, band in cost_bands.items()
            },
            "constraint_bands": {
                label: _robust_curve_payload(band)
                for label, band in constraint_bands.items()
            },
        },
        config=asdict(style),
    )


def _learning_dynamics_from_sidecar(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> Figure:
    """Reconstruct the flat sidecar `config` into a `LearningDynamicsStyle`
    and each persisted band back into a `RobustCurveWithBand`, then
    re-dispatch to the pure renderer."""
    cost_bands = _robust_bands_from_payload(data["cost_bands"])
    constraint_bands = _robust_bands_from_payload(data["constraint_bands"])
    return plot_learning_dynamics_with_constraint(
        cost_bands, constraint_bands, style=LearningDynamicsStyle(**config)
    )


register_sidecar_renderer("robust_learning_dynamics", _learning_dynamics_from_sidecar)
