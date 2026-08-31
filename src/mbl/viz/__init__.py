"""The consolidated presentation tier (REFACTOR_PLAN v3, Stage S5, T4.a):
one strictly layered `viz` namespace replacing the former `visualizations`
and `applications.visualization` packages.

Layers (each may import only the ones below it):

* `viz.style`     (L0) -- rc-params, palette, semantic color roles, the
  neutral series-style registry, and vector-figure/animation encoding.
* `viz.plots`     (L1) -- generic pure renderers: Result data in, Figure out.
* `viz.landscape` (L2) -- the Phase 2B cost-landscape toolkit (oracles,
  grid projection, overlays, landscape/signal renderers, animators).
* `viz.adapters`  (L3) -- the ONLY layer allowed to display or persist:
  the dual-format `FigureSink` (T4.e), the notebook report adapters, and
  the Streamlit-facing backend selection.

Importing this package has no side effects: backend selection (`Agg`) is an
adapter/entry-point concern (`viz.adapters.dashboard`, `tests/conftest.py`),
never an import-time one (N9).
"""

from .plots import (
    BenchmarkRecord,
    ConvergenceRecord,
    CostVsDepthStyle,
    MemoryBreakdown,
    TrainingCurveStyle,
    plot_computational_benchmarks,
    CocpOverlay,
    ContourLabels,
    plot_control_convergence_contour_2d,
    plot_convergence_curves,
    plot_cost_vs_unfolding_depth,
    plot_cumulative_cost_over_time,
    plot_empirical_vs_theoretical_cost,
    plot_high_dimensional_convergence,
    LossLandscapeLabels,
    LossLandscapeOverlay,
    plot_loss_landscape_3d,
    plot_matrix_evolution,
    plot_trajectory_comparison,
    plot_training_curves,
)
from .style import (
    CATEGORICAL_PALETTE,
    PLOT_STYLE,
    register_series_styles,
    save_animation,
    save_vector_figure,
    series_color,
    series_linestyle,
    style_3d_axes,
    styled_figure,
)

__all__ = [
    "BenchmarkRecord",
    "ConvergenceRecord",
    "MemoryBreakdown",
    "plot_computational_benchmarks",
    "CocpOverlay",
    "ContourLabels",
    "plot_control_convergence_contour_2d",
    "plot_convergence_curves",
    "CostVsDepthStyle",
    "plot_cost_vs_unfolding_depth",
    "plot_cumulative_cost_over_time",
    "plot_empirical_vs_theoretical_cost",
    "plot_high_dimensional_convergence",
    "LossLandscapeLabels",
    "LossLandscapeOverlay",
    "plot_loss_landscape_3d",
    "plot_matrix_evolution",
    "plot_trajectory_comparison",
    "plot_training_curves",
    "TrainingCurveStyle",
    "CATEGORICAL_PALETTE",
    "PLOT_STYLE",
    "register_series_styles",
    "save_animation",
    "save_vector_figure",
    "series_color",
    "series_linestyle",
    "style_3d_axes",
    "styled_figure",
]
