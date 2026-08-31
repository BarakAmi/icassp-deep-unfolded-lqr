"""viz layer L1 -- generic, pure renderers: frozen Result data (plain
arrays, named-series mappings, `BenchmarkRecord`s, `ConvergenceRecord`s) in,
`matplotlib` Figures out. Never imports engines, models, or solvers, never
displays, never touches the filesystem (REFACTOR_PLAN v3, T4.a/T4.b) --
persistence and display belong exclusively to `viz.adapters`.
"""

from .benchmarking import (
    BenchmarkRecord,
    MemoryBreakdown,
    plot_cartesian_performance_tradeoff,
    plot_computational_benchmarks,
)
from .convergence import (
    ConvergenceRecord,
    CocpOverlay,
    ContourLabels,
    plot_control_convergence_contour_2d,
    plot_convergence_curves,
    plot_high_dimensional_convergence,
)
from .cost_analysis import (
    CostVsDepthStyle,
    plot_cost_vs_unfolding_depth,
    plot_cumulative_cost_over_time,
    CostComparisonStyle,
    plot_empirical_vs_theoretical_cost,
)
from .depth_trajectories import (
    DepthTrajectoryAnimator,
    DepthTrajectoryStyle,
    plot_trajectory_vs_depth_with_error,
)
from .loss_landscape import (
    LossLandscapeLabels,
    LossLandscapeOverlay,
    plot_loss_landscape_3d,
)
from .matrix_evolution import plot_matrix_evolution
from .ood import (
    DepthAblationStyle,
    InteractionHeatmapStyle,
    LabeledScatterStyle,
    OODCostConstraintStyle,
    PhasePortraitStyle,
    plot_ood_cost_and_constraint,
    plot_ood_cost_vs_compute,
    plot_ood_depth_ablation,
    plot_ood_interaction_heatmap,
    plot_ood_mandate_scatter,
    plot_ood_phase_portrait,
)
from .regime_comparison import RegimeComparisonStyle, plot_regime_comparison
from .robust_training import (
    BandCurveStyle,
    LearningDynamicsStyle,
    plot_band_curves,
    plot_learning_dynamics_with_constraint,
)
from .trajectories import plot_trajectory_comparison
from .training_curves import TrainingCurveStyle, plot_training_curves

__all__ = [
    "BenchmarkRecord",
    "MemoryBreakdown",
    "plot_cartesian_performance_tradeoff",
    "plot_computational_benchmarks",
    "ConvergenceRecord",
    "plot_convergence_curves",
    "CocpOverlay",
    "ContourLabels",
    "plot_control_convergence_contour_2d",
    "plot_high_dimensional_convergence",
    "CostVsDepthStyle",
    "plot_cost_vs_unfolding_depth",
    "plot_cumulative_cost_over_time",
    "CostComparisonStyle",
    "plot_empirical_vs_theoretical_cost",
    "DepthTrajectoryAnimator",
    "DepthTrajectoryStyle",
    "plot_trajectory_vs_depth_with_error",
    "LossLandscapeLabels",
    "LossLandscapeOverlay",
    "LossLandscapeLabels",
    "LossLandscapeOverlay",
    "plot_loss_landscape_3d",
    "plot_matrix_evolution",
    "DepthAblationStyle",
    "InteractionHeatmapStyle",
    "LabeledScatterStyle",
    "OODCostConstraintStyle",
    "PhasePortraitStyle",
    "plot_ood_cost_and_constraint",
    "plot_ood_cost_vs_compute",
    "plot_ood_depth_ablation",
    "plot_ood_interaction_heatmap",
    "plot_ood_mandate_scatter",
    "plot_ood_phase_portrait",
    "RegimeComparisonStyle",
    "plot_regime_comparison",
    "BandCurveStyle",
    "LearningDynamicsStyle",
    "plot_band_curves",
    "plot_learning_dynamics_with_constraint",
    "plot_trajectory_comparison",
    "plot_training_curves",
    "TrainingCurveStyle",
]
