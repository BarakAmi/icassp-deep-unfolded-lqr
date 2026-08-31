"""The research workbench (REFACTOR_PLAN v3, T3.g): the notebook-facing API
that replaces the `applications.notebook_utils` God module, organized by
responsibility and with zero display side effects below the Tier-4 adapters:

* `loading` — run-reuse guards (`load_or_run` and friends);
* `setup` — parameter specs to ready-to-solve experiment bundles;
* `experiments` — thin execution assemblers (return data, never plot);
* `analysis` — result payloads to DataFrames / validation verdicts;
* `benchmarks` — phase-split offline/online time+memory measurement.

Rendering/display lives exclusively with the Tier-4 adapters
(`src.viz.adapters` -- Stage S5): every function here returns data, never a
figure.
"""

from .alignment_sweep import (
    LTVAlignmentPoint,
    LTVAlignmentSweepResult,
    run_ltv_alignment_sweep_study,
)
from .analysis import (
    COST_CONVENTIONS,
    CrossoverEstimate,
    NoiseCovariances,
    OneStepCostModel,
    build_benchmark_table,
    build_convergence_margin_table,
    build_unfolding_convergence_margin_table,
    compute_box_binding_fraction,
    compute_box_binding_fraction_per_timestep,
    compute_cumulative_average_cost,
    compute_gradient_descent_path,
    compute_loss_grid,
    compute_stage_cost_grid,
    describe_cocp_solver_choice,
    estimate_crossover,
    evaluate_cost_conventions,
    first_sustained_index,
    summarize_cost_comparison,
    validate_convergence,
)
from .benchmarks import (
    RICCATI_BENCHMARK_EXPERIMENT_NAME,
    InferenceBenchmarkSpec,
    OnlineInferenceRecord,
    RiccatiBenchmarkSpec,
    measure_analytic_gd_benchmark,
    measure_frozen_inference_benchmark,
    measure_riccati_synthesis_benchmark,
    measure_synthesized_controller_benchmark,
    run_benchmark_suite,
    torch_tensor_bytes,
)
from .depth_sweep import (
    BASELINE_LABELS,
    NB04_BASELINE_LABELS,
    NB05_BASELINE_LABELS,
    DepthSweepResult,
    run_box_constrained_depth_sweep_study,
    run_depth_sweep_study,
    run_ltv_depth_sweep_study,
    run_unfolding_depth_sweep_study,
)
from .experiments import (
    GDSolveSettings,
    M3ScatterStudy,
    run_analytic_controller_rollout,
    run_initializer_study,
    run_m3_scatter_study,
    run_topology_study,
    solve_signal_space_gd,
)
from .loading import (
    find_existing_run,
    load_or_run,
    load_run_artifacts,
)
from .replay import (
    COCPReferencePoint,
    UnfoldingLandscapeInputs,
    cocp_reference_point,
    load_contender_artifacts,
    load_training_history,
    render_training_log_replay,
    unfolding_landscape_inputs,
)
from .robust_training_sweep import (
    MIN_TRAINING_SEEDS,
    CurveWithBand,
    ModelAccess,
    RobustTrainingPoint,
    RobustTrainingResult,
    RobustTrainingSpec,
    TrainingCondition,
    UncertaintyAxis,
    run_robust_training_sweep,
)
from .spectral_analysis import PSpectrum, compute_p_spectrum, compute_principal_angles
from .robustness import (
    DimensionRobustnessSpec,
    assert_unfolded_dimension_robustness,
)
from .setup import (
    DisturbanceRealization,
    GaussianBatchSpec,
    ProblemDims,
    RunLocation,
    SamplerBundle,
    StochasticLQRExperiment,
    build_step_size_schedule,
    generate_marginally_stable_system,
    make_gaussian_batch_sampler,
    setup_stochastic_lqr_experiment,
)

__all__ = [
    "COST_CONVENTIONS",
    "CrossoverEstimate",
    "NoiseCovariances",
    "OneStepCostModel",
    "RICCATI_BENCHMARK_EXPERIMENT_NAME",
    "InferenceBenchmarkSpec",
    "OnlineInferenceRecord",
    "RiccatiBenchmarkSpec",
    "GDSolveSettings",
    "DisturbanceRealization",
    "GaussianBatchSpec",
    "ProblemDims",
    "RunLocation",
    "SamplerBundle",
    "build_benchmark_table",
    "build_convergence_margin_table",
    "build_unfolding_convergence_margin_table",
    "compute_box_binding_fraction",
    "compute_box_binding_fraction_per_timestep",
    "compute_cumulative_average_cost",
    "compute_gradient_descent_path",
    "compute_loss_grid",
    "compute_stage_cost_grid",
    "describe_cocp_solver_choice",
    "estimate_crossover",
    "evaluate_cost_conventions",
    "first_sustained_index",
    "summarize_cost_comparison",
    "validate_convergence",
    "measure_analytic_gd_benchmark",
    "measure_frozen_inference_benchmark",
    "measure_riccati_synthesis_benchmark",
    "measure_synthesized_controller_benchmark",
    "run_benchmark_suite",
    "torch_tensor_bytes",
    "BASELINE_LABELS",
    "NB04_BASELINE_LABELS",
    "NB05_BASELINE_LABELS",
    "DepthSweepResult",
    "run_box_constrained_depth_sweep_study",
    "run_depth_sweep_study",
    "run_ltv_depth_sweep_study",
    "run_unfolding_depth_sweep_study",
    "LTVAlignmentPoint",
    "LTVAlignmentSweepResult",
    "run_ltv_alignment_sweep_study",
    "M3ScatterStudy",
    "run_analytic_controller_rollout",
    "run_initializer_study",
    "run_m3_scatter_study",
    "run_topology_study",
    "solve_signal_space_gd",
    "find_existing_run",
    "load_or_run",
    "load_run_artifacts",
    "load_contender_artifacts",
    "load_training_history",
    "render_training_log_replay",
    "UnfoldingLandscapeInputs",
    "unfolding_landscape_inputs",
    "COCPReferencePoint",
    "cocp_reference_point",
    "DimensionRobustnessSpec",
    "assert_unfolded_dimension_robustness",
    "MIN_TRAINING_SEEDS",
    "CurveWithBand",
    "ModelAccess",
    "RobustTrainingPoint",
    "RobustTrainingResult",
    "RobustTrainingSpec",
    "TrainingCondition",
    "UncertaintyAxis",
    "run_robust_training_sweep",
    "PSpectrum",
    "compute_p_spectrum",
    "compute_principal_angles",
    "StochasticLQRExperiment",
    "build_step_size_schedule",
    "generate_marginally_stable_system",
    "make_gaussian_batch_sampler",
    "setup_stochastic_lqr_experiment",
]
