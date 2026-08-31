"""Analytic (non-learnable, closed-form) LQR controllers: the finite-horizon
Riccati recursion and its constraint-saturating variant — each exposed both
as its legacy single-phase controller and as its Stage-S2 two-phase
`Synthesizer` (see ``models.lifecycle``)."""

from .iterative_gd import (
    AnalyticalIterativeGDController,
    SolveSpec,
    build_riccati_gd_refinement,
)
from .riccati import (
    RiccatiController,
    RiccatiSynthesizer,
    SynthesizedRiccatiController,
    compute_state_covariance_trajectory,
    compute_theoretical_expected_cost,
    LocalCostToGoModel,
    evaluate_local_cost_to_go,
    finite_horizon_riccati,
    get_lqr_gradient_matrices,
    gradient_lipschitz_constant,
    problem_gradient_lipschitz_constant,
    get_riccati_control_policy,
    get_riccati_control_trajectory,
)
from .truncated_riccati import (
    SynthesizedTruncatedRiccatiController,
    TruncatedRiccatiController,
    TruncatedRiccatiSynthesizer,
)

__all__ = [
    "RiccatiController",
    "RiccatiSynthesizer",
    "SynthesizedRiccatiController",
    "compute_state_covariance_trajectory",
    "compute_theoretical_expected_cost",
    "LocalCostToGoModel",
    "evaluate_local_cost_to_go",
    "finite_horizon_riccati",
    "get_lqr_gradient_matrices",
    "gradient_lipschitz_constant",
    "problem_gradient_lipschitz_constant",
    "get_riccati_control_policy",
    "get_riccati_control_trajectory",
    "SynthesizedTruncatedRiccatiController",
    "TruncatedRiccatiController",
    "TruncatedRiccatiSynthesizer",
    "AnalyticalIterativeGDController",
    "SolveSpec",
    "build_riccati_gd_refinement",
]
