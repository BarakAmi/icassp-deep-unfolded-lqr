"""Research-notebook `Experiment` declarations (Tier-3, T3.d): unlike
`applications.standard_lqr`/`applications.box_constraint_lqr` (product case
studies under golden-master test), a study module is a single notebook's own
declaration -- new research questions become new study modules, never edits
to a benchmarked case study.
"""

from .nb03_unfolding import (
    NB03Config,
    TrainingMode,
    UnfoldedModelConfig,
    build_nb03_config,
    nb03_unfolding_experiment,
)
from .nb04_box_constrained import (
    NB04Config,
    BoxConstrainedFloors,
    build_nb04_config,
    compute_box_constrained_floors,
    nb04_box_constrained_experiment,
)
from .nb05_ltv_box_constrained import (
    NB05Config,
    LTVFloors,
    build_nb05_config,
    compute_ltv_floors,
    nb05_ltv_experiment,
)
from .nb06_ood_generalization import (
    CONTENDER_LABELS as NB06_CONTENDER_LABELS,
    NB06Config,
    build_nb06_config,
    nb06_ood_generalization_experiment,
)
from .nb07_robust_training import (
    NB07Config,
    build_nb07_config,
    nb07_compute_context,
    nb07_nominal_anchor_experiment,
    nominal_problem_and_batches,
    noise_family_condition_builder,
    noise_scale_condition_builder,
    plant_additive_condition_builder,
    plant_rotation_condition_builder,
    recipes_at_for,
    sample_size_condition_builder,
    six_training_sweep_contenders,
    train_horizon_condition_builder,
)

__all__ = [
    "NB03Config",
    "TrainingMode",
    "UnfoldedModelConfig",
    "build_nb03_config",
    "nb03_unfolding_experiment",
    "NB04Config",
    "BoxConstrainedFloors",
    "build_nb04_config",
    "compute_box_constrained_floors",
    "nb04_box_constrained_experiment",
    "NB05Config",
    "LTVFloors",
    "build_nb05_config",
    "compute_ltv_floors",
    "nb05_ltv_experiment",
    "NB06_CONTENDER_LABELS",
    "NB06Config",
    "build_nb06_config",
    "nb06_ood_generalization_experiment",
    "NB07Config",
    "build_nb07_config",
    "nb07_compute_context",
    "nb07_nominal_anchor_experiment",
    "nominal_problem_and_batches",
    "noise_family_condition_builder",
    "noise_scale_condition_builder",
    "plant_additive_condition_builder",
    "plant_rotation_condition_builder",
    "recipes_at_for",
    "sample_size_condition_builder",
    "six_training_sweep_contenders",
    "train_horizon_condition_builder",
]
