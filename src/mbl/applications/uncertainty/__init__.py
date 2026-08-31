"""Shared plant-perturbation, noise-family, and distance primitives for
uncertainty-axis research notebooks (docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md
Sec 3, resolved 2026-07-26): built once here so a later zero-shot-OOD study
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md) consumes the same
perturbation/noise/distance code rather than a second implementation of the
same rotation matrix.
"""

from .callback import DomainRandomizationCallback
from .dataset import FiniteTrajectoryDatasetSpec
from .distances import (
    DistributionDistance,
    distance_to_nominal_gaussian,
    lift_hellinger_to_dimension,
)
from .noise import ExoticBatchSpec, NoiseFamily
from .perturbations import (
    PerturbationDistribution,
    PerturbationKind,
    PlantPerturbation,
    rotation_matrix,
)
from .resync import UnsupportedResyncError, resync_to_plant

__all__ = [
    "DomainRandomizationCallback",
    "FiniteTrajectoryDatasetSpec",
    "DistributionDistance",
    "distance_to_nominal_gaussian",
    "lift_hellinger_to_dimension",
    "ExoticBatchSpec",
    "NoiseFamily",
    "PerturbationDistribution",
    "PerturbationKind",
    "PlantPerturbation",
    "rotation_matrix",
    "UnsupportedResyncError",
    "resync_to_plant",
]
