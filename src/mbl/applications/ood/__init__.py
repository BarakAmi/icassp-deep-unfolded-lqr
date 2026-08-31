"""Zero-shot OOD generalization machinery (NB06, docs/planning/
NB06_OOD_GENERALIZATION_PLAN.md): dynamics perturbations, exotic noise
families, and distribution-distance metrics for evaluating a nominally
trained controller under distribution shift.
"""

from .distances import (
    DistributionDistance,
    distance_to_nominal_gaussian,
    lift_hellinger_to_dimension,
)
from .noise import ExoticBatchSpec, NoiseFamily, family_scale_parameter
from .perturbations import (
    DynamicsPerturbation,
    DynamicsPerturbationKind,
    PerturbedLQRProblemFactory,
)
from .rehost import rehost_at_horizon

__all__ = [
    "DistributionDistance",
    "distance_to_nominal_gaussian",
    "lift_hellinger_to_dimension",
    "ExoticBatchSpec",
    "NoiseFamily",
    "family_scale_parameter",
    "DynamicsPerturbation",
    "DynamicsPerturbationKind",
    "PerturbedLQRProblemFactory",
    "rehost_at_horizon",
]
