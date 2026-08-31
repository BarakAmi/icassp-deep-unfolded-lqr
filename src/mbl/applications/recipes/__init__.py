"""The per-family `ModelRecipe` layer (REFACTOR_PLAN v3, T3.a) and its
registry (T2.f): controller families own their engine wiring; case studies
compose recipes as data."""

from .base import (
    RECIPE_REGISTRY,
    AnalyticRecipe,
    EngineHarness,
    EngineTrainedSynthesizer,
    FrozenControllerSynthesizer,
    ModelRecipe,
    TrainableRecipe,
    TrainedControllerArtifact,
    build_default_recipe_registry,
    null_harness,
    register_recipe,
)
from .analytic import RiccatiRecipe, TruncatedRiccatiRecipe
from .cocp import COCPLowerBoundRecipe, COCPRecipe
from .cocp_exact import ExactCOCPLowerBoundRecipe, ExactCOCPRecipe
from .neural import NeuralRecipe
from .unfolded import (
    FixedUnfoldedRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
    WarmStartSynthesizer,
    WarmStartUnfoldedRecipe,
)

__all__ = [
    "RECIPE_REGISTRY",
    "register_recipe",
    "build_default_recipe_registry",
    "EngineHarness",
    "ModelRecipe",
    "AnalyticRecipe",
    "TrainableRecipe",
    "TrainedControllerArtifact",
    "EngineTrainedSynthesizer",
    "FrozenControllerSynthesizer",
    "null_harness",
    "RiccatiRecipe",
    "TruncatedRiccatiRecipe",
    "NeuralRecipe",
    "UnfoldedKind",
    "UnfoldedRecipe",
    "FixedUnfoldedRecipe",
    "WarmStartUnfoldedRecipe",
    "WarmStartSynthesizer",
    "COCPRecipe",
    "ExactCOCPLowerBoundRecipe",
    "ExactCOCPRecipe",
    "COCPLowerBoundRecipe",
]
