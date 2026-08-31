"""Tier 3 — the specification grammar.

The typed description of an experiment, and the only place identity is derived
from. `ProblemSpec` arrives first because every other identifier is built on a
`ProblemID` (Stage 2 Phase A); `ContenderSpec` follows (Phase B).

**This tier may import `core/` and `store/` and nothing above them.** The
grammar exists to be reusable by the runner, the analysis tier and the command
line alike; letting it reach into `engine/`, `models/`, `experiments/` or
`applications/` is exactly what made `Experiment` un-reusable and is asserted
against in `tests/architecture/test_boundaries.py`, statically per module and
transitively at import time.

What the tier needs from above it is therefore **injected, never imported**:
`ContenderSpec` declares `Recipe` and `RecipeRegistry` as structural protocols
and the caller supplies the concrete registry. `mbl.experiments.ContenderSpec`
is the Tier-4 binding that supplies this project's own.
"""

from __future__ import annotations

from .analysis import AnalysisSpec, require_unique_analysis_ids
from .contender import ContenderSpec, Recipe, RecipeRegistry, Role
from .data import DEFAULT_DATA_KIND, DataSpec
from .errors import SpecificationError
from .evaluation import DEFAULT_METRICS, EvaluationSpec, RehostOverride
from .figure import (
    FigureSpec,
    require_resolvable_sources,
    require_unique_figure_ids,
)
from .gates import GATE_SCHEMA, GateKind, GateSpec, GateStage
from .identity import derive_measurement_id, derive_model_id
from .loader import (
    BUILD_KEY,
    SpecBindings,
    StudyDocument,
    load_study,
    load_tier_catalogue,
)
from .paths import replace_at
from .study import AxisLevel, Composition, StudyPoint, StudySpec, SweepAxis
from .problem import (
    COST_KEYS,
    STORAGE_DTYPE,
    SYSTEM_KEYS,
    GeneratorProvenance,
    ProblemData,
    ProblemSpec,
)
from .tiers import (
    AXIS_SUBSET_KEY,
    DEFAULT_TIER_CATALOGUE,
    PERMITTED_TIER_PATHS,
    SUBSETTING_TIER,
    ResolvedStudy,
    SmokeSubset,
    Tier,
    TierCatalogue,
    require_analysable_measurement,
    require_permitted_paths,
)
from .training import (
    PRECISION_REQUIREMENTS,
    BatchPlan,
    TrainingSpec,
    require_supported_precision,
)

__all__ = [
    "AXIS_SUBSET_KEY",
    "BUILD_KEY",
    "COST_KEYS",
    "DEFAULT_DATA_KIND",
    "DEFAULT_METRICS",
    "DEFAULT_TIER_CATALOGUE",
    "GATE_SCHEMA",
    "PERMITTED_TIER_PATHS",
    "PRECISION_REQUIREMENTS",
    "STORAGE_DTYPE",
    "SUBSETTING_TIER",
    "SYSTEM_KEYS",
    "AnalysisSpec",
    "AxisLevel",
    "BatchPlan",
    "Composition",
    "ContenderSpec",
    "DataSpec",
    "EvaluationSpec",
    "RehostOverride",
    "FigureSpec",
    "GateKind",
    "GateSpec",
    "GateStage",
    "GeneratorProvenance",
    "ProblemData",
    "ProblemSpec",
    "Recipe",
    "RecipeRegistry",
    "ResolvedStudy",
    "Role",
    "SmokeSubset",
    "SpecBindings",
    "SpecificationError",
    "StudyDocument",
    "StudyPoint",
    "StudySpec",
    "SweepAxis",
    "Tier",
    "TierCatalogue",
    "TrainingSpec",
    "derive_measurement_id",
    "derive_model_id",
    "load_study",
    "load_tier_catalogue",
    "replace_at",
    "require_analysable_measurement",
    "require_permitted_paths",
    "require_resolvable_sources",
    "require_supported_precision",
    "require_unique_analysis_ids",
    "require_unique_figure_ids",
]
