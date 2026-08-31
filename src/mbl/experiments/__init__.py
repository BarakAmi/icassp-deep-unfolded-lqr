"""The Research Operations layer (REFACTOR_PLAN v3, Stage S4): the
`Experiment` entity and `run_experiment` unified entry point (T3.d), the
per-contender, code-aware `ExperimentCache` (T3.e), the append-only
experiment history registry (T3.j), and the per-run logging binding (T3.k).
"""

from .bindings import BUILDERS, DEFAULT_SPEC_BINDINGS, ExperimentBindings
from .cache import (
    CachePolicy,
    CacheReadOnlyMissError,
    CachedContenderRun,
    ContenderPayload,
    ExperimentCache,
    TrackerBackedExperimentCache,
    compose_cache_key,
)
from .evaluation import evaluate_synthesized_controller
from .experiment import (
    ContenderResult,
    ContenderSpec,
    EvaluationProtocol,
    Experiment,
    ExperimentReport,
    problem_dimensions,
)
from .history import (
    REGISTRY_FILENAME,
    ExperimentHistoryRegistry,
    HistoryRecord,
)
from .notebook_bootstrap import (
    bootstrap_experiment_notebook,
    configure_notebook_logging,
    find_project_root,
)
from .provenance import (
    RESULTS_SCHEMA_VERSION,
    code_provenance_stamp,
    package_version,
)
from .run_logging import RUN_LOG_FILENAME, bind_run_log
from .runner import ExecutionServices, run_experiment
from .zero_shot import ShiftMetrics, evaluate_under_shift, synthesize_nominal_contenders

__all__ = [
    "BUILDERS",
    "DEFAULT_SPEC_BINDINGS",
    "ExperimentBindings",
    "CachePolicy",
    "CacheReadOnlyMissError",
    "CachedContenderRun",
    "ContenderPayload",
    "ExperimentCache",
    "TrackerBackedExperimentCache",
    "compose_cache_key",
    "evaluate_synthesized_controller",
    "ContenderResult",
    "ContenderSpec",
    "EvaluationProtocol",
    "Experiment",
    "ExperimentReport",
    "problem_dimensions",
    "REGISTRY_FILENAME",
    "ExperimentHistoryRegistry",
    "HistoryRecord",
    "bootstrap_experiment_notebook",
    "configure_notebook_logging",
    "find_project_root",
    "RESULTS_SCHEMA_VERSION",
    "code_provenance_stamp",
    "package_version",
    "RUN_LOG_FILENAME",
    "bind_run_log",
    "ExecutionServices",
    "run_experiment",
    "ShiftMetrics",
    "evaluate_under_shift",
    "synthesize_nominal_contenders",
]
