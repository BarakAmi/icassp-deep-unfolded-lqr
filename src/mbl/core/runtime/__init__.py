"""Tier 1 + 1.5 runtime substrate (REFACTOR_PLAN v3, Stage S1).

The single import root for the compute context (T1.b), the env-ingestion
settings surface (T1.c), the named tolerance policy (T1.d), the dual-mode
validation policy and its identity-law decorator surfaces (T1.5.b), and
the typed runtime exceptions.
"""

from .compute_context import (
    Backend,
    ComputeContext,
    Precision,
    default_compute_context,
)
from .exceptions import (
    DeviceUnavailableError,
    InvalidComputeContextError,
    UnsupportedBackendError,
)
from .modes import (
    ValidationMode,
    get_validation_mode,
    is_strict,
    set_validation_mode,
    validation_mode,
)
from .settings import env_flag, env_str
from .tolerances import DEFINITENESS_TOL, SYMMETRY_ATOL
from .validators import mode_gated, shape_checked, validate_boundary

__all__ = [
    "Backend",
    "ComputeContext",
    "Precision",
    "default_compute_context",
    "DeviceUnavailableError",
    "InvalidComputeContextError",
    "UnsupportedBackendError",
    "ValidationMode",
    "get_validation_mode",
    "is_strict",
    "set_validation_mode",
    "validation_mode",
    "env_flag",
    "env_str",
    "DEFINITENESS_TOL",
    "SYMMETRY_ATOL",
    "mode_gated",
    "shape_checked",
    "validate_boundary",
]
