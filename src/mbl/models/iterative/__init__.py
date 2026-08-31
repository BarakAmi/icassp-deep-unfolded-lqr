"""Shared, algorithm-agnostic iterative-refinement vocabulary (Phase 2.5):
control initializers, step-size providers, and gradient-descent refinement
mechanics, consumed by both `models.unfolded` (per-time-step, deep-unfolded
refinement) and `models.analytic.iterative_gd` (whole-horizon macro-sweep
refinement) -- one loop body, one step-size seam, shared by every iterative
LQR solver family."""

from .initializers import (
    ConstantInitializer,
    ControlInitializer,
    ControlInitMethod,
    SamplerInitializer,
    WarmStartInitializer,
    build_control_initializer,
)
from .refinement import (
    GradientDescentRefinement,
    IterativeRefinement,
    StepHook,
    StopCondition,
)
from .result import OptimizationResult
from .step_size import StepSizeProvider, StepSizeSchedule
from .sweeps import (
    GaussSeidelSweep,
    JacobiSweep,
    SweepContext,
    SweepResult,
    SweepStrategy,
)

__all__ = [
    "ControlInitializer",
    "ConstantInitializer",
    "WarmStartInitializer",
    "SamplerInitializer",
    "ControlInitMethod",
    "build_control_initializer",
    "StepSizeProvider",
    "StepSizeSchedule",
    "IterativeRefinement",
    "GradientDescentRefinement",
    "StepHook",
    "StopCondition",
    "OptimizationResult",
    "SweepStrategy",
    "SweepContext",
    "SweepResult",
    "JacobiSweep",
    "GaussSeidelSweep",
]
