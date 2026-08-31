"""viz layer L2 -- the Phase 2B algorithm/problem-agnostic cost-landscape
toolkit for control sequences: oracles, grid projection, overlays, static
landscape/signal renderers, their animated counterparts, and the grid
framing policies (REFACTOR_PLAN v3, T4.a -- relocated from
`applications.visualization`).

Everything here is a function of exactly two seams: a `CostOracle` (see
`types.CostOracle`) and plain ``baseline_U``/``candidate_U_history`` arrays
-- never a concrete controller, problem, or the string "Riccati"/"GD".
"""

from .animators import (
    FrameAnimator,
    LandscapeAnimationOptions,
    LandscapeAnimator,
    LineAnimator,
    RelativeErrorAnimator,
    SignalAnimator,
)
from .framing import compute_bowl_ranges, span_control_range
from .grids import GridProjector
from .landscapes import (
    ContourLandscapeRenderer,
    LineLandscapeRenderer,
    ScatterLandscapeRenderer,
    SurfaceLandscapeRenderer,
)
from .oracles import make_local_cost_to_go_oracle, make_rollout_cost_oracle
from .overlay import TrajectoryOverlay, select_iterations
from .signals import (
    ControlSignalRenderer,
    RelativeErrorRenderer,
    compute_aggregate_relative_error,
    compute_relative_error,
    plot_linked_signals_and_error,
    SignalStyle,
    plot_sparse_signals_and_errors,
)
from .types import (
    CostField,
    CostOracle,
    IterationSelection,
    SliceSpec,
    normalize_control,
    normalize_history,
)

__all__ = [
    # Seams / value objects
    "CostOracle",
    "CostField",
    "SliceSpec",
    "IterationSelection",
    "normalize_control",
    "normalize_history",
    # Oracle adapters
    "make_rollout_cost_oracle",
    "make_local_cost_to_go_oracle",
    # Grid projection & framing
    "GridProjector",
    "span_control_range",
    "compute_bowl_ranges",
    # Overlay (Asset 6)
    "TrajectoryOverlay",
    "select_iterations",
    # Assets 1 & 2 (static)
    "ControlSignalRenderer",
    "RelativeErrorRenderer",
    "plot_linked_signals_and_error",
    "SignalStyle",
    "plot_sparse_signals_and_errors",
    "compute_relative_error",
    "compute_aggregate_relative_error",
    # Assets 3a-5 (static)
    "LineLandscapeRenderer",
    "ContourLandscapeRenderer",
    "SurfaceLandscapeRenderer",
    "ScatterLandscapeRenderer",
    # Animated counterparts
    "FrameAnimator",
    "SignalAnimator",
    "RelativeErrorAnimator",
    "LandscapeAnimationOptions",
    "LandscapeAnimator",
    "LineAnimator",
]
