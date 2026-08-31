"""Open-loop control: a decision variable ``U`` (the whole control sequence)
refined directly, rather than via a per-time-step feedback law.

`OpenLoopGDController` (this package) is the channel-agnostic ``U``-holder,
usable by either gradient source:

- `models.analytic.iterative_gd.AnalyticalIterativeGDController` -- the
  engine-free, closed-form-local-gradient solver (preferred for LQR-shaped
  objectives).
- `.differentiable` (this package) -- the preserved autograd channel
  (`AnalyticalGradientDescent` + `engine.strategy.GradientDescentStrategy` +
  `engine.runner.Runner`), for objectives with no closed-form local
  gradient. See `.differentiable`'s own docstring for the full channel
  doctrine.

`AnalyticalGradientDescent` is re-exported here (from `.differentiable`) for
backward compatibility with existing notebooks/call sites that import it
from this top-level package.
"""

from .differentiable import AnalyticalGradientDescent
from .gd_controller import OpenLoopGDConfig, OpenLoopGDController
from ..iterative.step_size import StepSizeProvider, StepSizeSchedule

__all__ = [
    "OpenLoopGDConfig",
    "OpenLoopGDController",
    "AnalyticalGradientDescent",
    "StepSizeProvider",
    "StepSizeSchedule",
]
