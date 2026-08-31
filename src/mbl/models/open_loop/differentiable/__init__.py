"""The preserved autograd-differentiable open-loop channel (Phase 2.5): a
decision variable ``U`` refined by `AnalyticalGradientDescent` (a fixed-
schedule `torch.optim.Optimizer`) driven through the ML engine's
`GradientDescentStrategy` + `Runner` epoch loop.

**Channel doctrine -- when to use this vs. the analytical solver:**

- For a linear-quadratic (LQR-shaped) objective, prefer
  `models.analytic.iterative_gd.AnalyticalIterativeGDController`: it uses the
  exact, closed-form local Bellman gradient (no autograd graph), is
  engine-free (no `Runner`/`TrainingConfig`/`TrainingStrategy`), and is
  cheaper (one forward sweep serves both the gradient and the cost).
- Use **this** channel when the objective has **no closed-form local
  gradient** -- a non-quadratic penalty, a learned cost term, or (the
  motivating future case) a Covert-Control KL-divergence constraint on the
  control distribution. `RolloutModel` computes the trajectory's
  differentiable cost directly in torch; `AnalyticalGradientDescent` applies
  the same fixed-schedule update rule as the analytical solver, but derives
  its gradient from `.backward()` instead of a closed-form formula.

Both channels drive the identical open-loop `models.open_loop
.OpenLoopGDController` (the ``U``-holder) and the identical
`models.iterative.step_size.StepSizeSchedule` step-size geometry -- only the
gradient *source* and the *driver* (engine epoch loop vs. a native
`.solve()` method) differ.
"""

from .optimizer import AnalyticalGradientDescent

__all__ = ["AnalyticalGradientDescent"]
