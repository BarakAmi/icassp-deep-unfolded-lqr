"""The OOP refinement construct for open-loop analytical gradient descent:
`AnalyticalGradientDescent` is the fixed-schedule, non-learned counterpart to
a standard `torch.optim.Optimizer` -- the canonical PyTorch abstraction for
"apply a gradient update under a step-size policy", so it plugs into
`engine.strategy.GradientDescentStrategy` (which only calls
``zero_grad()``/``step()``) with zero changes to that strategy.
"""

from collections.abc import Iterable
from typing import Any

import torch

from ...iterative.step_size import StepSizeProvider


class AnalyticalGradientDescent(torch.optim.Optimizer):
    """Fixed-schedule gradient descent over a decision variable. On iteration
    ``i``, applies ``p <- p - alpha_i * p.grad``, where
    ``alpha_i = schedule.for_iteration(i)`` is a fixed, user-defined
    (broadcastable) step size -- the analytical, non-learned counterpart to a
    learnable optimizer. Exactly one ``.step()`` call is expected per Runner
    epoch, so this optimizer's internal iteration counter tracks the
    gradient-descent/epoch index ``i`` with no external syncing required.
    """

    def __init__(
        self, params: "Iterable[torch.Tensor]", schedule: StepSizeProvider
    ) -> None:
        """
        Args:
            params: An iterable of `requires_grad=True` leaf tensors (e.g.
                ``[controller.U]``) -- `torch.optim.Optimizer` accepts any
                such iterable, no `nn.Parameter`/`nn.Module` required.
            schedule: Yields the broadcastable per-iteration step size (see
                `StepSizeProvider`); fixed and non-learned.
        """
        super().__init__(params, defaults={})
        self._schedule = schedule
        self._iteration = 0

    @torch.no_grad()
    def step(  # type: ignore[override]  # never re-evaluates the loss (see docstring)
        self, closure: Any = None
    ) -> None:
        """Apply one in-place, leaf-preserving gradient-descent update to
        every parameter with a gradient, using this iteration's step size,
        then advance the internal iteration counter.

        Args:
            closure: Unused (present to match `torch.optim.Optimizer.step`'s
                signature); this optimizer never re-evaluates the loss.
        """
        alpha = self._schedule.for_iteration(self._iteration)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.sub_(alpha * p.grad)
        self._iteration += 1
