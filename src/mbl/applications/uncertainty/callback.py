"""`DomainRandomizationCallback` -- the training-loop mechanism behind every
NB07 uncertainty axis (docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec
2.1/5.2): at each epoch, draw a fresh (or held-fixed) perturbed plant from a
`PerturbationDistribution` and rewrite the LIVE `LinearSystem` the training
rollout simulates -- no `Runner`/`TrainingStrategy` edit required, because
`LinearSystem._state_transition_map` already reads its `A_t`/`B_t` fresh at
every rollout step (`core.system.linear_system.LinearSystem`).

Placed in `applications.uncertainty`, not `engine.callbacks`, deliberately:
it depends on `PerturbationDistribution`/`resync_to_plant`, both Tier-3
(`applications`) objects, and `engine` never imports upward from
`applications` anywhere in the tree -- keeping it here preserves that
one-way layering rather than inverting it for this one callback.
"""

from typing import Any

import numpy as np
import pandas as pd

from .perturbations import PerturbationDistribution
from .resync import resync_to_plant
from ...core.runtime import ComputeContext
from ...core.system.linear_system import TimeSeriesMatrix
from ...engine.callbacks import Callback
from ...engine.context import RunContext


class DomainRandomizationCallback(Callback):
    """Rewrites `context.model.problem.system`'s `A_t`/`B_t` every epoch from
    a `PerturbationDistribution`, optionally resynchronizing the controller's
    own internal model to match (the `ORACLE` model-access mode, NB07 plan
    Sec 4.2) -- `resync_controller=False` is `NOMINAL` mode: the plant drifts
    but the controller's internal model does not.

    The nominal `(A, B, W)` are restored onto `system` at `on_train_end`, so
    the shared `problem` object is left exactly as it was found once training
    completes -- any code that reads `problem` afterward (evaluation,
    `TrajectoryLoggingCallback`'s post-training rollout, a later contender
    sharing the same problem instance) sees the nominal plant, never a
    leftover randomized draw from the final training epoch.
    """

    def __init__(
        self,
        distribution: PerturbationDistribution,
        *,
        ctx: ComputeContext,
        resync_controller: bool,
        artifact_name: str = "domain_randomization_history",
    ) -> None:
        """
        Args:
            distribution: What each epoch draws its plant from.
            ctx: The compute context `resync_controller=True`'s resync
                rebuilds any new tensor at (dtype/device authority).
            resync_controller: `True` calls `resync_to_plant` after every
                draw (`ORACLE` mode); `False` leaves the controller's
                internal model untouched (`NOMINAL` mode).
            artifact_name: Artifact name the per-epoch realized-perturbation
                history (`spectral_radius`/`angle_degrees` per epoch) is
                saved under at `on_train_end`.
        """
        self._distribution = distribution
        self._ctx = ctx
        self._resync_controller = resync_controller
        self._artifact_name = artifact_name
        self._stream: Any = None
        self._nominal_A: Any = None
        self._nominal_B: Any = None
        self._history: list[dict[str, float]] = []

    def on_train_start(self, context: RunContext) -> None:
        """Capture the nominal `(A, B, W)` (for `on_train_end` restoration)
        and build the epoch-draw stream from them.

        Args:
            context: The current `RunContext`.
        """
        system = context.model.problem.system
        self._nominal_A = system.A_t.array
        self._nominal_B = system.B_t.array
        # W has no home on `LinearSystem` (process-noise covariance lives in
        # the batch sampler, not the dynamics); a fixed placeholder is
        # sufficient since every perturbation this module defines either
        # leaves W unchanged (ADDITIVE) or rotates an ISOTROPIC W (ROTATION,
        # whose covariance is invariant under rotation by construction) --
        # see `perturbations.PlantPerturbation._draw_rotation`'s docstring.
        n = system.dimensions.state_dim
        placeholder_w = np.eye(n)
        self._stream = self._distribution.build_stream(
            self._nominal_A, self._nominal_B, placeholder_w
        )

    def on_epoch_start(self, context: RunContext) -> None:
        """Draw this epoch's plant and rewrite `system.A_t`/`B_t` in place;
        under `resync_controller=True`, also resynchronize the controller.

        Args:
            context: The current `RunContext`.
        """
        A, B, _, realized = self._stream()
        system = context.model.problem.system
        system.A_t = TimeSeriesMatrix("A_t", A)
        system.B_t = TimeSeriesMatrix("B_t", B)
        if self._resync_controller:
            controller = getattr(context.model, "controller", context.model)
            resync_to_plant(controller, context.model.problem, self._ctx)
        self._history.append({"epoch": context.epoch, **realized})

    def on_train_end(self, context: RunContext) -> None:
        """Restore the nominal plant onto `system` and persist the per-epoch
        realized-perturbation history.

        Args:
            context: The current `RunContext`.
        """
        system = context.model.problem.system
        system.A_t = TimeSeriesMatrix("A_t", self._nominal_A)
        system.B_t = TimeSeriesMatrix("B_t", self._nominal_B)
        if self._resync_controller:
            controller = getattr(context.model, "controller", context.model)
            resync_to_plant(controller, context.model.problem, self._ctx)
        context.tracker.save_artifact(self._artifact_name, pd.DataFrame(self._history))
