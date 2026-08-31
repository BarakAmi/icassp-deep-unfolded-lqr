"""Pluggable mathematical "step" for the Runner: how one batch turns into a
scalar-metrics update. Concrete strategies range from standard backprop
(GradientDescentStrategy) to non-learnable closed-form solvers
(AnalyticalStrategy) -- the Runner itself never assumes gradients or an
optimizer exist.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import torch

from .context import RunContext
from .training_plan import OptimizerSpec, TrainingPlan
from ..core.utils import to_numpy
from ..core.profiling import profiled
from ..models.base import Controller
from ..models.unfolded.layerwise import LayerFreeze

type Batch = tuple[Any, Any, Any]
"""``(initial_state, process_noise, measurement_noise)``, each a batched
array/tensor consumed by a `TrainingStrategy`."""

type BatchSampler = Callable[[], Batch]
"""A zero-argument callable producing a fresh `Batch` (e.g. randomly sampled
initial states and noise)."""

type LossReduction = Callable[[torch.Tensor], torch.Tensor]
"""Reduces a per-sample/per-step cost tensor to a scalar loss (e.g. ``torch.mean``)."""


def chunk_bounds(size: int, microbatch: int | None) -> tuple[tuple[int, int], ...]:
    """Half-open `[low, high)` bounds one effective batch is executed in.

    `None`, or a microbatch at or above the batch, is **one** pass -- so a
    declaration that cannot chunk anything costs nothing and behaves exactly
    as its absence would. The last chunk carries the remainder, which is why
    `BatchPlan.accumulation_steps` is a `ceil`: dropping it would train on
    fewer trajectories than the specification declares, a silent change to the
    estimator rather than a rounding detail.
    """
    if microbatch is None or microbatch >= size:
        return ((0, size),)
    return tuple(
        (low, min(low + microbatch, size)) for low in range(0, size, microbatch)
    )


def accumulation_scale(reduction: LossReduction, chunk: int, total: int) -> float:
    """The share of the declared loss a chunk of `chunk` out of `total` carries.

    **Derived from the reduction, never assumed.** The declared loss is a
    weighted sum of per-trajectory costs with uniform weights -- `1/N` under
    `mean`, `1` under `sum` -- so a chunk must contribute the same weighted sum
    restricted to its own trajectories. Probing the reduction with a vector of
    ones recovers that weight for either:

    ==========  ==================  ==============
    reduction   this returns        which is
    ==========  ==================  ==============
    ``mean``    ``chunk / total``   the chunk's share of the batch
    ``sum``     ``1``               no rescaling at all
    ==========  ==================  ==============

    A uniform ``1 / len(chunks)`` is the natural misreading of Annex 06 §4.3's
    "each pass contributes its microbatch's share" and is wrong exactly where
    that section's own gate looks: at ``total = 100, microbatch = 30`` the
    chunks are 30/30/30/10 and ``1/K`` over-weights the last by **2.5x**.
    """
    if chunk == total:
        return 1.0
    ones_total = torch.ones(total, dtype=torch.float64)
    ones_chunk = torch.ones(chunk, dtype=torch.float64)
    return float(chunk * reduction(ones_total) / (total * reduction(ones_chunk)))


def accumulate_backward(
    model: torch.nn.Module,
    batch: Batch,
    reduction: LossReduction,
    microbatch: int | None,
) -> float:
    """Forward and backward one effective batch in chunks, accumulating into
    `.grad`, and return **the declared loss over the whole batch**.

    The invariant, and the only thing this function promises: *for any declared
    reduction and any microbatch, the accumulated gradient equals the gradient
    of the declared loss over the whole effective batch* (Annex 06 §4.3). The
    caller still owns everything that must happen once -- masking, clipping and
    the optimiser step -- because each of those applied per chunk would make its
    effect depend on the chunk size, which is the machine-dependent quantity
    D20 excludes from identity.

    Backward runs inside the loop rather than after it: freeing each chunk's
    graph as soon as it is consumed is the entire point, since the graph is
    what exceeds 24 GB at `n ~ 100-200`.
    """
    if microbatch is None:
        # The batch is not inspected at all when no chunking is asked for. A
        # `Batch` is only contractually a 3-tuple -- a strategy test may hand a
        # mock model `(None, None, None)` -- so reading a shape here would make
        # an unchunked step depend on something the type never promised. It
        # also keeps this path byte-identical to the pre-chunking one.
        _, _, _, cost = model(*batch)
        loss = reduction(cost)
        loss.backward()
        return float(loss.detach())

    initial_state, process_noise, measurement_noise = batch
    total = int(initial_state.shape[0])
    declared_loss = 0.0
    for low, high in chunk_bounds(total, microbatch):
        _, _, _, cost = model(
            initial_state[low:high],
            process_noise[low:high],
            measurement_noise[low:high],
        )
        scaled = reduction(cost) * accumulation_scale(reduction, high - low, total)
        scaled.backward()
        declared_loss += float(scaled.detach())
    return declared_loss


@dataclass(frozen=True)
class StepExecution:
    """How a step executes, as opposed to what it optimises.

    The three knobs a layer-wise run holds **constant across every phase**
    while `optimizer_spec` and `freeze` vary per phase, bundled because they
    travel together and because the alternative is a seven-argument
    constructor -- which this project gates at six rather than raising the
    limit to fit a feature (`PLR0913`).

    None of the three is signed. `loss_reduction` and `gradient_clip_norm`
    reach `ModelID` through the *plan*, which is a specification; this object
    is what the engine was handed to execute it with, and `microbatch` is
    excluded from identity outright (D20).
    """

    loss_reduction: LossReduction = torch.mean
    gradient_clip_norm: float | None = None
    microbatch: int | None = None


@runtime_checkable
class TrainingStrategy(Protocol):
    """One mathematical step, decoupled from the outer training loop."""

    trains_parameters: ClassVar[bool]
    """Whether this strategy ever updates learnable parameters (calls an
    optimizer). The authoritative, explicit signal `Runner` uses to derive
    `RunContext.is_trainable` -- deliberately not inferred via `nn.Module`
    parameter reflection, which is unreliable for this codebase's own
    `UnfoldedController` (its learnable parameters are never registered as
    standard submodule attributes, so `.parameters()` is empty regardless of
    whether they are actually frozen or trainable)."""

    def step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Perform one training step (forward/backward/update as applicable).

        Args:
            context: The current `RunContext` (model/tracker/config/epoch).
            batch: The batch to step on.

        Returns:
            This step's scalar metrics (e.g. ``{"loss": ...}``).
        """
        ...

    def evaluate_step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Perform one evaluation step (no parameter updates).

        Args:
            context: The current `RunContext`.
            batch: The batch to evaluate on.

        Returns:
            This step's scalar metrics (e.g. ``{"loss": ...}``).
        """
        ...


class GradientDescentStrategy:
    """Standard backprop step: zero_grad -> forward -> reduce -> backward -> step.

    Expects `model(initial_state, process_noise, measurement_noise)` to return
    `(X, Y, U, cost)`, matching UnfoldedController.forward.
    """

    trains_parameters: ClassVar[bool] = True

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_reduction: LossReduction = torch.mean,
        gradient_clip_norm: float | None = None,
        microbatch: int | None = None,
    ) -> None:
        """
        Args:
            model: A torch module whose ``forward(initial_state, process_noise,
                measurement_noise)`` returns ``(X, Y, U, cost)`` with `cost` part
                of the live autograd graph.
            optimizer: The optimizer stepped after each backward pass.
            loss_reduction: Reduces `model`'s returned `cost` tensor to a scalar
                loss; defaults to ``torch.mean``.
            gradient_clip_norm: Optional global gradient-norm clip applied to
                the optimizer's parameters before each step; ``None`` disables.
            microbatch: Trajectories per forward/backward pass, or ``None`` for
                one pass over the whole effective batch. A **resource**
                decision: it never reaches `ModelID` (D20), and it arrives from
                `EngineHarness` rather than from the signed `TrainingPlan`.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_reduction = loss_reduction
        self.gradient_clip_norm = gradient_clip_norm
        self.microbatch = microbatch

    @classmethod
    def from_plan(
        cls,
        model: torch.nn.Module,
        module: torch.nn.Module,
        plan: TrainingPlan,
        *,
        microbatch: int | None = None,
    ) -> "GradientDescentStrategy":
        """Build the strategy a declarative `TrainingPlan` describes (T3.b).

        The optimizer, loss reduction, and clipping are all derived from the
        plan -- the strategy actually executed IS the specification that gets
        signed and logged (the systemic C4 fix).

        Args:
            model: The forward/rollout module the strategy steps on (e.g. a
                `RolloutModel`).
            module: The trainable module supplying the optimizer's parameters
                (the `TrainableController.as_module()` seam, T2.b) -- often,
                but not necessarily, `model` itself.
            plan: The declarative training specification.

        Returns:
            The configured `GradientDescentStrategy`.
        """
        return cls(
            model=model,
            optimizer=plan.optimizer.build(module.parameters()),
            loss_reduction=plan.resolve_loss_reduction(),
            gradient_clip_norm=plan.gradient_clip_norm,
            # NOT from the plan: the plan is signed into `ModelID` and D20
            # excludes the microbatch from it. The caller sources it from
            # `EngineHarness`, which is execution-time resources and is signed
            # nowhere.
            microbatch=microbatch,
        )

    def step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Zero gradients, forward `batch` through `model`, reduce the returned
        cost, backpropagate, and step the optimizer.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"loss": <float>}``, the reduced loss for this step.
        """
        self.optimizer.zero_grad()
        loss = accumulate_backward(
            self.model, batch, self.loss_reduction, self.microbatch
        )
        # Once, on the ACCUMULATED gradient: a per-chunk clip would make the
        # threshold depend on the chunk size (Annex 06 §4.3).
        if self.gradient_clip_norm is not None:
            for group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.gradient_clip_norm)
        self.optimizer.step()
        return {"loss": loss}

    def evaluate_step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Forward `batch` through `model` under ``torch.no_grad()`` and report
        the reduced loss, without any parameter update.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"loss": <float>}``.
        """
        initial_state, process_noise, measurement_noise = batch
        with torch.no_grad():
            _, _, _, cost = self.model(initial_state, process_noise, measurement_noise)
            loss = self.loss_reduction(cost)
        return {"loss": float(loss)}


class LayerwiseGradientDescentStrategy:
    """One phase's masked backprop step for greedy layer-wise ("warm-start")
    deep-unfolded training (NB03 blueprint v2 SS3.4, the Freeze Contract).

    Sibling of `GradientDescentStrategy` -- NOT a subclass, and
    `GradientDescentStrategy` itself is never touched -- with the same
    forward/reduce/backward mechanics, but two load-bearing differences that
    together close the freeze leaks a naive "shared optimizer + gradient
    mask" design exhibits (empirically demonstrated in the blueprint's SS7.1:
    both weight decay AND residual Adam/AdamW momentum move a masked-frozen
    parameter under a shared optimizer):

    1. **A fresh, phase-owned optimizer.** `optimizer_spec` builds a NEW
       optimizer instance over `module.parameters()` at construction time,
       never accepting an already-built one from a caller -- so Adam's
       moment state (`m`, `v`) cannot carry over from an earlier phase's
       training into this one. A row trained in phase ``j`` and frozen in
       phase ``j+1`` therefore cannot drift from residual momentum.
    2. **Post-backward gradient masking.** `freeze.grad_masks` zeroes the
       frozen rows' gradients between `.backward()` and the optimizer step,
       so those rows receive an exact zero update every step -- combined
       with (1), and with `LayerwiseTrainingPlan`'s ``weight_decay`` guard
       (SS7.1), this makes the row freeze airtight under Adam/AdamW.

    The unified matrix parameter (no partial-row structure) freezes by a
    different, complementary mechanism: `freeze.trainable`'s whole-tensor
    `requires_grad_` toggle, applied idempotently every step -- a
    ``requires_grad=False`` parameter's ``.grad`` stays ``None``, so the
    optimizer skips it entirely (immune to weight decay by construction,
    verified empirically in the blueprint's SS7.1).
    """

    trains_parameters: ClassVar[bool] = True

    def __init__(
        self,
        model: torch.nn.Module,
        module: torch.nn.Module,
        optimizer_spec: OptimizerSpec,
        freeze: LayerFreeze,
        *,
        execution: StepExecution | None = None,
    ) -> None:
        """
        Args:
            model: A torch module whose ``forward(initial_state, process_noise,
                measurement_noise)`` returns ``(X, Y, U, cost)`` with `cost`
                part of the live autograd graph (e.g. a `RolloutModel`).
            module: The trainable module supplying the optimizer's
                parameters (the `TrainableController.as_module()` seam);
                its ``named_parameters()`` keys must match `freeze`'s.
            optimizer_spec: Builds THIS phase's own, fresh optimizer over
                `module.parameters()` at construction time -- never a
                shared/inherited instance (the anti-momentum-carryover law,
                point 1 of the class docstring).
            freeze: This phase's compiled freeze plan (row gradient masks
                plus a whole-tensor ``requires_grad`` plan), typically from
                `models.unfolded.layerwise.LayerFreezeBuilder`.
            loss_reduction: Reduces `model`'s returned `cost` tensor to a
                scalar loss; defaults to ``torch.mean``.
            gradient_clip_norm: Optional global gradient-norm clip applied
                to the currently-trainable parameters, AFTER masking -- a
                frozen row's zeroed gradient never inflates the clipped
                norm; ``None`` disables clipping.

        Raises:
            ValueError: If `optimizer_spec.hyperparameters` sets a nonzero
                ``weight_decay`` -- gradient masking cannot stop the
                optimizer from decaying a frozen row of a masked parameter
                (the freeze-contract guard, mirrored here at construction
                time -- the point where THIS phase's optimizer is actually
                built over `freeze` -- as defense-in-depth alongside
                `LayerwiseTrainingPlan`'s own identical guard, for callers
                that construct this strategy directly).
        """
        if optimizer_spec.hyperparameters.get("weight_decay", 0.0):
            raise ValueError(
                "LayerwiseGradientDescentStrategy forbids weight_decay in "
                "its OptimizerSpec: gradient masking cannot stop the "
                "optimizer from decaying frozen rows of a masked parameter "
                "(NB03 blueprint v2 SS7.1). Regularize active entries via a "
                "mask-aware mechanism instead of torch-level weight_decay."
            )
        self.model = model
        self.module = module
        self.freeze = freeze
        execution = execution if execution is not None else StepExecution()
        self.loss_reduction = execution.loss_reduction
        self.gradient_clip_norm = execution.gradient_clip_norm
        self.microbatch = execution.microbatch
        self.optimizer = optimizer_spec.build(module.parameters())
        self._apply_activation()

    def _apply_activation(self) -> None:
        """Toggle every named parameter's ``requires_grad`` per
        ``freeze.trainable`` -- idempotent and safe to call every step (the
        whole-tensor freeze mechanism for e.g. the unified matrix)."""
        for name, parameter in self.module.named_parameters():
            if name in self.freeze.trainable:
                parameter.requires_grad_(self.freeze.trainable[name])

    def step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Zero gradients, forward `batch`, reduce the cost, backpropagate,
        mask the frozen rows' gradients, clip, and step THIS phase's own
        optimizer.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"loss": <float>}``, the reduced loss for this step.
        """
        self._apply_activation()
        self.optimizer.zero_grad()
        loss = accumulate_backward(
            self.model, batch, self.loss_reduction, self.microbatch
        )
        # ONCE, on the accumulated gradient -- not per chunk. Masking inside the
        # loop is arithmetically the same only while the mask is constant across
        # chunks, and relying on that would make a future per-chunk mask
        # silently wrong.
        for name, parameter in self.module.named_parameters():
            mask = self.freeze.grad_masks.get(name)
            if mask is not None and parameter.grad is not None:
                parameter.grad.mul_(mask)
        if self.gradient_clip_norm is not None:
            trainable_params = [p for p in self.module.parameters() if p.requires_grad]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.gradient_clip_norm
                )
        self.optimizer.step()
        return {"loss": loss}

    def evaluate_step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Forward `batch` through `model` under ``torch.no_grad()`` and
        report the reduced loss, without any parameter update.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"loss": <float>}``.
        """
        initial_state, process_noise, measurement_noise = batch
        with torch.no_grad():
            _, _, _, cost = self.model(initial_state, process_noise, measurement_noise)
            loss = self.loss_reduction(cost)
        return {"loss": float(loss)}


class AnalyticalStrategy:
    """Wraps a non-learnable controller (e.g. RiccatiController, or an
    UnfoldedController whose parameters are all frozen) that computes its
    policy in closed form. There is no optimizer and no .backward() here --
    step() and evaluate_step() both simulate the system under the controller's
    (already-optimal or fixed) policy and report the resulting cost, proving
    the engine never forces gradient-based training where it isn't
    mathematically applicable.
    """

    trains_parameters: ClassVar[bool] = False

    def __init__(self, controller: Controller) -> None:
        """
        Args:
            controller: A non-learnable (or frozen) `Controller`, e.g.
                `RiccatiController`, whose `.problem` and
                `.get_control_policy()` drive the simulation.
        """
        self.controller = controller

    @profiled("analytical.simulate")
    def _simulate_and_score(self, batch: Batch) -> Mapping[str, float]:
        """Roll out `self.controller`'s policy on `batch` and report the
        resulting mean cost.

        Args:
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"cost": <float>}``, the batch-mean of `problem.cost`'s
            per-step-average output.
        """
        initial_state, process_noise, measurement_noise = batch
        problem = self.controller.problem
        policy = self.controller.get_control_policy()
        X, _, U = problem.system.run(
            policy, initial_state, process_noise, measurement_noise
        )
        # Cost is defined over the state trajectory X (horizon+1 steps), not the
        # observation trajectory Y (horizon steps) -- passing Y here would always
        # fail QuadraticCost's own horizon validation.
        #
        # to_numpy is a no-op for already-numpy X/U (e.g. RiccatiController),
        # and detaches+converts torch tensors (e.g. a torch-native controller
        # like UnfoldedController) -- QuadraticCost is numpy-only, and a torch
        # tensor whose graph still requires grad crashes outright on implicit
        # numpy conversion otherwise.
        cost = problem.cost(to_numpy(X), to_numpy(U))
        return {"cost": float(np.mean(cost))}

    def step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """Identical to `evaluate_step`: there is no learnable parameter to
        update, so a "training" step is just a simulation + score.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"cost": <float>}`` (see `_simulate_and_score`).
        """
        return self._simulate_and_score(batch)

    def evaluate_step(self, context: RunContext, batch: Batch) -> Mapping[str, float]:
        """See `step` -- both simulate and score, with no parameter update.

        Args:
            context: Unused (present to satisfy `TrainingStrategy`).
            batch: ``(initial_state, process_noise, measurement_noise)``.

        Returns:
            ``{"cost": <float>}`` (see `_simulate_and_score`).
        """
        return self._simulate_and_score(batch)
