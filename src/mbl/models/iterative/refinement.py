"""Iterative refinement mechanics (Phase 2.5): the shared, algorithm-agnostic
vocabulary mapping an initial control proposal to a refined one via one or
more gradient-descent steps -- consumed by both `models.unfolded`'s per-
time-step (deep-unfolded) refinement and `models.analytic.iterative_gd`'s
whole-horizon macro-sweep refinement, so the loop body exists exactly once.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence, cast

import torch

from .step_size import StepSizeProvider, StepSizeSchedule
from ...core.constraint.constraint import Constraint
from ...core.utils import ensure_positive_integer
from ...core.utils.signing import hash_array


type StepHook = Callable[[int, int, torch.Tensor, torch.Tensor], None]
"""``(k, t, u, grad) -> None``, invoked after each inner gradient-descent
iteration ``k`` at time step ``t``, e.g. for logging/visualization."""

type StopCondition = Callable[[int, torch.Tensor, torch.Tensor], bool]
"""``(k, u, grad) -> bool``, checked after each inner iteration; returning
``True`` stops the refinement early."""


class IterativeRefinement(ABC):
    """Stateless refinement: maps (t, y, u_init) -> u_refined."""

    def __init__(
        self,
        learnable_parameters: Mapping[str, Any] | None = None,
        static_parameters: Mapping[str, Any] | None = None,
        constraints: Sequence[Constraint] | None = None,
    ):
        """
        Args:
            learnable_parameters: Named learnable objects this refinement
                reads via `.get()` at call time (a live reference: values
                change between optimizer steps, so `.get()` is never cached
                across calls); defaults to ``{}`` (e.g. `GradientDescentRefinement`
                subclasses whose only learnable knob is the step size, held
                instead in its own dedicated constructor argument).
            static_parameters: Named non-learnable tensors/arrays (e.g.
                precomputed ``A``/``B``/``R`` stacks) used by the refinement;
                defaults to ``{}``.
            constraints: Optional feasible-set projections applied to `u`
                during refinement (see `apply_constraints`).

        The rarer behavior knobs — `apply_constraint_each` (projection
        cadence), `step_hook` (post-inner-iteration callback), and
        `stop_condition` (early-stop predicate) — are plain attributes with
        the defaults below; no current caller configures them at
        construction, so they were removed from the constructor surface
        (Stage S6 signature budget) while the seams themselves remain.
        """
        # Live reference — .get() always returns the current nn.Parameter values.
        # Do NOT call .get() here; parameters change between optimizer steps.
        self.learnable_parameters = learnable_parameters or {}
        self.static_parameters = static_parameters or {}
        self.constraints = constraints or []
        self.apply_constraint_each: int = 1
        self.step_hook: StepHook | None = None
        self.stop_condition: StopCondition | None = None

    def apply_constraints(self, u: torch.Tensor, k: int = -1) -> torch.Tensor:
        """Project `u` through every constraint in `self.constraints`, in
        order, if `k` is on the `apply_constraint_each` cadence.

        Args:
            u: Control to project, shape ``(batch, control_dim)``.
            k: The current inner iteration index (``-1`` for the initial,
                pre-refinement projection).

        Returns:
            `u`, projected through each constraint if applicable; otherwise
            `u` unchanged.
        """
        if self.constraints and (k + 1) % self.apply_constraint_each == 0:
            for constraint in self.constraints:
                u = cast(torch.Tensor, constraint(u))
        return u

    def apply_hook(self, u: torch.Tensor, k: int, t: int, *args: Any) -> None:
        """Invoke `self.step_hook` (if set) with the current iteration state.

        Args:
            u: The current (post-update) control.
            k: The current inner iteration index.
            t: The current time step.
            *args: Extra values forwarded to `step_hook` (e.g. the gradient).
        """
        if self.step_hook is not None:
            self.step_hook(k, t, u, *args)

    def should_stop(self, u: torch.Tensor, k: int, *args: Any) -> bool:
        """Evaluate `self.stop_condition` (if set) to decide early stopping.

        Args:
            u: The current (post-update) control.
            k: The current inner iteration index.
            *args: Extra values forwarded to `stop_condition` (e.g. the gradient).

        Returns:
            ``self.stop_condition(k, u, *args)`` if set, else ``False``.
        """
        if self.stop_condition is not None:
            return bool(self.stop_condition(k, u, *args))
        return False

    def on_rollout_start(self) -> None:
        """Lifecycle hook fired once at the start of each rollout (before the
        per-time-step calls). Default no-op; subclasses that cache per-forward
        state (e.g. RiccatiRefinement's horizon gradient matrices) override it to
        invalidate that cache so each forward rebuilds a fresh autograd graph."""

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: this refinement's own type -- lets a solve's
        provenance record (e.g. `models.iterative.OptimizationResult
        .signature`) capture which refinement strategy produced a given run.
        Subclasses with meaningful configuration (e.g. a step-size schedule)
        should override to include it.

        Returns:
            ``{"type": <class name>}`` by default.
        """
        return {"type": type(self).__name__}

    @abstractmethod
    def __call__(self, t: int, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Refine `u` at time step `t` given observation `y`.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.
            u: The initial control proposal, shape ``(batch, control_dim)``.

        Returns:
            The refined control, shape ``(batch, control_dim)``.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        ...


class GradientDescentRefinement(IterativeRefinement):
    """Refines `u` via a fixed number of gradient-descent iterations against
    a per-step objective supplied by `pre_iteration_hook`/`get_gradient`, at
    a step size supplied by an explicit, typed `StepSizeProvider` seam --
    fixed (`StepSizeSchedule`) or learnable (`models.unfolded.parameters
    .StepSizeParameter`, which structurally satisfies the same protocol via
    its own `for_iteration`)."""

    def __init__(
        self,
        step_size: StepSizeProvider,
        num_iterations: int,
        learnable_parameters: Mapping[str, Any] | None = None,
        static_parameters: Mapping[str, Any] | None = None,
        constraints: Sequence[Constraint] | None = None,
    ) -> None:
        """
        Args:
            step_size: Yields the broadcastable per-iteration step size (see
                `step_size.StepSizeProvider`).
            num_iterations: Number of gradient-descent iterations `__call__`
                runs (or an external driver runs via `refine_step`, one per
                call) -- previously implied by a step-size tensor's leading
                dimension; now explicit, since a learnable `StepSizeProvider`
                has no fixed length of its own to read.
            learnable_parameters: Named learnable objects OTHER than the step
                size (e.g. `RiccatiRefinement`'s ``"riccati_matrix"``); the
                step size itself is no longer looked up from this mapping --
                see `step_size` above. Defaults to ``{}``.
            static_parameters: See `IterativeRefinement.__init__`.
            constraints: See `IterativeRefinement.__init__`.

        Raises:
            ValueError: If `num_iterations` is not a positive ``int``.
        """
        super().__init__(
            learnable_parameters=learnable_parameters,
            static_parameters=static_parameters,
            constraints=constraints,
        )
        ensure_positive_integer(num_iterations, "num_iterations")
        self.step_size = step_size
        self.num_iterations = num_iterations

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: this refinement's type, iteration count, and --
        for a fixed (`StepSizeSchedule`) step size -- its shape and content
        hash (never embedded raw). A learnable step size (e.g.
        `StepSizeParameter`) is intentionally omitted here: its own
        `get_signature()` already has a home wherever it's tracked as a named
        learnable parameter (e.g. `UnfoldingConfig.parameters`).

        Returns:
            ``{"type":, "num_iterations":}``, plus ``"step_size_shape"``/
            ``"step_size_hash"`` when `step_size` is a `StepSizeSchedule`.
        """
        signature: dict[str, Any] = {
            "type": type(self).__name__,
            "num_iterations": self.num_iterations,
        }
        if isinstance(self.step_size, StepSizeSchedule):
            signature["step_size_shape"] = tuple(self.step_size.raw.shape)
            signature["step_size_hash"] = hash_array(
                self.step_size.raw.detach().cpu().numpy()
            )
        return signature

    @abstractmethod
    def pre_iteration_hook(self, t: int, y: torch.Tensor) -> tuple[Any, ...]:
        """Precompute this time step's fixed arguments to `get_gradient`
        (e.g. gradient-coefficient matrices), reused across every inner
        iteration.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.

        Returns:
            A tuple of extra arguments forwarded to `get_gradient` at every
            inner iteration.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    @abstractmethod
    def get_gradient(self, u: torch.Tensor, *args: Any) -> torch.Tensor:
        """Compute the gradient of the per-step objective w.r.t. `u`.

        Args:
            u: The current control iterate, shape ``(batch, control_dim)``.
            *args: This time step's fixed arguments, from `pre_iteration_hook`.

        Returns:
            The gradient, shape ``(batch, control_dim)``.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        pass

    def refine_step(
        self, iteration_index: int, u: torch.Tensor, *args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ONE gradient-descent update: ``u <- u - alpha_i * grad``,
        then project through `apply_constraints`. The single-iteration body
        shared by `__call__`'s inner loop (unfolded per-time-step refinement)
        and any external driver stepping through macro-iterations one call at
        a time (e.g. `models.analytic.iterative_gd`'s sweep strategies).

        The CALLER owns the meaning of `iteration_index`: `__call__` passes
        the inner unrolled index ``k``; an external driver passes its own
        macro-iteration index -- both simply select this iteration's step
        size via `self.step_size.for_iteration`.

        Args:
            iteration_index: Selects this iteration's step size.
            u: The current control iterate, shape ``(batch, control_dim)``.
            *args: This time step's fixed arguments, from `pre_iteration_hook`.

        Returns:
            ``(u_next, grad)``: the updated, constraint-projected control and
            the gradient it was computed from.
        """
        grad = self.get_gradient(u, *args)
        u_next = self.apply_constraints(
            u - self.step_size.for_iteration(iteration_index) * grad, iteration_index
        )
        return u_next, grad

    def __call__(self, t: int, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Run the fixed-iteration-count gradient descent: precompute this
        step's objective via `pre_iteration_hook`, then repeatedly apply
        `refine_step`, firing `apply_hook` after every iteration, with early
        exit via `should_stop`.

        Args:
            t: The current time step.
            y: Current observation, shape ``(batch, observation_dim)``.
            u: The initial control proposal, shape ``(batch, control_dim)``.

        Returns:
            The refined control after up to `self.num_iterations` iterations.
        """
        args = self.pre_iteration_hook(t, y)
        u = self.apply_constraints(u)  # initial projection before refinement

        for k in range(self.num_iterations):
            u, grad = self.refine_step(k, u, *args)
            self.apply_hook(u, k, t, grad)
            if self.should_stop(u, k, grad, *args):
                break

        return u
