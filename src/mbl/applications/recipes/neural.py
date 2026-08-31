"""Recipe for the recurrent neural policy family (`NeuralPolicy`)."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from .base import accept_enum_values, TrainableRecipe, register_recipe
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...engine.training_plan import TrainingPlan
from ...models.neural.nerual import NeuralConfig, NeuralPolicy, SequenceModelType


@register_recipe("neural")
@dataclass(frozen=True)
class NeuralRecipe(TrainableRecipe):
    """A sequence-model (default GRU) control policy, gradient-trained
    end-to-end through the differentiable rollout.

    Attributes:
        hidden_dim: The backbone's hidden state dimension.
        plan: This family's declarative `TrainingPlan` (T3.b).
        model_type: The sequence backbone family.
        init_seed: Seed of the policy's initial weights, or `None` -- the
            default -- to **inherit the training replicate** (Annex 01
            §2.3.3). `torch.nn.Module` parameter initialization draws from
            the ambient global RNG, which the producer seeds per replicate,
            so an unpinned policy is re-initialized by every replicate while
            a declared seed pins it across all of them. A declared seed
            constructs under a SCOPED fork of the global RNG, leaving the
            caller's own stream unaffected before and after (NB07 plan Sec
            2.4: two "identical" pinned recipes must not build differently-
            initialized policies because of unrelated prior draws).
        label: Instance name; defaults to ``"neural"``.
    """

    family: ClassVar[str] = "neural"

    hidden_dim: int
    plan: TrainingPlan
    model_type: SequenceModelType = SequenceModelType.GRU
    init_seed: int | None = None
    label: str = "neural"

    def __post_init__(self) -> None:
        """Accept `model_type` as its string value, so this family is reachable
        from the declarative surface (`unfolded.accept_enum_values`)."""
        accept_enum_values(self, model_type=SequenceModelType)

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> NeuralPolicy:
        """See `ModelRecipe.build_controller`.

        The policy's weights are authored at the context's precision (its
        backbone would otherwise default to float32 and a torch RNN raises
        on a dtype mismatch against the float64 system matrices) AND its
        device (constructing an `nn.Module` always defaults to CPU
        regardless of `ctx`, so a CUDA `ComputeContext` would otherwise
        leave the policy's parameters on CPU while every rollout tensor is
        CUDA-resident -- a device-mismatch `RuntimeError` on the very first
        forward pass; caught only by actually training a GRU on a CUDA
        machine, which no prior notebook's contender list had ever done).
        Its own ``apply_constraints`` already reads ``problem.constraints``,
        so a box-constrained problem needs no extra wiring here.

        A **declared** `init_seed` constructs under a scoped, seeded fork of
        the global torch RNG -- the global stream is restored to its prior
        state immediately after, whether construction succeeds or raises. An
        undeclared one draws from the ambient stream and advances it, which
        is exactly how it inherits the training replicate the producer seeded
        (Annex 01 §2.3.3); the advance is harmless because each point is
        seeded afresh before its own synthesis.
        """
        if self.init_seed is None:
            return self._policy(problem).to(
                dtype=ctx.torch_dtype, device=ctx.torch_device
            )
        saved_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.init_seed)
            policy = self._policy(problem)
        finally:
            torch.random.set_rng_state(saved_state)
        return policy.to(dtype=ctx.torch_dtype, device=ctx.torch_device)

    def _policy(self, problem: OptimalControlProblem) -> NeuralPolicy:
        """The construction itself, shared by the pinned and inherited paths."""
        return NeuralPolicy(
            problem,
            NeuralConfig(
                state_dim=problem.system.dimensions.state_dim,
                control_dim=problem.system.dimensions.control_dim,
                hidden_dim=self.hidden_dim,
                model_type=self.model_type,
            ),
        )
