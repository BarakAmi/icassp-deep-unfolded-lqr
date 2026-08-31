"""Declarative, signable torch-training specifications (REFACTOR_PLAN v3,
T3.b): `OptimizerSpec` and `TrainingPlan` are the single source of truth for
"which optimizer, at which hyperparameters, for how many epochs" -- the
engine *builds* its optimizer from the spec, and the logged/persisted
provenance is derived from the same spec, so the C4 defect (a
`TrainingConfig` constructed with one family's learning rate and logged for
every family) becomes structurally impossible rather than merely patched.

`TrainingState` is the resumability seam: a frozen value object a
checkpoint callback can emit so long deep-unfolding trainings become
pause/resume-capable without touching the Runner again (persisted through
the serializer registry today; the on-disk format switches to the T3.i
policy in Stage S4).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .config import TrainingConfig
from ..core.utils import ensure_positive_integer
from ..models.unfolded.layerwise import ParameterActivation

#: The optimizer families a spec may name. A dict (not an if-ladder) so new
#: optimizers are added by one entry, and the typed refusal below can name
#: every available choice.
_OPTIMIZER_FACTORIES: Mapping[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

#: Named scalar reductions of the per-batch cost tensor (the
#: `GradientDescentStrategy.loss_reduction` seam, made signable by name).
_LOSS_REDUCTIONS: Mapping[str, Any] = {
    "mean": torch.mean,
    "sum": torch.sum,
}


@dataclass(frozen=True)
class OptimizerSpec:
    """The optimizer's identity and hyperparameters, as inert data.

    Attributes:
        name: The optimizer family (a `_OPTIMIZER_FACTORIES` key).
        learning_rate: The learning rate the optimizer actually runs at --
            the one value C4 proved must never have two homes.
        hyperparameters: Extra constructor kwargs (e.g. ``betas``,
            ``weight_decay``), forwarded verbatim to the torch optimizer.
    """

    name: str
    learning_rate: float
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail-fast guards on the spec.

        Raises:
            ValueError: If `name` is not a registered optimizer family
                (the message names the available ones), or `learning_rate`
                is not positive.
        """
        if self.name not in _OPTIMIZER_FACTORIES:
            raise ValueError(
                f"Unknown optimizer {self.name!r}. "
                f"Available: {sorted(_OPTIMIZER_FACTORIES)}."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )

    def build(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Construct the optimizer this spec describes over `parameters`.

        This is the only place an optimizer is ever constructed from a spec,
        so the executed hyperparameters and the signed/logged ones cannot
        drift (the systemic C4 fix).

        Args:
            parameters: The trainable parameters (e.g.
                ``controller.as_module().parameters()``).

        Returns:
            The constructed ``torch.optim.Optimizer``.
        """
        factory = _OPTIMIZER_FACTORIES[self.name]
        return factory(  # type: ignore[call-arg]  # every registered factory accepts lr=
            parameters, lr=self.learning_rate, **dict(self.hyperparameters)
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full optimizer specification.

        Returns:
            ``{"type": "OptimizerSpec", "name":, "learning_rate":,
            "hyperparameters": {...}}``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "learning_rate": self.learning_rate,
            "hyperparameters": dict(self.hyperparameters),
        }


@dataclass(frozen=True)
class TrainingPlan:
    """One learned family's complete, declarative training specification.

    Each learned `ModelRecipe` owns its **own** plan; what gets logged and
    signed is the plan actually executed.

    Attributes:
        optimizer: The `OptimizerSpec` the engine builds its optimizer from.
        epochs: The number of training epochs.
        gradient_clip_norm: Optional global gradient-norm clip applied
            before each optimizer step; ``None`` disables clipping.
        loss_reduction: Named scalar reduction of the per-batch cost tensor
            (a `_LOSS_REDUCTIONS` key) -- signable, unlike a bare callable.
    """

    optimizer: OptimizerSpec
    epochs: int
    gradient_clip_norm: float | None = None
    loss_reduction: str = "mean"

    def __post_init__(self) -> None:
        """Fail-fast guards on the plan.

        Raises:
            ValueError: If `epochs` is not a positive integer,
                `gradient_clip_norm` is set but not positive, or
                `loss_reduction` is not a registered reduction name.
        """
        ensure_positive_integer(self.epochs, "epochs")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}."
            )
        if self.loss_reduction not in _LOSS_REDUCTIONS:
            raise ValueError(
                f"Unknown loss_reduction {self.loss_reduction!r}. "
                f"Available: {sorted(_LOSS_REDUCTIONS)}."
            )

    def resolve_loss_reduction(self) -> Any:
        """The named reduction as its torch callable.

        Returns:
            E.g. ``torch.mean`` for ``"mean"``.
        """
        return _LOSS_REDUCTIONS[self.loss_reduction]

    def training_config(
        self, *, batch_size: int, log_every: int = 1, seed: int | None = None
    ) -> TrainingConfig:
        """The run-level `TrainingConfig` for a run executing THIS plan.

        The C4 provenance law: every logged optimizer field
        (`learning_rate`, `optimizer_name`, `optimizer_kwargs`) is derived
        here, from the same `OptimizerSpec` that `build`s the live
        optimizer -- there is no second place a learning rate can come from.

        Args:
            batch_size: Trajectories per training batch.
            log_every: Metric-logging cadence in epochs.
            seed: Optional seed, logged for reproducibility.

        Returns:
            The derived `TrainingConfig`.
        """
        return TrainingConfig(
            batch_size=batch_size,
            learning_rate=self.optimizer.learning_rate,
            optimizer_name=self.optimizer.name,
            optimizer_kwargs=dict(self.optimizer.hyperparameters),
            log_every=log_every,
            seed=seed,
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full training specification.

        Returns:
            ``{"type": "TrainingPlan", "optimizer": {...}, "epochs":,
            "gradient_clip_norm":, "loss_reduction":}``.
        """
        return {
            "type": type(self).__name__,
            "optimizer": self.optimizer.get_signature(),
            "epochs": self.epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
            "loss_reduction": self.loss_reduction,
        }


@dataclass(frozen=True)
class TrainingState:
    """A resumable training snapshot (the T3.b checkpoint seam): everything
    needed to continue an interrupted run -- the epoch reached, the trained
    module weights, and the optimizer's own state (moments, step counts).

    Attributes:
        epoch: The last completed global epoch index.
        module_state_dict: ``nn.Module.state_dict()`` of the trained module
            (the in-memory checkpoint contract, T2.b).
        optimizer_state_dict: ``torch.optim.Optimizer.state_dict()``.
    """

    epoch: int
    module_state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]


@dataclass(frozen=True)
class PhaseSpec:
    """One compiled stage of a `LayerwiseTrainingPlan.compile` schedule:
    which parameters are active, for how many epochs, under what name (the
    `TrainingPhase.name` a `Runner` logs at INFO on phase transitions).

    Attributes:
        activation: Which step-size rows -- and whether the unified matrix
            -- train during this stage (see
            `models.unfolded.layerwise.ParameterActivation`).
        epochs: The number of epochs this stage's `TrainingPhase` runs for.
        name: The human-readable phase name.
    """

    activation: ParameterActivation
    epochs: int
    name: str

    def __post_init__(self) -> None:
        """Fail-fast guard: `epochs` must be a positive integer.

        Raises:
            ValueError: If `epochs` is not a positive ``int``.
        """
        ensure_positive_integer(self.epochs, "epochs")


@dataclass(frozen=True)
class LayerwiseTrainingPlan:
    """A greedy layer-wise ("warm-start") training schedule for a deep-
    unfolded controller: one phase per unfolding iteration -- training that
    iteration's step-size row(s) while the previously-trained rows stay
    frozen -- optionally followed by one end-to-end refinement phase that
    unfreezes everything (NB03 blueprint v2 §3, the flagship build).

    A sibling of `TrainingPlan`, not a subclass or an edit to it: `TrainingPlan`
    describes ONE phase's optimizer/epochs/clip/reduction; this describes a
    whole MULTI-phase schedule that `compile`s into the `PhaseSpec`s a
    recipe turns into `engine.runner.TrainingPhase`s -- the `Runner` itself
    needs no changes, since it already accepts ``phases: Sequence[TrainingPhase]``.

    Attributes:
        optimizer: The `OptimizerSpec` EVERY phase's own, freshly-built
            optimizer is constructed from (the C4 provenance law, and --
            critically -- the freeze contract's anti-momentum-carryover
            law: each phase gets its OWN optimizer instance built from this
            spec, so Adam state can never bleed from one phase into the
            next; see `__post_init__`'s ``weight_decay`` guard below).
        warmup_epochs_per_layer: Epoch count of each per-layer warm-up phase.
        refinement_epochs: Epoch count of the trailing end-to-end phase
            (every row + the matrix, if present, active); ``0`` disables it
            (pure greedy layer-wise, no refinement tail).
        train_matrix_from: When the unified Riccati-replacement matrix (if
            the controller has one) first becomes trainable: ``"refinement"``
            (only in the trailing end-to-end phase, the default -- the
            matrix trains only once every step-size row has its own
            warm-started value), ``"each"`` (every warm-up phase), or
            ``"last_layer"`` (the final warm-up phase onward).
        activation: Whether each warm-up phase ``j`` activates only row
            ``j`` (``"single"``, the classic greedy layer-wise default) or
            the cumulative prefix rows ``0..j`` (``"cumulative"``).
        gradient_clip_norm: Optional global gradient-norm clip applied
            before each phase's optimizer step (after grad masking, so a
            frozen row's zeroed gradient never inflates the clipped norm).
        loss_reduction: Named scalar reduction of the per-batch cost tensor
            (a `TrainingPlan`-compatible key: ``"mean"`` or ``"sum"``).
    """

    optimizer: OptimizerSpec
    warmup_epochs_per_layer: int
    refinement_epochs: int = 0
    train_matrix_from: Literal["refinement", "each", "last_layer"] = "refinement"
    activation: Literal["single", "cumulative"] = "single"
    gradient_clip_norm: float | None = None
    loss_reduction: str = "mean"

    def __post_init__(self) -> None:
        """Fail-fast guards on the schedule, including THE freeze-contract
        guard: a nonzero ``weight_decay`` is rejected outright, because
        gradient masking cannot stop the optimizer from decaying a frozen
        row of the shared step-size tensor (NB03 blueprint v2 §7.1 -- both
        Adam and AdamW were empirically shown to move a zero-gradient
        parameter when ``weight_decay > 0``). The unified matrix freezes via
        a whole-tensor ``requires_grad_(False)`` instead, which genuinely IS
        immune to weight decay (a `requires_grad=False` parameter's
        ``.grad`` stays ``None`` and every torch optimizer skips it
        entirely) -- but that path carries no per-row structure to leak.

        Raises:
            ValueError: If `warmup_epochs_per_layer` is not a positive
                integer, `refinement_epochs` is negative,
                `gradient_clip_norm` is set but not positive,
                `loss_reduction` is not a registered reduction name, or
                `optimizer.hyperparameters` sets a nonzero ``weight_decay``.
        """
        ensure_positive_integer(self.warmup_epochs_per_layer, "warmup_epochs_per_layer")
        if self.refinement_epochs < 0:
            raise ValueError(
                f"refinement_epochs must be non-negative, got {self.refinement_epochs}."
            )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}."
            )
        if self.loss_reduction not in _LOSS_REDUCTIONS:
            raise ValueError(
                f"Unknown loss_reduction {self.loss_reduction!r}. "
                f"Available: {sorted(_LOSS_REDUCTIONS)}."
            )
        if self.optimizer.hyperparameters.get("weight_decay", 0.0):
            raise ValueError(
                "LayerwiseTrainingPlan forbids weight_decay in its "
                "OptimizerSpec: gradient masking cannot stop the optimizer "
                "from decaying frozen rows of the shared step-size tensor "
                "(NB03 blueprint v2 SS7.1). Regularize active entries via a "
                "mask-aware mechanism instead of torch-level weight_decay."
            )

    def compile(self, num_layers: int, *, has_matrix: bool) -> tuple[PhaseSpec, ...]:
        """Compile this schedule into its ordered `PhaseSpec`s: one warm-up
        phase per layer, then an optional trailing refinement phase.

        Args:
            num_layers: The unfolded controller's depth (its number of
                unrolled gradient-descent iterations) -- one warm-up phase
                is produced per layer.
            has_matrix: Whether the controller carries a learnable unified
                matrix (kind ``LEARNED_STEP_SIZE_AND_MATRIX``); ``False``
                makes every phase's ``train_matrix`` vacuously ``False``.

        Returns:
            ``num_layers`` warm-up `PhaseSpec`s (named
            ``"warmup_layer_<j>"``), plus one ``"refinement"`` `PhaseSpec`
            when `refinement_epochs` is positive.

        Raises:
            ValueError: If `num_layers` is not a positive integer.
        """
        ensure_positive_integer(num_layers, "num_layers")
        phases = []
        for layer in range(num_layers):
            rows = (
                frozenset(range(layer + 1))
                if self.activation == "cumulative"
                else frozenset({layer})
            )
            train_matrix = has_matrix and (
                self.train_matrix_from == "each"
                or (self.train_matrix_from == "last_layer" and layer == num_layers - 1)
            )
            phases.append(
                PhaseSpec(
                    activation=ParameterActivation(
                        step_size_rows=rows, train_matrix=train_matrix
                    ),
                    epochs=self.warmup_epochs_per_layer,
                    name=f"warmup_layer_{layer}",
                )
            )
        if self.refinement_epochs > 0:
            phases.append(
                PhaseSpec(
                    activation=ParameterActivation(
                        step_size_rows="all", train_matrix=has_matrix
                    ),
                    epochs=self.refinement_epochs,
                    name="refinement",
                )
            )
        return tuple(phases)

    def resolve_loss_reduction(self) -> Any:
        """The named reduction as its torch callable.

        Returns:
            E.g. ``torch.mean`` for ``"mean"``.
        """
        return _LOSS_REDUCTIONS[self.loss_reduction]

    def training_config(
        self, *, batch_size: int, log_every: int = 1, seed: int | None = None
    ) -> TrainingConfig:
        """The run-level `TrainingConfig` for a run executing THIS schedule
        (the C4 provenance law, mirroring `TrainingPlan.training_config`):
        every logged optimizer field derives from `self.optimizer`, the
        SAME spec every phase's own optimizer is built from.

        Args:
            batch_size: Trajectories per training batch.
            log_every: Metric-logging cadence in epochs.
            seed: Optional seed, logged for reproducibility.

        Returns:
            The derived `TrainingConfig`.
        """
        return TrainingConfig(
            batch_size=batch_size,
            learning_rate=self.optimizer.learning_rate,
            optimizer_name=self.optimizer.name,
            optimizer_kwargs=dict(self.optimizer.hyperparameters),
            log_every=log_every,
            seed=seed,
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full schedule specification -- a
        warm-start run and an end-to-end run of the same optimizer/depth
        are DISTINCT cache identities (T3.e), since every schedule field
        participates.

        Returns:
            ``{"type": "LayerwiseTrainingPlan", "optimizer": {...},
            "warmup_epochs_per_layer":, "refinement_epochs":,
            "train_matrix_from":, "activation":, "gradient_clip_norm":,
            "loss_reduction":}``.
        """
        return {
            "type": type(self).__name__,
            "optimizer": self.optimizer.get_signature(),
            "warmup_epochs_per_layer": self.warmup_epochs_per_layer,
            "refinement_epochs": self.refinement_epochs,
            "train_matrix_from": self.train_matrix_from,
            "activation": self.activation,
            "gradient_clip_norm": self.gradient_clip_norm,
            "loss_reduction": self.loss_reduction,
        }
