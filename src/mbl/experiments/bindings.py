"""What the Tier-3 grammar may not import, supplied from Tier 4 (Stage 2 E3).

`spec/loader.py` reads a document and composes specifications. It may not
*construct* an `engine.TrainingPlan` or an `experiments.EvaluationProtocol`,
because Tier 3's boundary rule forbids it from importing `engine`,
`experiments` or `applications` -- which is the rule that keeps the grammar
reusable by the runner, the analysis tier and the command line alike.

So the grammar declares one structural seam, `spec.loader.SpecBindings`, and
this module is the concrete binding for this project. It is the same inversion
`experiments.ContenderSpec` already performs for the recipe registry, and it is
deliberately *one* seam rather than three: every table in a document carrying
the reserved key `_build` is handed here with the name it declared, so adding a
new buildable kind is a change to this file and to nothing in the grammar.

The vocabulary is small and each entry exists because a real study needs it:

| `_build` | Builds | Where a document uses it |
|---|---|---|
| `end_to_end` | `TrainingPlan` | `training.plan`, and `unfolded`/`cocp` contender configs |
| `layerwise` | `LayerwiseTrainingPlan` | `unfolded_warmstart` contender configs |
| `protocol` | `EvaluationProtocol` | `evaluation.protocol` |
| `gaussian` | `GaussianBatchSpec` | `evaluation.protocol.batch_spec` |

`state_dim` and `horizon` reach `gaussian` through the builder's `context`
rather than through the document. A document that could restate a dimension is
a document that could disagree with the problem's own matrices, and the whole
class of error is deleted by having no key for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..applications.factories import GaussianBatchSpec
from ..applications.recipes.base import build_default_recipe_registry
from ..engine.training_plan import (
    LayerwiseTrainingPlan,
    OptimizerSpec,
    TrainingPlan,
)
from ..models.registry import ControllerRegistry
from ..spec.errors import SpecificationError
from .experiment import EvaluationProtocol


def _require(declaration: Mapping[str, Any], key: str, kind: str) -> Any:
    if key not in declaration:
        raise SpecificationError(
            f"a {kind!r} declaration needs {key!r}; got {sorted(declaration)}"
        )
    return declaration[key]


def _optimizer(declaration: Mapping[str, Any], kind: str) -> OptimizerSpec:
    return OptimizerSpec(
        name=str(_require(declaration, "optimizer", kind)),
        learning_rate=float(_require(declaration, "learning_rate", kind)),
        hyperparameters=dict(declaration.get("hyperparameters", {})),
    )


def _end_to_end(declaration: Mapping[str, Any], _: Mapping[str, Any]) -> TrainingPlan:
    return TrainingPlan(
        optimizer=_optimizer(declaration, "end_to_end"),
        epochs=int(_require(declaration, "epochs", "end_to_end")),
        gradient_clip_norm=declaration.get("gradient_clip_norm"),
        loss_reduction=str(declaration.get("loss_reduction", "mean")),
    )


def _layerwise(
    declaration: Mapping[str, Any], _: Mapping[str, Any]
) -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=_optimizer(declaration, "layerwise"),
        warmup_epochs_per_layer=int(
            _require(declaration, "warmup_epochs_per_layer", "layerwise")
        ),
        refinement_epochs=int(declaration.get("refinement_epochs", 0)),
        train_matrix_from=str(declaration.get("train_matrix_from", "refinement")),  # type: ignore[arg-type]
        activation=str(declaration.get("activation", "single")),  # type: ignore[arg-type]
        gradient_clip_norm=declaration.get("gradient_clip_norm"),
        loss_reduction=str(declaration.get("loss_reduction", "mean")),
    )


#: Keys a batch declaration may never carry: they are the *problem's*, and a
#: document able to restate one is a document able to disagree with the
#: matrices it is scored against. Silently ignoring them is worse than
#: refusing, because the author has every reason to believe theirs applied.
DIMENSIONS_FROM_THE_PROBLEM = ("state_dim", "horizon")


def _gaussian(
    declaration: Mapping[str, Any], context: Mapping[str, Any]
) -> GaussianBatchSpec:
    restated = sorted(set(declaration) & set(DIMENSIONS_FROM_THE_PROBLEM))
    if restated:
        raise SpecificationError(
            f"a 'gaussian' declaration may not set {', '.join(restated)}: those "
            "are the problem's, and a batch drawn at a dimension the problem "
            "does not have would be scored against matrices it cannot multiply"
        )
    return GaussianBatchSpec(
        state_dim=int(context["state_dim"]),
        horizon=int(context["horizon"]),
        batch_size=int(_require(declaration, "batch_size", "gaussian")),
        seed=int(_require(declaration, "seed", "gaussian")),
        process_noise_std=float(_require(declaration, "process_noise_std", "gaussian")),
        initial_state_std=float(declaration.get("initial_state_std", 1.0)),
    )


def _protocol(
    declaration: Mapping[str, Any], context: Mapping[str, Any]
) -> EvaluationProtocol:
    batch_spec = _require(declaration, "batch_spec", "protocol")
    if not isinstance(batch_spec, (GaussianBatchSpec,)):
        raise SpecificationError(
            "protocol.batch_spec must itself declare what to build; add a "
            f"_build key to it, got {batch_spec!r}"
        )
    return EvaluationProtocol(
        batch_spec=batch_spec,
        n_batches=int(declaration.get("n_batches", 1)),
    )


#: The buildable vocabulary. Keyed by the `_build` value a document writes.
BUILDERS: dict[str, Any] = {
    "end_to_end": _end_to_end,
    "layerwise": _layerwise,
    "gaussian": _gaussian,
    "protocol": _protocol,
}


@dataclass(frozen=True)
class ExperimentBindings:
    """This project's concrete `spec.loader.SpecBindings`.

    Attributes:
        registry: The recipe registry contender families resolve through.
            Injected rather than looked up so a test can supply its own.
    """

    registry: ControllerRegistry = field(default_factory=build_default_recipe_registry)

    def build(
        self, kind: str, declaration: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Any:
        """Construct what `kind` names.

        Raises:
            SpecificationError: If `kind` is not in `BUILDERS`, or the
                declaration is incomplete. Both messages name the offender,
                because this is the surface an author is editing.
        """
        builder = BUILDERS.get(kind)
        if builder is None:
            raise SpecificationError(
                f"nothing is registered to build {kind!r}; available: "
                f"{sorted(BUILDERS)}"
            )
        try:
            return builder(declaration, context)
        except SpecificationError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            # The concrete types validate themselves and say so in their own
            # vocabulary; what they cannot know is which document key was being
            # built, which is the only thing an author can act on.
            raise SpecificationError(f"building {kind!r}: {error}") from error


#: The binding `load_study` is normally called with.
DEFAULT_SPEC_BINDINGS = ExperimentBindings()
