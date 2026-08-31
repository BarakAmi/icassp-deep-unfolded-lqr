"""Where identity is derived from the grammar (Stage 2 Phase D).

Two functions, and the important thing about the first one is a parameter it
does not have.

The defect this closes is one dictionary literal in
`experiments/experiment.py`, which keys a model on
``{problem, contender, evaluation, compute_context}``. Because the evaluation
protocol is in there, raising an evaluation batch count, changing an evaluation
seed or adding a metric invalidates every trained model and retrains it --
hours of accelerator time to answer a question about scoring.

`derive_model_id` takes a problem, a contender, a training declaration and a
seed. There is no argument through which an evaluation could be passed, so the
defect is not fixed by convention but made unrepresentable; the same property
is asserted of Tier 4's `model_id` in `tests/store/test_ids.py`, and carried up
to here because a correct `model_id` reached through a careless wrapper would be
no improvement at all.

The consequence is that distribution shift stops being a subsystem. A
measurement whose evaluation problem differs from the model's training problem
*is* a shifted result, produced by one rollout and zero training.
"""

from __future__ import annotations

from ..store.ids import MeasurementID, ModelID, measurement_id, model_id
from .contender import ContenderSpec
from .evaluation import EvaluationSpec
from .problem import ProblemSpec
from .training import TrainingSpec, require_supported_precision


def derive_model_id(
    problem: ProblemSpec,
    contender: ContenderSpec,
    training: TrainingSpec,
    seed: int,
) -> ModelID:
    """Identify one trained model.

    Note the absent parameter: there is no way to pass an evaluation protocol,
    an evaluation problem or a metric list, which is the entire point.

    This is also where `require_supported_precision` gets one of its two
    mandated call sites (the other is the study loader). It belongs here rather
    than on `TrainingSpec` because this is the choke point where a contender
    family and a training context are both in hand -- a `TrainingSpec` has no
    contender, and must not gain one, since a study declares one training plan
    for many contenders.

    Args:
        problem: The problem the model is trained on.
        contender: The controller family and its configuration.
        training: The fitting procedure, including its compute context.
        seed: The single training seed this model is fitted with. The
            declared seed *collection* stays in `training` and is stripped by
            Tier 4, so adding a sixth seed to a five-seed study trains one
            model rather than six.

    Returns:
        The `ModelID`.

    Raises:
        SpecificationError: If the contender family's numerics are
            incompatible with the training precision.
    """
    require_supported_precision(contender.family, training.ctx.precision_enum)
    return model_id(
        problem=problem.problem_id,
        contender=contender.get_signature(),
        training=training.get_signature(),
        ctx=training.ctx.get_signature(),
        seed=seed,
    )


def derive_measurement_id(model: ModelID, evaluation: EvaluationSpec) -> MeasurementID:
    """Identify one evaluation of one model.

    The evaluation problem travels on its own axis rather than inside the
    protocol tree, which is why `EvaluationSpec.get_signature()` omits it: a
    shifted measurement differs from a nominal one in that `ProblemID`, and the
    store can therefore answer "what has this model been evaluated on?" without
    unpacking a nested tree.

    Args:
        model: The already-derived identifier of the model being scored.
        evaluation: The scoring protocol, its context and its metrics.

    Returns:
        The `MeasurementID`.
    """
    return measurement_id(
        model=model,
        eval_problem=evaluation.problem.problem_id,
        eval_protocol=evaluation.get_signature(),
    )
