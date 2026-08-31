"""The three-level identity model: `ProblemID` -> `ModelID` -> `MeasurementID`.

Annex 01 §5. The whole re-architecture turns on one property, pinned by
`tests/store/test_ids.py`: **`model_id` has no parameter through which an
evaluation term could enter.** The old cache key was
``H(problem, contender, evaluation, ctx)``, so raising an evaluation batch count
or changing an evaluation seed retrained every model. Here the split is
structural rather than a convention someone must remember, because there is no
argument to pass an evaluation protocol to.

The consequence is that distribution shift stops being a subsystem: a
measurement whose `eval_problem` differs from the model's training problem *is*
a shifted result, computed from an already-trained model at the cost of one
rollout.

Identity is derived from **signature trees** -- plain JSON-shaped mappings --
never from live objects. The specification types that produce those trees arrive
with Tier 3; this module only needs their signatures, which is what keeps the
store independent of the grammar.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Self

#: Width of every identifier, in hex characters. Matches
#: `compute_signature_digest`, so ids from this module and legacy run signatures
#: are visually and structurally the same kind of thing.
ID_WIDTH = 16

_HEX = frozenset("0123456789abcdef")

#: Excluded from `ModelID` (§5 rule 2). A study declaring five seeds is five
#: models; adding a sixth must train one model, not six, so the declared
#: *collection* cannot appear in any individual model's identity -- only the one
#: seed that model was actually trained with, passed separately.
TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID = ("seeds",)

#: Excluded from `StudyID` (§5 rule 1). Renaming a study must not orphan its
#: results, so its own name is not part of what it is.
#:
#: `analyses` and `figures` join it for the same rule one step further out
#: (slice Phase B-1). A study's identity is what it *computes*; both of those
#: are derivations *from* what it computed. If declaring a figure moved the
#: identifier, `store/studies/<StudyID>/` would become a new directory, its
#: manifest would be orphaned, and `mbl run` would decide the whole study
#: needed re-running -- an author would learn that adding a plot costs a
#: retraining. Re-running an analysis is cheap and reads only the store, which
#: is exactly why it may be redeclared at will.
STUDY_KEYS_EXCLUDED_FROM_STUDY_ID = ("id", "analyses", "figures")

#: Excluded from `ModelID` (§5 rule 1 again, one level down -- Phase G-1). A
#: contender's `label` is the per-study instance name a figure axis is drawn
#: with; it reached identity through the recipe's own signature tree, so
#: renaming `unfolded_alpha` to `alpha` retrained every model that contender
#: had ever produced.
#:
#: Dropping it is provably lossless, which is why it is safe to do here rather
#: than by editing eight recipes. Measured over the whole registry: six
#: families default their label to a constant, and the two that derive one
#: (`unfolded_learned_step_size`, `unfolded_warmstart_learned_step_size_and_matrix`)
#: derive it from `kind`, which is signed on its own account. No family
#: distinguishes two configurations by label alone.
#:
#: Stripped **here** and not in `spec/`: the recipe tree keeps its `label`, so
#: `Experiment.contender_content_digest` -- the legacy cache key seven
#: notebooks still run on -- is untouched, and the blast radius is `ModelID`
#: alone.
CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID = ("label",)


class _Identifier(str):
    """A validated 16-hex-character identifier.

    Subclassing `str` keeps ids usable as dict keys, path segments and SQLite
    primary keys with no unwrapping, while the distinct subclasses stop a
    `MeasurementID` being passed where a `ModelID` is meant -- the two are
    indistinguishable as bare strings, and confusing them would silently
    associate a measurement with the wrong model.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        if len(value) != ID_WIDTH or not _HEX.issuperset(value):
            raise ValueError(
                f"{cls.__name__} must be {ID_WIDTH} lowercase hex characters, got {value!r}"
            )
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class ProblemID(_Identifier):
    """Identifies a fully specified optimal-control problem."""

    __slots__ = ()


class ModelID(_Identifier):
    """Identifies one trained or solved model. Carries **no** evaluation term."""

    __slots__ = ()


class MeasurementID(_Identifier):
    """Identifies one evaluation of one model, under one evaluation problem."""

    __slots__ = ()


class StudyID(_Identifier):
    """Identifies a study's content, independent of the name it is filed under."""

    __slots__ = ()


def _require_json_serialisable(tree: Mapping[str, Any]) -> None:
    """Reject anything JSON cannot represent, rather than hashing its `repr`.

    `compute_signature_digest` passes ``default=str``, which is right for the
    legacy run-signature path it serves but wrong for content addressing: it
    would quietly hash ``"<object at 0x7f...>"``, so the same logical input
    could produce different ids across processes, and two different objects
    could collide. An identifier derived from an address is not an identifier.

    Raises:
        TypeError: If any value in `tree` is not JSON-serialisable.
    """
    try:
        json.dumps(tree, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"signature contains a value that is not JSON-serialisable: {error}"
        ) from error


def _digest(tree: Mapping[str, Any]) -> str:
    """Canonical digest of a signature tree, strict about serialisability.

    `compute_signature_digest` is imported here rather than at module scope
    because it pulls in NumPy and PyTorch, and the identifier *types* above are
    plain `str` subclasses that need neither. Only code that actually derives an
    identifier is on the write path, where those libraries are loaded anyway;
    the `mbl` command line reads identifiers and would otherwise pay seconds of
    import for nothing.
    """
    from ..core.utils.signing import compute_signature_digest

    _require_json_serialisable(tree)
    return compute_signature_digest(tree)


def _without(tree: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in tree.items() if k not in keys}


def problem_id(problem_signature: Mapping[str, Any]) -> ProblemID:
    """Identify a problem from its signature tree.

    Args:
        problem_signature: The problem's `get_signature()` tree -- system, cost,
            constraints and horizon.

    Returns:
        The `ProblemID`.
    """
    return ProblemID(_digest(problem_signature))


def model_id(
    *,
    problem: str,
    contender: Mapping[str, Any],
    training: Mapping[str, Any],
    ctx: Mapping[str, Any],
    seed: int,
) -> ModelID:
    """Identify one trained model.

    Note the absent parameter: there is no way to pass an evaluation protocol or
    an evaluation problem, which is the entire point (§5 rule 3). Raising an
    evaluation batch count, changing an evaluation seed, or adding a metric
    therefore costs zero training.

    Args:
        problem: The `ProblemID` of the problem the model is trained on.
        contender: The contender's signature tree (family plus its config). Its
            `label` is dropped (`CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID`):
            renaming a contender must not orphan its models, exactly as
            renaming a study must not orphan its results.
        training: The training signature tree -- data distribution, plan, compute
            context. Any declared seed *collection* is dropped
            (`TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID`); the one seed this model
            used is passed as `seed`.
        ctx: The compute-context signature -- backend, device, precision. Present
            because precision genuinely changes the trained weights, so a
            float32 and a float64 model are different models.
        seed: The single training seed this model was fitted with.

    Returns:
        The `ModelID`.
    """
    return ModelID(
        _digest(
            {
                "problem": ProblemID(problem),
                "contender": _without(contender, CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID),
                "training": _without(training, TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID),
                "ctx": ctx,
                "seed": seed,
            }
        )
    )


def measurement_id(
    model: str,
    eval_problem: str,
    eval_protocol: Mapping[str, Any],
) -> MeasurementID:
    """Identify one evaluation of one model.

    `eval_problem` MAY differ from the training problem of `model`. When it
    does, the measurement *is* a distribution-shift result -- there is no
    separate mechanism, and that is what retires the out-of-distribution
    subsystem (§4.2b).

    Args:
        model: The `ModelID` being evaluated.
        eval_problem: The `ProblemID` evaluation runs on.
        eval_protocol: The evaluation protocol's signature tree -- data
            distribution, batch counts, metrics, statistics.

    Returns:
        The `MeasurementID`.
    """
    return MeasurementID(
        _digest(
            {
                "model": ModelID(model),
                "eval_problem": ProblemID(eval_problem),
                "eval_protocol": eval_protocol,
            }
        )
    )


def composite_study_id(studies: Sequence[str], *, kind: str) -> StudyID:
    """Identify an artifact COMPOSED from several studies' results.

    A figure assembled from more than one study has no study of its own, and
    the store has one shape for results — `store/studies/<id>/…`. Filing such
    an artifact anywhere else is an asymmetry nothing about it justifies (the
    author's ruling, 2026-08-14), so it gets an id derived from exactly what it
    composes: re-running the same sources writes to the same place, and
    changing any one of them moves it, which is the rule every other identity
    here obeys.

    Args:
        studies: The source `StudyID`s. Order does not matter — they are
            sorted, because "composed of A and B" is the same artifact as
            "composed of B and A" and two directories for it would be a bug.
        kind: What is composed, so two different figures over the same sources
            do not collide.

    Returns:
        The composite `StudyID`.

    Raises:
        ValueError: If no sources are given. An artifact composed of nothing
            is not composed.
    """
    if not studies:
        raise ValueError(
            f"a composite {kind!r} needs at least one source study; an "
            "artifact composed of nothing has no identity to derive"
        )
    return StudyID(_digest({"composes": sorted(studies), "kind": kind}))


def study_id(study_signature: Mapping[str, Any]) -> StudyID:
    """Identify a study by its content, never by its name.

    Args:
        study_signature: The study's signature tree. Its own `id` field is
            dropped (`STUDY_KEYS_EXCLUDED_FROM_STUDY_ID`) so that renaming a
            study does not orphan every result it has already produced.

    Returns:
        The `StudyID`.
    """
    return StudyID(
        _digest(_without(study_signature, STUDY_KEYS_EXCLUDED_FROM_STUDY_ID))
    )
