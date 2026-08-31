"""Acceptance tests for the three-level identity model (Annex 01 §5).

Written before `mbl.store.ids` exists, per the project's verification-hook rule.

The property that matters is that no evaluation term can reach a `ModelID` --
the defect that motivated the whole re-architecture was an evaluation knob
sitting inside the key that gates training. It is pinned by
`test_model_id_accepts_no_evaluation_parameter_at_all`, which inspects the
signature, **not** by the behavioural
`test_model_id_is_blind_to_every_evaluation_term`.

That distinction was established by mutation, not by reasoning: reintroducing
the original defect as a defaulted `eval_protocol` parameter shifted every id
uniformly and survived the entire behavioural suite. Both tests are kept -- the
structural one is the guard, the behavioural one documents the intent.
"""

import inspect

import pytest

from mbl.store.ids import (
    composite_study_id,
    CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID,
    MeasurementID,
    ModelID,
    ProblemID,
    StudyID,
    measurement_id,
    model_id,
    problem_id,
    study_id,
)

PROBLEM = {"system": {"kind": "lti", "A_hash": "sha256:aaa"}, "horizon": 100}
OTHER_PROBLEM = {"system": {"kind": "lti", "A_hash": "sha256:bbb"}, "horizon": 100}
CONTENDER = {"family": "unfolded", "config": {"depth": 10}}
TRAINING = {"data": {"kind": "gaussian", "batch_size": 8192}, "plan": {"epochs": 200}}
CTX = {"backend": "torch", "device": "cuda", "precision": "float32"}
EVAL_PROTOCOL = {"data": {"batch_size": 4096, "n_batches": 8, "seed": 12345}}


def _model(**overrides) -> ModelID:
    kwargs = {
        "problem": problem_id(PROBLEM),
        "contender": CONTENDER,
        "training": TRAINING,
        "ctx": CTX,
        "seed": 0,
    }
    kwargs.update(overrides)
    return model_id(**kwargs)


# --------------------------------------------------------------------------
# Shape and typing
# --------------------------------------------------------------------------


def test_ids_are_distinct_types_so_one_cannot_be_passed_for_another() -> None:
    assert not issubclass(ModelID, MeasurementID)
    assert not issubclass(MeasurementID, ModelID)
    assert not issubclass(ProblemID, ModelID)


def test_every_id_is_lowercase_hex_of_fixed_width() -> None:
    ids = [
        problem_id(PROBLEM),
        _model(),
        measurement_id(_model(), problem_id(PROBLEM), EVAL_PROTOCOL),
        study_id({"anything": 1}),
    ]
    for value in ids:
        assert len(value) == 16, value
        assert set(value) <= set("0123456789abcdef"), value


# --------------------------------------------------------------------------
# Determinism -- the store is content-addressed, so this is load-bearing
# --------------------------------------------------------------------------


def test_identical_inputs_give_identical_ids() -> None:
    assert _model() == _model()


def test_key_insertion_order_does_not_change_an_id() -> None:
    reordered = {"horizon": 100, "system": {"A_hash": "sha256:aaa", "kind": "lti"}}
    assert problem_id(PROBLEM) == problem_id(reordered)


def test_nested_reordering_does_not_change_an_id() -> None:
    a = model_id(
        problem=problem_id(PROBLEM),
        contender=CONTENDER,
        training=TRAINING,
        ctx=CTX,
        seed=3,
    )
    b = model_id(
        seed=3,
        ctx=dict(reversed(list(CTX.items()))),
        training=TRAINING,
        contender=CONTENDER,
        problem=problem_id(PROBLEM),
    )
    assert a == b


# --------------------------------------------------------------------------
# THE decisive property: ModelID carries no evaluation term
# --------------------------------------------------------------------------


def test_model_id_accepts_no_evaluation_parameter_at_all() -> None:
    """The guard is structural, and it has to be checked structurally.

    A behavioural test cannot see this defect: give `model_id` an extra
    evaluation parameter with a default and every id shifts *uniformly*, so
    comparing two calls that both omit it shows nothing. That mutation was
    applied deliberately and survived the rest of this file, which is why this
    test exists -- it pins the signature, so the defect cannot be reintroduced
    even with a default value.
    """
    parameters = inspect.signature(model_id).parameters
    assert set(parameters) == {"problem", "contender", "training", "ctx", "seed"}
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())


def test_model_id_is_blind_to_every_evaluation_term() -> None:
    """Changing anything about evaluation must not change a ModelID.

    This is the regression test for parent document §2.2: the old cache key was
    H(problem, contender, evaluation, ctx), so raising `n_batches` or changing an
    evaluation seed retrained every model. `model_id` has no parameter through
    which an evaluation term could enter, and this test pins that shape.
    """
    baseline = _model()
    for altered in (
        {"data": {"batch_size": 4096, "n_batches": 64, "seed": 12345}},
        {"data": {"batch_size": 1, "n_batches": 8, "seed": 999}},
        {"metrics": ["expected_cost", "constraint_activity"]},
    ):
        # The evaluation protocol changes the measurement, never the model.
        assert measurement_id(baseline, problem_id(PROBLEM), altered) != measurement_id(
            baseline, problem_id(PROBLEM), EVAL_PROTOCOL
        )
        assert _model() == baseline


def test_evaluating_on_a_different_problem_reuses_the_same_model() -> None:
    """Distribution shift is ordinary evaluation: eval_ProblemID may differ from
    the training ProblemID without touching the model (§4.2b)."""
    trained = _model()
    nominal = measurement_id(trained, problem_id(PROBLEM), EVAL_PROTOCOL)
    shifted = measurement_id(trained, problem_id(OTHER_PROBLEM), EVAL_PROTOCOL)

    assert nominal != shifted
    assert _model() == trained


# --------------------------------------------------------------------------
# Sensitivity: everything a model DOES depend on must move its id
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("problem", "0123456789abcdef"),
        ("contender", {"family": "unfolded", "config": {"depth": 20}}),
        (
            "training",
            {"data": {"kind": "gaussian", "batch_size": 4096}, "plan": {"epochs": 200}},
        ),
        ("ctx", {"backend": "torch", "device": "cpu", "precision": "float32"}),
        ("seed", 1),
    ],
)
def test_model_id_changes_when_a_model_level_input_changes(
    field: str, value: object
) -> None:
    assert _model(**{field: value}) != _model()


def test_precision_participates_so_float32_and_float64_are_different_models() -> None:
    f32 = _model(ctx={"backend": "torch", "device": "cuda", "precision": "float32"})
    f64 = _model(ctx={"backend": "torch", "device": "cuda", "precision": "float64"})
    assert f32 != f64


# --------------------------------------------------------------------------
# Seeds: the tuple is excluded, the individual seed is included (§5 rule 2)
# --------------------------------------------------------------------------


def test_a_seed_list_is_not_part_of_a_single_models_identity() -> None:
    """A five-seed study is five models. Adding a sixth seed must train one
    model, not six -- so the declared seed *collection* cannot appear in any
    individual ModelID."""
    five = {**TRAINING, "seeds": (0, 1, 2, 3, 4)}
    six = {**TRAINING, "seeds": (0, 1, 2, 3, 4, 5)}
    assert _model(training=five, seed=2) == _model(training=six, seed=2)


def test_different_seeds_are_different_models() -> None:
    assert _model(seed=0) != _model(seed=1)


# --------------------------------------------------------------------------
# Renaming must not orphan results (§5 rule 1)
# --------------------------------------------------------------------------


def test_a_studys_own_name_never_participates_in_its_id() -> None:
    a = study_id({"id": "box_lqr/depth_scaling", "problem": PROBLEM})
    b = study_id({"id": "box_lqr/renamed_later", "problem": PROBLEM})
    assert a == b


def test_a_studys_content_does_participate() -> None:
    a = study_id({"id": "s", "problem": PROBLEM})
    b = study_id({"id": "s", "problem": OTHER_PROBLEM})
    assert a != b


# --------------------------------------------------------------------------
# Negative controls: malformed input must be rejected, not silently hashed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("id_type", [ProblemID, ModelID, MeasurementID, StudyID])
@pytest.mark.parametrize(
    "bad",
    [
        "not-a-hex-id",  # non-hex characters
        "abc",  # too short
        "0" * 17,  # too long
        "0123456789ABCDEF",  # uppercase: would alias a distinct directory name
        "",  # empty
    ],
)
def test_every_id_type_rejects_malformed_input(id_type: type, bad: str) -> None:
    """Validated at construction, for every id type. An unvalidated id becomes a
    directory name, so `0123456789ABCDEF` and its lowercase form would be two
    store entries for one object on a case-insensitive filesystem."""
    with pytest.raises(ValueError, match="16 lowercase hex"):
        id_type(bad)


def test_a_well_formed_id_round_trips_through_its_type() -> None:
    for id_type in (ProblemID, ModelID, MeasurementID, StudyID):
        value = id_type("0123456789abcdef")
        assert value == "0123456789abcdef"
        assert isinstance(value, id_type)


def test_unserialisable_signature_content_raises_rather_than_stringifying() -> None:
    """`default=str` would quietly hash the *repr* of an object, so two
    different objects could collide on `<object at 0x...>`-style text or, worse,
    the same object could hash differently across processes."""
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        problem_id({"system": object()})


# --------------------------------------------------------------------------
# A contender's own name never participates either (§5 rule 1, Phase G-1)
# --------------------------------------------------------------------------


def test_renaming_a_contender_never_orphans_its_models() -> None:
    """The `StudySpec.id` rule, extended to contenders (plan §8.2, G-1).

    `label` is the per-study instance name a figure axis is drawn with. It
    reached `ModelID` through the recipe's signature tree, so renaming
    `unfolded_alpha` to `alpha` retrained every model that contender had ever
    produced -- which is Annex 01 §5 rule 1's defect, one level down from the
    study.
    """
    named = {**CONTENDER, "label": "unfolded_alpha"}
    renamed = {**CONTENDER, "label": "alpha"}
    assert _model(contender=named) == _model(contender=renamed)


def test_a_contender_with_no_label_is_the_same_model_as_one_with_any() -> None:
    """Stripping is by absence, not by normalisation to a default: a tree that
    never carried a label and one that carried any must agree, or a study
    composed in Python and the same one read from a document would differ."""
    assert _model(contender=CONTENDER) == _model(
        contender={**CONTENDER, "label": "anything"}
    )


def test_the_exclusion_is_by_the_name_tier_3_emits() -> None:
    """A by-name contract, in the same family as `seeds` and `id`. The grammar
    emits the recipe's tree verbatim and Tier 4 drops this key; renaming it on
    either side would silently put the label back into every identifier."""
    assert CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID == ("label",)


def test_everything_else_about_a_contender_still_counts() -> None:
    """Anti-vacuity. Dropping one key must not be dropping the tree: a
    contender that genuinely differs must still be a different model, or the
    test above would hold under an implementation that ignored contenders
    entirely."""
    assert _model(contender=CONTENDER) != _model(
        contender={**CONTENDER, "config": {"depth": 20}}
    )
    assert _model(contender=CONTENDER) != _model(
        contender={**CONTENDER, "family": "neural"}
    )


class TestCompositeStudyId:
    """A figure assembled from several studies has no study of its own, and the
    store has one shape for results. These pin the identity that lets it use
    that shape (the author's ruling, 2026-08-14)."""

    def test_the_order_of_the_sources_does_not_matter(self) -> None:
        """ "Composed of A and B" is the same artifact as "composed of B and A",
        and two directories for it would be a bug rather than a variant."""
        assert composite_study_id(["b", "a"], kind="panels") == composite_study_id(
            ["a", "b"], kind="panels"
        )

    def test_changing_any_source_moves_the_artifact(self) -> None:
        """The rule every other identity here obeys: re-running the same
        sources writes to the same place, and a different source does not."""
        base = composite_study_id(["a", "b", "c"], kind="panels")
        assert composite_study_id(["a", "b", "d"], kind="panels") != base
        assert composite_study_id(["a", "b"], kind="panels") != base

    def test_two_figures_over_the_same_sources_do_not_collide(self) -> None:
        assert composite_study_id(["a", "b"], kind="panels") != composite_study_id(
            ["a", "b"], kind="bars"
        )

    def test_composing_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one source"):
            composite_study_id([], kind="panels")
