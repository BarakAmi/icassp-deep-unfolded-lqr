"""Acceptance tests for semantic naming (Annex 02 §3).

The headline test is `test_reproduces_the_worked_example_from_the_annex`: the
annex publishes one fully worked name, so that name is the specification. If the
formatter and the document ever disagree, one of them is wrong and this fails.

Naming takes explicit descriptors rather than a specification object. The name's
fields (`u_max`, `depth`, `lr`) are grammar-level concepts and the grammar is
Tier 3, which does not exist yet; reaching into a spec shape from here would
couple Tier 4 to something unbuilt. The grammar supplies the descriptors later.
"""

from __future__ import annotations

import pytest

from mbl.store.naming import (
    ContenderName,
    ProblemName,
    TrainingName,
    semantic_name,
    split_semantic_name,
)

# The annex's worked example, verbatim.
ANNEX_PROBLEM = ProblemName(
    family="boxlqr", state_dim=7, control_dim=3, horizon=100, u_max=0.1, seed=0
)
ANNEX_CONTENDER = ContenderName(family="unfolded", kind="aP", depth=10)
ANNEX_TRAINING = TrainingName(
    optimizer="adam", learning_rate=1e-2, epochs=200, batch_size=8192, seed=0
)


# --------------------------------------------------------------------------
# The document is the specification
# --------------------------------------------------------------------------


def test_reproduces_the_worked_example_from_the_annex() -> None:
    name = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="3f9a1c",
    )
    assert (
        name
        == "boxlqr-n7m3-N100-u0.1-s0/unfolded-aP-J10/adam-lr1e-2-ep200-b8192-seed0#3f9a1c"
    )


def test_the_three_segments_are_ordered_by_discriminating_power() -> None:
    """Problem, then contender, then training -- the fields most likely to
    differ between two otherwise similar models come first, so a truncated
    display still distinguishes them."""
    problem, contender, training, digest = split_semantic_name(
        semantic_name(
            problem=ANNEX_PROBLEM,
            contender=ANNEX_CONTENDER,
            training=ANNEX_TRAINING,
            digest="3f9a1c",
        )
    )
    assert problem.startswith("boxlqr")
    assert contender.startswith("unfolded")
    assert training.startswith("adam")
    assert digest == "3f9a1c"


# --------------------------------------------------------------------------
# Determinism and predictability
# --------------------------------------------------------------------------


def test_the_same_descriptors_always_give_the_same_name() -> None:
    args = {
        "problem": ANNEX_PROBLEM,
        "contender": ANNEX_CONTENDER,
        "training": ANNEX_TRAINING,
        "digest": "3f9a1c",
    }
    assert semantic_name(**args) == semantic_name(**args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_dim", 8),
        ("control_dim", 4),
        ("horizon", 200),
        ("u_max", 0.2),
        ("seed", 1),
    ],
)
def test_a_differing_problem_field_changes_the_name(field: str, value: object) -> None:
    altered = ProblemName(**{**ANNEX_PROBLEM.__dict__, field: value})
    assert semantic_name(
        problem=altered, contender=ANNEX_CONTENDER, training=ANNEX_TRAINING, digest="d"
    ) != semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="d",
    )


def test_the_digest_is_what_guarantees_uniqueness() -> None:
    """The name is a label, never an identity: two genuinely different models can
    share a name if they differ only in a field naming does not render, and the
    digest is what separates them."""
    a = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="aaaaaa",
    )
    b = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="bbbbbb",
    )
    assert a != b
    assert a.rsplit("#", 1)[0] == b.rsplit("#", 1)[0]


# --------------------------------------------------------------------------
# Graceful degradation -- not every family has every field
# --------------------------------------------------------------------------


def test_an_analytic_contender_with_no_training_still_names_cleanly() -> None:
    """Truncated Riccati learns nothing: no optimizer, no epochs, no depth. The
    name must stay readable rather than emitting empty segments."""
    name = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ContenderName(family="truncated_riccati"),
        training=TrainingName(),
        digest="0a0a0a",
    )
    assert name == "boxlqr-n7m3-N100-u0.1-s0/truncated_riccati/-#0a0a0a"
    assert "--" not in name.split("#")[0].replace("/-", "")


def test_an_unconstrained_problem_omits_the_bound() -> None:
    name = semantic_name(
        problem=ProblemName(
            family="lqr", state_dim=7, control_dim=3, horizon=100, seed=0
        ),
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="d",
    )
    assert name.startswith("lqr-n7m3-N100-s0/")


# --------------------------------------------------------------------------
# Number formatting: compact, and identical for identical values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("learning_rate", "rendered"),
    [
        (1e-2, "lr1e-2"),
        (1e-3, "lr1e-3"),
        (0.1, "lr0.1"),
        (0.5, "lr0.5"),
        (3e-4, "lr0.0003"),
    ],
)
def test_learning_rates_render_compactly(learning_rate: float, rendered: str) -> None:
    name = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=TrainingName(optimizer="adam", learning_rate=learning_rate),
        digest="d",
    )
    assert rendered in name


def test_integral_floats_do_not_acquire_a_decimal_point() -> None:
    name = semantic_name(
        problem=ProblemName(
            family="lqr", state_dim=7, control_dim=3, horizon=100, u_max=1.0
        ),
        contender=ANNEX_CONTENDER,
        training=TrainingName(),
        digest="d",
    )
    assert "u1-" in name or name.startswith("lqr-n7m3-N100-u1/")


# --------------------------------------------------------------------------
# Safe as a path segment and as a CLI token
# --------------------------------------------------------------------------


def test_a_name_contains_no_character_unsafe_for_a_path_or_a_shell() -> None:
    name = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="3f9a1c",
    )
    assert not (set(name) & set(" \t\"'\\*?<>|:;$`()[]{}&"))


def test_a_family_name_with_awkward_characters_is_sanitised() -> None:
    name = semantic_name(
        problem=ProblemName(family="Box Constrained LQR!", state_dim=2, control_dim=1),
        contender=ContenderName(family="unfolded α+P"),
        training=TrainingName(),
        digest="d",
    )
    assert not (set(name) & set(" !+α"))
    assert name.startswith("box-constrained-lqr-n2m1/")


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_split_is_the_inverse_of_join() -> None:
    name = semantic_name(
        problem=ANNEX_PROBLEM,
        contender=ANNEX_CONTENDER,
        training=ANNEX_TRAINING,
        digest="3f9a1c",
    )
    problem, contender, training, digest = split_semantic_name(name)
    assert f"{problem}/{contender}/{training}#{digest}" == name


def test_splitting_something_that_is_not_a_semantic_name_raises() -> None:
    with pytest.raises(ValueError, match="not a semantic name"):
        split_semantic_name("boxlqr-n7m3")


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (1.0, "1"),
        (2.0, "2"),
        (0.5, "0.5"),
        (123456789.0, "123456789"),  # %g would give the LOSSY "1.23457e+08"
        (1e17, "1e17"),  # %g would give the UNSAFE "1e+17"
        (1.5e20, "1.5e20"),
    ],
)
def test_numbers_render_exactly_and_safely(value: float, rendered: str) -> None:
    """A name is lossy in *which fields it includes*, never in a value it does
    include. `%g` at default precision breaks that -- 123456789.0 becomes
    1.23457e+08, so two distinct bounds would share a name segment -- and it
    emits `+`, which is unsafe in a shell. Established by mutation: removing the
    integer shortcut survived until this test existed, because the only float
    tested was 1.0, where %g happens to agree.
    """
    name = semantic_name(
        problem=ProblemName(family="lqr", u_max=value),
        contender=ContenderName(family="c"),
        training=TrainingName(),
        digest="d",
    )
    assert f"u{rendered}/" in name
    assert "+" not in name
