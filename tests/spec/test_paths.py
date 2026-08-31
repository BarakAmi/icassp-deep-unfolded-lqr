"""Acceptance tests for writing into a specification by path — Phase F2a.

`replace_at` has existed since E2 and is exercised through the sweep algebra and
the tier catalogue. F2a adds `replace_at_all`, which is what lets a tier reach
the fitting plan that actually executes: it lives per contender, and until now
nothing could address "every contender that has one".

The checkpoint class is **degeneracy**. A fan-out has three outcomes that must
stay distinguishable — every element written, some elements skipped because the
path is not theirs, and *nothing* matched — and it is the third that carries the
defect §9.6.6 records: a path matching nobody is a knob nobody turns, which
provenance would otherwise report as one that was. `replace_at_all` reports the
count and leaves the policy to its caller, because the two callers disagree
about it for a reason (plan §F2a).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mbl.spec.errors import SpecificationError
from mbl.spec.paths import WILDCARD, replace_at, replace_at_all


@dataclass(frozen=True)
class _Plan:
    epochs: int = 10
    learning_rate: float = 0.01


@dataclass(frozen=True)
class _Contender:
    label: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Study:
    contenders: tuple[_Contender, ...]
    name: str = "probe"


def _study() -> _Study:
    """Two contenders that train and one that does not — the shape of every
    real study here, where three of NB04's six carry no plan at all."""
    return _Study(
        contenders=(
            _Contender("learned_a", {"plan": _Plan(), "depth": 3}),
            _Contender("learned_b", {"plan": _Plan(epochs=50), "depth": 5}),
            _Contender("analytic", {"depth": 5}),
        )
    )


# --------------------------------------------------------------------------
# The fan-out
# --------------------------------------------------------------------------


def test_a_wildcard_writes_every_element_that_has_the_path() -> None:
    study = _study()
    rebuilt, matched = replace_at_all(
        study, ["contenders", WILDCARD, "config", "plan", "epochs"], 2
    )

    assert matched == 2
    assert [c.config["plan"].epochs for c in rebuilt.contenders[:2]] == [2, 2]


def test_an_element_without_the_path_is_left_untouched() -> None:
    """Not an error: a `truncated_riccati` has no plan, and refusing would make
    a catalogue shared across studies unusable."""
    study = _study()
    rebuilt, _ = replace_at_all(
        study, ["contenders", WILDCARD, "config", "plan", "epochs"], 2
    )

    analytic = rebuilt.contenders[2]
    assert analytic == study.contenders[2]
    assert "plan" not in analytic.config, (
        "the write invented a field on a contender that never had one"
    )


def test_a_wildcard_matching_nothing_reports_zero_rather_than_raising() -> None:
    """The count is the deliverable. `replace_at_all` does not decide whether
    zero is an error, because the catalogue and a per-study overlay disagree
    about that for a reason (plan §F2a)."""
    study = _study()
    rebuilt, matched = replace_at_all(
        study, ["contenders", WILDCARD, "config", "plan", "epcohs"], 2
    )

    assert matched == 0
    assert rebuilt == study


def test_only_the_addressed_leaf_moves() -> None:
    study = _study()
    rebuilt, _ = replace_at_all(
        study, ["contenders", WILDCARD, "config", "plan", "epochs"], 7
    )

    assert [c.config["plan"].learning_rate for c in rebuilt.contenders[:2]] == [
        0.01,
        0.01,
    ]
    assert [c.config["depth"] for c in rebuilt.contenders] == [3, 5, 5]
    assert [c.label for c in rebuilt.contenders] == [
        "learned_a",
        "learned_b",
        "analytic",
    ]


def test_the_original_is_not_mutated() -> None:
    study = _study()
    before = [c.config.get("plan") for c in study.contenders]

    replace_at_all(study, ["contenders", WILDCARD, "config", "plan", "epochs"], 99)

    assert [c.config.get("plan") for c in study.contenders] == before


def test_the_sequence_keeps_its_type() -> None:
    """A study holds its contenders in a tuple, and a frozen dataclass rebuilt
    with a list where it declares a tuple compares unequal to itself."""
    rebuilt, _ = replace_at_all(
        _study(), ["contenders", WILDCARD, "config", "plan", "epochs"], 2
    )
    assert isinstance(rebuilt.contenders, tuple)


# --------------------------------------------------------------------------
# Degeneracy
# --------------------------------------------------------------------------


def test_a_wildcard_over_something_that_is_not_a_sequence_is_refused() -> None:
    """`study.name` is a string, and fanning out over its characters is not
    what any author meant. The message names the type reached; the *path* is
    named by the caller that knows it (`tiers`), which is the same layering
    `bindings.build` uses to report the document key."""
    with pytest.raises(SpecificationError, match="sequence"):
        replace_at_all(_study(), ["name", WILDCARD, "epochs"], 2)


def test_a_path_without_a_wildcard_behaves_exactly_like_replace_at() -> None:
    """One traversal, two entry points. Two implementations would drift, and
    the two callers would then disagree about what a path addresses."""
    study = _study()
    rebuilt, matched = replace_at_all(study, ["name"], "renamed")

    assert matched == 1
    assert rebuilt == replace_at(study, ["name"], "renamed")


def test_a_missing_field_outside_a_wildcard_still_raises() -> None:
    """Leniency is the wildcard's, and only for the elements it fans out over.
    A misspelled path with no `*` in it must fail exactly as it always has."""
    with pytest.raises(SpecificationError, match="nmae"):
        replace_at_all(_study(), ["nmae"], "renamed")


def test_the_wildcard_constant_is_the_one_the_sweep_algebra_uses() -> None:
    """`contenders.*` means the same thing in a sweep axis and in a tier
    override. Two spellings of one wildcard is a second grammar."""
    assert WILDCARD == "*"


# --------------------------------------------------------------------------
# Addressing ONE contender by its label (Annex 01 §2.3.1, 2026-08-12)
# --------------------------------------------------------------------------


def test_a_label_segment_writes_only_that_contender() -> None:
    """The ICASSP campaign measured exactly one learned family truncated at its
    declared budget and the author ruled per-family convergence — which a
    grammar that can say "all of them" and not "this one" cannot express. The
    others must come through untouched, or the ruling becomes "everyone gets
    the exception"."""
    study = _study()
    written, matched = replace_at_all(
        study, ["contenders", "learned_b", "config", "plan", "epochs"], 200
    )
    assert matched == 1
    epochs = {
        contender.label: getattr(contender.config.get("plan"), "epochs", None)
        for contender in written.contenders
    }
    assert epochs == {"learned_a": 10, "learned_b": 200, "analytic": None}


def test_a_label_that_names_nobody_refuses_as_a_missing_field() -> None:
    """It is not a fan-out over nobody — that is the wildcard's case and
    reports zero. A label nothing carries is a misspelling, and it must refuse
    with the message an ordinary bad segment gets rather than silently writing
    to no one, which is §9.6.6's knob-nobody-turns defect."""
    study = _study()
    with pytest.raises(SpecificationError, match="cannot set 'nosuch'"):
        replace_at_all(study, ["contenders", "nosuch", "config", "plan", "epochs"], 200)


def test_the_wildcard_still_fans_over_everyone() -> None:
    """The addressing is additive. A regression here would silently narrow
    every tier in the catalogue to whichever contender happened to be named."""
    study = _study()
    written, matched = replace_at_all(
        study, ["contenders", WILDCARD, "config", "plan", "epochs"], 7
    )
    assert matched == 2
    assert [
        getattr(contender.config.get("plan"), "epochs", None)
        for contender in written.contenders
    ] == [7, 7, None]


def test_a_label_addressed_write_leaves_the_original_alone() -> None:
    """Frozen dataclasses in, frozen dataclasses out: the sequence is rebuilt
    rather than mutated, and a study that shared a contender object with its
    rewrite would carry the edit into every earlier reference to it."""
    study = _study()
    replace_at_all(study, ["contenders", "learned_a", "config", "plan", "epochs"], 999)
    assert study.contenders[0].config["plan"].epochs == 10


def test_a_label_addressed_element_without_the_leaf_reports_zero() -> None:
    """`analytic` carries no plan. This module reports the count and leaves the
    policy to its caller — the same split the wildcard has, and for the same
    reason: a shared catalogue tolerates a path matching nothing where a
    per-study overlay treats it as a typo. Asserting the refusal here would put
    the decision in two places, and `_write_override`'s `strict` is where it
    already lives."""
    study = _study()
    written, matched = replace_at_all(
        study, ["contenders", "analytic", "config", "plan", "epochs"], 5
    )
    assert matched == 0
    assert written == study
