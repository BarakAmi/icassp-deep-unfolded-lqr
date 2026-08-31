"""Acceptance tests for `mbl run` — Stages 3–5 vertical slice, Phase A.

Written before the implementation. The producer has existed since Stage 2 Phase
F2 and has been reachable only from Python; this is the entry point that makes a
study runnable by a person. It is a *thin* command over `run_study`, deliberately
— Annex 06's DAG, queue and resource classes are deferred with reasons recorded
in the slice's plan, and a command that grew a scheduler would be the thing that
plan exists to prevent.

Two properties carry the phase, and the second is the one worth the phase:

* **`--dry-run` writes nothing and trains nothing.** Asserted against the store's
  bytes, not against the message: a preview that leaves a model behind is not a
  preview, and the failure would be invisible to any check of what was printed.
* **It reports what a real run would cost, and it is answerable from the
  specification alone.** Measured on the tracked study: loading, resolving and
  materialising all nine points with every identifier derived is 0.00 s, while
  one `build_controller` for a COCP contender is 1.37 s, because that is where
  G-4's canary runs. So a dry run that reaches `build_controller` is not slightly
  slower — it is a different kind of operation, and the suite pins the boundary
  rather than the timing.

The third editing level of Annex 01 §2.3.1 (`--set`) has been implemented in
`StudyDocument.resolve` since Phase E3 and reachable from no command line, which
is why it is tested here rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mbl.applications.recipes import cocp as cocp_recipes
from mbl.cli.app import DEFAULT_TIER, OK, REFUSED, main
from mbl.cli.run import plan_run
from mbl.store.content_store import ModelStore
from mbl.store.index import StoreIndex

STUDIES = Path(__file__).resolve().parents[2] / "studies"
NB04 = STUDIES / "box_lqr" / "depth_scaling.toml"
TIERS = STUDIES / "_tiers.toml"

#: NB04 at `smoke`: six contenders over a two-point depth axis, of which three
#: are depth-invariant, deduplicated by identity.
SMOKE_POINTS = 9


def _run(store: Path, *argv: str) -> int:
    return main(
        ["--store", str(store), "run", str(NB04), "--catalogue", str(TIERS), *argv]
    )


def _stored(store: Path) -> set[str]:
    root = store / "models"
    return {path.name for path in root.iterdir()} if root.is_dir() else set()


# -- the preview -------------------------------------------------------------


def test_a_dry_run_reports_the_work_and_does_none_of_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """§7.3's own gate wording: `--dry-run` correctly reports how many models
    must be trained versus reused.

    Asserted against the store's **bytes**, because that is the half a printed
    message cannot show. A preview that publishes a model is not a preview, and
    every other assertion here would still pass.
    """
    store = tmp_path / "store"
    assert _run(store, "--tier", "smoke", "--dry-run") == OK

    out = capsys.readouterr().out
    assert "would train" in out
    assert not _stored(store), "a dry run published a model"

    # The counts are asserted as VALUES, not as substrings of the line: the
    # rendered message contains both numbers in both orders, so `"9" in out`
    # holds whether the 9 is the count to train or the count reused.
    _, preview = plan_run(NB04, store=store, tier="smoke", catalogue=str(TIERS))
    assert (preview.points, preview.to_train, preview.reused) == (
        SMOKE_POINTS,
        SMOKE_POINTS,
        0,
    )


def test_a_dry_run_on_a_warm_store_reports_nothing_to_train(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The number that makes the preview worth having, and it is the whole
    economic claim of the re-architecture in one line of output."""
    store = tmp_path / "store"
    assert _run(store, "--tier", "smoke") == OK
    before = _stored(store)
    assert len(before) == SMOKE_POINTS

    capsys.readouterr()
    assert _run(store, "--tier", "smoke", "--dry-run") == OK
    assert _stored(store) == before, "a dry run changed a warm store"

    _, preview = plan_run(NB04, store=store, tier="smoke", catalogue=str(TIERS))
    assert (preview.to_train, preview.reused) == (0, SMOKE_POINTS), (
        "nothing left to train and everything reused -- and asserted as a pair, "
        "because the rendered line carries both numbers and a substring check "
        "cannot tell which of them it found"
    )


def test_a_dry_run_never_builds_a_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The phase's named trap, measured rather than timed.**

    `build_controller` is where G-4's canary runs — 1.37 s for one COCP
    contender against 0.00 s for the whole load-resolve-materialise path — so a
    dry run that reaches it is a different kind of operation from the one the
    flag promises. Pinned by making the call *fail*, which is stronger than
    asserting a duration and does not depend on the machine.

    Note which call is guarded: **not** recipe construction, which is free and
    which a dry run does perform to derive identifiers. The first draft of the
    plan named construction, and measuring corrected it.
    """

    def _refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a dry run built a controller")

    monkeypatch.setattr(cocp_recipes.COCPRecipe, "build_controller", _refuse)
    monkeypatch.setattr(cocp_recipes.COCPLowerBoundRecipe, "build_controller", _refuse)

    assert _run(tmp_path / "store", "--tier", "smoke", "--dry-run") == OK


# -- the real run ------------------------------------------------------------


def test_a_run_fills_the_store_and_a_second_one_costs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end closure through the command line, which is the only surface a
    person actually uses. The second run is the decisive half: the store is what
    makes it free, and a command that re-trained would make the store
    decorative."""
    store = tmp_path / "store"
    assert _run(store, "--tier", "smoke") == OK
    first = capsys.readouterr().out
    assert str(SMOKE_POINTS) in first

    published = _stored(store)
    assert len(published) == SMOKE_POINTS

    assert _run(store, "--tier", "smoke") == OK
    assert _stored(store) == published, "the second run republished something"
    assert len(list(StoreIndex(store).models())) == SMOKE_POINTS


def test_a_run_writes_names_a_person_can_read(tmp_path: Path) -> None:
    """F3's projection reaches the store through this command too — the point
    of the whole naming exercise is that `mbl models tree` is legible after a
    `mbl run`, not after a Python script."""
    store = tmp_path / "store"
    assert _run(store, "--tier", "smoke") == OK
    names = {
        ModelStore(store).get(model_id).spec["semantic_name"]
        for model_id in ModelStore(store).list_ids()
    }
    assert names and all(name.startswith("box_lqr-n4m2-N50-u0.5") for name in names)


# -- the three editing levels ------------------------------------------------


def test_the_invocation_level_reaches_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--set` is Annex 01 §2.3.1's third editing level. It has been implemented
    in `StudyDocument.resolve` since Phase E3 and reachable from no command
    line, so this is the first check that it is wired at all."""
    store = tmp_path / "store"
    assert (
        _run(store, "--tier", "smoke", "--set", "training.seeds=2", "--dry-run") == OK
    )
    out = capsys.readouterr().out
    assert str(SMOKE_POINTS * 2) in out, (
        "two replicates is twice the points; if --set reached nothing the count "
        "would be unchanged and this test would be the only thing that noticed"
    )


def test_an_override_outside_the_whitelist_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D15's forbidden column reaches the command line unchanged: a tier scales
    effort and never changes content, and `--set` is a tier write."""
    assert _run(
        tmp_path / "store", "--tier", "smoke", "--set", "problem.horizon=5"
    ) == (REFUSED)
    # stdout, not stderr: this command line treats its output as its return
    # value throughout (the adapter-layer carve-out), and a refusal that went
    # to a different stream from every other refusal would be the odd one.
    assert "problem.horizon" in capsys.readouterr().out


def test_an_unknown_tier_is_refused_naming_the_available_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mistyped `--tier` is the most likely way to reach this command wrongly,
    and the catalogue's own message already lists what exists."""
    assert _run(tmp_path / "store", "--tier", "smoak") == REFUSED
    assert "smoke" in capsys.readouterr().out


def test_a_missing_study_is_refused_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refused rather than reported as an empty run, which is what a store
    created at a mistyped path looks like."""
    assert (
        main(["--store", str(tmp_path / "store"), "run", str(tmp_path / "nope.toml")])
        == REFUSED
    )
    assert "nope.toml" in capsys.readouterr().out


def test_the_tier_defaults_to_standard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Annex 01 §2.3 marks `standard` the default, and a command that required
    `--tier` would make the default unreachable.

    The *output* is asserted, not only the exit code: every tier exits 0, so a
    command defaulting to `smoke` would satisfy an exit-code check while
    silently running a wiring check where a study was asked for.
    """
    assert (
        main(
            [
                "--store",
                str(tmp_path / "store"),
                "run",
                str(NB04),
                "--catalogue",
                str(TIERS),
                "--dry-run",
            ]
        )
        == OK
    )
    out = capsys.readouterr().out
    assert f"at {DEFAULT_TIER}:" in out
    assert DEFAULT_TIER == "standard"
