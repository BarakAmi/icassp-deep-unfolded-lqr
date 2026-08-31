"""End-to-end acceptance tests for the `mbl` command line (Annex 02 §5).

Every test drives `main(argv)` against a real store on disk, so what is asserted
is the behaviour a user gets, exit code included. Three properties are the ones
worth breaking a build over:

* **A destructive command that is declined must remove nothing.** Tested by
  answering the prompt with `n` and then checking the store is intact, not by
  checking that the message was printed.
* **A reference must resolve to exactly one model or to none.** An ambiguous
  prefix is refused; a `_` typed by a user is a literal, not a SQL wildcard.
* **The store root comes from `--store`, then `$MBL_STORE`, then the default.**
  All three are covered, because a precedence rule verified at one level is a
  rule verified nowhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mbl.cli.app import (
    DEFAULT_STORE,
    OK,
    REFUSED,
    STORE_ENV,
    USAGE,
    build_parser,
    main,
    parse_duration,
)
from mbl.store.content_store import (
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
)
from mbl.store.ids import MeasurementID, ModelID
from mbl.store.index import MeasurementRow, ModelRow, StoreIndex

PROBLEM_A = "0123456789abcdef"
PROBLEM_B = "fedcba9876543210"

#: Realistic, mutually distinguishable identifiers, so that a short prefix is
#: unique -- `3f9a1c` is the annex's own worked example of one.
MODEL_A = "3f9a1c8e2b4d5f60"
MODEL_B = "a1b2c3d4e5f60718"


def _build_store(root: Path) -> StoreIndex:
    """Two contenders on one problem, with nominal and shifted measurements."""
    models, measurements = ModelStore(root), MeasurementStore(root)
    index = StoreIndex(root)
    for n, (contender, training) in enumerate(
        [("unfolded-aP-J10", "adam-lr1e-2-ep200"), ("riccati-trunc", "-")]
    ):
        model = ModelID((MODEL_A, MODEL_B)[n])
        seed_part = f"-seed{n}" if training != "-" else ""
        models.put(
            ModelRecord(
                model_id=model,
                spec={"problem_id": PROBLEM_A, "epochs": 200 + n, "family": "unfolded"},
                weights={"alpha": torch.tensor([0.5], dtype=torch.float64)},
                history=None,
                log=None,
                provenance={"created_utc": f"2026-0{5 + n}-14T08:00:00"},
            )
        )
        index.upsert_model(
            ModelRow(
                model_id=model,
                semantic_name=f"boxlqr-n7m3/{contender}/{training}{seed_part}#{model[:6]}",
                problem_id=PROBLEM_A,
                family="unfolded" if n == 0 else "riccati",
                contender_id=contender,
                seed=n,
                state_dim=7,
                control_dim=3,
                horizon=100,
                created_utc=f"2026-0{5 + n}-14T08:00:00",
                wall_time_s=12.5,
                spec_json=json.dumps({"epochs": 200 + n, "problem_id": PROBLEM_A}),
            )
        )
        for k, problem in enumerate((PROBLEM_A, PROBLEM_B)):
            identifier = MeasurementID(f"{100 * (n + 1) + k:016x}")
            measurements.put(
                MeasurementRecord(
                    measurement_id=identifier,
                    spec={"model": str(model), "eval_problem": problem},
                    metrics={"expected_cost": 3.25 + n},
                    samples=None,
                    trace=None,
                    log=None,
                )
            )
            index.upsert_measurement(
                MeasurementRow(
                    measurement_id=identifier,
                    model_id=model,
                    eval_problem_id=problem,
                    created_utc="2026-06-02T08:00:00",
                    metrics={"expected_cost": 3.25 + n},
                )
            )
    return index


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    _build_store(root)
    return root


def _run(store: Path, *argv: str) -> int:
    return main(["--store", str(store), *argv])


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_tree_renders_the_store(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "tree") == OK
    out = capsys.readouterr().out
    assert "boxlqr-n7m3" in out
    assert "unfolded-aP" in out and "riccati-trunc" in out
    assert "(2 contenders, 2 models," in out
    assert "measurements: 1 nominal, 1 shifted" in out


def test_tree_reports_a_real_size(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header's size comes from the filesystem, so a store holding real
    checkpoints must not report zero."""
    _run(store, "models", "tree")
    header = capsys.readouterr().out.splitlines()[0]
    assert "0 B)" not in header
    assert header.rstrip().endswith("B)")


def test_a_filtered_tree_excludes_the_other_contenders_measurements(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Filtering narrows the models; the measurements shown must narrow with
    them, or the tree credits one contender with another's evaluations."""
    assert _run(store, "models", "tree", "--family", "riccati") == OK
    out = capsys.readouterr().out
    assert "unfolded-aP" not in out
    assert out.count("measurements:") == 1


def test_list_shows_every_model(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "list") == OK
    out = capsys.readouterr().out
    assert out.count("boxlqr-n7m3/") == 2


def test_listed_identifiers_are_distinct_and_resolvable(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The regression this exists for, found by running the command rather than
    by a unit test: a fixed eight-character truncation printed two models as the
    same value, and that value then resolved to neither of them. What the column
    shows must be what the next command accepts."""
    index = StoreIndex(store)
    for suffix in ("aaaaaaaa", "bbbbbbbb"):
        twin = ModelID("3f9a1c00" + suffix)
        index.upsert_model(
            ModelRow(
                twin, f"boxlqr-n7m3/twin/-#{twin[:6]}", PROBLEM_A, "twin", "twin", 0
            )
        )

    assert _run(store, "models", "list") == OK
    column = [line.split()[0] for line in capsys.readouterr().out.splitlines()[2:]]
    assert len(column) == 4
    assert len(set(column)) == 4
    for identifier in column:
        assert _run(store, "models", "show", identifier) == OK, identifier


def test_list_filters_by_family(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "list", "--family", "riccati") == OK
    out = capsys.readouterr().out
    assert "riccati-trunc" in out and "unfolded-aP" not in out


def test_list_since_excludes_earlier_models(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "list", "--since", "2026-06-01") == OK
    out = capsys.readouterr().out
    assert "riccati-trunc" in out and "unfolded-aP" not in out


def test_list_sorts_by_the_requested_key(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(store, "models", "list", "--sort", "created")
    ascending = capsys.readouterr().out
    _run(store, "models", "list", "--sort", "seed")
    by_seed = capsys.readouterr().out
    assert ascending.index("unfolded-aP") < ascending.index("riccati-trunc")
    assert by_seed.index("unfolded-aP") < by_seed.index("riccati-trunc")


# --------------------------------------------------------------------------
# Reference resolution
# --------------------------------------------------------------------------


def test_show_resolves_a_full_identifier(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "show", MODEL_A) == OK
    assert "unfolded-aP" in capsys.readouterr().out


def test_show_resolves_an_identifier_prefix(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Annex 02 §3's worked example, `mbl models show 3f9a1c`."""
    assert _run(store, "models", "show", "3f9a1c") == OK
    assert "unfolded-aP" in capsys.readouterr().out


def test_show_resolves_a_semantic_name_prefix(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Annex 02 §3: a command accepts a unique prefix of the name as readily as
    one of the hash."""
    assert _run(store, "models", "show", "boxlqr-n7m3/riccati") == OK
    out = capsys.readouterr().out
    assert "riccati-trunc" in out and "unfolded" not in out.splitlines()[0]


def test_show_resolves_an_alias(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = StoreIndex(store)
    with index.connect() as conn:
        conn.execute("INSERT INTO aliases VALUES (?,?)", ("flagship", MODEL_A))
    assert _run(store, "models", "show", "flagship") == OK
    assert "unfolded-aP" in capsys.readouterr().out


def test_an_identifier_prefix_is_not_shadowed_by_a_semantic_name(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reference can match one model by identifier and a different one by
    name. The identifier wins, because it is canonical and the name is a label;
    resolving to the name's model would hand the command a model the user did
    not ask for while looking like a successful lookup."""
    decoy = ModelID("cccccccccccccccc")
    StoreIndex(store).upsert_model(
        ModelRow(
            model_id=decoy,
            semantic_name=f"{MODEL_A[:6]}-decoy/c/-#cccccc",
            problem_id=PROBLEM_A,
            family="decoy",
            contender_id="decoy",
            seed=0,
        )
    )
    assert _run(store, "models", "show", MODEL_A[:6]) == OK
    out = capsys.readouterr().out
    assert MODEL_A in out and "decoy" not in out


def test_an_ambiguous_prefix_is_refused_and_names_the_candidates(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resolving to the first match would attach the command to the wrong
    model, and the user would have no way to notice."""
    assert _run(store, "models", "show", "boxlqr") == REFUSED
    out = capsys.readouterr().out
    assert "matches 2 models" in out
    assert MODEL_A in out and MODEL_B in out


def test_an_unknown_reference_is_refused(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "show", "nosuchthing") == REFUSED
    assert "no model matches" in capsys.readouterr().out


def test_an_underscore_is_a_literal_not_a_sql_wildcard(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`LIKE` treats `_` as 'any single character'. Unescaped, the typo
    `boxlqr-n7m_` would confidently resolve to `boxlqr-n7m3` -- a wrong answer
    presented as a right one."""
    assert _run(store, "models", "show", "boxlqr-n7m_") == REFUSED
    assert "no model matches" in capsys.readouterr().out


def test_a_percent_sign_is_a_literal_too(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other `LIKE` wildcard. `%` alone would otherwise match everything and
    report the whole store as an ambiguous prefix rather than as a bad
    reference."""
    assert _run(store, "models", "show", "%") == REFUSED
    assert "no model matches" in capsys.readouterr().out


# --------------------------------------------------------------------------
# show and diff
# --------------------------------------------------------------------------


def test_show_reports_measurement_counts(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(store, "models", "show", MODEL_A)
    assert "1 nominal, 1 shifted" in capsys.readouterr().out


def test_show_json_is_parseable(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "show", MODEL_A, "--json") == OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["model_id"] == MODEL_A
    assert payload["spec"]["epochs"] == 200


def test_diff_shows_only_the_differing_field(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "models", "diff", MODEL_A, MODEL_B) == OK
    out = capsys.readouterr().out
    assert "epochs" in out
    assert "problem_id" not in out.split("field")[-1]


def test_diff_of_a_model_with_itself_says_so(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Degenerate input: the answer is 'identical', not an empty table that
    reads as a failure."""
    assert _run(store, "models", "diff", MODEL_A, MODEL_A) == OK
    assert "identical" in capsys.readouterr().out


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def test_measurements_list_shows_every_measurement(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "measurements", "list") == OK
    out = capsys.readouterr().out
    assert out.count("nominal") == 2 and out.count("shifted") == 2


def test_measurements_shifted_filters_to_out_of_distribution_results(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "measurements", "list", "--shifted") == OK
    out = capsys.readouterr().out
    assert "shifted" in out and "nominal" not in out


def test_measurements_list_can_narrow_to_one_model(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "measurements", "list", "--model", "boxlqr-n7m3/riccati") == OK
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if "nominal" in ln or "shifted" in ln
    ]
    assert len(lines) == 2


def test_metrics_become_columns(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(store, "measurements", "list")
    out = capsys.readouterr().out
    assert "expected_cost" in out and "3.25" in out


# --------------------------------------------------------------------------
# store maintenance
# --------------------------------------------------------------------------


def test_stat_reports_each_class(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "store", "stat") == OK
    out = capsys.readouterr().out
    assert "models" in out and "measurements" in out and "index" in out
    assert "total" in out and "growth" in out


def test_verify_is_clean_on_a_healthy_store(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "store", "verify") == OK
    assert "unverifiable" in capsys.readouterr().out


def test_verify_exits_nonzero_on_corruption(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit code is what a scheduled check would act on, so it is asserted
    rather than the message."""
    weights = ModelStore(store).path(MODEL_A) / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
    assert _run(store, "store", "verify") == REFUSED
    assert "corrupt" in capsys.readouterr().out


def test_reindex_rebuilds_from_the_trees(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = StoreIndex(store)
    index.delete_measurements([m.measurement_id for m in index.measurements()])
    index.delete_models([m.model_id for m in index.models()])
    assert index.models() == []

    assert _run(store, "store", "reindex") == OK
    assert "2 models" in capsys.readouterr().out
    assert len(StoreIndex(store).models()) == 2


def test_reindex_preserves_the_queue(store: Path) -> None:
    """The one row in the index that is state rather than a projection. A
    reindex that dropped it would discard a multi-night batch."""
    index = StoreIndex(store)
    index.enqueue(entry_id="e1", study="depth-scaling", tier="standard")
    _run(store, "store", "reindex")
    assert [e["entry_id"] for e in StoreIndex(store).queue_entries()] == ["e1"]


# --------------------------------------------------------------------------
# gc
# --------------------------------------------------------------------------


def test_gc_refuses_and_explains_when_no_study_exists(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "store", "gc", "--yes") == REFUSED
    out = capsys.readouterr().out
    assert "refused" in out and "--partials" in out
    assert len(StoreIndex(store).measurements()) == 4


def _enrol(store: Path, member: str) -> None:
    with StoreIndex(store).connect() as conn:
        conn.execute(
            "INSERT INTO study_members VALUES (?,?,?)",
            ("study-a", "measurement", member),
        )


def test_gc_collects_what_no_study_references(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _enrol(store, f"{100:016x}")
    assert _run(store, "store", "gc", "--yes") == OK
    assert "collected" in capsys.readouterr().out
    assert [m.measurement_id for m in StoreIndex(store).measurements()] == [
        f"{100:016x}"
    ]


def test_a_declined_confirmation_removes_nothing(
    store: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety property that matters most: answering anything but yes must
    leave the store exactly as it was."""
    _enrol(store, f"{100:016x}")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert _run(store, "store", "gc") == REFUSED
    assert "cancelled" in capsys.readouterr().out
    assert len(StoreIndex(store).measurements()) == 4


def test_a_closed_stdin_is_not_taken_for_consent(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command run from a script with no terminal must decline, not proceed."""
    _enrol(store, f"{100:016x}")

    def _raise(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert _run(store, "store", "gc") == REFUSED
    assert len(StoreIndex(store).measurements()) == 4


def test_an_accepted_confirmation_collects(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrol(store, f"{100:016x}")
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert _run(store, "store", "gc") == OK
    assert len(StoreIndex(store).measurements()) == 1


def test_dry_run_removes_nothing_and_needs_no_confirmation(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _enrol(store, f"{100:016x}")
    assert _run(store, "store", "gc", "--dry-run") == OK
    assert "would free" in capsys.readouterr().out
    assert len(StoreIndex(store).measurements()) == 4


def test_gc_leaves_models_alone_without_the_flag(store: Path) -> None:
    _enrol(store, f"{100:016x}")
    _run(store, "store", "gc", "--yes")
    assert len(StoreIndex(store).models()) == 2


def test_gc_models_needs_the_explicit_flag(store: Path) -> None:
    _enrol(store, f"{100:016x}")
    assert _run(store, "store", "gc", "--models", "--yes") == OK
    assert [m.model_id for m in StoreIndex(store).models()] == [MODEL_A]


def test_partials_can_be_collected_on_a_store_with_no_studies(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The safe half of `gc` must stay usable while the study table is empty --
    otherwise the refusal above would make resume artifacts uncollectable."""
    partial = ModelStore(store).path(MODEL_A) / ".partial"
    partial.mkdir()
    (partial / "optimizer.safetensors").write_bytes(b"x" * 1024)

    assert _run(store, "store", "gc", "--partials", "--yes") == OK
    assert not partial.exists()
    assert len(StoreIndex(store).measurements()) == 4
    assert "collected" in capsys.readouterr().out


def test_combining_models_and_partials_is_a_usage_error(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(store, "store", "gc", "--models", "--partials") == USAGE
    assert "separately" in capsys.readouterr().out


def test_nothing_to_collect_is_reported_as_success(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for identifier in (100, 101, 200, 201):
        _enrol(store, f"{identifier:016x}")
    assert _run(store, "store", "gc", "--yes") == OK
    assert "nothing to collect" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Store root resolution and argument parsing
# --------------------------------------------------------------------------


def test_the_store_flag_wins_over_the_environment(
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The content is what is asserted, not the exit code: reading the wrong
    store succeeds just as cleanly as reading the right one, so a status check
    could not tell the two apart."""
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "elsewhere"))
    assert _run(store, "models", "list") == OK
    assert "boxlqr-n7m3/" in capsys.readouterr().out


def test_the_environment_is_used_when_no_flag_is_given(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(STORE_ENV, str(store))
    assert main(["models", "list"]) == OK
    assert "boxlqr-n7m3/" in capsys.readouterr().out


def test_the_default_is_the_store_of_the_project_you_are_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Renamed from "relative to the working directory", which is no longer
    what it does and was never what a reader wanted: the store belongs to the
    project, and `mbl run` from a subdirectory should not strand one there.

    The marker is written explicitly. A directory is a project because it
    carries `pyproject.toml`, not because it contains something called
    `store` — this repository has six directories called `store` or `studies`
    that are neither."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "probe"\n')
    _build_store(tmp_path / DEFAULT_STORE)
    monkeypatch.delenv(STORE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["models", "list"]) == OK
    assert "boxlqr-n7m3/" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("7d", 604_800.0), ("36h", 129_600.0), ("90m", 5_400.0), ("1w", 604_800.0)],
)
def test_durations_parse(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["7", "", "d", "7days", "-7d", "7.5d"])
def test_a_duration_without_a_unit_is_rejected(text: str) -> None:
    """`--older-than 7` meaning seconds where the user meant days would collect
    a week of recoverable work, so a bare number is refused rather than
    assumed."""
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_duration(text)


def test_every_subcommand_the_annex_lists_is_reachable() -> None:
    """The annex's §5 discovery and maintenance tables, minus `run`/`queue`,
    which need Tier 5. Enumerated rather than sampled: a subcommand that was
    never wired up would otherwise only be found by someone typing it."""
    parser = build_parser()
    for argv in (
        ["models", "list"],
        ["models", "show", "x"],
        ["models", "tree"],
        ["models", "diff", "a", "b"],
        ["measurements", "list"],
        ["store", "stat"],
        ["store", "verify"],
        ["store", "reindex"],
        ["store", "gc"],
    ):
        assert callable(parser.parse_args(argv).handler), argv


@pytest.mark.parametrize(
    "argv",
    [
        ["models", "list"],
        ["models", "tree"],
        ["models", "show", "x"],
        ["models", "diff", "a", "b"],
        ["measurements", "list"],
        ["store", "stat"],
        ["store", "verify"],
        ["store", "reindex"],
        ["store", "gc", "--yes"],
    ],
)
def test_a_missing_store_is_reported_and_not_created(
    argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Opening the index creates its directory, so a mistyped `--store` would
    otherwise produce a new empty store and then truthfully report it as empty.

    Every subcommand is checked rather than one: the guard lives in `main`
    precisely so it cannot be omitted, and that claim is only worth making if
    it is verified everywhere it claims to hold.
    """
    absent = tmp_path / "typo"
    assert main(["--store", str(absent), *argv]) == REFUSED
    assert "no store at" in capsys.readouterr().out
    assert not absent.exists()


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["models", "invent"])
    assert raised.value.code == USAGE
