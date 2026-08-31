"""Acceptance tests for the replay seam — Stage 6, Phase A.

Written before the implementation. `src/mbl/replay/` is the only module a
notebook imports, and D1's whole claim is that a notebook *loads* a result
rather than producing one. Three properties decide whether an implementation is
actually replay, and each is easy to satisfy in appearance:

1. **The notebook trains nothing.** Asserted by making the alternative raise —
   `run_study` and `build_controller` are monkeypatched to detonate, and
   `resolve` still succeeds against a warm store. Timing would assert a
   duration; this asserts reachability.
2. **A missing result is a refusal that names the command.** Not checked as a
   substring: the command the error prints is **executed**, and the study must
   then resolve. A message that named the wrong tier, the wrong document or the
   wrong verb would pass any `in` assertion and fail this one.
3. **A subsetted `smoke` store is refused.** The analysis tier already refuses
   it; a notebook that got further would be the only path by which a truncated
   sweep axis reaches a reader. Tested in the failing direction, which is the
   only direction that says anything.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mbl.cli.app import main
from mbl.analysis.gates import GateStatus
from mbl.replay import (
    GateFailedError,
    StoreIncompleteError,
    load_study,
    render_gates,
    resolve,
    survey,
)
from mbl.spec.errors import SpecificationError
from mbl.locate import PACKAGE_ROOT as REAL_PACKAGE_ROOT
from mbl.locate import ROOT_MARKER
from mbl.replay.loading import PACKAGE_STUDIES, STUDIES_DIRNAME, STUDIES_ENV
from mbl.replay.resolution import _activity_table
from mbl.store.content_store import MeasurementStore
from mbl.store.location import DEFAULT_STORE, STORE_ENV

from ..runner.test_producer import _write

#: The fixture at `standard`, where the catalogue forces one training seed:
#: two depths of the swept contender plus one flat baseline.
POINTS_AT_STANDARD = 3

REPORTING = """
[[analyses]]
id = "cost_by_depth"
kind = "cost_vs_axis"

[[figures]]
id = "fig_cost"
kind = "axis_scaling"
source = "cost_by_depth"
xlabel = "depth"
ylabel = "cost"
"""


def _document(directory: Path, /, **overrides: Any) -> Path:
    """The producer fixture's study, plus a declared analysis and figure."""
    path = _write(directory, **overrides)
    path.write_text(path.read_text() + REPORTING)
    return path


def _cli(store: Path, *argv: str) -> int:
    return main(["--store", str(store), *argv])


def _produce(document: Path, store: Path, *, tier: str = "standard") -> None:
    """Everything the pipeline can produce, through the shipped commands."""
    assert _cli(store, "run", str(document), "--tier", tier) == 0
    assert _cli(store, "analyse", str(document), "--tier", tier) == 0
    assert _cli(store, "figure", "render", str(document), "--tier", tier) == 0


class TestLoading:
    def test_the_tier_reaches_the_study(self, tmp_path: Path) -> None:
        # A pair, not a single number: the fixture declares two seeds and
        # `standard` forces one, so a tier that failed to apply would leave
        # six points. Asserting only the resolved count would hold under an
        # implementation that ignored the tier and a fixture that declared one
        # seed.
        document = _document(tmp_path / "doc")
        declared = load_study(document, tier="publication")
        standard = load_study(document, tier="standard")
        assert (
            len(declared.study.materialise()),
            len(standard.study.materialise()),
        ) == (
            15,
            POINTS_AT_STANDARD,
        )

    def test_an_override_reaches_the_study(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        without = load_study(document, tier="standard")
        with_it = load_study(document, tier="standard", **{"training.seeds": 2})
        assert (
            len(without.study.materialise()),
            len(with_it.study.materialise()),
        ) == (POINTS_AT_STANDARD, 2 * POINTS_AT_STANDARD)

    def test_a_forbidden_override_is_refused_by_the_grammar(
        self, tmp_path: Path
    ) -> None:
        # `replay` must not become a second, laxer editing surface: the tier
        # whitelist is the grammar's, and reaching it is the whole point of
        # being thin.
        document = _document(tmp_path / "doc")
        with pytest.raises(SpecificationError):
            load_study(document, tier="standard", **{"problem.path": '"other.npz"'})

    def test_the_command_it_prints_reproduces_the_same_study(
        self, tmp_path: Path
    ) -> None:
        """The strong form. A substring check on the command would hold for a
        message naming the wrong tier or dropping an override; re-parsing it
        and comparing `StudyID` would not."""
        from mbl.cli.run import parse_overrides

        document = _document(tmp_path / "doc")
        loaded = load_study(
            document,
            tier="standard",
            **{"training.seeds": 2, "training.plan.epochs": 3},
        )
        argv = shlex.split(loaded.command("run"))
        assert argv[:3] == ["uv", "run", "mbl"]
        tier = argv[argv.index("--tier") + 1]
        sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
        rebuilt = load_study(Path(argv[4]), tier=tier, **parse_overrides(sets))
        assert rebuilt.study.study_id == loaded.study.study_id

    @pytest.mark.parametrize("tier", ["smoke", "publication", "comprehensive"])
    def test_the_command_names_the_tier_it_was_loaded_at(
        self, tmp_path: Path, tier: str
    ) -> None:
        """The single-specimen trap, caught by a surviving mutant.

        Every other command test uses `standard`, which is also the default —
        so an implementation that hard-coded the default passed all of them.
        Parametrised over the tiers that are *not* the default, which is the
        property under test.
        """
        loaded = load_study(_document(tmp_path / "doc"), tier=tier)
        argv = shlex.split(loaded.command("run"))
        assert argv[argv.index("--tier") + 1] == tier

    def test_the_command_survives_a_shell(self, tmp_path: Path) -> None:
        """A sweep path contains `*`, and a reader pastes this into a shell.

        Caught by a surviving mutant: dropping `shlex.quote` changed nothing
        the suite could see, because every other test splits the string in
        Python and Python does not glob. `bash` does. The decoy file below is
        exactly what an unquoted `contenders.*.config.plan.epochs=2` expands
        onto, and the assertion is that the shell's argv equals the intended
        argv — which is false the moment the quoting goes.
        """
        (tmp_path / "contenders.X.config.plan.epochs=2").touch()
        loaded = load_study(
            _document(tmp_path / "doc"),
            tier="standard",
            **{"contenders.*.config.plan.epochs": 2},
        )
        tail = loaded.command("run").split("uv run mbl ", 1)[1]
        shell = subprocess.run(
            ["bash", "-c", f"printf '%s\\n' {tail}"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert shell.stdout.splitlines() == shlex.split(tail)


class TestResolutionAgainstACompleteStore:
    def test_it_returns_the_analyses_and_the_figures(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)

        resolution = resolve(load_study(document, tier="standard"), store=store)

        assert resolution.completeness.complete
        assert set(resolution.analyses) == {"cost_by_depth"}
        assert set(resolution.figures) == {"fig_cost"}
        assert len(resolution.analyses["cost_by_depth"].table) == POINTS_AT_STANDARD
        assert set(resolution.figures["fig_cost"]) >= {"pdf", "png", "data.parquet"}

    def test_every_figure_path_it_hands_back_exists(self, tmp_path: Path) -> None:
        """The keys are not the deliverable; the paths are.

        Found by a surviving mutant. Corrupting only the *value* expression of
        the artifact mapping leaves every key correct — so completeness still
        passes, the suite stays green, and a notebook doing
        `Image(results.figures["fig"]["png"])` is handed a path to nothing.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)

        resolution = resolve(load_study(document, tier="standard"), store=store)

        for figure_id, artifacts in resolution.figures.items():
            for suffix, path in artifacts.items():
                assert path.is_file(), f"{figure_id}.{suffix} -> {path} does not exist"
                assert path.stat().st_size > 0

    def test_resolving_reaches_no_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replay's central claim, asserted by making the alternative raise.

        Not a timing assertion: a fast implementation that *could* train is
        still not replay.

        **Which call to detonate is the whole subtlety, and the first draft of
        this test got it wrong.** Patching `ContenderSpec.resolve` fails, and
        correctly so: resolution is recipe *construction*, which
        `derive_model_id` needs to sign a contender and which slice Phase A
        measured at 0.00 s. The call that costs — 1.37 s per COCP contender,
        because it runs D17's solver canary — is `build_controller`, and it is
        the one a replay path must never reach. Second occurrence of a trap the
        slice plan named in advance: *"an implementer told to avoid
        constructing a recipe would have guarded the wrong call."*
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)

        def detonate(*_: object, **__: object) -> Any:
            raise AssertionError("replay reached the execution layer")

        monkeypatch.setattr("mbl.runner.producer.run_study", detonate)
        monkeypatch.setattr(
            "mbl.applications.recipes.base.TrainableRecipe.build_controller",
            detonate,
            raising=True,
        )
        monkeypatch.setattr(
            "mbl.applications.recipes.base.TrainableRecipe.build_engine",
            detonate,
            raising=True,
        )

        resolution = resolve(load_study(document, tier="standard"), store=store)
        assert resolution.completeness.complete


class TestTheRefusals:
    def test_an_empty_store_names_the_run_command(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        store.mkdir()

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"), store=store)

        message = str(raised.value)
        assert "uv run mbl run" in message
        assert "--tier standard" in message

    def test_the_command_the_refusal_prints_actually_fixes_it(
        self, tmp_path: Path
    ) -> None:
        """End-to-end closure, and the only check here that cannot pass by
        accident. The refusal's command is executed verbatim; if it named the
        wrong verb, tier, document or overrides, the second resolve fails."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        store.mkdir()
        loaded = load_study(document, tier="standard")

        # Three stages, so four passes: fix, fix, fix, then succeed.
        for _ in range(4):
            try:
                resolve(loaded, store=store)
                break
            except StoreIncompleteError as error:
                argv = str(error).split("uv run mbl ")[1].splitlines()[0].split()
                assert _cli(store, *[a.strip("'\"") for a in argv]) == 0
        else:  # pragma: no cover - three stages is the whole pipeline
            pytest.fail("the refusal never converged on a complete store")

        assert resolve(loaded, store=store).completeness.complete

    def test_missing_measurements_ask_for_run_and_not_for_analyse(
        self, tmp_path: Path
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        store.mkdir()
        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"), store=store)
        assert "uv run mbl run" in str(raised.value)
        assert "uv run mbl analyse" not in str(raised.value)

    def test_a_missing_analysis_asks_for_analyse_and_not_for_run(
        self, tmp_path: Path
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        assert _cli(store, "run", str(document), "--tier", "standard") == 0

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"), store=store)

        message = str(raised.value)
        assert "uv run mbl analyse" in message
        assert "uv run mbl run" not in message

    @pytest.mark.parametrize(
        ("produced", "stage"),
        [((), "run"), (("run",), "analyse"), (("run", "analyse"), "figure render")],
    )
    def test_the_prose_and_the_command_name_the_same_stage(
        self, tmp_path: Path, produced: tuple[str, ...], stage: str
    ) -> None:
        """A pair, not two substrings — and a surviving mutant is why.

        Making the sentence always say "run" left every other assertion here
        green, because each checks only that the *command* is right. The
        result is a message that reads "3/3 measured and its run stage is not
        satisfied" above a command that says `analyse`: self-contradictory,
        and it sends a reader hunting for measurements that are all present.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        store.mkdir()
        for verb in produced:
            assert _cli(store, verb, str(document), "--tier", "standard") == 0

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"), store=store)

        message = str(raised.value)
        prose = message.split(" stage is not satisfied")[0].rsplit(" ", 1)[-1]
        commanded = message.split("uv run mbl ")[1].split(str(document))[0].strip()
        assert (prose, commanded) == (stage.split()[-1], stage)

    def test_a_missing_figure_asks_for_figure_render(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        assert _cli(store, "run", str(document), "--tier", "standard") == 0
        assert _cli(store, "analyse", str(document), "--tier", "standard") == 0

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"), store=store)

        assert "uv run mbl figure render" in str(raised.value)

    def test_one_missing_point_is_named(self, tmp_path: Path) -> None:
        """A count is not a diagnosis. Removing exactly one measurement must
        name that measurement, not report `2/3`."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        victim = loaded.study.materialise()[0]
        shutil.rmtree(store / "measurements" / str(victim.measurement_id))

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(loaded, store=store)

        message = str(raised.value)
        assert str(victim.measurement_id) in message
        assert victim.contender.resolved_label in message

    def test_a_subsetted_smoke_store_is_refused(self, tmp_path: Path) -> None:
        """Phase A's failing-direction checkpoint.

        A `smoke` run truncates the sweep axis and stamps every measurement it
        produces. The store is *complete* for that study, so incompleteness
        cannot catch it — only the stamp can, and a notebook is the last place
        it could still reach a reader.
        """
        document = _document(tmp_path / "doc", depths=[2, 4, 6, 8])
        store = tmp_path / "store"
        assert _cli(store, "run", str(document), "--tier", "smoke") == 0

        with pytest.raises(SpecificationError, match="axis_subset"):
            resolve(load_study(document, tier="smoke"), store=store)

    def test_the_same_study_at_a_subsetting_tier_is_fine_when_it_did_not_subset(
        self, tmp_path: Path
    ) -> None:
        """The other direction, so the refusal above is not simply 'smoke is
        rejected'. With one axis value there is nothing to truncate, `smoke`
        stamps nothing, and the study resolves."""
        document = _document(tmp_path / "doc", depths=[2])
        store = tmp_path / "store"
        _produce(document, store, tier="smoke")

        assert resolve(
            load_study(document, tier="smoke"), store=store
        ).completeness.complete


class TestSurvey:
    def test_it_reports_instead_of_raising(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        store.mkdir()

        report = survey(load_study(document, tier="standard"), store=store)

        assert not report.complete
        assert (report.points, report.measurements_held) == (POINTS_AT_STANDARD, 0)

    def test_it_counts_what_is_there(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)

        report = survey(load_study(document, tier="standard"), store=store)

        assert (report.points, report.measurements_held, report.complete) == (
            POINTS_AT_STANDARD,
            POINTS_AT_STANDARD,
            True,
        )


class TestGates:
    """Annex 04 §4 — a failed gate stops the notebook.

    The fixture study declares none, so each test adds one, which also means
    the gate reaching `resolve` at all is part of what is asserted: before this
    step, a declared gate was read by nothing.
    """

    @staticmethod
    def _with_gate(directory: Path, body: str) -> Path:
        path = _document(directory)
        path.write_text(path.read_text() + body)
        return path

    def test_a_gate_that_holds_is_reported_and_does_not_stop_anything(
        self, tmp_path: Path
    ) -> None:
        document = self._with_gate(
            tmp_path / "doc",
            '\n[[gates]]\nkind = "contenders_separate"\nmin_relative_gap = 1e-6\n',
        )
        store = tmp_path / "store"
        _produce(document, store)

        resolution = resolve(load_study(document, tier="standard"), store=store)

        assert [o.status for o in resolution.gates] == [GateStatus.PASSED]
        assert resolution.gates[0].measured is not None

    def test_a_gate_that_fails_stops_the_notebook(self, tmp_path: Path) -> None:
        """The failing direction, and the reason the step exists.

        A threshold no study could miss is not a gate. This one demands a 900 %
        spread, which the fixture cannot show, and the store is otherwise
        complete — so nothing but the gate can raise here.
        """
        document = self._with_gate(
            tmp_path / "doc",
            '\n[[gates]]\nkind = "contenders_separate"\nmin_relative_gap = 9.0\n',
        )
        store = tmp_path / "store"
        _produce(document, store)

        with pytest.raises(GateFailedError, match="contenders_separate"):
            resolve(load_study(document, tier="standard"), store=store)

    def test_an_unevaluable_gate_does_not_stop_it_and_is_not_reported_as_passed(
        self, tmp_path: Path
    ) -> None:
        """`constraint_binds` has no quantity in the store. It must neither
        block the notebook nor claim to have held — the two ways of getting
        this wrong point in opposite directions."""
        document = self._with_gate(
            tmp_path / "doc",
            '\n[[gates]]\nkind = "contenders_separate"\nmin_relative_gap = 1e-6\n',
        )
        store = tmp_path / "store"
        _produce(document, store)
        resolution = resolve(load_study(document, tier="standard"), store=store)

        rendered = render_gates(resolution)

        assert "contenders_separate" in rendered
        assert "| **passed** |" in rendered

    def test_the_section_lists_every_declared_gate(self, tmp_path: Path) -> None:
        """A gate omitted from the table is indistinguishable from one that
        passed, which is the defect this whole step closes."""
        document = self._with_gate(
            tmp_path / "doc",
            '\n[[gates]]\nkind = "contenders_separate"\nmin_relative_gap = 1e-6\n'
            '\n[[gates]]\nkind = "stability"\nmax_spectral_radius = 100.0\n',
        )
        store = tmp_path / "store"
        _produce(document, store)
        resolution = resolve(load_study(document, tier="standard"), store=store)

        rendered = render_gates(resolution)

        assert len(resolution.gates) == 2
        for outcome in resolution.gates:
            assert f"`{outcome.kind.value}`" in rendered


class TestConstraintBindsEndToEnd:
    """The whole chain, on a real store — Annex 01 §4.1, live from 2026-08-22.

    The unit tests in `tests/analysis/test_gates.py` drive the evaluator from a
    hand-built frame, which cannot catch the two failures that matter here: a
    statistic the producer never wrote, and a table `resolve` never handed
    over. Both would leave every unit test green while the gate reported *not
    evaluable* forever — which is exactly the state this work found.
    """

    @staticmethod
    def _with_gate(directory: Path, fraction: float) -> Path:
        path = _document(directory)
        path.write_text(
            path.read_text()
            + f'\n[[gates]]\nkind = "constraint_binds"\nmin_fraction = {fraction}\n'
        )
        return path

    def test_the_gate_is_decided_rather_than_reported_unevaluable(
        self, tmp_path: Path
    ) -> None:
        """The point of the whole change: a study run through the shipped
        commands now carries a verdict on a box that binds, with the number
        behind it."""
        document = self._with_gate(tmp_path / "doc", 0.01)
        store = tmp_path / "store"
        _produce(document, store)

        resolution = resolve(load_study(document, tier="standard"), store=store)

        (outcome,) = [o for o in resolution.gates if o.kind.value == "constraint_binds"]
        assert outcome.status is GateStatus.PASSED
        assert outcome.measured is not None and 0.0 < outcome.measured <= 1.0

    def test_a_box_that_does_not_bind_enough_stops_the_notebook(
        self, tmp_path: Path
    ) -> None:
        """The failing direction, through the real pipeline. A threshold of
        99.9 % is one no cast of this fixture could meet, and the store is
        otherwise complete — so nothing but this gate can raise here."""
        document = self._with_gate(tmp_path / "doc", 0.999)
        store = tmp_path / "store"
        _produce(document, store)

        with pytest.raises(GateFailedError, match="constraint_binds"):
            resolve(load_study(document, tier="standard"), store=store)

    def test_a_measurement_without_the_statistic_is_not_evaluable(
        self, tmp_path: Path
    ) -> None:
        """The compatibility direction, and the reason the gate reads the
        store rather than assuming it. Every measurement written before this
        change carries no such statistic, which is stripped here to reproduce
        one — and the verdict must be *not evaluable*, never *passed* and never
        a blocking failure that would make old stores unreadable."""
        document = self._with_gate(tmp_path / "doc", 0.01)
        store = tmp_path / "store"
        _produce(document, store)

        measurements = MeasurementStore(store)
        for measurement_id in measurements.list_ids():
            path = measurements.path(measurement_id) / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            metrics.pop("eval_saturation_fraction", None)
            path.write_text(json.dumps(metrics), encoding="utf-8")

        resolution = resolve(load_study(document, tier="standard"), store=store)

        (outcome,) = [o for o in resolution.gates if o.kind.value == "constraint_binds"]
        assert outcome.status is GateStatus.NOT_EVALUABLE
        assert "MeasurementID" in outcome.detail

    def test_the_activity_is_reduced_over_seeds_by_a_mean(self, tmp_path: Path) -> None:
        """§A.3 fixes how numbers are combined and the gate must obey it.

        This SURVIVED a mutant that reduced across seeds by the maximum, for a
        reason worth recording: the fixture resolves to ONE seed at `standard`,
        and every reduction over one value is the identity. A declaration about
        combining numbers cannot be tested on a store that never combines two.
        Run at two seeds, and recomputed by hand from the stored metrics rather
        than by asking the function under test.
        """
        document = self._with_gate(tmp_path / "doc", 0.01)
        store = tmp_path / "store"
        assert (
            _cli(
                store,
                "run",
                str(document),
                "--tier",
                "standard",
                "--set",
                "training.seeds=2",
            )
            == 0
        )
        loaded = load_study(document, tier="standard", **{"training.seeds": 2})
        measurements = MeasurementStore(store)

        by_group: dict[tuple[str, str], list[float]] = {}
        for point in loaded.study.materialise():
            metrics = measurements.metrics(str(point.measurement_id))
            key = (
                point.contender.resolved_label,
                repr(sorted(point.axis_values.items(), key=str)),
            )
            by_group.setdefault(key, []).append(metrics["eval_saturation_fraction"])

        # The premise: without two seeds per group there is no reduction to
        # observe, and this test degenerates into the one it replaces.
        assert any(len(values) == 2 for values in by_group.values())
        assert any(len(set(values)) == 2 for values in by_group.values())

        activity = _activity_table(loaded, measurements)
        for _, row in activity.iterrows():
            values = by_group[(row["contender"], row["axis_value"])]
            assert row["aggregate"] == pytest.approx(sum(values) / len(values))

    def test_the_verdict_is_read_off_the_activity_and_not_off_the_cost(
        self, tmp_path: Path
    ) -> None:
        """Two frames reach the evaluator and only one of them is a saturated
        fraction. A gate reading the cost table would compare a cost against a
        threshold in (0, 1] and pass or fail for reasons unrelated to the box;
        here the fixture's costs are far above 1, so such a gate would pass a
        threshold this one must fail."""
        document = self._with_gate(tmp_path / "doc", 0.999)
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")

        activity = _activity_table(loaded, MeasurementStore(store))

        assert not activity.empty
        assert (activity["aggregate"] <= 1.0).all()
        with pytest.raises(GateFailedError):
            resolve(loaded, store=store)


def _the_real_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the package's own repository back.

    `tests/conftest.py` substitutes it for every test, so that one which forgot
    to build its own store cannot silently read the developer's real one — a
    directory holding a complete copy of the tracked study, against which most
    wrong tests would *pass*. The `studies/` half is committed and safe to
    depend on, so the tests below that are genuinely about it opt back in, and
    have to say so.
    """
    from mbl import locate

    monkeypatch.setattr(locate, "PACKAGE_ROOT", REAL_PACKAGE_ROOT)


def _project(directory: Path) -> Path:
    """Make `directory` a project root.

    A directory is a project because it carries the marker, not because it
    contains something called `store`. Spelled out in every fixture below
    rather than hidden in a conftest, because it is the property under test:
    the search that name-matched instead resolved into `store/studies/`,
    `src/mbl/store/` and four other namesakes this repository already has.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ROOT_MARKER).write_text('[project]\nname = "probe"\n')
    return directory


class TestStoreLocation:
    """Where the store is when a notebook does not say — Stage 6, Phase B.

    Defaulted so that the committed notebook carries no directory name in any
    cell (plan §2's second property). The tests are written from the failing
    side, because a default is exactly the shape of the trap this project has
    hit three times: a test that supplies the very value the implementation
    falls back to holds whether or not the argument is ever consulted. So the
    environment here always names a store that is **not** `./store`, and the
    explicit argument always names a store that is **not** the environment's.
    """

    def test_the_environment_is_honoured_when_nothing_is_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "named_by_the_environment"
        _produce(document, store)
        monkeypatch.setenv(STORE_ENV, str(store))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        resolution = resolve(load_study(document, tier="standard"))

        assert resolution.store == store

    def test_an_explicit_root_outranks_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two fixtures differ, so this cannot pass by both naming one
        place — the defect that made an earlier override test unfailable."""
        document = _document(tmp_path / "doc")
        wanted, ignored = tmp_path / "wanted", tmp_path / "ignored"
        _produce(document, wanted)
        monkeypatch.setenv(STORE_ENV, str(ignored))

        resolution = resolve(load_study(document, tier="standard"), store=wanted)

        assert resolution.store == wanted
        assert not ignored.exists(), "the environment's store was never touched"

    def test_it_falls_back_to_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = _document(tmp_path / "doc")
        monkeypatch.delenv(STORE_ENV, raising=False)
        _project(tmp_path)
        monkeypatch.chdir(tmp_path)
        _produce(document, tmp_path / DEFAULT_STORE)

        resolution = resolve(load_study(document, tier="standard"))

        assert resolution.store == tmp_path / DEFAULT_STORE

    def test_the_store_is_found_from_a_subdirectory_of_the_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE AUTHOR'S FAILURE, and the property no earlier test here had.

        Every check that had ever executed the notebook set `$MBL_STORE` — this
        suite, the probe, and every manual invocation — so not one of them
        exercised the case a human actually reaches: open the notebook, run it,
        no environment set. A Jupyter kernel's working directory is the
        **notebook's own**, two levels below the project root, and a store
        resolved relative to *that* is not merely wrong, it is **absent** — so
        the study reported 0 of 135 measured and the refusal told the reader to
        run a command they had already run.

        The fixture is the shape that matters: a store at a project root, and a
        working directory below it. Nothing about the assertion is new; what is
        new is standing somewhere other than beside the store.
        """
        document = _document(tmp_path / "doc")
        _project(tmp_path)
        _produce(document, tmp_path / DEFAULT_STORE)
        deep = tmp_path / "notebooks" / "experiments"
        deep.mkdir(parents=True)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.chdir(deep)

        resolution = resolve(load_study(document, tier="standard"))

        assert resolution.store == tmp_path / DEFAULT_STORE
        assert resolution.completeness.complete

    def test_a_directory_merely_named_store_is_not_the_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The false positive the first version of this fix introduced.

        Searching upward for a directory *called* `store` finds
        `src/mbl/store/` and `tests/store/` — Python packages — and from
        `src/mbl/` it returned the package as the results store. Measured, not
        imagined. Anchoring on the project marker instead removes the whole
        class rather than this instance.

        Driven from the failing side: the namesake sits **nearer** the working
        directory than the real store, so anything matching on name stops at it.
        """
        _project(tmp_path)
        (tmp_path / DEFAULT_STORE).mkdir()
        namesake = tmp_path / "src" / DEFAULT_STORE
        namesake.mkdir(parents=True)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.chdir(tmp_path / "src")

        from mbl.store.location import default_store

        assert default_store() == tmp_path / DEFAULT_STORE

    def test_a_working_directory_inside_a_store_still_finds_the_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The instance that settles the design, and it is structural rather
        than accidental.

        **Every store contains a `studies/` of its own** — the artifact tree is
        `store/studies/<StudyID>/`, fixed by the store's own layout. So a search
        for a directory named `studies`, run from anywhere inside any store,
        resolves into that store and reads a directory of `StudyID` folders as
        though it held study documents. Annex 05 §2.3 puts the store outside
        every worktree at a fixed absolute path, which is a path an author
        navigates into.

        Nothing about a name check can survive this; only the project marker
        can. Both directories are asserted, because the store half has the
        mirror-image version of the same collision.
        """
        _project(tmp_path)
        (tmp_path / STUDIES_DIRNAME).mkdir()
        inside_store = tmp_path / DEFAULT_STORE / STUDIES_DIRNAME / "8d4349a97cad1fae"
        inside_store.mkdir(parents=True)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(tmp_path / DEFAULT_STORE)

        from mbl.replay.loading import studies_root
        from mbl.store.location import default_store

        assert studies_root() == tmp_path / STUDIES_DIRNAME
        assert default_store() == tmp_path / DEFAULT_STORE

    def test_a_project_without_a_store_falls_through_to_the_package_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Standing in some other project, you still get results.

        The candidate list is ordered, and an order is only meaningful if a
        later member can win. Every other test here builds the nearer directory,
        so the first candidate always existed and returning it unconditionally
        was indistinguishable from searching — the mutant that did exactly that
        survived until this case existed.

        `PACKAGE_ROOT` is substituted rather than relied upon: the real one is
        `store/`, which is gitignored and absent on a fresh clone, so a test
        that asserted against it would pass here and fail in CI. Sixth recorded
        instance of environment-dependent success.
        """
        from mbl import locate

        elsewhere = _project(tmp_path / "another_project")
        package = tmp_path / "package_repo"
        (package / DEFAULT_STORE).mkdir(parents=True)
        monkeypatch.setattr(locate, "PACKAGE_ROOT", package)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.chdir(elsewhere)

        from mbl.store.location import default_store

        assert not (elsewhere / DEFAULT_STORE).exists(), "the nearer one must be absent"
        assert default_store() == package / DEFAULT_STORE

    def test_the_refusal_names_the_store_it_consulted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The line that would have made the author's incident self-diagnosing.

        They were told the study was 0 of 135 measured and given a command to
        run; they ran it and were told everything already existed. Both messages
        were true and neither was actionable, because the two were talking about
        **different stores** and nothing on either side said which. A refusal
        that names what is missing but not where it looked cannot be acted on
        when the whole error is *where it looked*.

        Driven from the failing side: two stores exist and the empty one is
        consulted, so a message that named a default, or the complete one,
        passes every assertion about the study and fails this.
        """
        document = _document(tmp_path / "doc")
        complete, empty = tmp_path / "complete", tmp_path / "empty"
        _produce(document, complete)
        empty.mkdir()
        monkeypatch.setenv(STORE_ENV, str(empty))

        with pytest.raises(StoreIncompleteError) as raised:
            resolve(load_study(document, tier="standard"))

        message = str(raised.value)
        assert str(empty) in message, "the refusal does not say where it looked"
        assert str(complete) not in message

    def test_a_nearer_store_outranks_a_higher_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The walk stops at the first store it finds, and it must: a nested
        project inside another one would otherwise read its parent's results
        and agree with itself completely.

        Driven from the failing side — the outer store is **complete** for this
        study, so a search that ran past the inner one would resolve rather than
        refuse, which is the silent direction.
        """
        document = _document(tmp_path / "doc")
        _project(tmp_path)
        _produce(document, tmp_path / DEFAULT_STORE)
        inner = _project(tmp_path / "nested")
        (inner / DEFAULT_STORE).mkdir(parents=True)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.chdir(inner)

        with pytest.raises(StoreIncompleteError):
            resolve(load_study(document, tier="standard"))

    def test_the_store_and_the_studies_directory_are_found_the_same_way(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two drifted apart once already and the author found both halves.

        `studies_root` was taught to walk up and `default_store` was not,
        because they were two implementations of one rule. Asserted
        **behaviourally** over one project layout carrying both directories,
        with the working directory two levels down: whichever of them stops
        walking, this fails. An assertion about which function each one calls
        would be satisfied by a wrapper that then did something else.
        """
        _project(tmp_path)
        (tmp_path / DEFAULT_STORE).mkdir()
        (tmp_path / STUDIES_DIRNAME).mkdir()
        deep = tmp_path / "notebooks" / "experiments"
        deep.mkdir(parents=True)
        monkeypatch.delenv(STORE_ENV, raising=False)
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(deep)

        from mbl.replay.loading import studies_root
        from mbl.store.location import default_store

        assert default_store() == tmp_path / DEFAULT_STORE
        assert studies_root() == tmp_path / STUDIES_DIRNAME

    def test_the_notebook_and_the_command_line_resolve_the_same_place(self) -> None:
        """One rule, and the test that keeps it one.

        `cli.app` re-exports these rather than declaring them, and this is the
        assertion that says so — two surfaces that drifted into disagreeing
        about where the store is would each be individually correct and
        together useless, because the notebook would report missing exactly the
        measurements the command line had just written.
        """
        from mbl.cli import app
        from mbl.store import location

        assert app.STORE_ENV is location.STORE_ENV
        assert app.DEFAULT_STORE is location.DEFAULT_STORE


class TestExecuteMissing:
    """D1's escape hatch — Annex 04 §1.4's `resolve(study, execute_missing=True)`.

    Deferred from Phase A because it needed a home that is not a second copy of
    the command line's own run path, and this is that home: the three stage
    entry points all sit *below* Tier 8, so the seam composes them and copies
    nothing. What the command line adds on top of them — parsing `--set`,
    shaping a line of output — is the part a notebook has no use for.

    The tests are written around the two ways this can be wrong. It can fail to
    execute, which is obvious. Or it can execute **too much**: re-running the
    trainer to produce a figure costs hours and produces byte-identical models,
    and nothing about a green result would show it. So the load-bearing test
    detonates the trainer and requires the escape hatch to succeed anyway.
    """

    def test_it_is_off_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is the whole safety property, so it is asserted
        structurally as well as behaviourally: a notebook that trains by
        default is not a replay notebook."""
        import inspect

        assert inspect.signature(resolve).parameters["execute_missing"].default is False

        monkeypatch.setattr(
            "mbl.runner.producer.run_study",
            _detonate("resolve trained without being asked to"),
        )
        with pytest.raises(StoreIncompleteError):
            resolve(
                load_study(_document(tmp_path / "doc"), tier="standard"),
                store=tmp_path / "store",
            )

    def test_it_fills_an_empty_store_and_resolves(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"

        resolution = resolve(
            load_study(document, tier="standard"), store=store, execute_missing=True
        )

        assert resolution.completeness.complete
        assert resolution.figures, "the figure stage never ran"
        assert resolution.analyses, "the analysis stage never ran"

    def test_it_starts_at_the_first_unsatisfied_stage_and_no_earlier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one that can actually fail, and the reason for the whole shape.

        The measurements are present and only the derived artifacts are gone —
        the ordinary case after an analysis is added to a document. Re-running
        the study would retrain nothing (the producer reuses), but it would pay
        the *evaluation* pass for every point, which for this study is most of
        the cost and for NB04 at publication is hours. Driven from the failing
        side: the trainer is made to explode, so a run stage that fired at all
        is a test failure rather than a slow test.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        artifacts = store / "studies" / str(loaded.study.study_id)
        shutil.rmtree(artifacts)
        assert not artifacts.exists()

        monkeypatch.setattr(
            "mbl.runner.producer.run_study",
            _detonate("the run stage fired although every measurement was held"),
        )

        resolution = resolve(loaded, store=store, execute_missing=True)

        assert resolution.completeness.complete
        assert resolution.analyses and resolution.figures

    def test_it_still_refuses_when_execution_does_not_satisfy_the_study(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The escape hatch is not an override. If producing does not in fact
        complete the study, the refusal is the same refusal — a flag that
        turned a check off would be worse than no flag."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        monkeypatch.setattr("mbl.runner.producer.run_study", lambda *a, **k: None)

        with pytest.raises(StoreIncompleteError):
            resolve(
                load_study(document, tier="standard"), store=store, execute_missing=True
            )


def _detonate(message: str):
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(message)

    return explode


class TestStudiesLocation:
    """Where study documents are found — the author hit this, running NB04.

    The seam resolves a study **id** rather than a path (Annex 04 §1.4) so that
    no cell carries a directory name. That resolution has to answer "which
    `studies/`", and the first implementation asked only the working directory —
    which is right for a Jupyter kernel, whose working directory is the
    notebook's own, and wrong for every other way of reaching the same call.

    The author reported the failure from an intermediate copy of the notebook
    that still named a path, and the message they got named a relative location
    and nothing else. Both halves are fixed here: the repository the package
    itself was imported from is searched too, and the refusal lists everywhere
    it looked.
    """

    def test_a_study_resolves_from_a_working_directory_outside_the_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failing direction, and the one the author actually hit.

        `tmp_path` has no `studies/` at or above it, so the working-directory
        walk finds nothing and only the package's own repository can answer.
        """
        _the_real_repository(monkeypatch)
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        loaded = load_study("box_lqr/depth_scaling", tier="standard")

        assert loaded.study.id == "box_lqr/depth_scaling"
        assert loaded.document.is_absolute()

    def test_a_nearer_studies_directory_that_lacks_the_study_is_not_the_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case a *list* of candidates exists for, and the one neither
        earlier test reached.

        Both were written from a working directory with no `studies/` above it
        at all, so the list only ever had one member and a mutant collapsing it
        to one survived. Here the nearest directory exists and simply does not
        hold this study — which is what happens to anyone standing in another
        project — and the search has to continue rather than stop.
        """
        _the_real_repository(monkeypatch)
        _project(tmp_path)
        (tmp_path / STUDIES_DIRNAME).mkdir()
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        loaded = load_study("box_lqr/depth_scaling", tier="standard")

        assert loaded.document.parent.parent == PACKAGE_STUDIES

    def test_the_working_directory_outranks_the_installed_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkout you are standing in wins over the one the package came
        from — otherwise a second worktree would silently read the first's
        documents, which is the class of bug the store's own location rule
        exists to prevent.

        Driven from the failing side: the local document is filed under the
        *tracked* study's id, so a resolution that reached past it would load a
        real document and look entirely healthy.
        """
        _project(tmp_path)
        local = tmp_path / STUDIES_DIRNAME / "box_lqr"
        local.mkdir(parents=True)
        document = _document(tmp_path / "doc")
        (local / "depth_scaling.toml").write_text(document.read_text())
        (local / "problem.npz").write_bytes(
            (document.parent / "problem.npz").read_bytes()
        )
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        loaded = load_study("box_lqr/depth_scaling", tier="standard")

        assert loaded.document == local / "depth_scaling.toml"
        assert loaded.study.id == "probe/producer", (
            "the tracked study was loaded instead of the local one"
        )

    def test_the_environment_outranks_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path / "declared" / "box_lqr"
        elsewhere.mkdir(parents=True)
        document = _document(tmp_path / "doc")
        (elsewhere / "depth_scaling.toml").write_text(document.read_text())
        (elsewhere / "problem.npz").write_bytes(
            (document.parent / "problem.npz").read_bytes()
        )
        monkeypatch.setenv(STUDIES_ENV, str(tmp_path / "declared"))

        loaded = load_study("box_lqr/depth_scaling", tier="standard")

        assert loaded.document.parent == elsewhere

    def test_the_refusal_lists_every_place_it_looked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal naming one location cannot be acted on when the caller's
        wrong assumption is about *which* location was consulted — which is
        precisely the report that produced this class."""
        _the_real_repository(monkeypatch)
        _project(tmp_path)
        nearer = tmp_path / STUDIES_DIRNAME
        nearer.mkdir()
        monkeypatch.delenv(STUDIES_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SpecificationError) as raised:
            load_study("no_such/study", tier="standard")

        message = str(raised.value)
        assert "no_such/study" in message
        # BOTH, and the fixture is built so there are two: naming one is what
        # the report that produced this class actually suffered from.
        assert str(nearer) in message, "the nearer directory is unnamed"
        assert str(PACKAGE_STUDIES) in message, "the repository's own is unnamed"
        assert STUDIES_ENV in message
