"""Acceptance tests for the generated notebook sections — Stage 6, Phase A.

Annex 04 §1.2 marks sections 3 (Protocol) and 7 (Provenance) **generated**, and
says why: prose drift becomes structurally impossible only if the prose is not
prose. So the tests here are not "does the string mention the tier" — a
hard-coded template mentions the tier too. They are **differential**: change the
study, and the rendered section must change with it; change nothing, and it must
not.

The provenance section gets the stronger treatment. It must name *every* model
and measurement the store holds for the study, so it is compared as a **set**
against the store itself. Asserting that one identifier appears would hold for a
renderer that emitted the first one and stopped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mbl.cli.app import main
from mbl.replay import load_study, render_protocol, render_provenance, resolve
from mbl.store.content_store import MeasurementRecord

from ..runner.test_producer import _write
from .test_resolution import REPORTING
from .test_sections import _with_a_second_instance


def _document(directory: Path, /, **overrides: Any) -> Path:
    path = _write(directory, **overrides)
    path.write_text(path.read_text() + REPORTING)
    return path


def _produce(document: Path, store: Path, *, tier: str = "standard") -> None:
    for argv in (
        ["run", str(document), "--tier", tier],
        ["analyse", str(document), "--tier", tier],
        ["figure", "render", str(document), "--tier", tier],
    ):
        assert main(["--store", str(store), *argv]) == 0


def _row(text: str, name: str) -> str:
    """The one table row named `name`, or a failure that says the row is gone."""
    assert name in text, f"the provenance table has no {name!r} row at all"
    return next(line for line in text.splitlines() if name in line)


def _untime(store: Path, points: Any) -> None:
    """Rewrite the given measurements the way the pre-clock producer wrote them.

    Through the store's own `put`, so the manifest is the manifest a record of
    that shape really has -- a hand-edited `spec.json` would leave a digest that
    `verify` rejects, and the test would then be asserting against a corrupt
    record rather than an older one.
    """
    from mbl.store.content_store import MeasurementStore

    measurements = MeasurementStore(store)
    for point in points:
        record = measurements.get(point.measurement_id)
        spec = dict(record.spec)
        provenance = {
            key: value
            for key, value in dict(spec["provenance"]).items()
            if key not in ("wall_time_s", "peak_vram_mb")
        }
        measurements.delete(point.measurement_id)
        measurements.put(
            MeasurementRecord(
                measurement_id=record.measurement_id,
                spec={**spec, "provenance": provenance},
                metrics=record.metrics,
                samples=record.samples,
                trace=record.trace,
                log=record.log,
            )
        )


#: The fixture's own learned plan, and the layer-wise plan substituted for it.
#: Held as literals compared against the document rather than as a `replace`
#: pattern trusted blind: a substitution that stops matching goes silently
#: inert, and the test then asserts against the unchanged fixture.
LEARNED_PLAN = """[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 8"""

LAYERWISE_PLAN = """[contenders.config.plan]
_build = "layerwise"
optimizer = "adam"
learning_rate = 0.05
warmup_epochs_per_layer = 2
refinement_epochs = 3"""


def _fitting_rows(text: str) -> list[str]:
    """The rows of §3's fitting table, and not of the cast table above it.

    Both tables key on the contender label, so a search over the whole section
    finds the cast row first and asserts nothing about the fitting -- which is
    how the first version of the test below passed against a section that had
    no fitting table at all.
    """
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("| contender | fitting |")
    )
    return [line for line in lines[start + 2 :] if line.startswith("| `")]


class TestProtocol:
    def test_it_names_every_contender(self, tmp_path: Path) -> None:
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        text = render_protocol(loaded)
        labels = {p.contender.resolved_label for p in loaded.study.materialise()}
        assert labels and all(label in text for label in labels)

    def test_it_is_generated_from_the_study_and_not_templated(
        self, tmp_path: Path
    ) -> None:
        """Differential. Two studies differing only in seed count must render
        two different protocols; a template that interpolated nothing renders
        one."""
        document = _document(tmp_path / "doc")
        one = render_protocol(load_study(document, tier="standard"))
        two = render_protocol(
            load_study(document, tier="standard", **{"training.seeds": 4})
        )
        assert one != two

    def test_the_training_is_stated_and_not_only_the_cast(self, tmp_path: Path) -> None:
        """Annex 04 §1.2 makes §3 "contenders, **training**, evaluation,
        statistics, seeds", and until 2026-08-21 it stated no training at all:
        no schedule, no optimiser, no budget, no initialisation. Two campaign
        figures differing in exactly those and in nothing else this section
        printed rendered identically."""
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        plan = dict(loaded.study.training.plan.get_signature())
        optimizer = dict(plan["optimizer"])

        text = render_protocol(loaded)

        assert "end-to-end" in text
        assert str(optimizer["name"]) in text
        assert repr(optimizer["learning_rate"]) in text
        assert repr(plan["epochs"]) in text

    def test_the_effective_plan_moves_when_the_budget_does(
        self, tmp_path: Path
    ) -> None:
        """Differential, and the one that matters: a template naming a plan it
        never read passes every `in` assertion above."""
        document = _document(tmp_path / "doc")
        declared = render_protocol(load_study(document, tier="standard"))
        raised = render_protocol(
            load_study(
                document, tier="standard", **{"contenders.*.config.plan.epochs": 3}
            )
        )
        assert declared != raised
        assert "epochs = `3`" in raised

    def test_a_contender_that_never_trains_is_said_not_to_be_fitted(
        self, tmp_path: Path
    ) -> None:
        """A closed-form contender has no plan, and a row that left the cell
        blank would read as "the section does not know" rather than as "there
        is nothing to know". The fixture's `baseline` is that case."""
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        unfitted = {
            point.contender.resolved_label
            for point in loaded.study.materialise()
            if "plan" not in point.contender.config
        }

        text = render_protocol(loaded)

        assert unfitted, "the fixture no longer has a contender that never trains"
        fitting = _fitting_rows(text)
        for label in unfitted:
            row = next(line for line in fitting if f"| `{label}` |" in line)
            assert "*not fitted*" in row

    def test_the_initialisation_is_stated_at_its_declared_precision(
        self, tmp_path: Path
    ) -> None:
        """A step size is identity-bearing: the campaign's `1/(2L)` literal and
        a rounded copy of it derive two different `ModelID`s, so a protocol
        section that prints `0.000361` has stated an experiment that never
        ran."""
        directory = tmp_path / "doc"
        document = _document(directory)
        literal = 0.0003605716441576073
        before = document.read_text()
        document.write_text(
            before.replace("step_size_init = 0.1", f"step_size_init = {literal!r}")
        )
        assert document.read_text() != before, "the fixture's step declaration moved"

        text = render_protocol(load_study(document, tier="standard"))

        assert repr(literal) in text
        assert "init method" not in text or "cold" in text

    def test_the_schedule_is_read_off_the_plan_and_not_assumed(
        self, tmp_path: Path
    ) -> None:
        """A layer-wise plan is the second schedule this project has, and it is
        the one a hard-coded word gets wrong. `end-to-end` printed of a study
        that warms up layer by layer is a §3 that describes an experiment
        nobody ran — the exact drift the section is generated to prevent."""
        directory = tmp_path / "doc"
        document = _document(directory)
        before = document.read_text()
        document.write_text(before.replace(LEARNED_PLAN, LAYERWISE_PLAN))
        assert document.read_text() != before, "the fixture's plan declaration moved"

        text = render_protocol(load_study(document, tier="standard"))

        assert "layer-wise" in text
        assert "warmup epochs per layer" in text

    def test_the_swept_axis_is_not_restated_beside_the_cast(
        self, tmp_path: Path
    ) -> None:
        """The contender table already carries the axis values; a second
        statement of them is a second thing to keep in step."""
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        swept = {
            path.rsplit(".", 1)[-1]
            for point in loaded.study.materialise()
            for path in point.axis_values
        }

        text = render_protocol(loaded)

        assert swept, "the fixture no longer sweeps anything"
        for leaf in swept:
            assert f"{leaf.replace('_', ' ')} = " not in text

    def test_a_spec_valued_axis_is_named_and_not_interpolated(
        self, tmp_path: Path
    ) -> None:
        """`SweepAxis.labels` exists because "a filename would name a category
        `..._rotA30` in a paper" and a specification interpolates as a repr.
        Neither §3 nor §7 performed that join until 2026-08-21: every mismatch
        document rendered a quarter of a megabyte of array repr into what was
        meant to be one table cell, with newlines that end the table at its
        first row. Found by rendering it."""
        document = _with_a_second_instance(tmp_path / "doc")

        text = render_protocol(load_study(document, tier="standard"))

        assert "nominal" in text and "shifted" in text
        assert "ProblemData(" not in text
        assert "array([" not in text
        assert len(text) < 8_000

    def test_it_is_deterministic(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        loaded = load_study(document, tier="standard")
        assert render_protocol(loaded) == render_protocol(loaded)

    def test_it_reports_the_tier_it_was_resolved_at(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        smoke = render_protocol(load_study(document, tier="smoke"))
        standard = render_protocol(load_study(document, tier="standard"))
        assert "smoke" in smoke and "standard" in standard
        assert smoke != standard

    def test_the_seed_line_says_how_many_and_not_only_which(
        self, tmp_path: Path
    ) -> None:
        """Found by rendering the real NB04 study: one seed rendered as the
        bare value `0`, and a table row reading `training seeds | 0` says
        *zero seeds* to every reader who has not read the code."""
        document = _document(tmp_path / "doc")
        one = render_protocol(load_study(document, tier="standard"))
        two = render_protocol(
            load_study(document, tier="standard", **{"training.seeds": 2})
        )
        assert "1 seed" in one
        assert "2 seeds" in two

    def test_a_distribution_is_stated_by_its_parameters_and_not_by_its_name(
        self, tmp_path: Path
    ) -> None:
        """Found by rendering the real NB04 study for §1: the row read
        `gaussian` and nothing else.

        That is a family, not a specification — the same objection Annex 04
        §1.2 raises about the problem, and no weaker here. Two studies differing
        only in their process-noise scale are two different experiments, and a
        section rendering them identically is prose drift with extra steps.
        Driven from the failing side: the two fixtures differ in **nothing** a
        name-only row could see.
        """
        quiet = load_study(_document(tmp_path / "a", eval_noise=0.25), tier="standard")
        loud = load_study(_document(tmp_path / "b", eval_noise=0.75), tier="standard")

        one, two = render_protocol(quiet), render_protocol(loud)

        assert one != two
        assert "0.25" in one and "0.75" in two

    def test_the_evaluation_distribution_reports_every_parameter_it_has(
        self, tmp_path: Path
    ) -> None:
        """Compared as a **set** against the sampler's own signature.

        The signature is what `MeasurementID` is derived from, so a parameter
        this row omits is one that can change a number without changing the
        row. Asserting that the noise scale appears would hold for a renderer
        that printed the noise scale and stopped.
        """
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        signature = loaded.study.evaluation.protocol.batch_spec.get_signature()
        expected = {
            name
            for name, value in signature.items()
            if name not in {"type", "state_dim", "horizon", "batch_size"}
            and isinstance(value, int | float)
        }

        text = render_protocol(loaded)
        row = next(
            line for line in text.splitlines() if "evaluation distribution" in line
        )

        assert expected, "the fixture's sampler must carry parameters to report"
        assert all(name in row for name in expected)


class TestProvenance:
    def test_it_names_every_identifier_the_study_resolves_to(
        self, tmp_path: Path
    ) -> None:
        """Compared as a SET against the study's own materialisation. A
        renderer that emitted one identifier, or that emitted the models and
        forgot the measurements, passes any `in` assertion and fails this."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        resolution = resolve(loaded, store=store)

        text = render_provenance(resolution)

        points = loaded.study.materialise()
        for point in points:
            assert str(point.model_id) in text
            assert str(point.measurement_id) in text

    def test_a_spec_valued_axis_names_its_rows(self, tmp_path: Path) -> None:
        """§7's axis column carries the same defect §3 did, and for the same
        reason: it interpolated the value. On a mismatch document that is one
        array repr per row, and §7 has one row per point."""
        document = _with_a_second_instance(tmp_path / "doc", reporting=False)
        store = tmp_path / "store"
        assert (
            main(["--store", str(store), "run", str(document), "--tier", "standard"])
            == 0
        )
        resolution = resolve(load_study(document, tier="standard"), store=store)

        text = render_provenance(resolution)

        assert "problem=nominal" in text and "problem=shifted" in text
        assert "array([" not in text

    def test_every_row_says_which_point_it_is(self, tmp_path: Path) -> None:
        """Found by rendering the real 27-point NB04 study, where eight rows
        all read `standard_pgd` with nothing to separate them.

        §7 exists so a figure can be traced to the models that produced it, and
        a table whose rows are indistinguishable cannot do that. Asserted
        structurally: the swept contender's rows must be pairwise different
        once the identifiers are stripped, which is false for a renderer that
        omits the axis value and true only for one that carries it.
        """
        document = _document(tmp_path / "doc", depths=[2, 4])
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")

        text = render_provenance(resolve(loaded, store=store))

        swept = [
            line.split("|")[1:3]  # contender, then the axis column
            for line in text.splitlines()
            if line.startswith("| `unfolded_a`")
        ]
        assert len(swept) == 2, "the fixture's swept contender has two depths"
        assert swept[0] != swept[1], (
            "two points of one contender render identically; the axis value is "
            "missing, so the table cannot say which model produced which number"
        )
        assert "2" in text and "4" in text

    def test_it_carries_the_study_identifier_and_the_stamp(
        self, tmp_path: Path
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        resolution = resolve(load_study(document, tier="standard"), store=store)

        text = render_provenance(resolution)

        assert str(resolution.loaded.study.study_id) in text
        assert "schema-" in text  # the provenance stamp the producer wrote

    def test_it_moves_when_the_store_does(self, tmp_path: Path) -> None:
        """Differential again, and it is the property that matters: a
        provenance section that did not change when the underlying models did
        would be the exact defect §7 exists to prevent."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        one = render_provenance(
            resolve(load_study(document, tier="standard"), store=store)
        )

        other = tmp_path / "other"
        _produce(_document(tmp_path / "doc2", depths=[3, 5]), other)
        two = render_provenance(
            resolve(
                load_study(
                    _document(tmp_path / "doc2", depths=[3, 5]), tier="standard"
                ),
                store=other,
            )
        )
        assert one != two


class TestProvenanceCompute:
    """Annex 04 §7 asks for the compute, and §7 did not report it.

    Found by the post-phase audit the standing rule requires — the annex laid
    against what the section actually renders. §7 names five things and Phase A
    delivered three: the specification hash, the identifiers and the package
    version (inside the stamp). "Total offline and online compute" was absent,
    and it is the one a thesis chapter is asked for most often.

    Both halves are recorded now (Annex 02 §2.2 gave the online pass a clock),
    which moves the interesting property rather than removing it: a store holds
    records written before that clock existed — 162 of them today — and a study
    is extended as often as it is re-run, so **the partially-timed store is the
    normal case**. The three states below are what the row must keep apart, for
    the same reason `GateStatus` keeps four rather than three.
    """

    def test_the_offline_compute_is_the_sum_the_records_carry(
        self, tmp_path: Path
    ) -> None:
        """Independent recomputation, off the model records rather than off the
        index: `mbl store reindex` exists because the index can be stale, and a
        provenance section that could disagree with the store it describes is
        the one section that must not."""
        from mbl.store.content_store import ModelStore

        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        resolution = resolve(loaded, store=store)

        models = ModelStore(store)
        expected = sum(
            float(models.get(p.model_id).provenance.get("wall_time_s") or 0.0)
            for p in loaded.study.materialise()
        )

        text = render_provenance(resolution)
        row = next(line for line in text.splitlines() if "offline compute" in line)

        assert expected > 0.0, "the fixture must have cost something to train"
        assert f"{expected:.1f}" in row

    def test_the_online_compute_is_the_sum_the_records_carry(
        self, tmp_path: Path
    ) -> None:
        """The half §7 could not report until the evaluation pass was timed."""
        from mbl.store.content_store import MeasurementStore

        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        resolution = resolve(loaded, store=store)

        measurements = MeasurementStore(store)
        points = loaded.study.materialise()
        expected = sum(
            float(measurements.get(p.measurement_id).spec["provenance"]["wall_time_s"])
            for p in points
        )

        row = _row(render_provenance(resolution), "online compute")

        assert expected > 0.0, "the fixture must have cost something to evaluate"
        assert f"{expected:.1f}" in row
        assert f"{len(points)} point(s)" in row

    def test_a_store_written_before_the_clock_existed_says_not_recorded(
        self, tmp_path: Path
    ) -> None:
        """The 162 stored measurements, whose provenance has no `wall_time_s`.

        Omitting the row, or reporting 0.0 s, would make "not measured"
        indistinguishable from "measured and negligible".
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        _untime(store, loaded.study.materialise())

        row = _row(render_provenance(resolve(loaded, store=store)), "online compute")

        assert "not recorded" in row

    def test_a_partially_timed_study_says_which_part(self, tmp_path: Path) -> None:
        """A study is extended as often as it is re-run, so a store with some
        timed points and some untimed ones is the ordinary case. Reporting the
        partial sum as if it covered everything understates the study's cost by
        exactly the part nobody measured."""
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        points = loaded.study.materialise()
        _untime(store, points[:1])

        row = _row(render_provenance(resolve(loaded, store=store)), "online compute")

        assert f"{len(points) - 1} of {len(points)} point(s)" in row


class TestAxisNaming:
    """The one column two axes share, and the absence a study can declare.

    Both defects were invisible until a study swept more than one problem, and
    both produced output that was *plausible* rather than obviously broken: a
    run of interleaved values that reads as one axis, and the word `None` where
    the document declares an absent world.
    """

    def test_two_axes_are_separated_and_named(self, tmp_path: Path) -> None:
        document = _with_a_second_instance(
            tmp_path / "doc", reporting=False, world=True
        )
        loaded = load_study(document, tier="standard")
        paths = {axis.path for axis in loaded.study.sweep}

        text = render_protocol(loaded)

        assert {"evaluation.problem", "training.problem"} <= paths
        row = next(line for line in text.splitlines() if "| `unfolded_a` |" in line)
        assert "evaluation.problem = " in row
        assert "training.problem = " in row

    def test_a_single_axis_keeps_its_leaf_name(self, tmp_path: Path) -> None:
        """The shortest *unique* suffix, so the common case stays short: a
        study with one depth axis says `num_iterations`, not its whole path."""
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")

        text = render_protocol(loaded)

        assert "num_iterations = " in text
        assert "contenders.*.config.num_iterations = " not in text

    def test_a_declared_absence_is_not_rendered_as_a_python_object(
        self, tmp_path: Path
    ) -> None:
        """The world axis's first position is *no world*, which the grammar
        spells as an absence. `None` in a table states an object where the
        study states nothing."""
        document = _with_a_second_instance(
            tmp_path / "doc", reporting=False, world=True
        )

        text = render_protocol(load_study(document, tier="standard"))

        assert "None" not in text
        assert "training.problem = —" in text
