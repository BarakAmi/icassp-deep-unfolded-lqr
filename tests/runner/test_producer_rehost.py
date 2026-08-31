"""Acceptance tests for the producer's aware rehost (Annex 01 §2.4.1, D24).

Under the corrected semantics (2026-08-08): an aware point rebuilds its
controller with the evaluation plant in its ONLINE expressions only — offline
artifacts (Riccati stacks, the declared step, every learned parameter) stay
frozen from the plant the model was told, a recipe that trains loads its
stored tensors into the rebuilt scaffold, and a declared per-plant override
reaches the rebuild instead of being clobbered by the stored constant tensor.
Every claim below is an equality or an inequality of published metrics,
because that is the surface a paper reads.

The comparisons lean on one fact about this suite's determinism: two studies
that resolve to the same controller construction, the same weights and the
same evaluation batches publish bit-identical metrics, so "equals" is exact
and "differs" is the anti-vacuity control.
"""

from __future__ import annotations

import json

import importlib
from pathlib import Path
from typing import Any

import pytest

from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import RunReport, run_study
from mbl.spec.loader import load_study
from mbl.store.index import StoreIndex

_producer = importlib.import_module("tests.runner.test_producer")

NOMINAL_SEED, SHIFTED_SEED = 0, 1

#: An analytic contender with stored-but-untrained weights: the family whose
#: rehost must NOT load them (the constant-step clobber, plan §5).
PGD = """
[[contenders]]
label = "pgd"
family = "unfolded_fixed"
role = "baseline"

[contenders.config]
num_iterations = 3
step_size_init = {pgd_step}
step_size_max = 1.0
horizon = 6
"""

#: The contender with no channel for the plant at all.
NEURAL = """
[[contenders]]
label = "neural"
family = "neural"

[contenders.config]
hidden_dim = 8
model_type = "GRU"
"""

OVERRIDE = """
[[evaluation.rehost_overrides]]
contender = "pgd"
problem = "rotated.npz"

[evaluation.rehost_overrides.config]
step_size_init = {step}
"""


def _document(
    directory: Path,
    name: str = "study.toml",
    *,
    problem_seed: int = NOMINAL_SEED,
    evaluation_lines: str = "",
    appendix: str = "",
    pgd_step: float = 0.1,
) -> Path:
    """The producer fixture's study, extended for the rehost comparisons.

    `problem_seed` decides which plant `problem.npz` *is*, so a study "trained
    on the rotated plant" needs no template surgery; `rotated.npz` is always
    written beside the document for shifted evaluations and overrides.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, "problem.npz", seed=problem_seed)
    _producer._problem_file(directory, "rotated.npz", seed=SHIFTED_SEED)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    marker = "[evaluation]\n"
    assert marker in text, "the producer fixture's evaluation table moved"
    if evaluation_lines:
        text = text.replace(marker, f"{marker}{evaluation_lines}\n", 1)
    text += PGD.format(pgd_step=pgd_step)
    path = directory / name
    path.write_text(text + appendix)
    return path


def _run(document: Path, store: Path) -> RunReport:
    study = load_study(document, bindings=DEFAULT_SPEC_BINDINGS).study
    return run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)


def _by_point(report: RunReport) -> dict[tuple[str, int, str], dict[str, float]]:
    """Metrics keyed by (label, seed, model_id) — same-store comparisons."""
    return {
        (o.label, o.seed, str(o.model_id)): dict(o.metrics) for o in report.outcomes
    }


def _by_label_seed(report: RunReport, label: str) -> dict[int, dict[str, float]]:
    """One contender's metrics keyed by seed — cross-store comparisons, valid
    only for contenders the depth axis does not multiply."""
    picked = {}
    for outcome in report.outcomes:
        if outcome.label == label:
            assert outcome.seed not in picked, "depth-swept label; key ambiguous"
            picked[outcome.seed] = dict(outcome.metrics)
    assert picked, f"no outcomes for {label!r}"
    return picked


class TestTheNoOpAnchor:
    def test_aware_on_the_training_problem_reproduces_blind_bit_for_bit(
        self, tmp_path: Path
    ) -> None:
        """With the evaluation problem equal to the training problem the
        rebuild is a semantic no-op, and the metrics must say so exactly —
        for the trained, the weightless-analytic and the constant-weight
        families alike."""
        directory, store = tmp_path / "docs", tmp_path / "store"
        blind = _run(_document(directory, "blind.toml"), store)
        aware = _run(
            _document(directory, "aware.toml", evaluation_lines='rehost = "aware"'),
            store,
        )

        assert not aware.trained, "an aware re-run of a stored study trains"
        assert _by_point(aware) == _by_point(blind)


class TestTheFullMode:
    """Annex 01 §2.4.1 as amended 2026-08-09: one doctrine with per-family
    consequences. A family that trains nothing has no offline stage to
    freeze — under `full` it re-synthesises whole against the plant in hand;
    a trainable family rebuilds by the identical algorithm as `aware`."""

    def test_full_on_the_training_problem_reproduces_blind_bit_for_bit(
        self, tmp_path: Path
    ) -> None:
        """The no-op anchor, and the reason the analytic re-synthesis runs
        through the SYNTHESIZER path: `build_controller` ignores the compute
        context and its gains differ by one ULP, which this equality would
        catch on the analytic rows."""
        directory, store = tmp_path / "docs", tmp_path / "store"
        blind = _run(_document(directory, "blind.toml"), store)
        full = _run(
            _document(directory, "full.toml", evaluation_lines='rehost = "full"'),
            store,
        )

        assert not full.trained, "a full re-run of a stored study trains"
        assert _by_point(full) == _by_point(blind)

    def test_a_trainable_family_rebuilds_exactly_as_aware(self, tmp_path: Path) -> None:
        """Deliberately the same code path today; this equality is what turns
        a future divergence into a conscious decision rather than drift."""
        shifted = 'problem = { path = "rotated.npz" }\n'
        aware = _run(
            _document(
                tmp_path / "aware", evaluation_lines=shifted + 'rehost = "aware"'
            ),
            tmp_path / "store_aware",
        )
        full = _run(
            _document(tmp_path / "full", evaluation_lines=shifted + 'rehost = "full"'),
            tmp_path / "store_full",
        )
        trainable = {"unfolded_a", "neural"}
        assert {
            key: value for key, value in _by_point(full).items() if key[0] in trainable
        } == {
            key: value for key, value in _by_point(aware).items() if key[0] in trainable
        }

    def test_an_analytic_family_re_synthesises_whole_against_the_plant_in_hand(
        self, tmp_path: Path
    ) -> None:
        """The defining property: an analytic contender under `full` equals a
        study genuinely declared on the shifted plant — the equality that was
        the OLD aware semantics' defect is this mode's contract — and differs
        from `aware` (whose offline stays frozen), which is the anti-vacuity
        control."""
        shifted = 'problem = { path = "rotated.npz" }\n'
        full = _run(
            _document(tmp_path / "full", evaluation_lines=shifted + 'rehost = "full"'),
            tmp_path / "store_full",
        )
        native = _run(
            _document(tmp_path / "native", problem_seed=SHIFTED_SEED),
            tmp_path / "store_native",
        )
        aware = _run(
            _document(
                tmp_path / "aware", evaluation_lines=shifted + 'rehost = "aware"'
            ),
            tmp_path / "store_aware",
        )
        for label in ("baseline", "pgd"):
            assert _by_label_seed(full, label) == _by_label_seed(native, label), label
            assert _by_label_seed(full, label) != _by_label_seed(aware, label), label

    def test_a_declared_override_reaches_the_full_re_synthesis(
        self, tmp_path: Path
    ) -> None:
        """§2.4.1: 'including its per-plant construction literals' — the
        override must move the analytic full cell, or the PGD would re-derive
        on the rotated plant with the nominal plant's declared step."""
        shifted = 'problem = { path = "rotated.npz" }\nrehost = "full"'
        plain = _run(
            _document(tmp_path / "plain", evaluation_lines=shifted),
            tmp_path / "store_plain",
        )
        overridden = _run(
            _document(
                tmp_path / "overridden",
                evaluation_lines=shifted,
                appendix=OVERRIDE.format(step=0.05),
            ),
            tmp_path / "store_overridden",
        )
        assert _by_label_seed(overridden, "pgd") != _by_label_seed(plain, "pgd")


class TestTheShiftedRebuild:
    def test_analytic_families_freeze_offline_and_use_the_plant_online(
        self, tmp_path: Path
    ) -> None:
        """The corrected semantics (Annex 01 §2.4.1, 2026-08-08): an informed
        analytic contender keeps its OFFLINE artifacts — the Riccati stack,
        the declared step — frozen from the told plant and uses the given
        plant only in its online expressions. So it must differ from blind
        (the rebuild happened) AND from a study genuinely declared on the
        shifted plant (no second recursion ran — the old semantics' equality,
        now the anti-vacuity control), and the truncated-Riccati rebuild must
        reproduce an independent NumPy frozen-P/live-gain construction."""
        aware = _run(
            _document(
                tmp_path / "aware",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
            ),
            tmp_path / "store_aware",
        )
        blind = _run(
            _document(
                tmp_path / "blind",
                evaluation_lines='problem = { path = "rotated.npz" }',
            ),
            tmp_path / "store_blind",
        )
        native = _run(
            _document(tmp_path / "native", problem_seed=SHIFTED_SEED),
            tmp_path / "store_native",
        )

        for label in ("baseline", "pgd"):
            assert _by_label_seed(aware, label) != _by_label_seed(native, label), label
            assert _by_label_seed(aware, label) != _by_label_seed(blind, label), label

        # The independent recomputation, on the plain NumPy path: P from the
        # nominal recursion, gains re-formed per step with the rotated
        # matrices, rolled out on the rotated plant over the study's own
        # evaluation batches.
        import numpy as np

        from mbl.core.kernels.riccati import (
            gains_for_cost_to_go,
            riccati_recursion,
        )
        from mbl.experiments.evaluation import evaluate_synthesized_controller
        from mbl.models.analytic.truncated_riccati import (
            SynthesizedTruncatedRiccatiController,
        )
        from mbl.spec.problem import ProblemSpec

        study = load_study(
            _document(
                tmp_path / "probe",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
            ),
            bindings=DEFAULT_SPEC_BINDINGS,
        ).study
        nominal = study.problem.build()
        rotated = ProblemSpec.load(tmp_path / "probe" / "rotated.npz").build()
        horizon = _producer.HORIZON

        def stacks(problem: Any) -> tuple[np.ndarray, np.ndarray]:
            system = problem.system
            return (
                np.stack([system.A_t[k] for k in range(horizon)]),
                np.stack([system.B_t[k] for k in range(horizon)]),
            )

        A_off, B_off = stacks(nominal)
        A_on, B_on = stacks(rotated)
        P_arr, _ = riccati_recursion(
            A_off,
            B_off,
            np.asarray(nominal.cost.Q),
            np.asarray(nominal.cost.R),
            horizon,
        )
        K_arr = gains_for_cost_to_go(
            P_arr, A_on, B_on, np.asarray(rotated.cost.R), horizon
        )
        # The rehosted problem's box, asserted rather than indexed blind: a
        # constrained problem that arrived here unconstrained would otherwise
        # fail inside the controller with a message about `None`, and this
        # probe's whole subject is which problem supplied what.
        assert rotated.constraints, "the rehosted problem carries no constraint"
        expected = SynthesizedTruncatedRiccatiController(
            P_arr=P_arr,
            K_arr=K_arr,
            constraint=rotated.constraints[0],
            context=study.training.ctx,
            synthesizer_signature={"type": "probe"},
        )
        protocol = _producer._protocol(study)
        metrics, _ = evaluate_synthesized_controller(
            expected, rotated, protocol.build_batches(study.evaluation.ctx)
        )
        measured = _by_label_seed(aware, "baseline")
        for seed_metrics in measured.values():
            assert seed_metrics["eval_expected_cost"] == pytest.approx(
                metrics["eval_expected_cost"], rel=1e-9
            )

    def test_the_trained_transplant_is_neither_blind_nor_retrained(
        self, tmp_path: Path
    ) -> None:
        """Three constructions of the unfolded contender, pairwise distinct:
        nominal scaffold + nominal weights (blind), shifted scaffold + nominal
        weights (aware — the transplant), shifted scaffold + shifted-trained
        weights (native). If aware equalled either neighbour, either the
        scaffold did not move or the weights did not load."""
        aware = _run(
            _document(
                tmp_path / "aware",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
            ),
            tmp_path / "store_aware",
        )
        blind = _run(
            _document(
                tmp_path / "blind",
                evaluation_lines='problem = { path = "rotated.npz" }',
            ),
            tmp_path / "store_blind",
        )
        native = _run(
            _document(tmp_path / "native", problem_seed=SHIFTED_SEED),
            tmp_path / "store_native",
        )

        def unfolded(report: RunReport) -> dict[tuple[str, int], Any]:
            return {
                (o.label, o.seed): dict(o.metrics)
                for o in report.outcomes
                if o.label == "unfolded_a"
            }

        assert unfolded(aware) != unfolded(blind)
        assert unfolded(aware) != unfolded(native)
        assert unfolded(blind) != unfolded(native)


class TestTheOverride:
    def test_a_declared_override_reaches_the_rebuilt_controller(
        self, tmp_path: Path
    ) -> None:
        """The override must change the informed score (it reached the
        rebuild — the stored constant-step tensor did not clobber it), and
        under the corrected semantics it must NOT reproduce a study genuinely
        declared at that step on the shifted plant, whose Riccati stack is
        the shifted plant's own rather than the frozen nominal one."""
        overridden = _run(
            _document(
                tmp_path / "overridden",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
                appendix=OVERRIDE.format(step=0.07),
            ),
            tmp_path / "store_overridden",
        )
        declared = _run(
            _document(
                tmp_path / "declared",
                problem_seed=SHIFTED_SEED,
                pgd_step=0.07,
            ),
            tmp_path / "store_declared",
        )
        bare = _run(
            _document(
                tmp_path / "bare",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
            ),
            tmp_path / "store_bare",
        )

        assert _by_label_seed(overridden, "pgd") != _by_label_seed(bare, "pgd")
        assert _by_label_seed(overridden, "pgd") != _by_label_seed(declared, "pgd")

    def test_an_override_applies_only_to_its_own_plant(self, tmp_path: Path) -> None:
        """Figure 2's own shape: a swept evaluation problem, aware everywhere,
        the override declared for the rotated plant alone. The nominal
        category must score exactly the blind nominal PGD (the no-op anchor,
        untouched by the override), and the rotated category must differ from
        the same sweep run without the override — the override applied there
        and only there."""
        sweep_block = (
            "\n[[sweep]]\n"
            'path = "evaluation.problem"\n'
            'values = ["problem.npz", "rotated.npz"]\n'
            'labels = ["nominal", "rotated"]\n'
        )
        swept = _run(
            _document(
                tmp_path / "swept",
                evaluation_lines='rehost = "aware"',
                appendix=sweep_block + OVERRIDE.format(step=0.07),
            ),
            tmp_path / "store_swept",
        )
        unoverridden = _run(
            _document(
                tmp_path / "unoverridden",
                evaluation_lines='rehost = "aware"',
                appendix=sweep_block,
            ),
            tmp_path / "store_unoverridden",
        )
        nominal = _run(
            _document(tmp_path / "nominal"),
            tmp_path / "store_nominal",
        )

        def per_seed_sets(report: RunReport) -> dict[int, set[Any]]:
            grouped: dict[int, set[Any]] = {}
            for outcome in report.outcomes:
                if outcome.label == "pgd":
                    grouped.setdefault(outcome.seed, set()).add(
                        frozenset(outcome.metrics.items())
                    )
            return grouped

        swept_sets = per_seed_sets(swept)
        bare_sets = per_seed_sets(unoverridden)
        for seed, nominal_metrics in _by_label_seed(nominal, "pgd").items():
            # The nominal category is untouched by the override in both runs.
            assert frozenset(nominal_metrics.items()) in swept_sets[seed]
            assert frozenset(nominal_metrics.items()) in bare_sets[seed]
        # The rotated categories differ between the two runs — the override
        # applied on its own plant and nowhere else.
        assert swept_sets != bare_sets


class TestNothingPublishes:
    def test_an_aware_run_adds_no_model_and_reuses_its_measurements(
        self, tmp_path: Path
    ) -> None:
        directory, store = tmp_path / "docs", tmp_path / "store"
        _run(_document(directory, "blind.toml"), store)
        before = _producer._record_digests(store)
        models_before = {k: v for k, v in before.items() if k.startswith("models/")}

        aware_document = _document(
            directory,
            "aware.toml",
            evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
        )
        first = _run(aware_document, store)
        after = _producer._record_digests(store)
        models_after = {k: v for k, v in after.items() if k.startswith("models/")}

        assert models_after == models_before, "an aware rebuild published a model"
        assert not first.trained
        assert first.evaluated, "the first aware run must actually roll out"

        second = _run(aware_document, store)
        assert not second.trained
        assert not second.evaluated, "the second aware run must reuse everything"


class TestTheContenderWithNoChannel:
    def test_the_gru_scores_identically_aware_and_blind(self, tmp_path: Path) -> None:
        """The recurrent baseline consumes no plant matrices at construction,
        so on the same shifted plant its aware and blind scores must be
        bit-identical — the 'no channel for the new knowledge' result, held
        as an equality rather than assumed."""
        directory, store = tmp_path / "docs", tmp_path / "store"
        blind = _run(
            _document(
                directory,
                "blind.toml",
                evaluation_lines='problem = { path = "rotated.npz" }',
                appendix=NEURAL,
            ),
            store,
        )
        aware = _run(
            _document(
                directory,
                "aware.toml",
                evaluation_lines='problem = { path = "rotated.npz" }\nrehost = "aware"',
                appendix=NEURAL,
            ),
            store,
        )

        assert not aware.trained
        assert _by_label_seed(aware, "neural") == _by_label_seed(blind, "neural")


class TestTheIndexCarriesWhatTheDiskCarries:
    """A run and a rebuild must produce the same index.

    Found by an audit rather than by a test (2026-08-13): the producer wrote
    `spec_json="{}"` for every measurement while `rebuild_from` wrote the real
    spec, so a query for a measurement's `rehost` came back empty **on every
    row** — a legal answer, silently wrong. It cost two mistaken conclusions in
    one sitting, first "no measurement is informed" and then "the informed
    cells never ran". Nothing inside this package reads that column, which is
    exactly why it survived; the model path has always written its spec on
    both routes.
    """

    def test_a_measurement_row_carries_its_spec(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        report = _run(_document(tmp_path, evaluation_lines='rehost = "full"'), store)
        assert report.outcomes
        index = StoreIndex(store)
        rows = [
            row
            for outcome in report.outcomes
            for row in index.measurements(model_id=outcome.model_id)
        ]
        assert rows, "the run published no measurement to the index"
        for row in rows:
            spec = json.loads(row.spec_json or "{}")
            assert spec, f"measurement {row.measurement_id} indexed with an empty spec"
            assert spec["model"] == str(row.model_id)
            assert spec["eval_problem"] == str(row.eval_problem_id)

    def test_the_declared_rehost_survives_into_the_index(self, tmp_path: Path) -> None:
        """The field the audit actually needed. `blind` signs nothing and is
        absent by design, so the assertion is on the mode that IS recorded."""
        store = tmp_path / "store"
        report = _run(_document(tmp_path, evaluation_lines='rehost = "full"'), store)
        index = StoreIndex(store)
        modes = {
            json.loads(row.spec_json or "{}").get("eval_protocol", {}).get("rehost")
            for outcome in report.outcomes
            for row in index.measurements(model_id=outcome.model_id)
        }
        assert modes == {"full"}
