"""Acceptance tests for the rehost declaration (Annex 01 §2.4.1, D24).

Written before the implementation. `EvaluationSpec` gains `rehost` — blind by
default, aware meaning the controller is rebuilt against the *evaluation*
problem — plus per-plant `rehost_overrides` carrying authored construction
literals. Three properties carry the suite:

* **Aware moves every `MeasurementID` and no `ModelID`.** An unsigned mode
  would be worse than inert: `materialise()` deduplicates points on derived
  identifiers, so two points differing only in `rehost` would silently
  collapse into one.
* **Blind signs nothing.** A blind spec's signature must be byte-identical to
  one predating the fields — asserted on the exact key set, not on a hash —
  so the grammar growing orphans no stored measurement.
* **Every refusal names its subject**: an unknown mode, a duplicate override,
  an override in an all-blind document, an override naming a contender no
  declaration carries, and an override whose plant no materialised point
  evaluates on (the gate-firability rule, applied here).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study

_producer = importlib.import_module("tests.runner.test_producer")

NOMINAL, SHIFTED = "problem.npz", "rotated.npz"
MEASUREMENT_LEVEL = "evaluation.problem"

#: The producer fixture's two contenders, by resolved label.
TRAINED, ANALYTIC = "unfolded_a", "baseline"


def _document(
    directory: Path,
    *,
    evaluation_lines: str = "",
    sweep: str | None = None,
    appendix: str = "",
) -> Path:
    """The producer fixture's study, with the evaluation table extended.

    Two frozen problems are written beside it (D19). `evaluation_lines` is
    injected directly under `[evaluation]`; `appendix` is appended at the end
    of the document, which is where TOML puts `[[evaluation.rehost_overrides]]`
    blocks declared after the contenders.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, NOMINAL, seed=0)
    _producer._problem_file(directory, SHIFTED, seed=1)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    marker = "[evaluation]\n"
    assert marker in text, "the producer fixture's evaluation table moved"
    if evaluation_lines:
        text = text.replace(marker, f"{marker}{evaluation_lines}\n", 1)
    if sweep is not None:
        head, found, _ = text.partition("[[sweep]]")
        assert found, "the producer fixture's sweep block moved"
        text = f"{head}[[sweep]]\n{sweep}\n" if sweep else head
    (directory / "study.toml").write_text(text + appendix)
    return directory / "study.toml"


def _study(directory: Path, **kwargs: Any) -> Any:
    return load_study(
        _document(directory, **kwargs), bindings=DEFAULT_SPEC_BINDINGS
    ).study


def _shift_sweep() -> str:
    return (
        f'path = "{MEASUREMENT_LEVEL}"\n'
        f'values = ["{NOMINAL}", "{SHIFTED}"]\n'
        'labels = ["nominal", "rotated"]'
    )


def _override(
    contender: str = ANALYTIC, problem: str = SHIFTED, config: str = "horizon = 6"
) -> str:
    return (
        "\n[[evaluation.rehost_overrides]]\n"
        f'contender = "{contender}"\n'
        f'problem = "{problem}"\n'
        "\n[evaluation.rehost_overrides.config]\n"
        f"{config}\n"
    )


def _ids(study: Any) -> tuple[set[str], set[str]]:
    points = study.materialise()
    return (
        {str(point.model_id) for point in points},
        {str(point.measurement_id) for point in points},
    )


class TestTheDeclarationParses:
    def test_the_default_is_blind(self, tmp_path: Path) -> None:
        study = _study(tmp_path / "doc")
        assert study.evaluation.rehost == "blind"
        assert study.evaluation.rehost_overrides == ()

    def test_aware_is_read_from_the_document(self, tmp_path: Path) -> None:
        study = _study(tmp_path / "doc", evaluation_lines='rehost = "aware"')
        assert study.evaluation.rehost == "aware"

    def test_full_is_read_from_the_document(self, tmp_path: Path) -> None:
        """Annex 01 §2.4.1 as amended 2026-08-09: the doctrine's second half
        joins the closed set as a NEW mode, never a correction to `aware`."""
        study = _study(tmp_path / "doc", evaluation_lines='rehost = "full"')
        assert study.evaluation.rehost == "full"

    def test_an_unknown_mode_is_refused_naming_the_permitted_set(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SpecificationError, match="blind"):
            _study(tmp_path / "doc", evaluation_lines='rehost = "rebuild"')


class TestIdentity:
    """Aware moves every MeasurementID, no ModelID, and the StudyID."""

    def test_aware_moves_every_measurement_id_and_no_model_id(
        self, tmp_path: Path
    ) -> None:
        blind_models, blind_measurements = _ids(_study(tmp_path / "blind"))
        aware_models, aware_measurements = _ids(
            _study(tmp_path / "aware", evaluation_lines='rehost = "aware"')
        )

        # The SAME SET, not merely the same count.
        assert aware_models == blind_models
        assert aware_measurements.isdisjoint(blind_measurements)
        assert len(aware_measurements) == len(blind_measurements)

    def test_the_study_id_moves(self, tmp_path: Path) -> None:
        blind = _study(tmp_path / "blind")
        aware = _study(tmp_path / "aware", evaluation_lines='rehost = "aware"')
        assert str(blind.study_id) != str(aware.study_id)

    def test_full_measurements_are_disjoint_from_blind_and_aware_alike(
        self, tmp_path: Path
    ) -> None:
        """The whole reason `full` is a new mode: the rebuild algorithm is
        not signed into `MeasurementID`, so the corrected analytic law must
        arrive under identifiers no stored record answers to — three modes,
        three disjoint measurement sets, one shared model set."""
        blind_models, blind_measurements = _ids(_study(tmp_path / "blind"))
        aware_models, aware_measurements = _ids(
            _study(tmp_path / "aware", evaluation_lines='rehost = "aware"')
        )
        full_models, full_measurements = _ids(
            _study(tmp_path / "full", evaluation_lines='rehost = "full"')
        )
        assert full_models == blind_models == aware_models
        assert full_measurements.isdisjoint(blind_measurements)
        assert full_measurements.isdisjoint(aware_measurements)
        assert len(full_measurements) == len(blind_measurements)

    def test_a_blind_signature_has_no_rehost_key_at_all(self, tmp_path: Path) -> None:
        """The conditional-signing rule, asserted on the exact key set: a
        blind spec must sign byte-identically to one predating the field, or
        every stored measurement is orphaned by the grammar growing."""
        signature = _study(tmp_path / "doc").evaluation.get_signature()
        assert set(signature) == {"type", "protocol", "ctx", "metrics"}

    def test_an_aware_signature_carries_the_mode_and_the_overrides(
        self, tmp_path: Path
    ) -> None:
        study = _study(
            tmp_path / "doc",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
            appendix=_override(),
        )
        signature = study.evaluation.get_signature()
        assert signature["rehost"] == "aware"
        assert signature["rehost_overrides"], "the override must sign"

    def test_an_override_moves_the_measurement_ids_it_matches(
        self, tmp_path: Path
    ) -> None:
        bare = _study(
            tmp_path / "bare",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
        )
        overridden = _study(
            tmp_path / "overridden",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
            appendix=_override(),
        )
        _, bare_measurements = _ids(bare)
        _, overridden_measurements = _ids(overridden)
        assert bare_measurements != overridden_measurements

    def test_the_override_value_moves_the_measurement_id(self, tmp_path: Path) -> None:
        """Not merely its presence: two documents differing only in the
        replaced VALUE must derive different measurement identifiers, or the
        store would answer a 0.06-step document with a 0.07-step measurement."""
        seven = _study(
            tmp_path / "seven",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
            appendix=_override(config="horizon = 7"),
        )
        eight = _study(
            tmp_path / "eight",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
            appendix=_override(config="horizon = 8"),
        )
        _, seven_measurements = _ids(seven)
        _, eight_measurements = _ids(eight)
        assert seven_measurements != eight_measurements


class TestTheOverrideIsARealDeclaration:
    def test_the_plant_is_resolved_beside_the_document(self, tmp_path: Path) -> None:
        study = _study(
            tmp_path / "doc",
            evaluation_lines='rehost = "aware"',
            sweep=_shift_sweep(),
            appendix=_override(),
        )
        (override,) = study.evaluation.rehost_overrides
        assert override.contender == ANALYTIC
        assert str(override.problem.problem_id)
        assert dict(override.config) == {"horizon": 6}

    def test_a_missing_plant_file_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError, match="absent.npz"):
            _study(
                tmp_path / "doc",
                evaluation_lines='rehost = "aware"',
                sweep=_shift_sweep(),
                appendix=_override(problem="absent.npz"),
            )

    def test_an_unknown_contender_label_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError, match="nobody"):
            _study(
                tmp_path / "doc",
                evaluation_lines='rehost = "aware"',
                sweep=_shift_sweep(),
                appendix=_override(contender="nobody"),
            )

    def test_a_duplicate_pair_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError, match="once"):
            _study(
                tmp_path / "doc",
                evaluation_lines='rehost = "aware"',
                sweep=_shift_sweep(),
                appendix=_override() + _override(),
            )

    def test_an_override_in_an_all_blind_document_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The inert-declaration rule: a blind document never reads its
        overrides, and a declaration nothing reads is refused, not ignored."""
        with pytest.raises(SpecificationError, match="blind"):
            _study(
                tmp_path / "doc",
                sweep=_shift_sweep(),
                appendix=_override(),
            )

    def test_an_override_no_point_evaluates_on_is_refused(self, tmp_path: Path) -> None:
        """The gate-firability rule. Without the shift sweep no point ever
        evaluates on the rotated plant, so the override can never fire."""
        study = _study(
            tmp_path / "doc",
            evaluation_lines='rehost = "aware"',
            appendix=_override(),
        )
        with pytest.raises(SpecificationError, match="rotated.npz|no point"):
            study.materialise()


class TestTheSmokeSubsetAndTheOverride:
    """A tier's truncation must not make a valid document refuse itself.

    The smoke subset keeps an axis's endpoints, so a five-position zipped
    sweep loses its middle positions — including, for Figure 2's shape, the
    one aware position the PGD override is declared for. The truncation is
    the tier's doing, not the author's, so the resolution prunes the dead
    override instead of carrying a study that fails its own firability rule;
    and the surviving labels must stay aligned with the surviving values,
    because their join is position.
    """

    def _five_position_document(self, directory: Path) -> Path:
        sweep = (
            'path = "evaluation.problem"\n'
            f'values = ["{NOMINAL}", "{SHIFTED}", "{SHIFTED}", "{NOMINAL}", "{SHIFTED}"]\n'
            'labels = ["nominal", "s-blind", "s-aware", "n-again", "s-last"]'
        )
        second = (
            '\n[[sweep]]\npath = "evaluation.rehost"\n'
            'values = ["blind", "blind", "aware", "aware", "blind"]\n'
            'compose = "zip"\n'
        )
        return _document(
            directory,
            sweep=sweep,
            appendix=second + _override(),
        )

    def test_the_truncation_prunes_the_override_it_killed(self, tmp_path: Path) -> None:
        from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE

        document = load_study(
            self._five_position_document(tmp_path / "doc"),
            bindings=DEFAULT_SPEC_BINDINGS,
        )
        # The full document fires the override (position 2 is shifted+aware).
        assert document.study.materialise()
        assert document.study.evaluation.rehost_overrides

        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "smoke").study
        # Endpoints kept: positions 0 (nominal, blind) and 4 (shifted, blind)
        # — no aware position survives, so the override is pruned and the
        # truncated study still materialises.
        assert resolved.evaluation.rehost_overrides == ()
        assert resolved.materialise()

    def test_the_truncation_keeps_labels_aligned_with_values(
        self, tmp_path: Path
    ) -> None:
        from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE

        document = load_study(
            self._five_position_document(tmp_path / "doc"),
            bindings=DEFAULT_SPEC_BINDINGS,
        )
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "smoke").study
        axis = resolved.sweep[0]
        assert len(axis.values) == 2
        # The first and last labels, because the first and last values were
        # kept; before the fix the labels were dropped altogether, silently
        # renaming every surviving category.
        assert axis.labels == ("nominal", "s-last")

    def test_a_tier_that_truncates_nothing_prunes_nothing(self, tmp_path: Path) -> None:
        from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE

        document = load_study(
            self._five_position_document(tmp_path / "doc"),
            bindings=DEFAULT_SPEC_BINDINGS,
        )
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "publication").study
        assert len(resolved.evaluation.rehost_overrides) == 1
        assert resolved.sweep[0].labels[2] == "s-aware"


class TestTheModeIsAnOrdinaryScalarPath:
    """`evaluation.rehost` sweeps like any scalar path, which is what makes
    Annex 01 §2.5's blind/aware zip illustration expressible in one study."""

    def test_sweeping_it_pairs_blind_and_aware_in_one_study(
        self, tmp_path: Path
    ) -> None:
        # The baseline carries NO axis at all: the swept document replaces the
        # fixture's depth axis, so comparing against the depth-swept study
        # would measure the removed axis rather than the added one.
        blind_models, blind_measurements = _ids(_study(tmp_path / "base", sweep=""))
        swept = _study(
            tmp_path / "swept",
            sweep='path = "evaluation.rehost"\nvalues = ["blind", "aware"]',
        )
        models, measurements = _ids(swept)

        assert models == blind_models
        assert len(measurements) == 2 * len(blind_measurements)
        assert blind_measurements < measurements
