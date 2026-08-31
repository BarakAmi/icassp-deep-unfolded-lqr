"""Acceptance for the `step_is_inverse_lipschitz` gate (Figure-3 plan Phase 2).

The campaign plan's finding: the analytic PGD's declared literal exceeded the
classical stability limit 2/L on four of the seven frozen plants, the box hid
the divergence completely, and a depth-scaling figure whose analytic baseline
worsens with depth is not measuring depth. The surface that survived
measurement is a frozen literal plus a gate that recomputes L from the study's
own problem and refuses a document whose declared step is not exactly 1/L —
a step computed at build time would make `ModelID` a function of which LAPACK
path ran (`eigvalsh`, `eigvals`, `norm`, `svd` and the closed 2x2 form
disagree by exactly one ULP on the n = 4 plant).

Written before the implementation, in the idiom of `test_spec_valued_axis`.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.problem import ProblemSpec

_producer = importlib.import_module("tests.runner.test_producer")

REPO = Path(__file__).resolve().parents[2]
FROZEN_N4 = REPO / "studies" / "icassp" / "icassp_n4m2_N100_u0p1_s0.npz"

#: The campaign's authored literals for the frozen n = 4 plant
#: (`tools/report_pgd_step.py`, quoted in fig1/fig2 documents and the plan).
FROZEN_N4_L = 17.535383914972115
FROZEN_N4_STEP = 0.05702755097059363


def _gate_block(*labels: str) -> str:
    named = ", ".join(f'"{label}"' for label in labels)
    return f'\n[[gates]]\nkind = "step_is_inverse_lipschitz"\ncontenders = [{named}]\n'


def _document(
    directory: Path, *, step: float, gate: str, sweep: str | None = None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, "problem.npz", seed=0)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    text = text.replace("step_size_init = 0.1", f"step_size_init = {step!r}")
    if sweep is not None:
        head, marker, _ = text.partition("[[sweep]]")
        assert marker, "the producer fixture's sweep block moved"
        text = f"{head}{sweep}"
    path = directory / "study.toml"
    path.write_text(text + gate)
    return path


def _load(path: Path) -> Any:
    return load_study(path, bindings=DEFAULT_SPEC_BINDINGS).study


def _materialise(path: Path) -> Any:
    return _load(path).materialise()


def _fixture_step(directory: Path) -> float:
    """The fixture plant's exact 1/L, through the gate's own kernel."""
    from mbl.spec.gates import inverse_lipschitz_step

    directory.mkdir(parents=True, exist_ok=True)
    plant = _producer._problem_file(directory, "problem.npz", seed=0)
    return inverse_lipschitz_step(ProblemSpec.load(plant))


class TestTheKernelIsTheAuthoringPath:
    def test_the_frozen_plant_reproduces_the_campaign_literals(self) -> None:
        """The gate's kernel must land on the exact floats the documents
        declare, or every tracked study would be refused by its own gate."""
        from mbl.spec.gates import inverse_lipschitz_constant, inverse_lipschitz_step

        spec = ProblemSpec.load(FROZEN_N4)
        assert inverse_lipschitz_constant(spec) == FROZEN_N4_L
        assert inverse_lipschitz_step(spec) == FROZEN_N4_STEP

    def test_the_models_wrapper_still_agrees_bit_for_bit(self) -> None:
        """`models/analytic` delegates to the core kernel; the two paths must
        be one computation, not two that happen to agree today."""
        from mbl.models.analytic import problem_gradient_lipschitz_constant
        from mbl.spec.gates import inverse_lipschitz_constant

        spec = ProblemSpec.load(FROZEN_N4)
        assert inverse_lipschitz_constant(spec) == problem_gradient_lipschitz_constant(
            spec.build()
        )


class TestTheGateDecides:
    def test_the_exact_literal_passes(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        points = _materialise(
            _document(tmp_path / "doc", step=step, gate=_gate_block("unfolded_a"))
        )
        assert points

    def test_one_ulp_off_is_refused_quoting_the_expected_literal(
        self, tmp_path: Path
    ) -> None:
        step = _fixture_step(tmp_path / "probe")
        wrong = float(np.nextafter(step, 1.0))
        with pytest.raises(SpecificationError) as caught:
            _materialise(
                _document(tmp_path / "doc", step=wrong, gate=_gate_block("unfolded_a"))
            )
        assert repr(step) in str(caught.value)
        assert "unfolded_a" in str(caught.value)

    def test_the_historical_default_is_refused(self, tmp_path: Path) -> None:
        """0.05 is the literal every earlier study declared, and the one the
        campaign measured limit-cycling above 2/L; the gate exists for it."""
        with pytest.raises(SpecificationError):
            _materialise(
                _document(tmp_path / "doc", step=0.05, gate=_gate_block("unfolded_a"))
            )


class TestTheRefusalsNameTheirSubject:
    def test_an_unknown_contender_label_is_refused(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError, match="no_such_contender"):
            _load(
                _document(
                    tmp_path / "doc", step=step, gate=_gate_block("no_such_contender")
                )
            )

    def test_a_contender_without_the_field_is_refused(self, tmp_path: Path) -> None:
        """The inert-declaration rule: a gate naming a contender it cannot
        check is a check the study file claims and nobody performs."""
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError, match="baseline"):
            _load(
                _document(
                    tmp_path / "doc",
                    step=step,
                    gate=_gate_block("unfolded_a", "baseline"),
                )
            )

    def test_an_empty_contender_list_is_refused(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError):
            _load(_document(tmp_path / "doc", step=step, gate=_gate_block()))


class TestTheZipPairsPlantWithLiteral:
    """The Figure-3 shape: the step literal rides the plant axis as its zipped
    companion, and the gate must evaluate the PAIRS AS MATERIALISED — the
    plan names reading the document instead of the positions as the gate's
    blind spot."""

    def _two_plant_document(self, directory: Path, steps: tuple[float, float]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        _producer._problem_file(directory, "problem.npz", seed=0)
        _producer._problem_file(directory, "second.npz", seed=1)
        sweep = (
            "[[sweep]]\n"
            'path = "problem"\n'
            'values = ["problem.npz", "second.npz"]\n'
            'labels = ["first", "second"]\n'
            "\n"
            "[[sweep]]\n"
            'path = "contenders.*.config.step_size_init"\n'
            f"values = [{steps[0]!r}, {steps[1]!r}]\n"
            'applies_to = ["unfolded_a"]\n'
            'compose = "zip"\n'
        )
        return _document(
            directory,
            step=0.1,  # the contender block's literal; the axis overrides it
            gate=_gate_block("unfolded_a"),
            sweep=sweep,
        )

    def _steps(self, directory: Path) -> tuple[float, float]:
        from mbl.spec.gates import inverse_lipschitz_step

        directory.mkdir(parents=True, exist_ok=True)
        first = _producer._problem_file(directory, "problem.npz", seed=0)
        second = _producer._problem_file(directory, "second.npz", seed=1)
        return (
            inverse_lipschitz_step(ProblemSpec.load(first)),
            inverse_lipschitz_step(ProblemSpec.load(second)),
        )

    def test_the_correct_pairing_passes(self, tmp_path: Path) -> None:
        steps = self._steps(tmp_path / "probe")
        assert steps[0] != steps[1], "two plants with one L prove nothing"
        points = _materialise(self._two_plant_document(tmp_path / "doc", steps))
        assert points

    def test_the_swapped_pairing_is_refused(self, tmp_path: Path) -> None:
        first, second = self._steps(tmp_path / "probe")
        with pytest.raises(SpecificationError):
            _materialise(self._two_plant_document(tmp_path / "doc", (second, first)))


def _fraction_gate_block(fraction: str, *labels: str) -> str:
    named = ", ".join(f'"{label}"' for label in labels)
    return (
        f'\n[[gates]]\nkind = "step_is_inverse_lipschitz"\n'
        f"contenders = [{named}]\nfraction = {fraction}\n"
    )


class TestTheFractionParameterisesTheLaw:
    """Annex 01 §4 (2026-08-09): the advisors' half-step law reaches the gate
    as `fraction = 0.5`, exact and conditionally signed."""

    def test_the_halved_literal_passes_at_fraction_half(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        halved = step / 2
        assert halved * 2 == step, "halving must be exact for the law to be"
        points = _materialise(
            _document(
                tmp_path / "doc",
                step=halved,
                gate=_fraction_gate_block("0.5", "unfolded_a"),
            )
        )
        assert points

    def test_the_full_literal_is_refused_at_fraction_half(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError, match=re.escape(repr(step / 2))):
            _materialise(
                _document(
                    tmp_path / "doc",
                    step=step,
                    gate=_fraction_gate_block("0.5", "unfolded_a"),
                )
            )

    def test_one_ulp_off_the_halved_literal_is_refused(self, tmp_path: Path) -> None:
        """The plan's trap: the gate multiplies the kernel's value by the
        fraction and never re-derives 1/(2L), whose last ULP can differ --
        so a literal one ULP off the exact product must refuse."""
        step = _fixture_step(tmp_path / "probe")
        wrong = float(np.nextafter(step / 2, 1.0))
        with pytest.raises(SpecificationError, match="step_is_inverse_lipschitz"):
            _materialise(
                _document(
                    tmp_path / "doc",
                    step=wrong,
                    gate=_fraction_gate_block("0.5", "unfolded_a"),
                )
            )

    def test_a_fraction_outside_the_closed_set_is_refused(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError, match="fraction"):
            _load(
                _document(
                    tmp_path / "doc",
                    step=step,
                    gate=_fraction_gate_block("0.25", "unfolded_a"),
                )
            )

    @pytest.mark.parametrize("multiple", [2.0, 4.0, 8.0])
    def test_the_multiples_above_one_pass_on_their_own_exact_literal(
        self, tmp_path: Path, multiple: float
    ) -> None:
        """Annex 01 §4 as widened 2026-08-12: a study may declare a step ABOVE
        `1/L` — `4.0` is the classical limit `2/L` itself — so that a sweep can
        ask where stability ends without giving up the literal check. Exactness
        is what makes it admissible, and it is asserted rather than assumed."""
        step = _fixture_step(tmp_path / "probe")
        scaled = step * multiple
        assert scaled / multiple == step, "the scaling must be exact for the law"
        points = _materialise(
            _document(
                tmp_path / f"doc{multiple:g}",
                step=scaled,
                gate=_fraction_gate_block(f"{multiple:g}", "unfolded_a"),
            )
        )
        assert points

    @pytest.mark.parametrize("multiple", [2.0, 4.0, 8.0])
    def test_one_ulp_off_a_widened_multiple_is_still_refused(
        self, tmp_path: Path, multiple: float
    ) -> None:
        """Widening the set must not widen the tolerance: the bit-equality law
        holds at every member, or the new ones are a hole rather than a gate."""
        step = _fixture_step(tmp_path / "probe")
        wrong = float(np.nextafter(step * multiple, 0.0))
        with pytest.raises(SpecificationError, match="step_is_inverse_lipschitz"):
            _materialise(
                _document(
                    tmp_path / f"doc{multiple:g}",
                    step=wrong,
                    gate=_fraction_gate_block(f"{multiple:g}", "unfolded_a"),
                )
            )

    def test_the_undeclared_overshoot_is_still_refused(self, tmp_path: Path) -> None:
        """The failure this gate exists for is UNDECLARED: the campaign's own
        0.05 sitting 5-24x above 2/L. Widening the set admits a step above 1/L
        only when the document SAYS so — declaring `fraction = 1.0` and writing
        a doubled literal must still refuse."""
        step = _fixture_step(tmp_path / "probe")
        with pytest.raises(SpecificationError, match=re.escape(repr(step))):
            _materialise(
                _document(
                    tmp_path / "doc",
                    step=step * 2.0,
                    gate=_fraction_gate_block("1.0", "unfolded_a"),
                )
            )

    def test_an_explicit_default_fraction_signs_nothing(self, tmp_path: Path) -> None:
        """`fraction = 1.0` and no `fraction` are one claim; two spellings
        deriving two StudyIDs is exactly what the schema's canonicalisation
        exists to prevent (conditional signing, the D24 pattern)."""
        step = _fixture_step(tmp_path / "probe")
        absent = _load(
            _document(tmp_path / "absent", step=step, gate=_gate_block("unfolded_a"))
        )
        explicit = _load(
            _document(
                tmp_path / "explicit",
                step=step,
                gate=_fraction_gate_block("1.0", "unfolded_a"),
            )
        )
        assert str(absent.study_id) == str(explicit.study_id)

    def test_an_off_default_fraction_moves_the_study_and_no_model(
        self, tmp_path: Path
    ) -> None:
        step = _fixture_step(tmp_path / "probe")
        full = _load(
            _document(tmp_path / "full", step=step, gate=_gate_block("unfolded_a"))
        )
        halved = _load(
            _document(
                tmp_path / "halved",
                step=step / 2,
                gate=_fraction_gate_block("0.5", "unfolded_a"),
            )
        )
        assert str(full.study_id) != str(halved.study_id)
        # The step literal itself moves the stepper's ModelID (it is config);
        # the FRACTION must not reach any model beyond that -- asserted on a
        # contender the gate does not name, whose config is untouched.
        untouched = {
            str(p.model_id)
            for p in full.materialise()
            if p.contender.resolved_label != "unfolded_a"
        }
        assert untouched == {
            str(p.model_id)
            for p in halved.materialise()
            if p.contender.resolved_label != "unfolded_a"
        }


class TestTheGateSignsTheStudyAndNothingElse:
    def test_study_id_moves_and_no_model_id_does(self, tmp_path: Path) -> None:
        step = _fixture_step(tmp_path / "probe")
        without = _load(_document(tmp_path / "without", step=step, gate=""))
        with_gate = _load(
            _document(tmp_path / "with", step=step, gate=_gate_block("unfolded_a"))
        )

        assert str(without.study_id) != str(with_gate.study_id)
        assert {str(p.model_id) for p in without.materialise()} == {
            str(p.model_id) for p in with_gate.materialise()
        }


class TestStepSizeMaxRefusesInsteadOfClamping:
    """The identity defect on the campaign plan's record: `step_size_init =
    500` and `= 1000` were two ModelIDs for one executed model, because the
    sigmoid reparameterisation clamped both onto the same tensor in silence."""

    def test_a_document_with_init_at_max_is_refused_naming_the_position(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            SpecificationError, match=r"contenders\[0\].*step_size_init"
        ):
            _load(_document(tmp_path / "doc", step=1.0, gate=""))

    def test_a_document_with_init_above_max_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError):
            _load(_document(tmp_path / "doc", step=500.0, gate=""))

    def test_a_non_positive_init_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError):
            _load(_document(tmp_path / "doc", step=0.0, gate=""))

    def test_the_parameter_config_is_guarded_for_non_recipe_callers(self) -> None:
        """`build_unfolded_controller` is reached by workbench and notebook
        modules that pass through no recipe; the config tier guards them."""
        import torch

        from mbl.models.unfolded.parameters import StepSizeParameterConfig

        with pytest.raises(ValueError):
            StepSizeParameterConfig(
                name="step_size",
                num_iterations=3,
                action_dim=2,
                alpha_init=1.0,
                alpha_max=1.0,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
        with pytest.raises(ValueError):
            StepSizeParameterConfig(
                name="step_size",
                num_iterations=3,
                action_dim=2,
                alpha_init=-0.1,
                alpha_max=1.0,
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
