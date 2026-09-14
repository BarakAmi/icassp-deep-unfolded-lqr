"""Acceptance tests for `axis_scaling` and the figure runner — C2.

Written before the implementation. The phase's checkpoint is a **negative
control twice** (slice plan, Phase C), and both halves are here:

* the greyscale gate must **fail** on a deliberately unseparable figure and
  pass on the real one — a gate verified only in the passing direction is a
  gate nobody has tested;
* the render path must run with the store's **model directory renamed away**,
  which is the only way to prove it reads no model.

The property that shapes the encodings is §B.3.1's: **a figure that drops a
contender must not repaint the survivors.** It is asserted by rendering the
same study twice, once with every series and once with one dropped, and
comparing the drawn colour, linestyle and marker of everything that remained.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import axis_scaling
from mbl.present.encodings import BOUND_COLOR, encode_series
from mbl.present.greyscale import GreyscaleError, encodings_of
from mbl.present.profiles import resolve_profile
from mbl.present.registry import (
    FigureContext,
    registered_figures,
    resolve_figure,
)
from mbl.present.runner import render_figures
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "standard_pgd": Role.BASELINE,
    "truncated_riccati": Role.BASELINE,
    "cocp_lower_bound": Role.BOUND,
}
ORDER = tuple(ROLES)


def _table(*, intervals: bool = False) -> pd.DataFrame:
    """Two swept contenders, one swept baseline, one flat baseline, one bound."""
    rows: list[dict[str, Any]] = []
    for label, base in (
        ("unfolded_alpha", 3.0),
        ("unfolded_alpha_p", 3.6),
        ("standard_pgd", 3.2),
    ):
        for depth in (1, 2, 4, 8):
            aggregate = base - 0.1 * depth
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_value": float(depth),
                    "aggregate": aggregate,
                    "interval_low": aggregate - 0.05 if intervals else np.nan,
                    "interval_high": aggregate + 0.05 if intervals else np.nan,
                }
            )
    for label, value in (("truncated_riccati", 2.9), ("cocp_lower_bound", 1.2)):
        rows.append(
            {
                "contender": label,
                "role": ROLES[label].value,
                "axis_value": np.nan,
                "aggregate": value,
                "interval_low": np.nan,
                "interval_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _context(table: pd.DataFrame, **config: Any) -> FigureContext:
    return FigureContext(
        figure_id="fig_cost_vs_depth",
        table=table,
        config=config,
        profile=resolve_profile("thesis"),
        series_order=ORDER,
        roles=ROLES,
    )


def _drawn(figure: Any) -> dict[str, tuple[Any, Any, Any]]:
    return {
        encoding.label.split(" (")[0]: (
            encoding.color,
            encoding.linestyle,
            encoding.marker,
        )
        for encoding in encodings_of(figure)
    }


class TestTheRegistry:
    def test_the_kind_resolves_by_its_annex_name(self) -> None:
        assert resolve_figure("axis_scaling") is axis_scaling
        assert "axis_scaling" in registered_figures()

    def test_an_unregistered_kind_names_what_is_available(self) -> None:
        with pytest.raises(SpecificationError, match="axis_scaling"):
            resolve_figure("no_such_figure")


class TestWhatIsDrawn:
    def test_every_declared_series_appears(self) -> None:
        figure = axis_scaling(_context(_table()))
        assert set(_drawn(figure)) == set(ORDER)
        plt.close(figure)

    def test_a_depth_invariant_contender_is_a_flat_line(self) -> None:
        figure = axis_scaling(_context(_table()))
        axes = figure.axes[0]
        flat = [
            line
            for line in axes.get_lines()
            if str(line.get_label()).startswith("truncated_riccati")
        ]
        assert len(flat) == 1
        assert len(set(np.asarray(flat[0].get_ydata()))) == 1
        plt.close(figure)

    def test_a_bound_carries_its_value_in_the_legend(self) -> None:
        # §B.3.1: a bound is annotated with its value, so the axis rule may
        # clip it without hiding it.
        figure = axis_scaling(_context(_table()))
        labels = {str(line.get_label()) for line in figure.axes[0].get_lines()}
        assert "cocp_lower_bound (1.2)" in labels
        plt.close(figure)

    def test_a_bound_is_grey_and_never_a_palette_slot(self) -> None:
        figure = axis_scaling(_context(_table()))
        assert _drawn(figure)["cocp_lower_bound"][0] == BOUND_COLOR
        plt.close(figure)

    def test_an_interval_is_drawn_when_the_analysis_produced_one(self) -> None:
        with_bands = axis_scaling(_context(_table(intervals=True)))
        without = axis_scaling(_context(_table(intervals=False)))
        assert len(with_bands.axes[0].collections) > len(without.axes[0].collections)
        plt.close(with_bands)
        plt.close(without)

    def test_no_band_is_drawn_when_there_is_no_interval(self) -> None:
        # One training seed produces no across-seed interval at all (§A.3), and
        # a band drawn from nothing would be a claim the analysis refused.
        figure = axis_scaling(_context(_table(intervals=False)))
        # `len(...) == 0` rather than `== []`: matplotlib returns an
        # `ArtistList` view, which is not a list and compares unequal to one.
        assert len(figure.axes[0].collections) == 0
        plt.close(figure)


class TestSectionB5TheAxisRule:
    def test_the_axis_is_scaled_to_the_curves_not_the_bound(self) -> None:
        """The measured complaint about this project's own NB04 figure.

        The bound sits at 1.2 and the contenders live in 2.4–3.5. An axis
        that included the bound would compress the entire result into the top
        third of the figure.
        """
        figure = axis_scaling(_context(_table()))
        low, high = figure.axes[0].get_ylim()
        assert low > 2.0
        assert high < 3.7
        plt.close(figure)

    def test_an_explicit_ylim_wins(self) -> None:
        figure = axis_scaling(_context(_table(), ylim=(0.0, 10.0)))
        assert figure.axes[0].get_ylim() == (0.0, 10.0)
        plt.close(figure)

    def test_the_axis_does_not_depend_on_draw_order(self) -> None:
        """`ax.dataLim` grows as artists are added, so an axis read from it
        would move if the flat references were drawn first. The range is read
        from the table, which no draw order can change."""
        table = _table()
        forward = axis_scaling(_context(table))
        reversed_rows = table.iloc[::-1].reset_index(drop=True)
        backward = axis_scaling(_context(reversed_rows))
        assert forward.axes[0].get_ylim() == pytest.approx(backward.axes[0].get_ylim())
        plt.close(forward)
        plt.close(backward)


class TestDroppingASeriesRepaintsNothing:
    def test_the_survivors_keep_every_channel(self) -> None:
        """§B.3.1, the sentence this module's whole shape follows from."""
        everything = axis_scaling(_context(_table()))
        full = _drawn(everything)
        plt.close(everything)

        kept = [label for label in ORDER if label != "unfolded_alpha"]
        subset = axis_scaling(_context(_table(), series=kept))
        partial = _drawn(subset)
        plt.close(subset)

        assert set(partial) == set(kept), "the selection did not take effect"
        for label in kept:
            assert partial[label] == full[label], label

    def test_adding_a_contender_after_the_others_repaints_none_of_them(
        self,
    ) -> None:
        """A study that gains a contender at the END keeps every earlier one.

        The weaker claim the first draft made -- that a study's PREFIX encodes
        identically to the whole -- is false by construction and should be:
        the unregistered-colour fallback claims the first slot nobody
        registered, so a study with fewer contenders has fewer claims. What
        must hold is that appending never disturbs what came before, which is
        what an author actually does.
        """
        before = encode_series(ORDER, ROLES)
        after = encode_series((*ORDER, "extra"), {**ROLES, "extra": Role.CONTENDER})
        for label in ORDER:
            assert before[label] == after[label], label

    def test_no_two_contenders_share_a_colour(self) -> None:
        # Measured on the real study: `standard_pgd` is unregistered and the
        # positional fallback handed it `unfolded_learned_step_size`'s own
        # blue, so two contenders of one figure were the same colour. The
        # triple encoding kept them separable and the gate passed, which is
        # why this needs its own assertion.
        styles = encode_series(ORDER, ROLES)
        drawn = [
            style.color
            for label, style in styles.items()
            if ROLES[label] is not Role.BOUND
        ]
        assert len(set(drawn)) == len(drawn)

    def test_two_baselines_do_not_collide(self) -> None:
        """REVERSED 2026-08-07, and the test is stronger for it.

        It used to assert that the two baselines SHARE a linestyle and are kept
        apart by the marker alone — §B.3.1's role table read as one style per
        role, which gives two baselines one line and leaves one channel doing
        the work. §B.3.1 is now corrected against §B.4.2, whose "slot *i* takes
        palette entry *i*, linestyle *i* and marker *i*" was always the rule:
        the two baselines take two slots and therefore differ in every channel.

        What it checks is unchanged — that two series of one role are separable
        — and it now checks it of the channel that carries it.
        """
        styles = encode_series(ORDER, ROLES)
        first, second = styles["standard_pgd"], styles["truncated_riccati"]
        assert first.linestyle != second.linestyle
        assert first.marker != second.marker
        assert first.color != second.color


class TestTheGreyscaleGateOnARealFigure:
    def test_the_real_figure_passes(self) -> None:
        from mbl.present.greyscale import require_greyscale_separable

        figure = axis_scaling(_context(_table()))
        require_greyscale_separable(figure, "fig_cost_vs_depth")
        plt.close(figure)

    def test_a_deliberately_unseparable_figure_fails(self) -> None:
        """The negative control the phase's checkpoint demands.

        Every contender forced to one role, so the encoder gives them one
        linestyle, and one registered colour, so the marker is the only
        channel left — then the markers are removed. Three series that are one
        series in print.
        """
        from mbl.present.greyscale import require_greyscale_separable

        figure, axes = plt.subplots()
        for label in ("unfolded_alpha", "unfolded_alpha_p", "standard_pgd"):
            axes.plot([1, 2], [1, 2], label=label, color="#2a78d6", linestyle="-")
        with pytest.raises(GreyscaleError, match="unfolded_alpha"):
            require_greyscale_separable(figure, "fig_broken")
        plt.close(figure)


class TestTheTableIsTheContract:
    def test_a_table_missing_a_column_is_refused_by_name(self) -> None:
        table = _table().drop(columns=["interval_low"])
        with pytest.raises(SpecificationError, match="interval_low"):
            axis_scaling(_context(table))

    def test_an_empty_table_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="empty"):
            axis_scaling(_context(_table().iloc[0:0]))

    def test_selecting_an_absent_series_is_refused(self) -> None:
        # Not skipped: a figure quietly missing the contender its caption is
        # about is the plausible-and-wrong output this architecture prevents.
        with pytest.raises(SpecificationError, match="typo"):
            axis_scaling(_context(_table(), series=["typo"]))


class TestTheRunner:
    def _study_with_figure(self, tmp_path: Path):
        from mbl.analysis import run_analyses
        from mbl.experiments import DEFAULT_SPEC_BINDINGS
        from mbl.runner.producer import run_study
        from mbl.spec.analysis import AnalysisSpec
        from mbl.spec.figure import FigureSpec
        from mbl.spec.loader import load_study
        import dataclasses

        from ..runner.test_producer import _write

        document = load_study(_write(tmp_path / "doc"), bindings=DEFAULT_SPEC_BINDINGS)
        study = dataclasses.replace(
            document.study,
            analyses=(AnalysisSpec(id="cost_by_depth", kind="cost_vs_axis"),),
            figures=(
                FigureSpec(
                    id="fig_cost_vs_depth",
                    kind="axis_scaling",
                    source="cost_by_depth",
                ),
            ),
        )
        store = tmp_path / "store"
        run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
        run_analyses(study, store=store)
        return study, store

    def test_it_writes_four_artifacts(self, tmp_path: Path) -> None:
        study, store = self._study_with_figure(tmp_path)

        outcomes = render_figures(study, store=store)

        assert [o.figure_id for o in outcomes] == ["fig_cost_vs_depth"]
        assert all(path.is_file() for path in outcomes[0].artifacts.paths)
        assert outcomes[0].profile == "thesis"

    def _study_drawing_a_bound(self, tmp_path: Path):
        """The same study, with its analysis declaring a box-aware floor.

        The fixture plant is a real box-constrained instance, so the bound is
        computed rather than stubbed.
        """
        import dataclasses

        from mbl.analysis import run_analyses
        from mbl.spec.analysis import AnalysisSpec

        study, store = self._study_with_figure(tmp_path)
        study = dataclasses.replace(
            study,
            analyses=(
                AnalysisSpec(
                    id="cost_by_depth",
                    kind="cost_vs_axis",
                    config={"bounds": ["finite_horizon_box"]},
                ),
            ),
        )
        run_analyses(study, store=store)
        return study, store

    def test_a_study_with_no_bounds_carries_no_bound_names(
        self, tmp_path: Path
    ) -> None:
        """The spec says what this figure draws, and nothing else.

        The first version seeded every spec with the whole bound registry, so a
        study drawing no bound still carried two entries about nothing.
        `test_display_names.py` caught it by asserting the mapping exactly,
        which is the assertion worth keeping.
        """
        from mbl.analysis.bounds import BOUND_DISPLAY

        study, store = self._study_with_figure(tmp_path)
        render_figures(study, store=store)

        directory = store / "studies" / str(study.study_id) / "figures"
        spec = json.loads((directory / "fig_cost_vs_depth.spec.json").read_text())
        assert not set(spec["display_names"]) & set(BOUND_DISPLAY)

    def test_the_stored_spec_carries_the_bound_names_too(self, tmp_path: Path) -> None:
        """A bound has no `display` of its own, and the spec is what a rebuild reads.

        Both were true and both were needed: the live render learned the bound
        names and `<id>.spec.json` did not, because the two were separate
        comprehensions over the contenders. A rebuild then put the registry key
        back into the legend of a figure that had just been fixed. Asserted on
        the WRITTEN spec rather than on the function, because the spec is the
        artifact a reviewer re-renders from.
        """
        from mbl.analysis.bounds import BOUND_DISPLAY

        study, store = self._study_drawing_a_bound(tmp_path)
        render_figures(study, store=store)

        directory = store / "studies" / str(study.study_id) / "figures"
        spec = json.loads((directory / "fig_cost_vs_depth.spec.json").read_text())
        assert (
            spec["display_names"].get("finite_horizon_box")
            == (BOUND_DISPLAY["finite_horizon_box"])
        ), "a rebuild would draw the bound as its own registry key"
        # ...and the contenders are still there, which the "bounds first" order
        # is what protects: a study may name a contender after a bound.
        for contender in study.contenders:
            assert (
                spec["display_names"][contender.resolved_label]
                == contender.resolved_display
            )

    def test_the_stored_spec_records_the_profile_without_pinning_it(
        self, tmp_path: Path
    ) -> None:
        study, store = self._study_with_figure(tmp_path)
        render_figures(study, store=store, style="ieee-2col")

        directory = store / "studies" / str(study.study_id) / "figures"
        spec = json.loads((directory / "fig_cost_vs_depth.spec.json").read_text())
        assert spec["rendered_at_profile"] == "ieee-2col"
        assert "width_in" not in json.dumps(spec)

    def test_the_data_artifact_is_the_analysis_table(self, tmp_path: Path) -> None:
        """§B.1.1: a copy, frozen beside the spec that plotted it."""
        from mbl.store.study_artifacts import StudyArtifactStore

        study, store = self._study_with_figure(tmp_path)
        render_figures(study, store=store)

        directory = store / "studies" / str(study.study_id) / "figures"
        drawn = pd.read_parquet(directory / "fig_cost_vs_depth.data.parquet")
        analysed = (
            StudyArtifactStore(store)
            .get_analysis(str(study.study_id), "cost_by_depth")
            .table
        )
        pd.testing.assert_frame_equal(drawn, analysed)

    def test_it_renders_with_the_model_directory_renamed_away(
        self, tmp_path: Path
    ) -> None:
        """The second negative control the checkpoint demands.

        A figure that reached for a model could not be rebuilt once the
        checkpoints were gone, which is the property that makes the
        data-plus-spec pair worth more than a pickle.
        """
        study, store = self._study_with_figure(tmp_path)
        (store / "models").rename(store / "models-renamed")

        outcomes = render_figures(study, store=store)

        assert outcomes[0].artifacts.pdf.is_file()

    def test_a_figure_whose_analysis_was_never_run_is_refused(
        self, tmp_path: Path
    ) -> None:
        import dataclasses

        study, store = self._study_with_figure(tmp_path)
        shutil.rmtree(store / "studies")
        with pytest.raises(SpecificationError, match="mbl analyse"):
            render_figures(study, store=store)
        del dataclasses

    def test_selecting_an_undeclared_figure_is_refused(self, tmp_path: Path) -> None:
        study, store = self._study_with_figure(tmp_path)
        with pytest.raises(SpecificationError, match="typo"):
            render_figures(study, store=store, only=["typo"])

    def test_a_study_declaring_no_figures_renders_nothing(self, tmp_path: Path) -> None:
        import dataclasses

        study, store = self._study_with_figure(tmp_path)
        assert render_figures(dataclasses.replace(study, figures=()), store=store) == ()


class TestTheRegistryIsPopulatedWithoutImportingTheRenderer:
    def test_importing_the_runner_alone_registers_the_kind(self) -> None:
        """The defect only running it on real data could find.

        Every test in this file imports `axis_scaling` directly, so the
        registry was always populated by the time `resolve_figure` ran — and
        `runner.py` did not import the kinds at all. A caller that had only
        imported the runner got "no figure is registered under kind
        'axis_scaling'; available: (none)".

        A subprocess is the only honest form: within this interpreter the
        module is already imported, so the assertion cannot fail here.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mbl.present.runner\n"
                "from mbl.present.registry import registered_figures\n"
                "print(registered_figures())",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "axis_scaling" in result.stdout

    def test_the_package_export_registers_it_too(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mbl.present\nprint(mbl.present.registered_figures())",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "axis_scaling" in result.stdout


class TestTheRoleIsVisibleWithoutColour:
    def test_a_baseline_and_a_contender_take_different_linestyles(self) -> None:
        """Mutation testing found this untested.

        `test_two_baselines_do_not_collide` asserts the two baselines *agree*,
        which stays true when every role collapses to one linestyle. What must
        also hold is that they DIFFER from a contender's — otherwise a reader
        with the colour removed cannot tell an attained non-learned baseline
        from a learned contender, which is the distinction §B.3.1 reserves an
        encoding for.
        """
        styles = encode_series(ORDER, ROLES)
        assert (
            styles["truncated_riccati"].linestyle != styles["unfolded_alpha"].linestyle
        )
        assert styles["cocp_lower_bound"].linestyle not in {
            styles["unfolded_alpha"].linestyle,
            styles["truncated_riccati"].linestyle,
        }


class TestTheGateRunsBeforeAnythingIsWritten:
    def test_an_unseparable_figure_leaves_no_artifact(self, tmp_path: Path) -> None:
        """§B.4's "a figure that fails does not merge", one step earlier.

        Mutation testing moved the gate after the write and nothing failed:
        every test asserted that the gate raised, and none asserted that the
        store was untouched when it did. A figure that would lose a series in
        print must not reach the store at all.
        """
        import dataclasses

        from mbl.present import registry as registry_module

        study, store = TestTheRunner()._study_with_figure(tmp_path)
        figures = store / "studies" / str(study.study_id) / "figures"
        assert not figures.exists()

        def unseparable(context: FigureContext) -> Any:
            figure, axes = plt.subplots()
            for label in ("a", "b"):
                axes.plot([1, 2], [1, 2], label=label, color="#2a78d6", linestyle="-")
            return figure

        original = registry_module._REGISTRY["axis_scaling"]
        registry_module._REGISTRY["axis_scaling"] = unseparable
        try:
            with pytest.raises(GreyscaleError):
                render_figures(study, store=store)
        finally:
            registry_module._REGISTRY["axis_scaling"] = original
        del dataclasses

        assert not figures.exists() or not list(figures.iterdir())
