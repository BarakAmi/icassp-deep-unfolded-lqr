"""Acceptance tests for the greyscale gate and the four-artifact writer — C1.

Written before the implementation.

**The greyscale gate is tested in the failing direction first.** Annex 03
§B.4.1 says so explicitly, and this project has shipped a guard verified only
one way before. A separability check that has never refused anything is a
check nobody has tested, so the suite below builds a deliberately unseparable
pair — same colour, same linestyle, same marker — and asserts the refusal
*before* it asserts that a real figure passes.

**The gate reads the rendered figure.** `encodings_of` walks the axes' own
artists rather than a declaration, because a renderer that ignored its style
arguments would satisfy any check of those arguments. The test that pins this
draws a figure whose *declared* encodings differ and whose *drawn* ones do not.

For the writer, the property that matters is atomicity in the direction that
can actually happen: the spec is serialised last, so a spec that cannot be
written is the interleaving that would otherwise leave three new files beside
one old one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from mbl.present.artifacts import (
    FIGURE_SUFFIXES,
    RASTER_DPI,
    read_figure_data,
    read_figure_spec,
    write_figure_artifacts,
)
from mbl.present.profiles import resolve_profile
from mbl.present.greyscale import (
    LUMINANCE_FLOOR,
    GreyscaleError,
    SeriesDriftError,
    SeriesEncoding,
    UnreadableSeriesError,
    encodings_by_axes,
    encodings_of,
    figure_conflicts,
    printed_color,
    greyscale_conflicts,
    relative_luminance,
    require_greyscale_separable,
)


def _figure(*series: tuple[str, str, str, str]) -> Figure:
    """One axes carrying `(label, colour, linestyle, marker)` per series."""
    figure, axes = plt.subplots()
    for label, color, linestyle, marker in series:
        axes.plot(
            [1, 2, 3],
            [1, 2, 3],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )
    axes.legend()
    return figure


def _table() -> pd.DataFrame:
    return pd.DataFrame({"contender": ["a", "b"], "aggregate": [1.0, 2.0]})


class TestRelativeLuminance:
    def test_black_and_white_are_the_endpoints(self) -> None:
        assert relative_luminance("#000000") == pytest.approx(0.0)
        assert relative_luminance("#ffffff") == pytest.approx(1.0)

    def test_it_is_the_wcag_definition_not_a_channel_mean(self) -> None:
        # Pure green is far brighter than pure blue under WCAG's weights; a
        # naive (r+g+b)/3 would call them identical, which is exactly the
        # error that makes a palette look separable and print flat.
        assert relative_luminance("#00ff00") > relative_luminance("#0000ff")
        assert relative_luminance("#00ff00") == pytest.approx(0.7152, abs=1e-4)

    def test_the_srgb_linearisation_is_applied(self) -> None:
        """Mutation testing removed the gamma step and nothing failed.

        Every colour above is fully saturated, and linearisation maps 0 to 0
        and 1 to 1 — so a fixture of pure channels cannot see it at all. A
        mid-grey can: 0.5 encoded is 0.2159 linear, and skipping the step
        would read 0.5. That factor is the difference between a palette that
        prints separable and one that does not.
        """
        assert relative_luminance("#808080") == pytest.approx(0.2159, abs=1e-3)
        assert relative_luminance("#808080") != pytest.approx(0.5, abs=1e-2)


class TestTheGateRefuses:
    def test_an_identically_encoded_pair_is_refused(self) -> None:
        """The negative control §B.4.1 demands, and the first assertion here.

        Same colour, same linestyle, same marker: two series that are one
        series in print, and one series on screen too.
        """
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#2a78d6", "-", "o"))
        with pytest.raises(GreyscaleError, match="alpha"):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_pair_separated_only_by_hue_is_refused(self) -> None:
        # The subtler and more common failure: two colours a reader can tell
        # apart on screen that collapse to the same grey in print.
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#a0522d", "-", "o"))
        conflicts = greyscale_conflicts(encodings_of(figure))
        assert conflicts and conflicts[0].separation < LUMINANCE_FLOOR
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_the_message_names_both_series_and_the_number(self) -> None:
        # An author is told which two series to re-encode, not that "the
        # figure failed".
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#2a78d6", "-", "o"))
        with pytest.raises(GreyscaleError) as caught:
            require_greyscale_separable(figure, "probe")
        message = str(caught.value)
        assert "alpha" in message and "beta" in message
        assert "0.000" in message
        plt.close(figure)


class TestTheGateAccepts:
    def test_a_differing_linestyle_passes_whatever_the_colour(self) -> None:
        # Triple encoding is the rule: a channel colour removal cannot touch
        # is separability, however close the two greys are.
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#2a78d6", "--", "o"))
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_differing_marker_passes(self) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#2a78d6", "-", "s"))
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_large_luminance_gap_passes_with_no_other_channel(self) -> None:
        figure = _figure(("alpha", "#000000", "-", "o"), ("beta", "#eda100", "-", "o"))
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_unlabelled_artists_are_chrome_rather_than_series(self) -> None:
        # matplotlib's underscore convention marks an artist as excluded from
        # the legend. Two gridlines are not two series, and a gate that
        # counted them would refuse every figure that has any.
        figure, axes = plt.subplots()
        axes.plot([1, 2], [1, 2], label="alpha", color="#000000")
        axes.plot([1, 2], [2, 3], color="#000000")
        axes.axhline(1.5, color="#000000")
        assert [encoding.label for encoding in encodings_of(figure)] == ["alpha"]
        require_greyscale_separable(figure, "probe")
        plt.close(figure)


class TestTheGateReadsWhatWasDrawn:
    def test_it_reads_the_artists_and_not_a_declaration(self) -> None:
        """A renderer that ignores its own style arguments must still fail.

        The declaration here says two linestyles; the drawing says one,
        because the second `plot` overrides it. A gate built on the declared
        encodings would pass this figure.
        """
        figure, axes = plt.subplots()
        declared = [("alpha", "-"), ("beta", "--")]
        for label, _ignored_linestyle in declared:
            axes.plot([1, 2], [1, 2], label=label, color="#2a78d6", linestyle="-")
        axes.legend()

        drawn = encodings_of(figure)
        assert {encoding.linestyle for encoding in drawn} == {"-"}
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_pure_pair_check_needs_no_figure(self) -> None:
        # The pure half is separately usable, which is what lets a renderer
        # choose encodings before it draws anything.
        pair = (
            SeriesEncoding("a", "#2a78d6", "-", "o"),
            SeriesEncoding("b", "#2a78d6", "-", "o"),
        )
        assert len(greyscale_conflicts(pair)) == 1
        assert greyscale_conflicts(pair[:1]) == ()


class TestTheGateSeesEveryPanelAndEveryArtist:
    """Annex 03 §B.4.1's per-axes reading and §B.4.3's one-label-one-encoding.

    Every test here except the neutrality ones fails against the pooled,
    lines-only gate that preceded them.
    """

    @staticmethod
    def _panels(*series: tuple[str, str, str, str]):
        """Two stacked panels, every series drawn and LABELLED on both."""
        figure, axes = plt.subplots(2, 1)
        for panel in axes:
            for label, color, linestyle, marker in series:
                panel.plot(
                    [1, 2],
                    [1, 2],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
        return figure, axes

    def test_a_series_on_two_panels_is_one_series_not_a_conflicting_pair(
        self,
    ) -> None:
        """The defect the broken axis would have hit head-on. Pooling every
        Axes pairs a series with its own copy on the other panel: measured,
        two series on two panels produced SIX conflicts, two of them self-pairs
        reporting a separation of 0.000."""
        figure, _ = self._panels(
            ("alpha", "#2a78d6", "-", "o"), ("beta", "#eda100", "--", "s")
        )
        assert figure_conflicts(figure) == ()
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_labelling_only_one_panel_is_not_the_remedy(self) -> None:
        """Both panels stay under the gate. If the lower panel's labels were
        suppressed to satisfy a pooled check, this count would be 2."""
        figure, _ = self._panels(
            ("alpha", "#2a78d6", "-", "o"), ("beta", "#eda100", "--", "s")
        )
        assert len(encodings_by_axes(figure)) == 2
        assert len(encodings_of(figure)) == 4
        plt.close(figure)

    def test_an_unseparable_pair_on_one_panel_of_two_still_raises(self) -> None:
        figure, axes = plt.subplots(2, 1)
        axes[0].plot([1, 2], [1, 2], label="alpha", color="#2a78d6", linestyle="-")
        axes[0].plot([1, 2], [2, 3], label="beta", color="#2a78d6", linestyle="-")
        axes[1].plot([1, 2], [1, 2], label="gamma", color="#000000", linestyle="-")
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_panel_that_repaints_every_series_is_refused(self) -> None:
        """§B.4.3. A lower panel painted in one colour satisfies a per-axes
        pair check on the upper panel and says nothing about itself."""
        figure, axes = plt.subplots(2, 1)
        for label, color, linestyle, marker in (
            ("alpha", "#2a78d6", "-", "o"),
            ("beta", "#eda100", "--", "s"),
        ):
            axes[0].plot(
                [1, 2],
                [1, 2],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            axes[1].plot(
                [1, 2], [1, 2], label=label, color="C0", linestyle="-", marker="o"
            )
        with pytest.raises(SeriesDriftError) as raised:
            require_greyscale_separable(figure, "probe")
        assert "alpha" in str(raised.value) and "beta" in str(raised.value)
        plt.close(figure)

    def test_an_errorbar_series_is_seen_at_all(self) -> None:
        """The trap that would have turned the print law off for Figure 1.

        `ax.errorbar(label=...)` puts the label on the CONTAINER and leaves
        every Line2D at `_nolegend_`. Measured against the lines-only gate,
        `encodings_of` returned the empty tuple and the gate passed a figure it
        had checked nothing on.
        """
        figure, axes = plt.subplots()
        for label in ("alpha", "beta"):
            axes.errorbar(
                [1, 2],
                [1, 2],
                yerr=[0.1, 0.1],
                label=label,
                color="#2a78d6",
                linestyle="-",
                marker="o",
                capsize=2,
            )
        assert len(encodings_of(figure)) == 2
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_an_unlabelled_errorbar_companion_does_not_double_count(self) -> None:
        """The form the error-bar renderer actually uses: a labelled `plot`
        for the series, an unlabelled `errorbar(fmt="none")` for the caps.
        The companion registers as `_container0` and must contribute nothing,
        or every series would conflict with itself."""
        figure, axes = plt.subplots()
        axes.plot(
            [1, 2], [1, 2], label="alpha", color="#2a78d6", linestyle="-", marker="o"
        )
        axes.errorbar([1, 2], [1, 2], yerr=[0.1, 0.1], fmt="none", capsize=2)
        assert [e.label for e in encodings_of(figure)] == ["alpha"]
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_bar_is_a_series_the_gate_can_read(self) -> None:
        """Figure 2 and Figure 4 are bar charts, and the gate walked
        `get_lines()` only -- so a bar figure cleared the black-and-white
        publication gate WITHOUT BEING CHECKED."""
        figure, axes = plt.subplots()
        axes.bar([0], [1], label="alpha", color="#2a78d6")
        axes.bar([1], [1], label="beta", color="#2a78d6")
        assert len(encodings_of(figure)) == 2
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_two_bars_separated_by_hatch_pass(self) -> None:
        figure, axes = plt.subplots()
        axes.bar([0], [1], label="alpha", color="#2a78d6")
        axes.bar([1], [1], label="beta", color="#2a78d6", hatch="//")
        require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_translucent_bar_is_compared_as_it_prints(self) -> None:
        """A fill at alpha 0.3 is not its own colour on paper. Black at 0.3
        over white prints at luminance 0.7, not 0.0 -- so comparing the
        declared colour would pass a pair that merges in print."""
        assert printed_color("#000000", 0.3) == pytest.approx((0.7, 0.7, 0.7))
        assert printed_color("#000000", None) == pytest.approx((0.0, 0.0, 0.0))

    def test_the_gate_itself_composites_and_not_merely_the_helper(self) -> None:
        """The verdict must flip, not just the helper's return value.

        Asserting `printed_color` alone leaves the WIRING untested: a mutant
        that compares `face.get_facecolor()` directly passes such a suite
        untouched, because the helper it no longer calls still works.

        Measured on this pair: black at alpha 0.1 and 50 % grey at alpha 0.2
        are 0.214041 apart as DECLARED -- comfortably over LUMINANCE_FLOOR --
        and both print as (0.9, 0.9, 0.9), luminance 0.787412, exactly
        0.000000 apart. One bar in print, two by the declaration.
        """
        figure, axes = plt.subplots()
        axes.bar([0], [1], label="alpha", color="#000000", alpha=0.1)
        axes.bar([1], [1], label="beta", color="0.5", alpha=0.2)
        assert abs(relative_luminance("#000000") - relative_luminance("0.5")) > (
            LUMINANCE_FLOOR
        )
        with pytest.raises(GreyscaleError, match="0.000"):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_an_unset_hatch_and_an_empty_hatch_are_one_shape(self) -> None:
        """Both bars are unhatched, so shape separates nothing and colour must.

        Measured: `get_hatch()` is `None` when never set and `np.str_('')`
        when `hatch=""` is declared. Formatted naively those are `'None'` and
        `''` -- two different strings for two bars a reader cannot tell apart,
        which would satisfy §B.4.1's clause 1 and let an identical pair
        through on a difference that exists only in the object model.
        """
        figure, axes = plt.subplots()
        axes.bar([0], [1], label="alpha", color="#2a78d6")
        axes.bar([1], [1], label="beta", color="#2a78d6", hatch="")
        with pytest.raises(GreyscaleError):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_a_labelled_artist_the_gate_cannot_read_is_refused(self) -> None:
        """Refused, never skipped: a labelled thing stepped over is a series
        that escaped the gate while it reported green."""
        figure, axes = plt.subplots()

        class _Opaque:
            patches = ()
            lines = ()

            def get_label(self) -> str:
                return "alpha"

        axes.containers.append(_Opaque())
        with pytest.raises(UnreadableSeriesError, match="alpha"):
            encodings_of(figure)
        plt.close(figure)

    def test_a_single_axes_figure_is_unchanged(self) -> None:
        """Neutrality. Same tuple, same order, as before the grouping."""
        figure = _figure(("alpha", "#2a78d6", "-", "o"), ("beta", "#eda100", "--", "s"))
        assert [e.label for e in encodings_of(figure)] == ["alpha", "beta"]
        assert encodings_by_axes(figure) == (encodings_of(figure),)
        plt.close(figure)


class TestTheFourArtifacts:
    def test_all_four_land_with_one_stem(self, tmp_path: Path) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        written = write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="fig_cost",
            table=_table(),
            spec={"kind": "axis_scaling"},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        assert [path.name for path in written.paths] == [
            f"fig_cost{suffix}" for suffix in FIGURE_SUFFIXES
        ]
        assert all(path.is_file() for path in written.paths)

    def test_a_shipped_figure_carries_no_clock_and_no_build(
        self, tmp_path: Path
    ) -> None:
        """A figure a reviewer redraws must come back as the same file.

        Rendering was already deterministic; the environment was not. Two
        renders differed in **exactly three bytes** -- the `/CreationDate`
        matplotlib stamps into a PDF -- and in nothing else. Three bytes is
        enough for `git status` to report a modified file after an identical
        redraw, and a reader who cannot tell "the same" from "not the same"
        without opening both has lost the property the artifact exists for.

        **Asserted on the two fields, not on two renders being equal.** The
        first version of this test rendered twice and compared the bytes, and
        it could not fail: both renders happen in the same second, so the
        stamp agrees with itself. Measured -- putting the stamp back left it
        passing. What can fail is the presence of the fields themselves.

        They are different properties and both are wanted. `/CreationDate` is a
        clock, so it breaks reproducibility across TIME. The raster's
        `Software` chunk is matplotlib's version, so it breaks it across
        BUILDS: the same figure redrawn on a reviewer's newer matplotlib would
        differ in bytes for a reason that has nothing to do with the results.
        """
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="fig_cost",
            table=_table(),
            spec={"kind": "axis_scaling"},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        vector = (tmp_path / "fig_cost.pdf").read_bytes()
        raster = (tmp_path / "fig_cost.png").read_bytes()
        assert b"/CreationDate" not in vector, (
            "the vector carries a clock; an identical redraw will not match"
        )
        assert b"Software" not in raster, (
            "the raster carries the renderer's version; an identical redraw on "
            "another build will not match"
        )

    def test_two_identical_renders_agree_byte_for_byte(self, tmp_path: Path) -> None:
        """The end-to-end form of the check above.

        Weak on its own -- two renders a moment apart agree even with a clock
        in them -- so it stands beside the field assertions rather than instead
        of them. It is here because the fields are a mechanism and this is the
        property: something else stamped into the file later would pass that
        check and fail this one.
        """
        rendered = {}
        for run in ("first", "second"):
            figure = _figure(("alpha", "#2a78d6", "-", "o"))
            write_figure_artifacts(
                figure,
                directory=tmp_path / run,
                figure_id="fig_cost",
                table=_table(),
                spec={"kind": "axis_scaling"},
                profile=resolve_profile("thesis"),
            )
            plt.close(figure)
            rendered[run] = {
                suffix: (tmp_path / run / f"fig_cost{suffix}").read_bytes()
                for suffix in (".pdf", ".png")
            }
        for suffix in (".pdf", ".png"):
            first, second = rendered["first"][suffix], rendered["second"][suffix]
            assert first == second, (
                f"{suffix} differs in "
                f"{sum(a != b for a, b in zip(first, second))} byte(s)"
            )

    def test_the_raster_is_at_the_annex_s_resolution(self, tmp_path: Path) -> None:
        # 600 dpi, not `viz.style.io`'s 150 -- a figure at 150 is a preview.
        from PIL import Image

        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        written = write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="fig_cost",
            table=_table(),
            spec={},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        with Image.open(written.paths[1]) as image:
            # The literal, NOT `RASTER_DPI`. Mutation testing lowered the
            # constant to 150 and this test still passed, because both sides
            # of the comparison moved together -- a tautology dressed as a
            # measurement.
            assert image.info["dpi"][0] == pytest.approx(600, abs=1)
        assert RASTER_DPI == 600

    def test_the_data_artifact_is_the_table_it_was_given(self, tmp_path: Path) -> None:
        """§B.1.1: a copy, so a later re-analysis cannot change what this
        figure claims to have plotted."""
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        table = _table()
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="fig_cost",
            table=table,
            spec={},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        pd.testing.assert_frame_equal(read_figure_data(tmp_path, "fig_cost"), table)

    def test_the_spec_round_trips(self, tmp_path: Path) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        spec = {"kind": "axis_scaling", "style": "thesis", "series": ["alpha"]}
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="f",
            table=_table(),
            spec=spec,
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        assert read_figure_spec(tmp_path, "f") == spec

    def test_the_spec_is_plain_json(self, tmp_path: Path) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="f",
            table=_table(),
            spec={"kind": "axis_scaling"},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        raw = (tmp_path / "f.spec.json").read_text(encoding="utf-8")
        assert json.loads(raw)["kind"] == "axis_scaling"


class TestTheWriteIsAtomic:
    def test_a_failed_spec_leaves_the_previous_render_whole(
        self, tmp_path: Path
    ) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        first = pd.DataFrame({"contender": ["a"], "aggregate": [1.0]})
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="f",
            table=first,
            spec={"generation": 1},
            profile=resolve_profile("thesis"),
        )

        with pytest.raises((TypeError, ValueError)):
            write_figure_artifacts(
                figure,
                directory=tmp_path,
                figure_id="f",
                table=pd.DataFrame({"contender": ["z"], "aggregate": [99.0]}),
                spec={"bad": object()},
                profile=resolve_profile("thesis"),
            )
        plt.close(figure)

        # The spec alone is not the assertion: the spec is written LAST, so it
        # survives a non-atomic writer too. Mutation testing wrote the other
        # three straight into place and this test passed. What must hold is
        # that the whole BUNDLE is the first render -- the data artifact
        # included, since that is what a reader would compare against the spec.
        assert read_figure_spec(tmp_path, "f") == {"generation": 1}
        pd.testing.assert_frame_equal(read_figure_data(tmp_path, "f"), first)

    def test_no_staging_directory_survives(self, tmp_path: Path) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="f",
            table=_table(),
            spec={},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            f"f{suffix}" for suffix in sorted(FIGURE_SUFFIXES)
        ]

    @pytest.mark.parametrize("figure_id", ["../escape", "a/b", "", "."])
    def test_an_id_that_is_not_one_segment_is_refused(
        self, tmp_path: Path, figure_id: str
    ) -> None:
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        with pytest.raises(ValueError):
            write_figure_artifacts(
                figure,
                directory=tmp_path,
                figure_id=figure_id,
                table=_table(),
                spec={},
                profile=resolve_profile("thesis"),
            )
        plt.close(figure)


class TestRebuildNeedsOnlyTwoFiles:
    def test_the_spec_and_the_data_read_without_a_store(self, tmp_path: Path) -> None:
        """§B.1's promise, in its weakest and therefore strongest form.

        The two files are read from a bare directory with no store, no
        analysis and no study document anywhere near it.
        """
        figure = _figure(("alpha", "#2a78d6", "-", "o"))
        write_figure_artifacts(
            figure,
            directory=tmp_path / "loose",
            figure_id="f",
            table=_table(),
            spec={"kind": "axis_scaling"},
            profile=resolve_profile("thesis"),
        )
        plt.close(figure)

        assert read_figure_spec(tmp_path / "loose", "f")["kind"] == "axis_scaling"
        assert len(read_figure_data(tmp_path / "loose", "f")) == 2

    def test_an_unrendered_figure_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="absent"):
            read_figure_spec(tmp_path, "absent")


class TestTheProfileReachesTheWriter:
    def test_a_figure_rendered_inside_and_written_outside_is_still_type_42(
        self, tmp_path: Path
    ) -> None:
        """The runner's real shape, and the defect only real data found.

        matplotlib reads `pdf.fonttype` when `savefig` runs, not when the
        figure is built. Every earlier test both rendered and wrote inside one
        `profile_context`, so all of them passed while the runner — which
        renders inside and writes outside — emitted a Type 3 PDF. The writer
        now takes the profile and opens it around the save, so the guarantee
        is structural rather than a property of the caller's indentation.
        """
        import re

        from mbl.present.profiles import profile_context

        profile = resolve_profile("thesis")
        with profile_context(profile):
            figure = _figure(("alpha", "#2a78d6", "-", "o"))
        # The context is CLOSED here, exactly as it is in `render_figures`.
        written = write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="f",
            table=_table(),
            spec={},
            profile=profile,
        )
        plt.close(figure)

        subtypes = set(re.findall(rb"/Subtype\s*/(\w+)", written.pdf.read_bytes()))
        assert b"Type3" not in subtypes
        assert b"CIDFontType2" in subtypes
