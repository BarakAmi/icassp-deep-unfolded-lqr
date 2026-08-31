"""Acceptance tests for the style profiles — slice Phase C, step C1.

Written before the implementation. Annex 03 §B.2 parameterises the target
venue rather than guessing it: a profile fixes width, typography and sizing,
and the figure spec stays profile-independent so one declaration renders at
every venue width.

**The property this suite exists for is §B.2.1's, and it is asserted on a
rendered file.** Every PDF this repository has ever emitted carries Type 3
fonts — measured: `PLOT_STYLE` sets no `pdf.fonttype`, so matplotlib's default
of 3 stands under `styled_figure()`. §B.2 has always forbidden that, and IEEE
and ACM submission checkers reject it. A test of `rc_params()["pdf.fonttype"]`
would pass against a writer that dropped the setting on the floor, so the
assertion opens the PDF and looks for the font subtype that is actually in it.

The second property is quieter and is a machine-dependence trap this project
has now paid for seven times: **a profile naming a font the machine does not
have makes matplotlib warn**, and the repository's pytest gate is
zero-warning. Only DejaVu is installed on the development machine; CI's may
differ. Every fallback chain therefore ends in a font matplotlib itself ships.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from mbl.present.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    StyleProfile,
    profile_context,
    resolve_profile,
)


def _render(path: Path, profile: StyleProfile) -> Path:
    with profile_context(profile):
        figure, axes = plt.subplots()
        axes.plot([1, 2, 3], [1, 4, 9], label=r"$\alpha$ curve")
        axes.set_xlabel("depth $J$")
        axes.legend()
        figure.savefig(path)
        plt.close(figure)
    return path


class TestTheCatalogue:
    def test_every_profile_annex_b2_names_is_present(self) -> None:
        assert set(PROFILES) == {
            "thesis",
            "ieee-1col",
            "ieee-2col",
            "neurips",
            "ieee-paper",
            "talk",
        }

    def test_the_default_is_thesis(self) -> None:
        assert DEFAULT_PROFILE == "thesis"
        assert resolve_profile(None).name == "thesis"

    def test_the_widths_are_the_annex_s(self) -> None:
        widths = {name: profile.width_in for name, profile in PROFILES.items()}
        assert widths == {
            "thesis": 6.0,
            "ieee-1col": 3.5,
            "ieee-2col": 7.16,
            "neurips": 5.5,
            "ieee-paper": 9.0,
            "talk": 10.0,
        }

    def test_the_base_font_sizes_are_the_annex_s(self) -> None:
        sizes = {name: profile.base_font_pt for name, profile in PROFILES.items()}
        assert sizes == {
            "thesis": 10.0,
            "ieee-1col": 8.0,
            "ieee-2col": 8.0,
            "neurips": 9.0,
            "ieee-paper": 16.0,
            "talk": 16.0,
        }

    def test_an_unknown_profile_names_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="ieee-2col"):
            resolve_profile("ieee-3col")


class TestTypeFortyTwoOnARenderedFile:
    def test_the_pdf_carries_no_type_3_font(self, tmp_path: Path) -> None:
        """§B.2.1, asserted where it is true or false rather than in a dict.

        A Type 3 font appears in the PDF's own object stream as
        `/Subtype /Type3`. Measured, matplotlib at `pdf.fonttype: 42` writes
        `/Type0` with a `/CIDFontType2` descendant — the composite TrueType
        form — and not the bare `/TrueType` a first draft of this test looked
        for. Reading the bytes is the only assertion a writer that dropped the
        rcParam could not satisfy.
        """
        import re

        raw = _render(tmp_path / "figure.pdf", resolve_profile("thesis")).read_bytes()
        subtypes = set(re.findall(rb"/Subtype\s*/(\w+)", raw))

        assert b"Type3" not in subtypes
        assert b"CIDFontType2" in subtypes

    def test_the_current_default_would_have_failed_it(self, tmp_path: Path) -> None:
        """The control that makes the test above mean something.

        Rendering the same figure under the repository's existing style emits
        a Type 3 font. If this ever stops being true the assertion above has
        become a tautology and this failure says so.
        """
        from mbl.viz.style.theme import styled_figure

        path = tmp_path / "legacy.pdf"
        with styled_figure():
            figure, axes = plt.subplots()
            axes.plot([1, 2, 3], [1, 4, 9], label="curve")
            axes.legend()
            figure.savefig(path)
            plt.close(figure)

        import re

        assert b"Type3" in set(re.findall(rb"/Subtype\s*/(\w+)", path.read_bytes()))

    @pytest.mark.parametrize("name", sorted(PROFILES))
    def test_every_profile_embeds_truetype(self, tmp_path: Path, name: str) -> None:
        import re

        raw = _render(tmp_path / f"{name}.pdf", resolve_profile(name)).read_bytes()
        assert b"Type3" not in set(re.findall(rb"/Subtype\s*/(\w+)", raw))


class TestTheProfileReachesTheFigure:
    def test_the_figure_width_is_the_profile_s(self) -> None:
        with profile_context(resolve_profile("ieee-2col")):
            assert plt.rcParams["figure.figsize"][0] == pytest.approx(7.16)

    def test_the_base_font_size_is_the_profile_s(self) -> None:
        with profile_context(resolve_profile("ieee-1col")):
            assert plt.rcParams["font.size"] == pytest.approx(8.0)

    def test_the_fallback_chain_reaches_matplotlib(self) -> None:
        # Mutation testing found nothing asserted this: a profile could name a
        # face and never request it, and every venue would render in the house
        # default while the catalogue said otherwise.
        with profile_context(resolve_profile("thesis")):
            assert plt.rcParams["font.serif"][0] == "Times New Roman"
            assert plt.rcParams["font.serif"][-1] == "DejaVu Serif"
        with profile_context(resolve_profile("talk")):
            assert plt.rcParams["font.sans-serif"][0] == "Helvetica"
            assert plt.rcParams["font.family"] == ["sans-serif"]

    def test_two_profiles_give_two_widths(self, tmp_path: Path) -> None:
        # The property `mbl figure rebuild --style ieee-2col` sells: the same
        # declaration renders at a different width. Asserted on the rendered
        # figures rather than on the rcParams.
        sizes = []
        for name in ("ieee-1col", "ieee-2col"):
            with profile_context(resolve_profile(name)):
                figure, axes = plt.subplots()
                axes.plot([1, 2], [1, 2])
                sizes.append(figure.get_size_inches()[0])
                plt.close(figure)
        assert sizes[0] != sizes[1]

    def test_the_context_restores_what_it_changed(self) -> None:
        before = dict(plt.rcParams)
        with profile_context(resolve_profile("talk")):
            pass
        after = dict(plt.rcParams)
        assert before["figure.figsize"] == after["figure.figsize"]
        assert before["font.size"] == after["font.size"]


class TestNoProfileWarnsOnThisMachine:
    @pytest.mark.parametrize("name", sorted(PROFILES))
    def test_rendering_emits_no_warning(self, tmp_path: Path, name: str) -> None:
        """The seventh instance of the environment-dependent trap.

        A profile naming a font the machine lacks makes matplotlib warn, and
        the repository's pytest gate is zero-warning — so a profile that is
        correct on a machine with Times installed would break CI on one
        without it. Every fallback chain ends in a font matplotlib ships.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _render(tmp_path / f"{name}.pdf", resolve_profile(name))

    def test_every_fallback_chain_ends_in_a_shipped_font(self) -> None:
        # The structural form of the same claim: asserted against the fonts
        # matplotlib installs with itself, so it holds on a machine this test
        # has never run on.
        shipped = {"DejaVu Serif", "DejaVu Sans"}
        for profile in PROFILES.values():
            assert profile.font_fallbacks[-1] in shipped, profile.name


class TestTheProfileIsNotAPerFigureDefault:
    def test_a_profile_is_frozen(self) -> None:
        # §B.2.1: a `FigureSpec.config` may carry annotations and series
        # selection, and may not carry a width or a font size, or "no figure
        # is scaled after export" stops being enforceable.
        profile = resolve_profile("thesis")
        with pytest.raises(Exception):
            profile.width_in = 3.0  # type: ignore[misc]

    def test_a_lookup_returns_the_same_profile_every_time(self) -> None:
        # Frozen and shared: a caller that mutated one would restyle every
        # figure the process renders afterwards.
        assert resolve_profile("thesis") is PROFILES["thesis"]
        assert resolve_profile("thesis") == resolve_profile("thesis")


class TestTheDataInkIsSizedByTheProfile:
    """§B.2, amended 2026-08-07 on the author's review: "sizing" includes the
    marks, not only the type.

    The omission this covers was invisible because a setting nobody writes
    raises nothing. No profile declared `lines.markersize`, so every figure
    this project ever rendered used matplotlib's default **6.0 pt** at every
    venue — 0.60x the base font at `thesis` and 0.75x at the IEEE profiles,
    where the figure is typeset at 8 pt. A marker three quarters the height of
    the surrounding type competes with the type.
    """

    def test_every_profile_declares_a_marker_size(self) -> None:
        for name, profile in PROFILES.items():
            assert "lines.markersize" in profile.rc_params(), name

    def test_it_scales_with_the_base_font_rather_than_being_fixed(self) -> None:
        """The property that makes `--style` a re-render: a figure whose marks
        suit `thesis` and are wrong at `ieee-1col` cannot be re-styled."""
        sizes = {
            name: profile.rc_params()["lines.markersize"]
            for name, profile in PROFILES.items()
        }
        fonts = {name: profile.base_font_pt for name, profile in PROFILES.items()}
        ratios = [sizes[name] / fonts[name] for name in PROFILES]
        assert max(ratios) == pytest.approx(min(ratios)), sizes
        assert len(set(sizes.values())) > 1, "a fixed size is not a scaled one"

    def test_the_marker_no_longer_competes_with_the_type(self) -> None:
        """Measured before: 0.60x at `thesis`, 0.75x at the IEEE profiles.
        The floor and ceiling here are what papers in this area draw at."""
        for name, profile in PROFILES.items():
            ratio = profile.rc_params()["lines.markersize"] / profile.base_font_pt
            assert 0.3 <= ratio <= 0.55, (name, ratio)

    def test_the_line_is_thinner_than_the_house_style_drew_it(self) -> None:
        """`PLOT_STYLE` declares 2.0 pt for every venue and every figure. At
        an 8 pt IEEE column that is a heavy line."""
        from mbl.viz.style.theme import PLOT_STYLE

        for name, profile in PROFILES.items():
            width = profile.rc_params()["lines.linewidth"]
            if profile.base_font_pt <= 10.0:
                assert width < PLOT_STYLE["lines.linewidth"], (name, width)

    def test_a_figure_actually_draws_at_those_sizes(self) -> None:
        """Asserted on the artists, not on the dictionary: a setting correct in
        `rc_params` and never applied would satisfy every test above."""
        from matplotlib import pyplot as plt

        profile = PROFILES["ieee-2col"]
        with profile_context(profile):
            figure, axes = plt.subplots()
            (line,) = axes.plot([0, 1], [0, 1], marker="o")
            try:
                assert line.get_markersize() == pytest.approx(
                    profile.rc_params()["lines.markersize"]
                )
                assert line.get_linewidth() == pytest.approx(
                    profile.rc_params()["lines.linewidth"]
                )
            finally:
                plt.close(figure)
