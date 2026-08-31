"""Style profiles — Annex 03 §B.2.

The target venue was not fixable in advance, so it is **parameterised** rather
than guessed. A profile fixes width, typography and sizing; a `FigureSpec` is
profile-independent, which is what makes `mbl figure rebuild <id> --style
ieee-2col` a re-render rather than a re-execution.

**A profile is not a bag of per-figure defaults** (§B.2.1). Width and base font
size are exactly what make a figure droppable into a manuscript *unscaled*, so
they live here and nowhere else; a figure that could override its own width
would make "no figure is scaled after export" unenforceable one figure at a
time.

Two things this module fixes that the repository had wrong:

**Every PDF this project has emitted carries Type 3 fonts.** `viz.style.theme
.PLOT_STYLE` sets neither `pdf.fonttype` nor `ps.fonttype`, so matplotlib's
default of 3 stands — the format IEEE and ACM submission checkers reject, and
the one §B.2 has always forbidden in writing. Font embedding is typography, so
it belongs to the profile, and the acceptance test asserts it on the **bytes of
a rendered PDF**: a setting correct in the rcParam dictionary and lost by the
writer would satisfy any test of the dictionary.

**A font the machine does not have makes matplotlib warn**, and this
repository's pytest gate is zero-warning. Only DejaVu is installed on the
development machine and CI's may differ, so every fallback chain here ends in a
font matplotlib itself ships. That is the seventh instance of the
environment-dependent failure this project has recorded, and the first one
caught before it shipped.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from matplotlib import rc_context

from ..viz.style.theme import PLOT_STYLE

#: What a figure's height is, as a fraction of its width, when the profile
#: does not say. §B.2 fixes the *width* per venue and is silent on height,
#: because height is a property of the chart rather than of the page. This is
#: the golden ratio, and a chart that needs another shape says so in its own
#: renderer rather than in the profile.
DEFAULT_ASPECT = 1 / 1.618

#: Relative type scale, applied to the profile's base size. Held here rather
#: than per profile so that changing a venue's base size rescales the whole
#: figure coherently -- which is the property §B.2's "no figure is scaled after
#: export" rule depends on.
TYPE_SCALE: Mapping[str, float] = {
    "axes.titlesize": 1.1,
    "axes.labelsize": 1.0,
    "legend.fontsize": 0.9,
    "xtick.labelsize": 0.85,
    "ytick.labelsize": 0.85,
}

#: Data-ink sizes, as multiples of the profile's base font (§B.2, added
#: 2026-08-07 on the author's review of ICASSP Figure 1).
#:
#: **The omission this replaces was invisible, because a setting nobody writes
#: raises nothing.** No profile declared `lines.markersize`, so every figure
#: this project has rendered used matplotlib's default **6.0 pt** at every
#: venue — 0.60x the base font at `thesis` and **0.75x** at the two IEEE
#: profiles, where the whole figure is typeset at 8 pt. A marker three quarters
#: the height of the surrounding type competes with the type, which is what the
#: author reports as a clumsy figure. `lines.linewidth` had the opposite
#: problem: declared once in the house style, at 2.0 pt for every venue.
#:
#: Scaled rather than fixed for the same reason `TYPE_SCALE` is: a figure whose
#: markers suit `thesis` and are wrong at `ieee-1col` cannot be re-styled, and
#: `mbl figure rebuild --style` is the whole point of a profile-independent
#: `FigureSpec`. The multipliers put `ieee-2col` at 3.6 pt markers on 1.28 pt
#: lines and `thesis` at 4.5 on 1.6, which is the range this class of figure
#: is drawn at in the venues it targets.
MARK_SCALE: Mapping[str, float] = {
    "lines.markersize": 0.45,
    "lines.linewidth": 0.16,
    "lines.markeredgewidth": 0.09,
}

#: PDF font type 42 is TrueType embedding; 3 is the one §B.2 forbids and
#: matplotlib's default.
TRUETYPE = 42


@dataclass(frozen=True)
class StyleProfile:
    """One venue's page geometry and typography.

    Attributes:
        name: The profile's key, as `--style` spells it.
        width_in: Figure width in inches — the venue's column or text width.
        base_font_pt: Body text size; every other size is `TYPE_SCALE` times
            this one.
        font_family: matplotlib's family bucket, ``"serif"`` or
            ``"sans-serif"``.
        font_fallbacks: Preferred faces, most specific first. **The last entry
            must be a font matplotlib ships**, or a machine without the others
            warns — and the repository's pytest gate is zero-warning.
        aspect: Height as a fraction of width.
    """

    name: str
    width_in: float
    base_font_pt: float
    font_family: str
    font_fallbacks: tuple[str, ...]
    aspect: float = DEFAULT_ASPECT

    @property
    def figsize(self) -> tuple[float, float]:
        """Width and height in inches."""
        return (self.width_in, self.width_in * self.aspect)

    def rc_params(self) -> dict[str, Any]:
        """The matplotlib settings this profile imposes.

        Layered over `viz.style.theme.PLOT_STYLE` rather than replacing it:
        the grid, the line widths and the mathtext font are the project's
        house style and are the same at every venue. What a profile changes is
        the page — size, type scale, face — plus the font *embedding* §B.2
        requires and `PLOT_STYLE` never set.
        """
        scaled = {
            key: self.base_font_pt * factor
            for key, factor in (*TYPE_SCALE.items(), *MARK_SCALE.items())
        }
        return {
            **PLOT_STYLE,
            **scaled,
            "figure.figsize": self.figsize,
            "font.size": self.base_font_pt,
            "font.family": self.font_family,
            self._fallback_key: list(self.font_fallbacks),
            # §B.2, and §B.2.1's measured defect: without this, every PDF
            # this project writes carries Type 3 fonts.
            #
            # `ps.fonttype` is deliberately NOT set beside it. Mutation
            # testing removed it and nothing failed, which is the definition
            # of decoration -- this project emits PDF, PNG, SVG and GIF and no
            # PostScript at all, and matplotlib's SVG writer converts text to
            # paths, so there is no second embedding decision to make. A knob
            # for a format nothing writes is the failure F2a, G-2 and the
            # microbatch each recorded.
            "pdf.fonttype": TRUETYPE,
        }

    @property
    def _fallback_key(self) -> str:
        return "font.serif" if self.font_family == "serif" else "font.sans-serif"


def _serif(name: str, width: float, size: float, *, preferred: str) -> StyleProfile:
    return StyleProfile(
        name=name,
        width_in=width,
        base_font_pt=size,
        font_family="serif",
        font_fallbacks=(preferred, "Nimbus Roman", "Liberation Serif", "DejaVu Serif"),
    )


#: The five profiles of Annex 03 §B.2.
PROFILES: Mapping[str, StyleProfile] = {
    "thesis": _serif("thesis", 6.0, 10.0, preferred="Times New Roman"),
    "ieee-1col": _serif("ieee-1col", 3.5, 8.0, preferred="Times New Roman"),
    "ieee-2col": _serif("ieee-2col", 7.16, 8.0, preferred="Times New Roman"),
    # The geometry the ICASSP submission is actually typeset at, recovered
    # from the hand export that produced its PDFs: nine inches wide, sixteen
    # point body, and an aspect that gives the 9x5 in page those figures use.
    #
    # It exists because the alternative was worse. Those PDFs were made by an
    # untracked notebook calling `figure_from_artifacts` and then a bare
    # `plt.savefig`, which is outside every mechanism that sets `pdf.fonttype`
    # -- so all three carry the Type 3 fonts §B.2 forbids and IEEE rejects,
    # while all 43 PDFs in the store are clean. A venue the repository can
    # name is a venue the repository can render correctly.
    "ieee-paper": StyleProfile(
        name="ieee-paper",
        width_in=9.0,
        base_font_pt=16.0,
        font_family="serif",
        font_fallbacks=(
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ),
        aspect=5.0 / 9.0,
    ),
    "neurips": _serif("neurips", 5.5, 9.0, preferred="Times New Roman"),
    "talk": StyleProfile(
        name="talk",
        width_in=10.0,
        base_font_pt=16.0,
        font_family="sans-serif",
        font_fallbacks=("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"),
    ),
}

#: What a figure renders at when nothing says otherwise (§B.2's own default).
DEFAULT_PROFILE = "thesis"


def resolve_profile(name: str | None) -> StyleProfile:
    """The profile `name` refers to, or the default.

    Args:
        name: A profile key, or `None` for `DEFAULT_PROFILE`.

    Returns:
        The profile.

    Raises:
        KeyError: If no such profile exists, naming the ones that do — this is
            the surface a person types at.
    """
    key = DEFAULT_PROFILE if name is None else name
    profile = PROFILES.get(key)
    if profile is None:
        raise KeyError(
            f"no style profile {key!r}; available: {', '.join(sorted(PROFILES))}"
        )
    return profile


@contextmanager
def profile_context(profile: StyleProfile) -> Iterator[None]:
    """Render inside `profile`, restoring every setting afterwards.

    `rc_context` rather than a global `rcParams.update`, for the reason
    `viz.style.theme.styled_figure` uses one: a notebook that rendered one
    figure at `talk` and left the settings behind would silently restyle every
    figure after it.
    """
    with rc_context(profile.rc_params()):
        yield
