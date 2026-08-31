"""NB05's cross-regime comparison figure
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 7/11b): a grouped bar
chart -- one group per contender, one bar per LTV regime within each group
-- for a fixed/matched distinct-slice count ``D``, so the reader can
directly compare "reuse" (periodic) against "dwell time" (block-constant)
against "no structure" (fully varying) at the SAME representational
complexity, per contender.

A genuinely new chart shape (no existing renderer in this tree groups two
categorical dimensions -- contender AND regime -- into one barplot), so it
gets its own small module rather than stretching
`benchmarking.plot_cartesian_performance_tradeoff`'s single-category-per-bar
shape to fit. Regime identity is carried by BAR COLOR (a small, fixed
`CATEGORICAL_PALETTE` slice, consistent order every time this is called);
contender identity is carried by X-POSITION/tick label, not color -- so this
figure never collides with the `MODEL_FAMILY_COLORS` identity every other
figure in the notebook already uses for contenders.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..style import CATEGORICAL_PALETTE, place_legend_outside, styled_figure


@dataclass(frozen=True)
class RegimeComparisonStyle:
    """Presentation knobs for `plot_regime_comparison` beyond its data
    inputs (PLR0913: keeps the renderer under the 6-argument cap).

    Attributes:
        xlabel: Shared x-axis label.
        ylabel: Shared y-axis label.
        title: Figure title.
        legend_inside: Draw the legend inside the axes instead of the
            default external placement.
        group_width: Fractional width (of one contender slot) the whole
            group of per-regime bars occupies, leaving the remainder as
            inter-group whitespace.
    """

    xlabel: str = "Contender"
    ylabel: str = "Estimated average cost"
    title: str = "Contender Cost Across LTV Regimes (matched D)"
    legend_inside: bool = False
    group_width: float = 0.8


def plot_regime_comparison(
    costs: Mapping[str, Mapping[str, float]],
    *,
    regime_order: Sequence[str] | None = None,
    style: RegimeComparisonStyle = RegimeComparisonStyle(),
) -> Figure:
    """Grouped bar chart: one group per contender (x-axis), one bar per
    regime within each group (color-coded).

    Args:
        costs: contender label -> {regime label -> cost}. Every contender
            must declare the SAME set of regime labels (a matched-D
            comparison is meaningless if a contender is missing a regime
            another one has).
        regime_order: Fixed left-to-right bar order within each group (and
            hence the fixed color assignment -- the FIRST regime always
            gets `CATEGORICAL_PALETTE[0]`, etc., regardless of `costs`'
            own dict order); defaults to the regime labels' first
            appearance in `costs`' first contender.
        style: labeling/legend knobs (see `RegimeComparisonStyle`).

    Returns:
        The Figure.

    Raises:
        ValueError: If `costs` is empty, or any contender's regime label
            set disagrees with the first contender's.
    """
    if not costs:
        raise ValueError("costs must contain at least one contender.")

    contenders = list(costs)
    regimes = (
        list(regime_order) if regime_order is not None else list(costs[contenders[0]])
    )
    for label, per_regime in costs.items():
        if set(per_regime) != set(regimes):
            raise ValueError(
                f"Contender {label!r} has regimes {sorted(per_regime)}, "
                f"expected {sorted(regimes)} (every contender must share the "
                "same regime set for a matched-D comparison)."
            )

    n_groups, n_bars = len(contenders), len(regimes)
    x = np.arange(n_groups)
    bar_width = style.group_width / n_bars

    with styled_figure():
        fig, ax = plt.subplots(figsize=(1.8 * n_groups + 4, 6.5))
        for j, regime in enumerate(regimes):
            offsets = x + (j - (n_bars - 1) / 2) * bar_width
            values = [costs[contender][regime] for contender in contenders]
            ax.bar(
                offsets,
                values,
                width=bar_width * 0.92,
                color=CATEGORICAL_PALETTE[j % len(CATEGORICAL_PALETTE)],
                label=regime,
            )
        ax.set_xticks(x)
        # A mild rotation + right-anchored label so adjacent contender names
        # never collide (e.g. "unfolded_alpha" / "unfolded_alpha_p" abut at
        # 0 deg for anything but very wide figures) -- harmless even when
        # there IS room to spare.
        ax.set_xticklabels(contenders, rotation=15, ha="right")
        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(fig, ax, inside=style.legend_inside)

    return fig
