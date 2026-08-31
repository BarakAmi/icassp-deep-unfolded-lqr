"""Phase F acceptance tests for `viz.plots.regime_comparison`
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 7/11b): a genuinely new
chart shape (grouped bar: contender x regime), so its data contract --
consistent regime sets across every contender, a fixed color/order via
`regime_order` -- is exercised directly, not just "renders without error".
"""

import pytest
from matplotlib.figure import Figure

from mbl.viz.plots.regime_comparison import (
    RegimeComparisonStyle,
    plot_regime_comparison,
)

_COSTS = {
    "truncated_riccati": {"periodic": 2.1, "block_constant": 2.4, "fully_varying": 2.9},
    "standard_pgd": {"periodic": 2.0, "block_constant": 2.3, "fully_varying": 2.8},
    "unfolded_alpha": {"periodic": 1.95, "block_constant": 2.25, "fully_varying": 2.7},
    "unfolded_alpha_p": {"periodic": 1.7, "block_constant": 2.5, "fully_varying": 3.5},
}


def test_returns_a_figure_with_one_axes() -> None:
    fig = plot_regime_comparison(_COSTS)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_bar_count_equals_contenders_times_regimes() -> None:
    fig = plot_regime_comparison(_COSTS)
    (ax,) = fig.axes
    assert len(ax.patches) == len(_COSTS) * 3  # 4 contenders x 3 regimes


def test_legend_has_one_entry_per_regime() -> None:
    fig = plot_regime_comparison(_COSTS)
    (ax,) = fig.axes
    legend = ax.get_legend() or fig.legends[0]
    assert len(legend.get_texts()) == 3


def test_xtick_labels_are_the_contender_names_in_order() -> None:
    fig = plot_regime_comparison(_COSTS)
    (ax,) = fig.axes
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == list(_COSTS)


def test_empty_costs_rejected() -> None:
    with pytest.raises(ValueError, match="at least one contender"):
        plot_regime_comparison({})


def test_inconsistent_regime_sets_rejected() -> None:
    with pytest.raises(ValueError, match="regimes"):
        plot_regime_comparison(
            {"a": {"periodic": 1.0, "block_constant": 2.0}, "b": {"periodic": 1.0}}
        )


def test_regime_order_controls_bar_left_to_right_position() -> None:
    """The FIRST bar in each group (leftmost x-offset) must correspond to
    `regime_order[0]`, not dict insertion order."""
    fig = plot_regime_comparison(
        _COSTS, regime_order=["fully_varying", "block_constant", "periodic"]
    )
    (ax,) = fig.axes
    # Three containers, one per regime, added in `regime_order`'s order.
    assert len(ax.containers) == 3
    first_container_x = [patch.get_x() for patch in ax.containers[0].patches]
    second_container_x = [patch.get_x() for patch in ax.containers[1].patches]
    assert all(
        a < b for a, b in zip(first_container_x, second_container_x, strict=True)
    )


def test_style_labels_and_title_are_applied() -> None:
    style = RegimeComparisonStyle(xlabel="X", ylabel="Y", title="T")
    fig = plot_regime_comparison(_COSTS, style=style)
    (ax,) = fig.axes
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    assert ax.get_title() == "T"
