"""§B.5's axis rule keys on ROLE, not on whether a row happens to be flat.

The annex says the axis is "scaled to the data under study, not to the
**reference lines**", and that a **bound** far outside the contenders' range is
drawn clipped with its value in the legend. `_apply_axis_rule` implemented that
as ``table[table["axis_value"].notna()]`` -- "has a swept axis value" as a proxy
for "is not a reference".

The proxy held only while every depth-invariant series was a reference or a
bound. ICASSP Figure 1 is the first study where it is not: `truncated_riccati`
is `role = "baseline"` -- an attained, non-learned policy, and the number a
learned box-aware controller has to beat -- and it carries no unrolling depth,
so it has no axis value. Measured on the rendered figure: the depth-swept
contenders span 8.2742-8.8240, the upper panel was scaled to exactly that, and
Truncated-Riccati at **9.1272** was drawn off the canvas. It appeared in the
legend and nowhere on the axes.

Two depth-invariant CONTENDERS (the GRU at 8.3620, COCP at 8.2571) were visible
only because they happened to fall inside the swept range -- which is the
clearest statement of why this is a defect and not a rule: whether a contender
is shown depended on luck.

So the rule becomes what the annex already said: rows whose role is CONTENDER or
BASELINE are the data under study and set the limits whether or not they are
flat; REFERENCE and BOUND remain context and are clipped with their value in the
legend, which is §B.3.1's own encoding for them.
"""

import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from mbl.present.axis_scaling import _apply_axis_rule


def _table(rows: list[tuple[str, str, float | None, float]]) -> pd.DataFrame:
    """A minimal tidy table: (contender, role, axis_value, aggregate)."""
    return pd.DataFrame(
        {
            "contender": [r[0] for r in rows],
            "role": [r[1] for r in rows],
            "axis_value": [np.nan if r[2] is None else r[2] for r in rows],
            "aggregate": [r[3] for r in rows],
            "across_seed_spread": [0.0] * len(rows),
            "interval_low": [r[3] for r in rows],
            "interval_high": [r[3] for r in rows],
        }
    )


def _axes():
    return Figure().add_subplot(1, 1, 1)


#: ICASSP Figure 1's own shape, reduced to the four rows that decide the limits.
FIGURE_ONE = [
    ("unfolded_alpha", "contender", 1.0, 8.8018),
    ("unfolded_alpha", "contender", 10.0, 8.8240),
    ("unfolded_alpha_pj", "contender", 10.0, 8.2742),
    ("truncated_riccati", "baseline", None, 9.1272),
    ("riccati_unconstrained", "reference", None, 1.8603),
]


def test_a_flat_baseline_is_inside_the_axis() -> None:
    """The defect, as an executable claim. Truncated-Riccati at 9.1272 was
    drawn off a panel that stopped at ~8.83."""
    axes = _axes()
    _apply_axis_rule(axes, _table(FIGURE_ONE), {})
    low, high = axes.get_ylim()
    assert high >= 9.1272, (
        f"a baseline contender at 9.1272 is outside the axis [{low}, {high}]; "
        "it would appear in the legend and nowhere on the figure"
    )


def test_a_flat_contender_is_inside_the_axis() -> None:
    """The GRU and COCP carry no depth either. On the real figure they were
    visible only because they happened to land inside the swept range."""
    rows = [*FIGURE_ONE, ("neural", "contender", None, 12.5)]
    axes = _axes()
    _apply_axis_rule(axes, _table(rows), {})
    assert axes.get_ylim()[1] >= 12.5, axes.get_ylim()


def test_a_reference_far_below_still_does_NOT_set_the_axis() -> None:
    """The half of §B.5 that must survive the fix, and the reason the rule
    exists at all: the unconstrained Riccati at 1.8603 is an order of magnitude
    below every contender, and scaling to it would compress the whole result
    into a band. It stays context, clipped, with its value in the legend."""
    axes = _axes()
    _apply_axis_rule(axes, _table(FIGURE_ONE), {})
    low, _ = axes.get_ylim()
    assert low > 1.8603, (
        f"axis bottom {low} reaches down to the reference at 1.8603; §B.5's "
        "third rule is that a reference does not set the scale"
    )


def test_a_bound_far_outside_still_does_NOT_set_the_axis() -> None:
    """`role = "bound"` is context by the same rule -- COCP-LB is drawn with the
    reference-line grammar, not the contender grammar (§B.3.1)."""
    rows = [*FIGURE_ONE, ("sdp_floor", "bound", None, 0.5)]
    axes = _axes()
    _apply_axis_rule(axes, _table(rows), {})
    assert axes.get_ylim()[0] > 0.5, axes.get_ylim()


def test_the_inversion_still_scales_to_whatever_it_is_given() -> None:
    """§B.5.1 clause 5: the lower panel of a broken figure exists to hold the
    references, so `include_flat=True` scales TO them. Unchanged by this fix,
    and asserted because a panel holding only references is exactly the case
    that once took no limits at all and let matplotlib choose."""
    axes = _axes()
    _apply_axis_rule(
        axes,
        _table([("riccati_unconstrained", "reference", None, 1.8603)]),
        {},
        include_flat=True,
    )
    low, high = axes.get_ylim()
    assert low <= 1.8603 <= high, (low, high)
    assert high - low < 0.5, f"span {high - low} for a single level at 1.8603"


def test_a_declared_ylim_still_wins() -> None:
    axes = _axes()
    _apply_axis_rule(axes, _table(FIGURE_ONE), {"ylim": (0.0, 100.0)})
    assert axes.get_ylim() == pytest.approx((0.0, 100.0))


def test_a_table_of_only_references_takes_no_limits_without_the_inversion() -> None:
    """The early return must survive: with nothing but context and no
    inversion there is no data under study to scale to, and inventing limits
    would be worse than leaving them."""
    axes = _axes()
    before = axes.get_ylim()
    _apply_axis_rule(
        axes, _table([("riccati_unconstrained", "reference", None, 1.86)]), {}
    )
    assert axes.get_ylim() == before
