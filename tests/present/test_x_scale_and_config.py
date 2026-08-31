"""A declared x-scale, and a figure config that refuses what it cannot read.

Two work items that belong together, because the first would become an
instance of the second.

**The scale.** Annex 03 §B.5.1 draws the line: "the axis stays **linear**; a
break is a linear axis with a gap, and a figure whose data genuinely spans
decades takes a **logarithmic axis** instead." Two figures in the ICASSP
campaign span decades on the *x* axis and neither can be drawn today — the GRU
ablation's widths (8…256, six values, log base 2) and Figure 3's state
dimensions (4…50). `xscale` appears nowhere under `src/mbl/present`; the only
implementations are legacy. One mechanism serves both.

**The refusal.** Measured before this existed: `xscale`, `x_scale`, `log_x`,
`logx`, `scale`, `axis_value` and an outright `typo_key` were *all* accepted in
silence and *all* did nothing — a figure declaring any of them rendered a
byte-identical PNG and raised nothing. So a new key added for the campaign
would have joined six inert declarations this project has already shipped, and
the author would have read a log axis into a linear figure.

The scale is **declared, never derived** — the same rule §B.5.1 applies to the
break, and for the same reason: a scale inferred from the data moves whenever a
seed or a contender changes, and a figure whose axis moved with no document
edit is a figure whose two renders cannot be compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import CONFIG_KEYS, axis_scaling
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError

ROLES = {"a": Role.CONTENDER, "b": Role.CONTENDER}
ORDER = tuple(ROLES)

#: The ablation's own axis: six widths, log base 2.
WIDTHS = (8, 16, 32, 64, 128, 256)


def _table(values: tuple[int, ...] = WIDTHS, *, labels: tuple[str, ...] = ("a",)):
    rows: list[dict[str, Any]] = []
    for label in labels:
        for value in values:
            rows.append(
                {
                    "contender": label,
                    "role": Role.CONTENDER.value,
                    "axis_value": float(value),
                    "aggregate": 3.0 - 0.1 * np.log2(max(value, 1)),
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _figure(table: pd.DataFrame | None = None, **config: Any) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig",
            table=_table() if table is None else table,
            config=config,
            profile=resolve_profile("thesis"),
            series_order=ORDER,
            roles=ROLES,
        )
    )


class TestTheScaleIsDeclared:
    def test_nothing_declared_is_linear(self) -> None:
        figure = _figure()
        assert figure.axes[0].get_xscale() == "linear"
        plt.close(figure)

    def test_log_is_honoured(self) -> None:
        figure = _figure(xscale="log")
        assert figure.axes[0].get_xscale() == "log"
        plt.close(figure)

    def test_log2_is_a_log_axis_at_base_two(self) -> None:
        """The ablation's axis. Base 2 is not decoration: the widths are
        8, 16, 32, 64, 128, 256, and on a base-10 axis their ticks fall where
        no datum is."""
        figure = _figure(xscale="log2")
        axes = figure.axes[0]
        assert axes.get_xscale() == "log"
        # Read the base off the axis rather than trusting the call: a renderer
        # that set `log` and dropped the base would pass an xscale check.
        assert axes.xaxis.get_transform().base == 2
        plt.close(figure)

    def test_the_declaration_moves_the_drawn_ticks(self) -> None:
        """Perturbation, not invocation. `get_xscale()` reports what was asked
        for; the ticks report what a reader sees."""
        linear = _figure()
        linear.canvas.draw()
        linear_ticks = [t.get_text() for t in linear.axes[0].get_xticklabels()]
        plt.close(linear)

        logged = _figure(xscale="log2")
        logged.canvas.draw()
        log_ticks = [t.get_text() for t in logged.axes[0].get_xticklabels()]
        plt.close(logged)

        assert linear_ticks != log_ticks

    def test_an_unknown_scale_is_refused_by_name(self) -> None:
        with pytest.raises(SpecificationError, match="symlog|xscale"):
            _figure(xscale="symlog")

    def test_a_log_axis_over_a_non_positive_value_is_refused(self) -> None:
        """A log axis cannot show 0 or a negative, and matplotlib does not say
        so — it drops the point and draws the rest, which is a figure missing a
        mark its table still carries (§B.1.2 clause 2)."""
        table = _table(values=(0, 1, 2, 4))
        with pytest.raises(SpecificationError, match="positive|0"):
            _figure(table, xscale="log")

    def test_a_linear_axis_over_the_same_values_is_fine(self) -> None:
        # The boundary from the passing side: the refusal is about the SCALE,
        # not about the data.
        figure = _figure(_table(values=(0, 1, 2, 4)))
        assert figure.axes[0].get_xscale() == "linear"
        plt.close(figure)


class TestItComposesWithTheBrokenAxis:
    def test_a_log_x_and_a_broken_y_are_independent(self) -> None:
        """§B.5.1 breaks the VALUE axis; the scale here is the swept axis. A
        renderer that applied the scale to only the first panel would leave a
        broken figure with two different x axes under one shared x."""
        table = _table()
        table.loc[len(table)] = {
            "contender": "b",
            "role": Role.CONTENDER.value,
            "axis_value": np.nan,
            "aggregate": 0.5,
            "interval_low": np.nan,
            "interval_high": np.nan,
        }
        figure = _figure(table, xscale="log2", ybreak=[0.6, 2.0])
        assert len(figure.axes) == 2
        assert {axes.get_xscale() for axes in figure.axes} == {"log"}
        plt.close(figure)


class TestUnknownKeysAreRefused:
    """Every one of these was accepted in silence and did nothing."""

    @pytest.mark.parametrize(
        "key", ["x_scale", "log_x", "logx", "scale", "axis_value", "typo_key"]
    )
    def test_a_misspelled_or_invented_key_is_refused(self, key: str) -> None:
        with pytest.raises(SpecificationError, match=key):
            _figure(**{key: "log"})

    def test_the_refusal_names_what_is_available(self) -> None:
        """A refusal an author cannot act on is barely better than silence."""
        with pytest.raises(SpecificationError) as raised:
            _figure(typo_key=1)
        message = str(raised.value)
        for key in ("xscale", "series", "ybreak"):
            assert key in message, key

    @pytest.mark.parametrize("key", sorted(CONFIG_KEYS))
    def test_every_declared_key_is_actually_accepted(self, key: str) -> None:
        """The other direction, and the one that keeps `CONFIG_KEYS` honest: a
        key listed as known but not read would be an inert declaration wearing
        a validator's blessing."""
        values: dict[str, Any] = {
            "series": ["a"],
            "xlabel": "x",
            "ylabel": "y",
            "title": "t",
            "ylim": (0.0, 5.0),
            "ybreak": None,
            # §B.5.1 clause 3's re-weighting is refused WITHOUT a break (one
            # panel already has the whole height), so it cannot stand alone
            # any more than `ybreak` can — the skip below is its answer too.
            "ypanel_weights": None,
            "xscale": "linear",
            # "bar" and not "none": §A.3.2 rule 4 pairs `none` with a table
            # carrying the spread, and this fixture declares no table -- so
            # `none` here would be testing the refusal rather than the key.
            "dispersion": "bar",
        }
        config = {key: values[key]}
        if config[key] is None:
            pytest.skip(f"{key} has no non-trivial value that stands alone")
        figure = _figure(**config)
        plt.close(figure)

    def test_the_known_set_is_exactly_what_the_renderer_reads(self) -> None:
        """Structural, and the reason this test exists at all: `CONFIG_KEYS`
        is a hand-written list beside the code that reads the keys, so the two
        can drift. Grepping the module for `config.get(...)` and comparing is
        the census that notices."""
        import importlib
        import re
        from pathlib import Path

        # `import mbl.present.axis_scaling as module` binds the re-exported
        # FUNCTION, because `present/__init__.py` shadows the submodule with
        # it — so that spelling reads `__file__` off a function and raises.
        #
        # BOTH modules, because the renderer delegates: `series` is read by
        # `select_series` in `selection.py`, and a census of the renderer alone
        # reports it as declared-but-never-read. The effective read set spans
        # the helper, which is the thing an author's key actually reaches.
        sources = "".join(
            Path(str(importlib.import_module(name).__file__)).read_text(
                encoding="utf-8"
            )
            for name in ("mbl.present.axis_scaling", "mbl.present.selection")
        )
        read = set(re.findall(r'config(?:\.get)?[.(\[]\s*"(\w+)"', sources))
        assert read <= CONFIG_KEYS, f"read but not declared known: {read - CONFIG_KEYS}"
        assert CONFIG_KEYS <= read, (
            f"declared known but never read: {CONFIG_KEYS - read}"
        )


class TestTheTrackedFigureIsUnchanged:
    def test_a_figure_declaring_nothing_new_renders_as_before(self) -> None:
        # Neutrality. The tracked study declares xlabel/ylabel/title only, so
        # the validator must pass it untouched.
        figure = _figure(xlabel="depth", ylabel="cost", title="Cost vs depth")
        assert figure.axes[0].get_xscale() == "linear"
        plt.close(figure)


class TestALogAxisIsReadable:
    """A mark a reader cannot put a number to is a mark the figure did not
    really make."""

    def test_the_ticks_sit_at_the_plotted_values(self) -> None:
        """matplotlib's `LogLocator` thins a six-value base-2 axis to
        2^4, 2^6, 2^8 — measured — so three of six widths sat at no tick and a
        reader could not say which one they were looking at. That is §B.1.2's
        ruler problem arriving through the axis instead of the table."""
        figure = _figure(xscale="log2")
        figure.canvas.draw()
        shown = [
            float(text.get_text())
            for text in figure.axes[0].get_xticklabels()
            if text.get_text()
        ]
        assert shown == [float(width) for width in WIDTHS]
        plt.close(figure)

    def test_the_labels_are_plain_numbers_not_powers(self) -> None:
        # `2^{4}` is the width 16 written in a notation the study document does
        # not use; a reader matching the figure to the table should not have to
        # exponentiate.
        figure = _figure(xscale="log2")
        figure.canvas.draw()
        texts = [t.get_text() for t in figure.axes[0].get_xticklabels() if t.get_text()]
        assert "16" in texts
        assert not any("mathdefault" in text or "^" in text for text in texts)
        plt.close(figure)

    def test_a_linear_axis_keeps_matplotlibs_own_ticks(self) -> None:
        # The change is scoped to a DECLARED log scale; a linear figure must
        # keep the autoscaled ticks every existing figure was drawn with.
        figure = _figure()
        figure.canvas.draw()
        shown = [
            float(t.get_text().replace("−", "-"))
            for t in figure.axes[0].get_xticklabels()
            if t.get_text()
        ]
        assert shown != [float(width) for width in WIDTHS]
        plt.close(figure)
