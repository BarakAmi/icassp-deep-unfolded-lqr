"""The across-seed interval must be a function of its inputs (Phase A2).

`scipy.stats.bootstrap` draws 10,000 resamples from a generator it creates
itself when none is given, so two analyses of one unchanged store returned
different confidence intervals — measured, up to 1.1e-3 apart run to run, and
up to 2.8e-2 against the tables the campaign stored. Every other column
re-derived at exactly 0.0.

Nothing published moves: the paper's figures declare `dispersion: none` and
draw no band. It matters anyway, for two reasons. A reviewer handed the result
bundle and told they can recompute every number will diff two parquet files and
find a mismatch in a correct bundle. And an acceptance test that compares a
recomputed analysis against a stored one has to special-case these two columns
forever, which is a permanent exemption bought to avoid a one-line fix.

The generator is constructed per call rather than threaded through the
reduction, and that is the property worth pinning: an interval must depend on
the values it summarises and on nothing else — not on how many rows were
reduced before it, and so not on the order of a table.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbl.analysis.reduction import Declared, interval_of


def _spec(resamples: int = 2_000) -> Declared:
    return Declared(
        axis_path="training.plan.epochs",
        quantity="trajectory_cost",
        aggregate="mean",
        dispersion="std",
        level=0.95,
        resamples=resamples,
    )


PER_SEED = [8.4412, 8.4390, 8.4501, 8.4377, 8.4468, 8.4423]


class TestTheIntervalIsAFunctionOfItsInputs:
    def test_two_calls_on_one_input_agree_exactly(self) -> None:
        spec = _spec()
        first = interval_of(PER_SEED, spec)
        second = interval_of(PER_SEED, spec)
        assert first == second, (
            f"{first} then {second}: the BCa bootstrap drew from an unseeded "
            "generator, so the interval moves on every analysis of unchanged "
            "data."
        )

    def test_the_interval_does_not_depend_on_what_was_reduced_before_it(
        self,
    ) -> None:
        """Order-independence, which a single shared generator would break.

        A table is reduced row by row. If one generator were threaded through
        the reduction, row five's interval would depend on rows one to four,
        and re-ordering a table — which changes no measurement — would change
        a published number.
        """
        spec = _spec()
        alone = interval_of(PER_SEED, spec)
        for other in ([1.0, 2.0, 3.0, 4.0], [90.0, 91.5, 89.2, 92.7]):
            interval_of(other, spec)
        after = interval_of(PER_SEED, spec)
        assert alone == after

    def test_it_is_still_a_real_interval_and_not_a_frozen_constant(self) -> None:
        """The anti-vacuity control.

        Seeding must not be achieved by degenerating the statistic. The
        interval has to bracket the mean with positive width, and it has to
        MOVE when the data moves — a constant would satisfy both determinism
        tests above and mean nothing.
        """
        spec = _spec()
        low, high = interval_of(PER_SEED, spec)
        assert low < float(np.mean(PER_SEED)) < high
        assert high - low > 0.0
        shifted = [value + 5.0 for value in PER_SEED]
        moved_low, moved_high = interval_of(shifted, spec)
        assert moved_low > low and moved_high > high

    @pytest.mark.parametrize("resamples", [500, 2_000])
    def test_determinism_holds_at_more_than_one_resample_count(
        self, resamples: int
    ) -> None:
        spec = _spec(resamples)
        assert interval_of(PER_SEED, spec) == interval_of(PER_SEED, spec)
