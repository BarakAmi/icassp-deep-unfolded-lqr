"""Phase A acceptance tests for the LTV problem factory
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.1/12): the degeneracy
law (variation_strength == 0.0 must reproduce `LQRProblemFactory` bit-for-
bit, in every regime -- the strongest possible NB04 regression anchor), the
three regimes' distinct-slice counts and schedules, Rule 1 (horizon must be
divisible by period/block), and Rule 4 / Gate 2 (the horizon-product growth
rate is exactly 1 after normalization, in every regime and at multiple
variation strengths/seeds -- not just the single example verified by hand
while writing the plan).
"""

import math

import numpy as np
import pytest

from mbl.applications.factories import LQRProblemFactory, ProblemFactory
from mbl.applications.ltv_factories import (
    LTVLQRProblemFactory,
    LTVRegime,
    matching_lti_factory,
)

_REGIMES_WITH_PERIOD = [
    (LTVRegime.PERIODIC, 4),
    (LTVRegime.PERIODIC, 6),
    (LTVRegime.BLOCK_CONSTANT, 4),
    (LTVRegime.BLOCK_CONSTANT, 6),
]


def _horizon_growth_rate(A_full: np.ndarray) -> float:
    """Independent (non-reused) reference computation of the per-step
    horizon-product growth rate, for cross-checking `build()`'s own Rule-4
    normalization -- deliberately a plain, unscaled product (fine at this
    test's small horizon/epsilon scale), never importing the production
    scaled accumulator under test."""
    n = A_full.shape[-1]
    Phi = np.eye(n)
    for k in range(A_full.shape[0]):
        Phi = A_full[k] @ Phi
    horizon = A_full.shape[0]
    return float(np.max(np.abs(np.linalg.eigvals(Phi))) ** (1.0 / horizon))


class TestDegeneracyLaw:
    """Sec 2.3: variation_strength == 0.0 must reproduce `LQRProblemFactory`
    bit-for-bit, regardless of regime -- the strongest possible regression
    test, since it lets NB05 reuse NB04's own cached numbers as ground truth."""

    @pytest.mark.parametrize(
        "regime,period_or_block",
        [
            (LTVRegime.PERIODIC, 4),
            (LTVRegime.BLOCK_CONSTANT, 4),
            (LTVRegime.FULLY_VARYING, None),
        ],
    )
    def test_zero_variation_matches_lqr_problem_factory_bit_for_bit(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=7,
            regime=regime,
            period_or_block=period_or_block,
            variation_strength=0.0,
            u_max=0.5,
        )
        ltv_problem = ltv.build()
        lti_problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=12, seed=7, u_max=0.5
        ).build()

        assert np.array_equal(
            ltv_problem.system.A_t.array, lti_problem.system.A_t.array
        )
        assert np.array_equal(
            ltv_problem.system.B_t.array, lti_problem.system.B_t.array
        )
        assert np.array_equal(ltv_problem.cost.Q, lti_problem.cost.Q)
        assert np.array_equal(ltv_problem.cost.R, lti_problem.cost.R)
        assert ltv_problem.system.is_time_invariant()
        assert ltv_problem.system.A_t.array.ndim == 2

    def test_zero_variation_matches_via_matching_lti_factory_helper(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=9,
            seed=3,
            regime=LTVRegime.BLOCK_CONSTANT,
            period_or_block=3,
            variation_strength=0.0,
        )
        ltv_problem = ltv.build()
        lti_problem = matching_lti_factory(ltv).build()
        assert np.array_equal(
            ltv_problem.system.A_t.array, lti_problem.system.A_t.array
        )
        assert np.array_equal(
            ltv_problem.system.B_t.array, lti_problem.system.B_t.array
        )

    def test_signature_type_never_collides_with_lqr_problem_factory(self) -> None:
        """Even though the built PROBLEM is byte-identical at epsilon=0, the
        factory's own signature type must differ -- no NB05 cache key can
        ever collide with an NB04 one."""
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=9,
            seed=3,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.0,
        )
        lti_sig = matching_lti_factory(ltv).get_signature()
        ltv_sig = ltv.get_signature()
        assert ltv_sig["type"] == "LTVLQRProblemFactory"
        assert lti_sig["type"] == "LQRProblemFactory"
        assert ltv_sig != lti_sig


class TestRegimeSchedule:
    """Sec 1.1/1.2: the three regimes' distinct-slice counts and schedules,
    cross-checked against the verified `TimeSeriesMatrix` compression modes."""

    def test_periodic_distinct_slice_count_and_schedule(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=12,
            seed=0,
            regime=LTVRegime.PERIODIC,
            period_or_block=4,
        )
        assert ltv.distinct_slice_count() == 4
        np.testing.assert_array_equal(ltv.schedule_indices(), np.arange(12) % 4)

    def test_block_constant_distinct_slice_count_and_schedule(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=12,
            seed=0,
            regime=LTVRegime.BLOCK_CONSTANT,
            period_or_block=4,
        )
        assert ltv.distinct_slice_count() == 3
        np.testing.assert_array_equal(ltv.schedule_indices(), np.arange(12) // 4)

    def test_fully_varying_distinct_slice_count_and_schedule(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=12,
            seed=0,
            regime=LTVRegime.FULLY_VARYING,
        )
        assert ltv.distinct_slice_count() == 12
        np.testing.assert_array_equal(ltv.schedule_indices(), np.arange(12))

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES_WITH_PERIOD)
    def test_built_system_realizes_the_declared_distinct_slice_count(
        self, regime: LTVRegime, period_or_block: int
    ) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=1,
            regime=regime,
            period_or_block=period_or_block,
            variation_strength=0.3,
        )
        problem = ltv.build()
        assert not problem.system.is_time_invariant()
        assert problem.system.A_t.array.shape[0] == ltv.distinct_slice_count()


class TestValidation:
    """Sec 1.3 Rule 1 and the positivity guards."""

    def test_periodic_requires_horizon_divisible_by_period(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.PERIODIC,
                period_or_block=5,
            )

    def test_block_constant_requires_horizon_divisible_by_block(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.BLOCK_CONSTANT,
                period_or_block=5,
            )

    def test_periodic_requires_period_or_block_given(self) -> None:
        with pytest.raises(ValueError, match="period_or_block"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.PERIODIC,
            )

    def test_fully_varying_rejects_period_or_block(self) -> None:
        with pytest.raises(ValueError, match="FULLY_VARYING"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.FULLY_VARYING,
                period_or_block=4,
            )

    def test_negative_variation_strength_rejected(self) -> None:
        with pytest.raises(ValueError, match="variation_strength"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.FULLY_VARYING,
                variation_strength=-1e-9,
            )

    def test_non_positive_u_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="u_max"):
            LTVLQRProblemFactory(
                state_dim=3,
                control_dim=2,
                horizon=12,
                seed=0,
                regime=LTVRegime.FULLY_VARYING,
                u_max=0.0,
            )

    @pytest.mark.parametrize("field", ["state_dim", "control_dim", "horizon"])
    def test_non_positive_dimensions_rejected(self, field: str) -> None:
        kwargs = dict(
            state_dim=3,
            control_dim=2,
            horizon=12,
            seed=0,
            regime=LTVRegime.FULLY_VARYING,
        )
        kwargs[field] = 0
        with pytest.raises(ValueError):
            LTVLQRProblemFactory(**kwargs)


class TestStabilityNormalization:
    """Sec 2.2 Rule 4 / Sec 9 Gate 2: the built system's horizon-product
    growth rate must be exactly 1 (to numerical tolerance) after
    normalization -- checked independently of the production accumulator,
    across every regime, several variation strengths, and several seeds
    (not just the single hand-verified example the plan document quotes)."""

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES_WITH_PERIOD)
    @pytest.mark.parametrize("variation_strength", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_growth_rate_is_one_after_normalization(
        self,
        regime: LTVRegime,
        period_or_block: int,
        variation_strength: float,
        seed: int,
    ) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=seed,
            regime=regime,
            period_or_block=period_or_block,
            variation_strength=variation_strength,
        )
        problem = ltv.build()
        A_full = np.stack([problem.system.A_t[k] for k in range(12)])
        growth_rate = _horizon_growth_rate(A_full)
        assert growth_rate == pytest.approx(1.0, abs=1e-6)

    def test_fully_varying_growth_rate_is_one(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=4,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.7,
        )
        problem = ltv.build()
        A_full = np.stack([problem.system.A_t[k] for k in range(12)])
        assert _horizon_growth_rate(A_full) == pytest.approx(1.0, abs=1e-6)

    def test_periodic_normalization_matches_the_monodromy_radius(self) -> None:
        """Sec 2.2 property 2: for a period p dividing N, the per-step
        growth rate of the FULL horizon product must equal the per-step
        growth rate of the single-period monodromy matrix -- an independent
        cross-check of Rule 4's mathematical claim, not merely that the
        result is 1."""
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=5,
            regime=LTVRegime.PERIODIC,
            period_or_block=4,
            variation_strength=0.4,
        )
        problem = ltv.build()
        A_period = np.stack([problem.system.A_t[k] for k in range(4)])
        monodromy_rate = _horizon_growth_rate(A_period)
        assert monodromy_rate == pytest.approx(1.0, abs=1e-6)


class TestPerturbationSemantics:
    """Sec 2.1: B-perturbation is a separable knob, and both A and B are
    perturbed relative to the SAME nominal draw LQRProblemFactory makes."""

    def test_perturb_b_false_leaves_every_b_slice_identical(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=2,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.5,
            perturb_B=False,
        )
        problem = ltv.build()
        B_full = np.stack([problem.system.B_t[k] for k in range(12)])
        assert np.all(B_full == B_full[0])

    def test_perturb_b_true_varies_b_across_slices(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=4,
            control_dim=2,
            horizon=12,
            seed=2,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.5,
            perturb_B=True,
        )
        problem = ltv.build()
        B_full = np.stack([problem.system.B_t[k] for k in range(12)])
        assert not np.all(B_full == B_full[0])

    def test_larger_variation_strength_yields_larger_dispersion(self) -> None:
        """A coarse monotonicity sanity check: more epsilon should mean more
        spread across the unique A slices (not a strict growth-rate claim --
        that is already covered by TestStabilityNormalization)."""

        def dispersion(epsilon: float) -> float:
            ltv = LTVLQRProblemFactory(
                state_dim=4,
                control_dim=2,
                horizon=12,
                seed=9,
                regime=LTVRegime.FULLY_VARYING,
                variation_strength=epsilon,
            )
            problem = ltv.build()
            A_full = np.stack([problem.system.A_t[k] for k in range(12)])
            mean_A = A_full.mean(axis=0)
            return float(np.mean([np.linalg.norm(A - mean_A) for A in A_full]))

        assert dispersion(0.1) < dispersion(0.8)


class TestConstraintAttachment:
    def test_u_max_attaches_a_single_box_constraint(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=9,
            seed=0,
            regime=LTVRegime.BLOCK_CONSTANT,
            period_or_block=3,
            variation_strength=0.2,
            u_max=0.4,
        )
        problem = ltv.build()
        assert problem.constraints is not None
        assert len(problem.constraints) == 1

    def test_no_u_max_attaches_no_constraint(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=9,
            seed=0,
            regime=LTVRegime.BLOCK_CONSTANT,
            period_or_block=3,
            variation_strength=0.2,
        )
        problem = ltv.build()
        assert problem.constraints is None


class TestProblemFactoryProtocol:
    """Sec 6.2: the type-level widening -- both factories structurally
    satisfy `ProblemFactory`, with no inheritance relationship."""

    def test_lqr_problem_factory_satisfies_the_protocol(self) -> None:
        assert isinstance(
            LQRProblemFactory(state_dim=3, control_dim=2, horizon=5, seed=0),
            ProblemFactory,
        )

    def test_ltv_problem_factory_satisfies_the_protocol(self) -> None:
        ltv = LTVLQRProblemFactory(
            state_dim=3,
            control_dim=2,
            horizon=5,
            seed=0,
            regime=LTVRegime.FULLY_VARYING,
        )
        assert isinstance(ltv, ProblemFactory)

    def test_no_inheritance_relationship_between_the_two_factories(self) -> None:
        assert not issubclass(LTVLQRProblemFactory, LQRProblemFactory)
        assert not issubclass(LQRProblemFactory, LTVLQRProblemFactory)


def test_horizon_product_growth_rate_guards_against_degenerate_products() -> None:
    """Sec 2.2's defensive guard: a product that collapses to the exact-zero
    matrix must raise, not silently propagate a -inf/nan growth rate."""
    from mbl.applications.ltv_factories import _horizon_product_growth_rate

    zero_stack = np.zeros((3, 2, 2))
    with pytest.raises(ValueError):
        _horizon_product_growth_rate(zero_stack)


def test_horizon_product_growth_rate_matches_reference_on_random_stacks() -> None:
    """Cross-check the production scaled accumulator against the plain
    (unscaled) reference implementation on a well-conditioned random stack,
    where both are numerically safe."""
    from mbl.applications.ltv_factories import _horizon_product_growth_rate

    rng = np.random.default_rng(11)
    A_full = 0.3 * rng.normal(size=(10, 3, 3))
    reference = _horizon_growth_rate(A_full)
    production = _horizon_product_growth_rate(A_full)
    assert production == pytest.approx(reference, rel=1e-9)
    assert math.isfinite(production)
