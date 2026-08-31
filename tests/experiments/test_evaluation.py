"""Acceptance tests for the evaluation payload — slice Phase B, step B0.

Written before the implementation. The requirement is
[Annex 03 §A.3.1](../../docs/architecture/03_analysis_and_visual_standard.md):
a stored measurement must retain the **per-trajectory** cost, because §A.3
names evaluation trajectories as the within-seed aggregation unit and §A.4's
paired procedures have no operand without them.

Two properties carry this suite, and the second is the one that is easy to
break while believing the phase went well:

* **The per-trajectory vector is an ADDITION.** `eval_expected_cost` keeps its
  `BATCH_MEAN` association order. `PER_SAMPLE` and `BATCH_MEAN` are distinct
  named orders with separate golden-master heritage
  (`core.kernels.quadratic`), so re-deriving the scalar from the per-trajectory
  vector would move every stored number in its last bits and buy nothing.
  The behavioural guard against that mistake was written first, measured, and
  **rejected as unfailable** — `TestTheScalarMetricIsUnmoved` records the
  measurement that killed it and pins the order where it is declared instead.

* **The pairing key must survive.** §A.4 pairs contenders trajectory by
  trajectory, which is only valid because the batches are common random
  numbers. `(batch_index, trajectory_index)` is therefore an identity that has
  to mean the same realisation for every contender, and it is asserted as such
  rather than assumed from the protocol's docstring.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

import mbl.experiments.evaluation as evaluation_module
from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments.evaluation import evaluate_synthesized_controller
from mbl.core.constraint.activity import SATURATION_TOLERANCE
from mbl.models.analytic.riccati import RiccatiSynthesizer
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from mbl.models.guards import require_linear_quadratic

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
HORIZON = 10
N_BATCHES = 3
BATCH_SIZE = 64


def _problem(*, seed: int = 0, u_max: float = 0.3) -> OptimalControlProblem:
    return LQRProblemFactory(
        state_dim=4, control_dim=2, horizon=HORIZON, seed=seed, u_max=u_max
    ).build()


def _artifact(problem: OptimalControlProblem):
    constraint = problem.constraints[0]  # type: ignore[index]
    return TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(problem, CTX)


def _batches(
    problem: OptimalControlProblem,
    *,
    n_batches: int = N_BATCHES,
    batch_size: int = BATCH_SIZE,
    seed: int = 1,
):
    spec = GaussianBatchSpec(
        state_dim=problem.system.dimensions.state_dim,
        horizon=HORIZON,
        batch_size=batch_size,
        seed=seed,
        process_noise_std=0.5,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    return tuple(sampler() for _ in range(n_batches))


def _roll_out(artifact, problem: OptimalControlProblem, batches):
    """Both reductions, per batch, from an independent rollout.

    Deliberately not a call into `evaluate_synthesized_controller`: a test that
    recomputed a quantity by asking the thing under test for it would assert
    that the function equals itself. The artifact here is analytic, so
    `make_policy`'s freshness law reproduces the same policy every time and the
    rollout is bit-identical to the one under test.
    """
    _, cost = require_linear_quadratic(problem)
    Q = torch.as_tensor(time_invariant_slice(cost.Q))
    R = torch.as_tensor(time_invariant_slice(cost.R))

    batch_means: list[float] = []
    per_sample: list[np.ndarray] = []
    for initial_state, process_noise, measurement_noise in batches:
        policy = artifact.make_policy()
        with torch.no_grad():
            X, _, U = problem.system.run(
                policy, initial_state, process_noise, measurement_noise
            )
            kwargs = {
                "conventions": cost.conventions,
            }
            batch_means.append(
                float(
                    total_quadratic_cost(
                        Q.to(dtype=X.dtype),
                        R.to(dtype=U.dtype),
                        X,
                        U,
                        reduction=CostReduction.BATCH_MEAN,
                        **kwargs,
                    )
                )
            )
            per_sample.append(
                total_quadratic_cost(
                    Q.to(dtype=X.dtype),
                    R.to(dtype=U.dtype),
                    X,
                    U,
                    reduction=CostReduction.PER_SAMPLE,
                    **kwargs,
                )
                .numpy()
                .astype(np.float64)
            )
    return np.asarray(batch_means, dtype=np.float64), np.stack(per_sample)


class TestThePerTrajectoryPayload:
    def test_a_measurement_carries_one_cost_per_trajectory(self) -> None:
        problem = _problem()
        _, arrays = evaluate_synthesized_controller(
            _artifact(problem), problem, _batches(problem)
        )

        assert "eval_trajectory_costs" in arrays
        assert arrays["eval_trajectory_costs"].shape == (N_BATCHES, BATCH_SIZE)

    def test_the_shape_follows_the_protocol_rather_than_a_constant(self) -> None:
        problem = _problem()
        _, arrays = evaluate_synthesized_controller(
            _artifact(problem),
            problem,
            _batches(problem, n_batches=5, batch_size=32),
        )

        assert arrays["eval_trajectory_costs"].shape == (5, 32)

    def test_the_values_are_the_independent_rollout_s_per_sample_totals(self) -> None:
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem)

        _, arrays = evaluate_synthesized_controller(artifact, problem, batches)
        _, expected = _roll_out(artifact, problem, batches)

        np.testing.assert_array_equal(arrays["eval_trajectory_costs"], expected)

    def test_the_batch_payload_is_retained_for_its_existing_consumers(self) -> None:
        # `workbench/replay.py` and `experiments/runner.py` read this key. The
        # per-trajectory vector is an addition; removing the batch payload
        # would be a silent breaking change to a path this phase does not own.
        problem = _problem()
        _, arrays = evaluate_synthesized_controller(
            _artifact(problem), problem, _batches(problem)
        )

        assert arrays["eval_batch_costs"].shape == (N_BATCHES,)


class TestTheScalarMetricIsUnmoved:
    """Which named association order produced `eval_expected_cost`.

    **The behavioural version of this test was written first and rejected on a
    measurement.** The obvious guard against "the implementer re-derived the
    scalar from the per-trajectory vector" is to assert that the metric differs
    from `eval_trajectory_costs.mean()`. On this fixture the two orders agree
    **bit-for-bit** — both read `3.1399457625146496` over 3 batches of 64
    float64 trajectories — so that assertion passes or fails by coincidence,
    and the coincidence is a property of the summation the runner's libraries
    happen to perform. Asserting it would have reproduced the one-ULP class of
    CI flake this project has already paid for once.

    So the order is pinned where it is actually declared: at the call. The spy
    records every `reduction` the evaluation asks for, which is deterministic,
    machine-independent, and fails loudly against exactly the mistake the
    annex forbids.
    """

    def test_the_scalar_is_reduced_by_batch_mean_at_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem)
        seen: list[CostReduction] = []
        original = evaluation_module.total_quadratic_cost

        def recording(*args: object, **kwargs: object):
            seen.append(cast(CostReduction, kwargs["reduction"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(evaluation_module, "total_quadratic_cost", recording)
        evaluate_synthesized_controller(artifact, problem, batches)

        assert seen.count(CostReduction.BATCH_MEAN) == N_BATCHES
        assert seen.count(CostReduction.PER_SAMPLE) == N_BATCHES

    def test_the_reduction_is_passed_explicitly_and_not_defaulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `total_quadratic_cost`'s own default is PER_SAMPLE, so an omitted
        # keyword would silently give the scalar the wrong order while the
        # spy above still counted the calls. The metric must name what it wants.
        problem = _problem()
        artifact = _artifact(problem)
        original = evaluation_module.total_quadratic_cost

        def recording(*args: object, **kwargs: object):
            assert "reduction" in kwargs
            return original(*args, **kwargs)

        monkeypatch.setattr(evaluation_module, "total_quadratic_cost", recording)
        evaluate_synthesized_controller(artifact, problem, _batches(problem))

    def test_the_scalar_keeps_its_association_order(self) -> None:
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem)

        metrics, _ = evaluate_synthesized_controller(artifact, problem, batches)
        batch_means, _ = _roll_out(artifact, problem, batches)

        assert metrics["eval_expected_cost"] == float(batch_means.mean())

    def test_the_two_orders_nevertheless_agree_to_tolerance(self) -> None:
        # What an analysis may rely on, stated as a number rather than as a
        # hope: aggregating the per-trajectory vector answers the same question
        # as the stored scalar, to floating-point tolerance and not exactly.
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem)

        metrics, arrays = evaluate_synthesized_controller(artifact, problem, batches)

        assert float(arrays["eval_trajectory_costs"].mean()) == pytest.approx(
            metrics["eval_expected_cost"], rel=1e-12
        )

    def test_the_batch_payload_still_means_the_batch_mean(self) -> None:
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem)

        _, arrays = evaluate_synthesized_controller(artifact, problem, batches)
        batch_means, _ = _roll_out(artifact, problem, batches)

        np.testing.assert_array_equal(arrays["eval_batch_costs"], batch_means)


class TestThePairingKeyIsCommonAcrossContenders:
    def test_the_same_index_is_the_same_realisation_for_two_artifacts(self) -> None:
        # §A.4 pairs trajectory by trajectory, and that is only valid because
        # the batches are common random numbers. Two artifacts that differ must
        # therefore produce DIFFERENT costs at the same index while having been
        # handed the SAME realisation -- which is what makes the difference a
        # paired one.
        problem = _problem()
        batches = _batches(problem)
        tight = _artifact(_problem(u_max=0.05))
        loose = _artifact(problem)

        _, tight_arrays = evaluate_synthesized_controller(tight, problem, batches)
        _, loose_arrays = evaluate_synthesized_controller(loose, problem, batches)

        assert (
            tight_arrays["eval_trajectory_costs"].shape
            == loose_arrays["eval_trajectory_costs"].shape
        )
        assert not np.array_equal(
            tight_arrays["eval_trajectory_costs"],
            loose_arrays["eval_trajectory_costs"],
        )

    def test_one_realisation_reaches_exactly_one_index(self) -> None:
        # The rollout must not reorder or reshape trajectories on its way out:
        # index i of the payload is the trajectory built from index i of the
        # sampled initial states. Perturbing a single realisation must move a
        # single cost.
        problem = _problem()
        artifact = _artifact(problem)
        batches = _batches(problem, n_batches=1)

        _, before = evaluate_synthesized_controller(artifact, problem, batches)

        initial_state, process_noise, measurement_noise = batches[0]
        perturbed = initial_state.clone()
        perturbed[7] = perturbed[7] * 2.0 + 1.0
        _, after = evaluate_synthesized_controller(
            artifact, problem, ((perturbed, process_noise, measurement_noise),)
        )

        moved = ~np.isclose(
            before["eval_trajectory_costs"][0],
            after["eval_trajectory_costs"][0],
            rtol=0.0,
            atol=0.0,
        )
        assert moved.sum() == 1
        assert bool(moved[7])


class TestTheMeasurementPassIsDeterministic:
    """A measured policy is in eval mode, and nothing else has to remember.

    `torch.no_grad()` keeps the rollout out of the autograd graph; it does
    **not** turn off dropout, and the two are routinely confused. A module left
    in training mode after `engine.train()` returns therefore scores under live
    dropout, and the same policy on the same batch gives a different number
    every call — measured on the campaign's GRU, **38.986540 and 38.988693**.

    That is not a small error, it is an unreproducible measurement: the number
    entering the store under a `MeasurementID` is one draw of a random
    variable, and re-running the identical study answers differently.

    Today the stack cannot build a multi-layer GRU (`num_layers` is unreachable
    from `NeuralRecipe`) and torch's `GRU` ignores `dropout` at one layer, so
    the defect is currently latent. This is the guard the layer ablation needs
    in place BEFORE it trains anything, and it is written against a module that
    is stochastic in training mode regardless of family.
    """

    class _StochasticController(torch.nn.Module):
        """A controller whose output is randomised in training mode only."""

        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

        def as_module(self) -> torch.nn.Module:
            return self

        def make_policy(self):
            def policy(t, y):  # noqa: ANN001, ANN202 -- the ControlPolicy shape
                del t
                control = y[..., :2] * 0.1
                return self.dropout(control)

            return policy

    class _Artifact:
        """The `TrainedControllerArtifact` shape: a `controller` attribute."""

        def __init__(self, controller: object) -> None:
            self.controller = controller

        def make_policy(self):
            return self.controller.make_policy()

    def test_a_module_left_training_is_measured_in_eval_mode(self) -> None:
        problem = _problem()
        batches = _batches(problem, n_batches=1, batch_size=16)
        controller = self._StochasticController()
        controller.train()  # exactly the state `engine.train()` leaves behind
        artifact = self._Artifact(controller)

        first, _ = evaluate_synthesized_controller(artifact, problem, batches)
        second, _ = evaluate_synthesized_controller(artifact, problem, batches)

        assert first["eval_expected_cost"] == second["eval_expected_cost"]

    def test_the_pass_actually_switches_the_mode(self) -> None:
        # Structural, because equality above could also hold by luck on a
        # fixture whose randomness happened not to bite.
        problem = _problem()
        batches = _batches(problem, n_batches=1, batch_size=16)
        controller = self._StochasticController()
        controller.train()
        assert controller.training

        evaluate_synthesized_controller(self._Artifact(controller), problem, batches)

        assert not controller.training

    def test_an_analytic_artifact_with_no_module_is_untouched(self) -> None:
        """`TrainedControllerArtifact.as_module` exists unconditionally and
        delegates, so a NumPy-native analytic controller answers `hasattr` and
        then raises on the call. The lookup goes through `controller`, exactly
        as `_weights_of` in the producer does."""
        problem = _problem()
        artifact = _artifact(problem)
        metrics, _ = evaluate_synthesized_controller(
            artifact, problem, _batches(problem)
        )
        assert metrics["eval_expected_cost"] > 0.0


class TestTheConstraintActivityStatistic:
    """Annex 03 §A.3.1a — retained content, added 2026-08-22.

    The requirement it closes had run for thirty study documents: they declared
    a `constraint_binds` gate and the store carried no quantity it could
    compare, so the analysis tier reported it not evaluable on every one.

    The properties that matter are the ones that would still let a wrong number
    into the store while every count agreed:

    * the payload must be **recomputable independently** from the same rollout,
      not merely present and plausible;
    * it must be **absent, never zero**, where the problem declares no box;
    * and retaining it must not move the cost the measurement already reported.
    """

    _COLUMNS = ("eval_max_abs_control", "eval_control_saturation")

    def test_it_is_retained_per_trajectory_on_the_costs_own_key(self) -> None:
        problem = _problem()
        batches = _batches(problem)
        _, arrays = evaluate_synthesized_controller(
            _artifact(problem), problem, batches
        )
        for column in self._COLUMNS:
            assert arrays[column].shape == arrays["eval_trajectory_costs"].shape

    def test_the_retained_values_are_the_rollouts_own(self) -> None:
        """Recomputed from an independent rollout rather than compared against
        the function's own output, which would assert that it equals itself."""
        problem = _problem()
        batches = _batches(problem)
        u_max = float(problem.constraints[0].u_max)  # type: ignore[index]

        expected_max: list[np.ndarray] = []
        expected_saturated: list[np.ndarray] = []
        artifact = _artifact(problem)
        for initial_state, process_noise, measurement_noise in batches:
            policy = artifact.make_policy()
            with torch.no_grad():
                _, _, U = problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                )
            expected_max.append(U.abs().amax(dim=(1, 2)).numpy())
            at_bound = U.abs() >= (1.0 - SATURATION_TOLERANCE) * u_max
            expected_saturated.append(
                at_bound.to(dtype=U.dtype).mean(dim=(1, 2)).numpy()
            )

        _, arrays = evaluate_synthesized_controller(artifact, problem, batches)
        assert arrays["eval_max_abs_control"] == pytest.approx(np.stack(expected_max))
        assert arrays["eval_control_saturation"] == pytest.approx(
            np.stack(expected_saturated)
        )

    def test_a_clipped_baseline_is_feasible_and_binds(self) -> None:
        """The statistic must be able to say something. A truncated-Riccati
        loop is feasible by construction and saturates wherever the free loop
        would exceed the box, so a payload reporting no activity at all — or
        an infeasible maximum — is broken rather than uninformative."""
        problem = _problem(u_max=0.05)
        u_max = float(problem.constraints[0].u_max)  # type: ignore[index]
        metrics, _ = evaluate_synthesized_controller(
            _artifact(problem), problem, _batches(problem)
        )
        assert metrics["eval_max_abs_control"] <= u_max
        assert metrics["eval_saturation_fraction"] > 0.0

    def test_the_maximum_is_a_maximum_and_the_fraction_is_a_mean(self) -> None:
        """The two scalars reduce differently and must not be reduced alike: a
        mean of per-trajectory maxima would report a policy feasible on the
        strength of its quiet trajectories.

        **Scored on an UNCONSTRAINED controller against a boxed problem**, and
        that is the whole of this test. Written first against the clipped
        baseline, it SURVIVED a mutant replacing the maximum with a mean —
        because a clipping policy pins *every* trajectory to the bound, so all
        192 per-trajectory maxima were exactly 0.3 and the two reductions
        differed by 7e-17, inside `pytest.approx`. The assertion held while the
        reduction was wrong; the guard written to prove it could fail passed on
        that same 7e-17. The unconstrained loop is where the two genuinely
        separate, and it is the row the figures actually carry.
        """
        problem = _problem()
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, CTX)
        metrics, arrays = evaluate_synthesized_controller(
            artifact, problem, _batches(problem)
        )
        maxima = arrays["eval_max_abs_control"]
        # The premise, asserted rather than hoped for: without a real spread
        # this test is the degenerate one it replaces.
        assert maxima.max() > 1.5 * maxima.mean()
        assert metrics["eval_max_abs_control"] == pytest.approx(maxima.max())
        assert metrics["eval_saturation_fraction"] == pytest.approx(
            arrays["eval_control_saturation"].mean()
        )

    def test_the_tolerance_travels_with_the_fraction(self) -> None:
        """A saturated fraction quoted without its tolerance says nothing, and
        a reader of a stored measurement has nowhere else to look."""
        problem = _problem()
        metrics, _ = evaluate_synthesized_controller(
            _artifact(problem), problem, _batches(problem)
        )
        assert metrics["eval_saturation_tolerance"] == SATURATION_TOLERANCE

    def test_it_is_absent_and_not_zero_without_a_box(self) -> None:
        """A saturated fraction of zero is a claim about a box, and an
        unconstrained problem has none. Absence is the honest answer and it is
        what makes the gate report *not evaluable* rather than *failed*."""
        problem = LQRProblemFactory(
            state_dim=4, control_dim=2, horizon=HORIZON, seed=0, u_max=None
        ).build()
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, CTX)
        metrics, arrays = evaluate_synthesized_controller(
            artifact, problem, _batches(problem)
        )
        assert not {*self._COLUMNS, "eval_saturation_tolerance"} & {
            *arrays,
            *metrics,
        }
        assert "eval_trajectory_costs" in arrays

    def test_retaining_it_did_not_move_the_cost(self) -> None:
        """The addition is a payload and nothing else. The scalar keeps its
        `BATCH_MEAN` association order, so a measurement re-run after this
        change must reproduce the number it reported before it — which is the
        acceptance check the Figure-5 backfill is built on."""
        problem = _problem()
        batches = _batches(problem)
        expected, _ = _roll_out(_artifact(problem), problem, batches)
        metrics, _ = evaluate_synthesized_controller(
            _artifact(problem), problem, batches
        )
        assert metrics["eval_expected_cost"] == float(np.mean(expected))
