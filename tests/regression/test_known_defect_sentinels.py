"""Stage S0 sentinels for the four known structural defects (REFACTOR_PLAN
v3 §1.1, C1-C4).

These tests originally FROZE THE DEFECTIVE BEHAVIOR AS IT EXISTED at S0.
They are the sanctioned exception ledger of the golden-master harness
(``test_golden_master.py``): each C-finding is intentionally corrected in a
later stage (C3 -> S1 [RESOLVED], C1/C2 -> S2 [RESOLVED], C4 -> S3), and
when that happens the matching sentinel MUST fail -- that failure is the
signal to replace the sentinel with the corresponding new acceptance test,
as a deliberate, reviewed change. If one of these fails at any other moment,
the behavior drifted unintentionally.

C3 was fixed in Stage S1 (T1.d named tolerance policy + explicit
forwarding); C1 (pure lifecycle protocols + `SynthesizerBase` default) and
C2 (the flag-honoring dual-backend quadratic kernel) were fixed in Stage S2;
C4 (per-recipe `TrainingPlan`/`OptimizerSpec` provenance) was fixed in Stage
S3. Every sentinel below is now the acceptance test asserting the corrected
behavior -- the exception ledger is closed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.applications.standard_lqr.app import StandardLQRApp
from mbl.applications.standard_lqr.config import StandardLQRConfig
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import default_compute_context
from mbl.core.system.linear_system import LinearSystem
from mbl.core.utils.validation import (
    validate_positive_definite,
    validate_positive_semi_definite,
)
from mbl.models.analytic.riccati import RiccatiController, RiccatiSynthesizer
from mbl.models.base import Controller
from mbl.persistence.local_tracker import LocalExperimentTracker


def _small_problem(
    *, include_terminal_cost: bool, is_time_averaged: bool
) -> tuple[OptimalControlProblem, RiccatiController, int]:
    """A deliberately small but non-trivial seeded LQR instance (n=2, m=1,
    T=8, marginally scaled A) for the C1/C2 sentinels."""
    horizon = 8
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 2))
    A /= np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.standard_normal((2, 1))
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(2)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(1)[None], horizon, axis=0)
    cost = QuadraticCost(
        Q=Q,
        R=R,
        include_terminal_cost=include_terminal_cost,
        is_time_averaged=is_time_averaged,
    )
    problem = OptimalControlProblem(system=system, cost=cost)
    return problem, RiccatiController(problem, horizon), horizon


class TestC1SignatureDisciplineResolved:
    """C1 RESOLVED (S2, T2.a): the deprecated `Controller` Protocol is now
    *pure* -- the concrete `get_signature` default that embedded
    ``self.problem.get_signature()`` (contradicting its own docstring, and
    never structurally inherited anyway) is gone. The correct, genuinely
    inheritable default lives on `models.lifecycle.SynthesizerBase`, and the
    lifecycle signature law holds: no signature at any phase embeds
    ``problem.*``. This replaces the S0 sentinel that froze the defective
    body."""

    def test_protocol_no_longer_carries_the_defective_default(self):
        """A pure Protocol body returns None when invoked explicitly; the
        pre-S2 defective body returned a dict embedding problem.*."""
        _, riccati, _ = _small_problem(
            include_terminal_cost=False, is_time_averaged=True
        )
        assert Controller.get_signature(riccati) is None

    def test_synthesizer_signature_is_specification_time_only(self):
        """`SynthesizerBase`'s default + overrides carry type and spec-time
        config only -- never problem.*, never live parameters."""
        _, _, horizon = _small_problem(
            include_terminal_cost=False, is_time_averaged=True
        )
        signature = RiccatiSynthesizer(horizon).get_signature()
        assert signature == {"type": "RiccatiSynthesizer", "horizon": horizon}

    def test_synthesized_artifact_signature_adds_provenance_not_problem(self):
        """`SynthesizedController.get_signature` = synthesizer signature +
        synthesis provenance (the effective compute context) -- and still no
        problem.* duplication anywhere in the tree."""
        problem, _, horizon = _small_problem(
            include_terminal_cost=False, is_time_averaged=True
        )
        artifact = RiccatiSynthesizer(horizon).synthesize(problem)
        signature = artifact.get_signature()
        assert signature["synthesizer"] == {
            "type": "RiccatiSynthesizer",
            "horizon": horizon,
        }
        assert signature["compute_context"] == default_compute_context().get_signature()

        def flat_keys(mapping):
            for key, value in mapping.items():
                yield key
                if isinstance(value, dict):
                    yield from flat_keys(value)

        assert "problem" not in set(flat_keys(signature))

    def test_concrete_controller_override_is_still_clean(self):
        """The legacy controllers' own signatures are unchanged (golden
        signature digests depend on them)."""
        _, riccati, horizon = _small_problem(
            include_terminal_cost=False, is_time_averaged=True
        )
        assert riccati.get_signature() == {
            "type": "RiccatiController",
            "horizon": horizon,
        }


class TestC2RolloutModelHonorsCostFlags:
    """C2 RESOLVED (S2, T2.d): `RolloutModel._differentiable_cost` now flows
    through the dual-backend quadratic kernel's ``BATCH_MEAN`` reduction and
    honors the cost's declared `include_terminal_cost`/`is_time_averaged`
    unconditionally -- the training loss IS the declared objective for every
    flag combination, not only for the default ``(False, True)``
    coincidence. This replaces the S0 sentinel that froze the
    flag-ignoring behavior."""

    @staticmethod
    def _rollout_and_costs(
        include_terminal_cost: bool, is_time_averaged: bool
    ) -> tuple[float, float]:
        """Returns (the batch mean of RolloutModel's per-trajectory costs,
        the declared QuadraticCost convention's batch-mean final value) on one
        seeded float64 batch.

        The rollout returns one cost per trajectory since Annex 06 §4.3's
        correction; reducing here keeps this class testing what it is for --
        that the cost FLAGS are honoured -- rather than the batch axis.
        """
        problem, riccati, horizon = _small_problem(
            include_terminal_cost=include_terminal_cost,
            is_time_averaged=is_time_averaged,
        )
        rng = np.random.default_rng(1)
        x0 = torch.as_tensor(rng.normal(size=(16, 2)), dtype=torch.float64)
        w = torch.as_tensor(
            0.1 * rng.normal(size=(16, horizon, 2)), dtype=torch.float64
        )
        v = torch.zeros((16, horizon, 2), dtype=torch.float64)

        X, _, U, rollout_cost = RolloutModel(riccati)(x0, w, v)
        declared = problem.cost(X.numpy(), U.numpy())[-1]
        return float(rollout_cost.mean()), float(declared)

    @pytest.mark.parametrize(
        ("include_terminal_cost", "is_time_averaged"),
        [(False, True), (True, True), (True, False), (False, False)],
    )
    def test_rollout_cost_tracks_the_declared_objective(
        self, include_terminal_cost, is_time_averaged
    ):
        """The flag-honoring law: for EVERY flag combination the training
        loss equals the declared cost convention's final cumulative value
        (up to floating-point association order -- the two ride different
        named reduction orders of the same objective)."""
        rollout_cost, declared = self._rollout_and_costs(
            include_terminal_cost, is_time_averaged
        )
        np.testing.assert_allclose(rollout_cost, declared, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize(
        ("include_terminal_cost", "is_time_averaged"),
        [(True, True), (True, False), (False, False)],
    )
    def test_flipping_a_flag_now_changes_the_training_loss(
        self, include_terminal_cost, is_time_averaged
    ):
        """The silent drift C2 named is dead: a non-default flag combination
        trains against a genuinely different objective value."""
        rollout_cost, _ = self._rollout_and_costs(
            include_terminal_cost, is_time_averaged
        )
        default_rollout_cost, _ = self._rollout_and_costs(False, True)
        assert abs(rollout_cost - default_rollout_cost) > 1e-6

    def test_default_flags_reproduce_the_pre_fix_loss_to_within_one_ulp(self):
        """Golden-master continuity (scenarios.py `J_opt` NOTE).

        **This asserted bit-for-bit equality until 2026-08-04 and no longer
        can.** Under the default flags the kernel's streamed `BATCH_MEAN` order
        *was* the pre-S2 operation sequence -- mean over batch and time -- so
        the frozen `J_opt` survived S2 by construction. `PER_SAMPLE` followed by
        `.mean()` is a third association order of the same objective, and it is
        the order the golden path now takes.

        Measured on this fixture: 2.220e-16 absolute, **0.80 ULP**. The frozen
        `J_opt` golden still passes at its own 1e-12 tolerance, so continuity
        holds -- what is gone is the stronger "by construction" claim, and
        saying so is the point of a sentinel. Asserted at four ULP: tight enough
        to fail on a real divergence, loose enough not to fail on association
        order."""
        _, riccati, horizon = _small_problem(
            include_terminal_cost=False, is_time_averaged=True
        )
        rng = np.random.default_rng(1)
        x0 = torch.as_tensor(rng.normal(size=(16, 2)), dtype=torch.float64)
        w = torch.as_tensor(
            0.1 * rng.normal(size=(16, horizon, 2)), dtype=torch.float64
        )
        v = torch.zeros((16, horizon, 2), dtype=torch.float64)

        rollout = RolloutModel(riccati)
        X, _, U, rollout_cost = rollout(x0, w, v)
        Q = rollout._Q.to(dtype=X.dtype)
        R = rollout._R.to(dtype=U.dtype)
        legacy_order = (
            torch.einsum("bti,ij,btj->bt", X[:, :-1], Q, X[:, :-1]).mean()
            + torch.einsum("bti,ij,btj->bt", U, R, U).mean()
        )
        assert torch.allclose(
            rollout_cost.mean(),
            legacy_order,
            rtol=4 * torch.finfo(torch.float64).eps,
            atol=0.0,
        )


class TestC3DefinitenessAtolForwarded:
    """C3 RESOLVED (S1, T1.d): `validate_positive_definite` /
    `validate_positive_semi_definite` now forward `atol` to the eigenvalue
    check exactly as their docstrings always promised, with an explicit
    `tol` override for the eigenvalue side and defaults drawn from the named
    tolerance policy (`SYMMETRY_ATOL`/`DEFINITENESS_TOL`). This replaces the
    S0 sentinel that froze the defective non-forwarding behavior."""

    def test_psd_loosened_atol_reaches_eigenvalue_check(self):
        """min eigenvalue -1e-7 with atol=1e-6: passes, because the loosened
        tolerance now governs the eigenvalue check (-1e-7 > -1e-6). Under the
        C3 defect this raised (the check ignored atol and ran at 1e-9)."""
        matrix = np.diag([1.0, -1e-7])
        assert validate_positive_semi_definite("M", matrix, atol=1e-6) is matrix

    def test_pd_tightened_atol_reaches_eigenvalue_check(self):
        """min eigenvalue 1e-12 with atol=1e-15: passes, because the
        tightened tolerance now governs the strict check (1e-12 > 1e-15).
        Under the C3 defect this raised (the check demanded > 1e-9)."""
        matrix = np.diag([1.0, 1e-12])
        assert validate_positive_definite("M", matrix, atol=1e-15) is matrix

    def test_default_call_sites_behave_exactly_as_before_the_fix(self):
        """No-argument calls keep the pre-S1 thresholds (symmetry 1e-12,
        eigenvalue 1e-9) -- the golden-master surface is untouched."""
        with pytest.raises(ValueError, match="positive definite"):
            validate_positive_definite("M", np.diag([1.0, 1e-12]))
        assert validate_positive_semi_definite("M", np.diag([1.0, -1e-10])) is not None

    def test_explicit_tol_overrides_atol_for_the_eigenvalue_check(self):
        """The (atol, tol) pair is fully explicit: tol wins the eigenvalue
        side, so a loose symmetry atol with a strict eigenvalue tol still
        rejects a slightly indefinite matrix."""
        matrix = np.diag([1.0, -1e-7])
        with pytest.raises(ValueError, match="positive semidefinite"):
            validate_positive_semi_definite("M", matrix, atol=1e-6, tol=1e-9)


class TestC4OptimizerProvenanceResolved:
    """C4 RESOLVED (S3, T3.b): each learned family owns a declarative
    `TrainingPlan`/`OptimizerSpec`; the live optimizer is BUILT from the
    spec (`OptimizerSpec.build`) and the logged `TrainingConfig` is DERIVED
    from the same spec (`TrainingPlan.training_config`) -- absolute
    provenance: there is no second place a learning rate can come from, so
    the pre-S3 misreport (one
    ``TrainingConfig(learning_rate=cfg.unfolded_learning_rate)`` logged for
    EVERY family) is structurally impossible, not just patched. This
    replaces the S0 sentinel that froze the defective behavior."""

    @staticmethod
    def _engine_for(app, name, problem, models, tmp_path):
        tracker = LocalExperimentTracker(tmp_path, f"c4_{name}")
        return app.build_engine(name, models[name], problem, tracker)

    @pytest.fixture()
    def app_and_models(self, tmp_path):
        cfg = StandardLQRConfig(
            state_dim=2,
            control_dim=1,
            horizon=6,
            batch_size=4,
            num_unfolding_iterations=2,
            unfolded_epochs=1,
            neural_hidden_dim=8,
            neural_epochs=1,
        )
        assert cfg.neural_learning_rate != cfg.unfolded_learning_rate
        app = StandardLQRApp(
            tracker_factory=lambda name: LocalExperimentTracker(tmp_path, name),
            config=cfg,
        )
        problem = app.build_problem()
        return cfg, app, problem, app.build_models(problem)

    @pytest.mark.parametrize(
        ("name", "lr_field"),
        [
            ("neural", "neural_learning_rate"),
            ("unfolded_learned_step_size", "unfolded_learning_rate"),
        ],
    )
    def test_logged_learning_rate_is_the_executed_learning_rate(
        self, app_and_models, tmp_path, name, lr_field
    ):
        """The absolute-provenance law, per family: executed optimizer lr ==
        logged TrainingConfig lr == the family's own configured lr."""
        cfg, app, problem, models = app_and_models
        runner = self._engine_for(app, name, problem, models, tmp_path)

        optimizer = runner.phases[0].strategy.optimizer
        optimizer_lr = optimizer.param_groups[0]["lr"]
        assert optimizer_lr == getattr(cfg, lr_field)
        assert runner.config.learning_rate == optimizer_lr
        assert runner.config.optimizer_name == "adam"
        assert isinstance(optimizer, torch.optim.Adam)

    def test_families_no_longer_share_one_logged_learning_rate(
        self, app_and_models, tmp_path
    ):
        """The exact C4 symptom, inverted: neural and unfolded runs now log
        DIFFERENT learning rates (each its own), where the defect logged the
        unfolded value for both."""
        cfg, app, problem, models = app_and_models
        neural = self._engine_for(app, "neural", problem, models, tmp_path)
        unfolded = self._engine_for(
            app, "unfolded_learned_step_size", problem, models, tmp_path
        )
        assert neural.config.learning_rate != unfolded.config.learning_rate
        assert neural.config.learning_rate == cfg.neural_learning_rate
        assert unfolded.config.learning_rate == cfg.unfolded_learning_rate

    def test_analytic_runs_carry_no_learning_rate_at_all(
        self, app_and_models, tmp_path
    ):
        """A non-trainable run has no optimizer to describe: its config
        carries ``None`` (and `ExperimentTrackingCallback` omits the
        optimizer-only fields from its metadata regardless)."""
        _, app, problem, models = app_and_models
        runner = self._engine_for(app, "analytic", problem, models, tmp_path)
        assert runner.config.learning_rate is None
