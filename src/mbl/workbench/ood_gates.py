"""Regime/discriminability/stationarity pre-flight gates for NB06
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 3.11): three fail-fast,
value-reporting checks a zero-shot OOD run must pass BEFORE its sweeps are
read as meaning anything.

The executed v1 run passed the box-binding gate that existed
(`workbench.analysis.compute_box_binding_fraction`, one-sided:
``binding > 0.01``) while sitting at 77.6% binding and 84.3% nominal
saturation for every contender -- a regime in which every controller is
effectively bang-bang and cannot express its own architecture, so every OOD
curve in that run was six near-coincident lines (NB06 plan Sec 11 S1). The
SAME configuration's per-step-averaged cost nearly doubled from horizon 15
to 60 (Sec 11 S5), meaning the closed loop never reaches a stationary
distribution -- invalidating the infinite-horizon SDP floor as a reference
and making the horizon axis uninterpretable on its own terms.

Each gate here is a pure function of ALREADY-MEASURED numbers (this module
computes nothing about a problem or policy itself -- it only judges values
the caller measured with existing machinery), and returns a small frozen
dataclass carrying the measured value, the threshold, and the pass/fail
verdict, so a notebook can print a full diagnostic line -- including a
named, actionable fix -- rather than a bare assertion that gives no clue
which Control-Panel knob to turn.

Not exported from `workbench/__init__.py`'s aggregation, deliberately
mirroring `workbench.ood_sweep`'s own precedent: that package's eager
re-export chain (`applications.studies` -> `experiments` -> ...) closes a
real circular-import cycle when `experiments` is the first module a fresh
process touches (see `viz.plots.ood`'s module docstring for the full
account) -- every OOD-specific module is imported by its own path instead
(``from mbl.workbench.ood_gates import ...``), exactly how the notebook
already imports `ood_sweep`.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxBindingGateResult:
    """Verdict of `check_box_binding_window` (NB06 plan Sec 3.11).

    Attributes:
        fraction: The measured box-binding fraction (e.g. from
            `workbench.analysis.compute_box_binding_fraction`).
        window: The ``(low, high)`` pass window.
        passed: Whether `fraction` fell inside `window`.
    """

    fraction: float
    window: tuple[float, float]
    passed: bool

    def diagnosis(self) -> str:
        """A human-readable line: the measured value, the window, and --
        on failure -- which Control-Panel knob to turn."""
        low, high = self.window
        verdict = "PASS" if self.passed else "FAIL"
        line = (
            f"[Box-binding gate] {verdict}: fraction={self.fraction:.1%}, "
            f"window=[{low:.0%}, {high:.0%}]"
        )
        if self.passed:
            return line
        if self.fraction > high:
            return (
                f"{line} -- box binds on ALMOST EVERY entry; every contender "
                "degenerates to bang-bang and cannot express its own "
                "architecture. Raise U_MAX or lower PROCESS_NOISE_STD."
            )
        return (
            f"{line} -- box barely binds; the constrained framing is "
            "nearly vacuous. Lower U_MAX or raise PROCESS_NOISE_STD."
        )


def check_box_binding_window(
    binding_fraction: float, *, window: tuple[float, float] = (0.05, 0.60)
) -> BoxBindingGateResult:
    """Is the box binding often enough to matter, but not so often every
    contender is forced bang-bang (NB06 plan Sec 3.11, gate 1)?

    Args:
        binding_fraction: The measured fraction (e.g. from
            `workbench.analysis.compute_box_binding_fraction` on the
            UNCONSTRAINED policy's rollout).
        window: The ``(low, high)`` pass window; the v1 run's 77.6% sits
            above the default high end.

    Returns:
        The `BoxBindingGateResult`.
    """
    low, high = window
    return BoxBindingGateResult(
        fraction=binding_fraction,
        window=window,
        passed=low <= binding_fraction <= high,
    )


@dataclass(frozen=True)
class SeparationGateResult:
    """Verdict of `check_nominal_separation` (NB06 plan Sec 3.11).

    Attributes:
        spread: ``max(nominal_costs) - min(nominal_costs)``.
        mc_stderr: The Monte-Carlo standard error the spread is judged
            against.
        ratio: ``spread / mc_stderr`` (``inf`` if `mc_stderr` is ``0``).
        threshold: The minimum required `ratio`.
        passed: Whether `ratio >= threshold`.
    """

    spread: float
    mc_stderr: float
    ratio: float
    threshold: float
    passed: bool

    def diagnosis(self) -> str:
        """A human-readable line: the measured spread/ratio and, on
        failure, what it means."""
        verdict = "PASS" if self.passed else "FAIL"
        line = (
            f"[Nominal-separation gate] {verdict}: spread={self.spread:.4g}, "
            f"mc_stderr={self.mc_stderr:.4g}, ratio={self.ratio:.2f}x "
            f"(need >= {self.threshold:.1f}x)"
        )
        if self.passed:
            return line
        return (
            f"{line} -- the contenders are indistinguishable from Monte-Carlo "
            "noise at the nominal point; every OOD curve below would compare "
            "six copies of the same number. Raise eval_batches/batch_size to "
            "shrink mc_stderr, or revisit the problem's difficulty."
        )


def check_nominal_separation(
    nominal_costs: Mapping[str, float], mc_stderr: float, *, threshold: float = 3.0
) -> SeparationGateResult:
    """Do the contenders' nominal costs actually differ by more than
    Monte-Carlo noise (NB06 plan Sec 3.11, gate 2)?

    Args:
        nominal_costs: label -> nominal (in-distribution) cost, one entry
            per contender.
        mc_stderr: The Monte-Carlo standard error of a single contender's
            cost estimate (e.g. ``cost_std / sqrt(effective_sample_size)``
            from a pooled `experiments.zero_shot.ShiftMetrics`).
        threshold: The minimum required ``spread / mc_stderr`` ratio.

    Returns:
        The `SeparationGateResult`.

    Raises:
        ValueError: If `nominal_costs` is empty.
    """
    if not nominal_costs:
        raise ValueError("nominal_costs must contain at least one contender.")
    values = list(nominal_costs.values())
    spread = max(values) - min(values)
    ratio = spread / mc_stderr if mc_stderr > 0 else math.inf
    return SeparationGateResult(
        spread=spread,
        mc_stderr=mc_stderr,
        ratio=ratio,
        threshold=threshold,
        passed=ratio >= threshold,
    )


@dataclass(frozen=True)
class StationarityGateResult:
    """Verdict of `check_closed_loop_stationarity` (NB06 plan Sec 3.11).

    Attributes:
        cost_at_half: Per-step-averaged cost at half the nominal horizon.
        cost_at_full: Per-step-averaged cost at the nominal horizon.
        relative_change: ``abs(cost_at_full - cost_at_half) / cost_at_half``.
        tolerance: The maximum allowed `relative_change`.
        passed: Whether `relative_change <= tolerance`.
    """

    cost_at_half: float
    cost_at_full: float
    relative_change: float
    tolerance: float
    passed: bool

    def diagnosis(self) -> str:
        """A human-readable line: the measured costs/change and, on
        failure, what it means."""
        verdict = "PASS" if self.passed else "FAIL"
        line = (
            f"[Stationarity gate] {verdict}: cost(N/2)={self.cost_at_half:.4g}, "
            f"cost(N)={self.cost_at_full:.4g}, relative_change="
            f"{self.relative_change:.1%} (need <= {self.tolerance:.0%})"
        )
        if self.passed:
            return line
        return (
            f"{line} -- the per-step-averaged cost is still moving between "
            "N/2 and N: the closed loop has not reached a stationary "
            "distribution, so the infinite-horizon SDP floor is not a valid "
            "reference and the horizon axis measures accumulation, not "
            "generalization. Raise U_MAX (more control authority) or lower "
            "PROCESS_NOISE_STD."
        )


def check_closed_loop_stationarity(
    cost_at_half: float, cost_at_full: float, *, tolerance: float = 0.05
) -> StationarityGateResult:
    """Has the per-step-averaged cost converged between half and full
    horizon (NB06 plan Sec 3.11, gate 3)?

    Args:
        cost_at_half: Per-step-averaged cost evaluated at ``horizon=N/2``
            (e.g. via `applications.ood.perturbations.PerturbedLQRProblemFactory`
            with ``horizon_override=N//2`` and an identity perturbation, plus
            `experiments.zero_shot.evaluate_under_shift`).
        cost_at_full: The same, at the nominal ``horizon=N``.
        tolerance: The maximum allowed relative change.

    Returns:
        The `StationarityGateResult`.

    Raises:
        ValueError: If `cost_at_half` is not strictly positive (the
            relative-change denominator would be undefined or misleading).
    """
    if cost_at_half <= 0:
        raise ValueError(f"cost_at_half must be > 0, got {cost_at_half}.")
    relative_change = abs(cost_at_full - cost_at_half) / cost_at_half
    return StationarityGateResult(
        cost_at_half=cost_at_half,
        cost_at_full=cost_at_full,
        relative_change=relative_change,
        tolerance=tolerance,
        passed=relative_change <= tolerance,
    )
