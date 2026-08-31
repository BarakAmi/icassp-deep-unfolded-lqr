# Cross-Machine Numerical Reproducibility: which quantities can be frozen, and which cannot

**Status:** describes implemented behaviour and a known limitation. Written 2026-07-31 after
the fourth occurrence of the same failure.

This document exists because one class of defect has now cost four separate debugging sessions,
each of which began by assuming a code regression and ended by discovering that a test was
asserting something no two machines owe each other. It states the mechanism, gives a
**measurement procedure** that settles the question in minutes instead of hours, and records
what must be re-checked when the affected code is rewritten.

Scope: [`tests/regression/test_golden_master.py`](../../tests/regression/test_golden_master.py)
and [`tests/regression/scenarios.py`](../../tests/regression/scenarios.py), plus the
degeneracy controls in
[`tests/applications/test_p_resolution_contenders.py`](../../tests/applications/test_p_resolution_contenders.py)
and
[`tests/models/unfolded/test_iteration_varying_riccati.py`](../../tests/models/unfolded/test_iteration_varying_riccati.py).

---

## 1. The rule

> **A test may assert equality only of quantities that are determined by the specification.**
> A quantity computed by an iterative procedure that has not converged is determined by the
> specification *and by the arithmetic*, and the arithmetic differs between machines.

Two machines running identical source produce identical results only where the arithmetic is
identical. It is not: floating-point addition is not associative, so a different BLAS build, a
different instruction set, a different thread count or a different reduction order gives a
different last bit. That difference is ~1e-7 relative in float32 — negligible in itself, and
irrelevant *unless the computation amplifies it*.

Whether a computation amplifies it is not a matter of opinion. It is measurable, and §3 gives
the procedure.

---

## 2. What went wrong, concretely

The golden-value harness was specified in
`foundational_architecture_consolidation_v3.md` §7.1,
which named four things to freeze:

1. `QuadraticCost.__call__` outputs,
2. `finite_horizon_riccati` P/K stacks,
3. one `OptimizationResult.J_history` per sweep topology,
4. the seeded `metadata.json` signature digests.

The implementation froze those **and a fifth category the plan never asked for**: individual
ensemble members of the gradient-descent solve — `U_final_head`, `X_final_head`,
`U_history_member0`, per variant.

The `signal_space_gd` scenario is float32, `alpha = 0.02`, `horizon = 100`, `max_iters = 100`,
and **deliberately does not converge** (`converged is False`, which the suite asserts itself).
An unconverged iterate is a point part-way along a trajectory; perturb the input in the last
bit and you land somewhere else along it. The measured consequence, from a single float32 ULP
(9.08e-8 relative) injected into the process noise:

| Frozen quantity | Category | Violation ratio |
|---|---|---|
| `A`, `B`, `P_arr`, `K_arr`, `U_opt*`, `x0_batch_head` | §7.1 (2) + setup | 0 |
| `{variant}__J_history` | §7.1 (3), batch-averaged objective | 0.019 |
| the four cost-convention curves (`standard_lqr`) | §7.1 (1) | 6.3e-4 |
| `{variant}__U_final_mean` / `X_final_mean` | batch reductions | 1.7 – 28 |
| `{variant}__U_final_head` | **individual members** | 42 – 100 |
| `{variant}__U_history_member0` | **one member's path** | 101 – 112 |
| `{variant}__X_final_head` | **individual members** | 780 – **1,257** |

Everything §7.1 specified is reproducible to within a factor of two of the arithmetic itself.
Everything added beyond §7.1 is not. **`random_jacobi__X_final_head` moves 7.3 % of its own
magnitude from one input ULP.**

Fifteen of the channel's forty arrays are therefore held to `rtol = 1e-5` when a single ULP
already violates it, by up to 7,331x. The suite passed for months because CI runners happened
to agree; when one did not, it failed on `cold_jacobi__U_final_head` — not because that array
is special, but because `numpy.testing.assert_allclose` raises on the first mismatch in sorted
key order, so every `X_final_*` was never even reached.

### 2.1 The same mechanism, second form

Three tests assert `torch.equal` — *bit* identity — between two mathematically equivalent
computations performed by **different op sequences**: a stacked `(J, T, …)` kernel against its
per-slice equivalent, and `P[j]` selected from a stack against `c_j · P`. The operands are
bit-identical by construction; the outputs are two different reduction orders of the same sum.
Bit identity there is a property of the BLAS, not of the mathematics.

---

## 3. The measurement procedure — use this first

Do **not** begin by trying to reproduce the other machine. That was attempted here across
batched-versus-single matmul, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` at 1–8,
`MKL_ENABLE_INSTRUCTIONS` at AVX512/AVX2/AVX/SSE4_2, and `ATEN_CPU_CAPABILITY` at
default/avx2/avx512. All were bit-identical locally; none reproduced anything. Reproduction
requires the other machine's CPU, which is not obtainable.

Perturb the **input** instead and measure how far the output moves. Two details decide whether
the answer is usable, and **both were got wrong on the first attempt**:

**Move every entry, not one.** Two hosts disagree in the last bit of *every* entry at once.
Measured on `cold_jacobi__X_final_meansq`, a single-entry perturbation moved it by 5.5e-7 while
the real cross-runner disagreement was 2.5e-5 — a factor of 46. A single-entry probe understates
systematically, and that understatement shipped once and reddened CI on the next pull request.

```python
rng = np.random.default_rng(seed)
towards = np.where(rng.integers(0, 2, size=w.shape).astype(bool),
                   dtype.type(np.inf), dtype.type(-np.inf))
w = np.nextafter(w.astype(dtype), towards)      # every entry, independent directions
```

**Measure the violation ratio, not the relative movement.** `assert_allclose` compares against
`atol + rtol·|desired|`, so the quantity that decides pass or fail is

```
ratio = max |fresh - golden| / (atol + rtol * |golden|)
```

A ratio of 1 is exactly the contract. Relative movement alone is misleading in both directions:
`w_mean` is the mean of zero-mean noise, so it moves enormously in relative terms and not at all
in the terms it is judged by — a relative-movement rule flagged it as a problem when it has
1,600× of room.

Repeat over at least three sign patterns and take the worst. Then:

| Violation ratio | Verdict |
|---|---|
| ≤ 0.5 | The contract holds with room for an unluckier sign pattern than the ones sampled. |
| 0.5 – 1 | Inside the contract, but one bad draw from failing. Exclude, or tighten the computation. |
| > 1 | **The quantity is not reproducible.** No tolerance is both safe and meaningful. |

The last row is the important one, and the reason is that widening the tolerance to accommodate
such a quantity produces a test that cannot fail on anything short of total breakage. **A
tolerance loose enough to accommodate a chaotic quantity is worse than not asserting it at all**,
because it presents as coverage and is unfailable.

The tool is committed: `tests/regression/measure_ulp_amplification.py`, with `--write`
regenerating `tests/regression/fixtures/ulp_movement.json`. That file records the tolerances the
ratios were computed under, so editing `TOLERANCES` without re-measuring is caught by a test
rather than discovered later.

## 4. Current state, and why it is contained rather than redesigned

The affected channel is a legacy asset. It was built for REFACTOR_PLAN v3, whose stages S0–S6
are complete, and it reaches the solver through `mbl.workbench`, which
Stage 6 of the renewed architecture
explicitly retires. Redesigning it would be investment in something already scheduled for
deletion.

It is therefore **contained**: the measured-unreproducible arrays are excluded from elementwise
comparison, each with its measured amplification recorded beside it as the justification.
Nothing is re-baselined and no fixture value is rewritten — the exclusion states which
quantities the harness was never able to promise.

**What the containment costs.** Batch statistics are invariant to which ensemble member is
which. With the per-member arrays no longer compared, a defect that permutes or mis-indexes
members — or that silently applies the wrong sweep order, and Jacobi versus Gauss-Seidel is one
of the four variants under test — produces identical statistics and passes unnoticed. This is
the one real gap the containment leaves, and §5 is where it gets closed.

---

## 5. What must be verified at the end of development

When Stage 6 replaces `workbench` and this channel is rewritten or deleted, the following must
be discharged. Until then this section is the outstanding obligation.

1. **Re-run the §3 measurement on every quantity the replacement suite freezes**, before
   capturing any fixture. A quantity above 100x amplification must not be frozen at all. This
   is a *pre*-capture step: capturing first and measuring afterwards is how the present
   situation arose.
2. **Restore member-permutation sensitivity.** Add a per-member *cost* array — costs are
   well conditioned (~2x) yet member-indexed, so a permutation or sweep-order defect still
   shows. Verify its amplification before freezing it.
3. **Confirm the sweep-order defect is actually detectable.** Mutate the sweep strategy
   (Jacobi ↔ Gauss-Seidel) and assert the suite goes red. This is the negative control for
   item 2, and without it item 2 is an assumption.
4. **Re-examine the three `torch.equal` degeneracy controls** (§2.1). The claim they encode —
   that R2 at J=1 degenerates to R1, and that each per-iteration slice equals the single-P
   refinement — is worth keeping. Split it: assert the *constructed parameters* are bit-identical
   (structural, exact, portable) and the *forward outputs* agree at a measured tolerance.
5. **Never re-baseline to make CI green.** No baseline satisfies two runners. If a golden
   disagrees across machines, the question is always whether the quantity should be frozen, not
   what number to write down.
6. **Measure that each surviving detector still fails.** Binary-search the smallest relative
   perturbation the contract still rejects, and record it next to the tolerance. A tolerance
   without that number is a guess.

---

## 6. What this obliges us to do in exchange

The containment buys a green CI at the price of the gap in §4. Three standing obligations
follow, and they apply to new code as much as to the replacement suite:

- **No new frozen quantity without a prior amplification measurement.** This is the cheapest
  possible discipline — one script, three injection points, two minutes — and it would have
  prevented all four occurrences.
- **State the working precision *and* the conditioning.** The existing tolerance table is keyed
  on a channel's declared dtype, which describes the arithmetic and says nothing about whether
  the computation amplifies it. Both are needed; the dtype alone is what made an unconverged
  iterate look like an ordinary float32 quantity.
- **Prefer a well-conditioned proxy to a loose tolerance.** Where a quantity of interest is
  chaotic, freeze something determined that would still move if it broke — the objective, a
  batch statistic, a per-member cost — rather than the quantity itself at a tolerance wide
  enough to survive.

---

## 7. Related

- `foundational_architecture_consolidation_v3.md` §7.1
  — the golden-value harness as specified.
- Annex 05 §11 — the six verification
  checkpoint classes (**D18**). The §3 procedure is an *independent recomputation* checkpoint;
  §5 item 3 is a *negative control*.
- Renewed Project Architecture — Stage 6
  retires `workbench`, which is what makes containment the proportionate response here.
