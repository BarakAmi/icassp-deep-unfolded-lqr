# The parallel campaign, measured

`icassp_exact_convex` re-derives the two convex-policy contenders of the ICASSP
campaign — solved as the structured problems they are rather than through a
general convex solver — and reuses every other result beside them. **The ICASSP
campaign is not edited, re-run or superseded**; where both versions run, its
stored result is the control this one is measured against.

Regenerate any row with `mbl run studies/icassp_exact_convex/<document>.toml
--tier <tier>`, then `mbl analyse` and `mbl figure render`. The reuse is the
point: the first run of each document trains only what is new.

## The names here are the campaign's, not the paper's

This document ships inside the paper's artifact repository, so the mapping has
to be stated rather than assumed:

| this document | the paper |
|---|---|
| `fig1_depth` | Figure 1 |
| `fig2_angle_blind` + `_told` + `_world` | Figure 2, composed of the three |
| **`fig5_stress_depth`** | **Figure 3** |
| `icassp_exact_convex_cost_grid_n4` | the cost table |
| `fig3_large_depth` | **nothing — not printed, and not released** |

**So the section below headed "`fig3_large_depth`" is not the paper's Figure
3**, and neither is "Table 2", which is measured at that study's instance. Both
are campaign results; their models are not in the released bundle and their
documents are not in the released tree.

Renaming the documents to match the paper was considered and refused. A study's
identifiers are derived from the document that declares it, so `fig5` becoming
`fig3` would move every `StudyID`, `ModelID` and `MeasurementID` the paper
reports and detach them from 25.6 hours of compute. The paper prints figures;
the store records documents; this table is the join.

### And the same for what the figures LABEL

`display` is what a reader is shown and takes no part in any identifier, so
these were changed after the results existed — measured before and after: both
StudyIDs and all 450 ModelIDs unchanged.

| stored `label` | drawn until 2026-09-06 | drawn now |
|---|---|---|
| `cocp_exact` | COCP (exact) | **COCP** |
| `finite_horizon_box` | `finite_horizon_box` | **Finite-horizon box bound** |

**"(exact)" was ours.** It separated this contender from the cvxpylayers-backed
`cocp` of the ICASSP campaign that it replaces — a distinction a reader of the
paper has not been given and does not need in a legend. It survives where it
belongs: in the joining `label`, and in this document.

**The bound was printing its registry key**, because a bound is not a contender
and so carried no `display` at all. The two figures that draw one were the only
two that could show it, and mathtext rendered the underscores as spacing, so it
appeared as "finite horizon box". Bound names now live beside the bound kinds
in `src/mbl/analysis/bounds.py`, and a test asserts the two key sets are equal
so a new bound cannot ship nameless.

## What the campaign actually cost

| document | tier | points | trained | reused |
|---|---|---|---|---|
| `fig1_depth` | publication_b16k | 225 | **10** | 215 |
| `fig2_angle_blind` | publication_b16k | 210 | **0** | 210 |
| `fig2_angle_told` | publication_b16k | 210 | **0** | 210 |
| `fig2_angle_world` | publication_b16k | 210 | **25** | 185 |
| `fig3_large_depth` | publication | 225 | **10** | 215 |
| `fig5_stress_depth` | publication | 225 | **225** | **0** |

**260 models across 1305 points.** `fig5_stress_depth` is the one document that reuses nothing, and that is its check rather than its cost: it declares a new plant, a new `ProblemID` moves every `ModelID`, and **any** reuse at all would have meant the box did not move.

Of the other five, **35 models across 1080 points.** The two Figure-2 conditions train nothing at
all: their convex policy is fitted on the nominal plant, which Figure 1 had
already produced, so identity alone reuses it across documents. `fig2_angle_world`
trains 25 rather than the 30 its own dry run predicted, for the same reason —
its unrotated cell is that same model.

## Figure 1 — the A/B, at n = 4

Same plant, same seeds, same evaluation. The only difference is how the per-step
QP is solved and differentiated, and whether the policy carries the linear term.

| campaign | contender | seeds | mean cost | across-seed σ | synthesis (provenance) |
|---|---|---|---|---|---|
| ICASSP | `cocp` | 5 | 8.270511 | 0.000309 | 2511.5 s |
| **exact** | `cocp_exact` | 5 | **8.270458** | 0.000296 | **75.6 s** |
| ICASSP | `cocp_lower_bound` | 5 | 8.272536 | 0 | 0.08 s |
| **exact** | `cocp_exact_lower_bound` | 5 | 8.272530 | 0 | 0.02 s |

**The dispersion column is the standard deviation of the per-seed aggregates —
the quantity the figures draw.** An earlier version of this table gave the
*range* (0.00078 and 0.00070) under the same heading, which is 2.4× the error
bar beside it on the figure. §A.3.2's rule that the three dispersions must not
share a name applies to prose as much as to code.

**The synthesis column is PROVENANCE, not a benchmark**: it is what the training
run recorded on a shared machine, and the ratio it gives (33.2×) is not the
measured one. The benchmarked figure is **29.3×**, from the cost grid, and it is
the number to quote — see "Table 1's rows, re-measured" below.

**−0.0006 % on cost**, with across-seed dispersions equal to the fourth decimal:
the two are statistically indistinguishable. **The two frozen policies agree to
0.000006 (7e-5 %)** — 8.272536 against 8.272530, one derived from the
semidefinite program and one from the dual. They are the same policy to five
decimal places, which is a stronger statement than the one this paragraph used
to make: an earlier version reported them 0.0029 apart, from a value for the
ICASSP frozen policy that appears nowhere in the store.

## `fig3_large_depth` — the campaign result at n = 100, m = 30

*Not the paper's Figure 3. This study is not printed and not released;
it is here because it is what the exact solver made possible.*

The ICASSP version draws seven curves. **Neither convex contender was in it**,
and neither could have been: the trained one was excluded by a contract that
rested on the reference's error and would have cost ~5 days per seed regardless,
the frozen one by a 4.9-hour synthesis that returned a value 0.030 *above* the
optimum it claimed to bound.

| contender | mean cost | across-seed spread | synthesis |
|---|---|---|---|
| **`cocp_exact`** | **60.1089** | 0.0007 | 4786–5026 s |
| best learned contender (UF-αP, J = 10) | 60.1166 | — | ~415 s |
| **`cocp_exact_lower_bound`** (frozen) | 60.7128 | **0** (deterministic) | **7.0 s** |
| clipped Riccati baseline | 64.1507 | — | — |
| recurrent baseline | 405.9 | — | ~336 s |

Two things worth stating plainly. **The convex policy is competitive at this
dimension** — marginally ahead of the best learned contender, which is a result
the ICASSP campaign could not have obtained at any price. And **the frozen
policy, synthesised in seven seconds and trained not at all, beats every fixed
unfolding and the clipped baseline**, landing within 1 % of the trained
contenders.

## The floor, drawn at last

Figure 1 and `fig3_large_depth` now carry a **box-aware lower bound**, which
the ICASSP campaign
records omitting and left as its plan's open decision: the bound available then
was an infinite-horizon steady-state average while the figures draw a
finite-horizon time-average, and *"plotting it beside these numbers would
bracket the truth with two different quantities"*. Computing the floor in the
evaluated convention closes that decision.

| | n = 4 | n = 100 |
|---|---|---|
| infinite-horizon bound (the semidefinite program's own quantity) | 6.14967 | 56.01282 |
| **finite-horizon bound — what these figures draw** | **5.58741** | **55.45806** |
| best attained contender | 8.27046 | 60.10890 |
| dual stationarity of the drawn bound | 1.68e-09 | 1.13e-09 |

Both conventions happen to sit below the best contender at these two instances,
so either would *look* right — and at n = 100 the infinite-horizon one already
sits **above** the unconstrained curve (55.430). Right by luck is not right, and
the renderer does not check which quantity a number is.

Two corrections travelled with it. **The frozen policy is a `BASELINE`, not a
`BOUND`**: it is a real, feasible, box-respecting policy whose attained cost is
an *upper* bound, and `role = "bound"` on it is verbatim the error the role
exists to prevent. And **Figure 1's axis breaks were re-authored**: the new floor
lands inside the range the ICASSP document declares empty, and the renderer
refused it by name rather than hiding a mark.

## What the speed-up is, at each size

| | n = 4 | n = 100 |
|---|---|---|
| the QP solve alone | ~470× | ~2100× |
| **end to end, real training** | **33×** | **~84×** |

The end-to-end factor is far below the isolated one and rises with dimension,
which is what the proposal predicted and refused to quote the larger number
against: once the QP is cheap, what remains is the rollout every contender
shares. At n = 100 the QP was 183× more expensive and the rollout was not, so
more of the saving survives. The n = 100 figure is against the ~5-day-per-seed
projection the exclusion rested on, not against a run that was ever made.

**Both grids in full — every contender, offline and online, time and memory —
are committed as [their own document](icassp_exact_convex_cost_tables.md)**, as
the ICASSP table is. What follows is the reading of them, not a substitute.

## Table 1's rows, re-measured — and a headline ratio that inverts

The same grid at **Figure 1's own instance and substrate** (n = 4, CPU/float64,
J = 3), same driver, same emitter, so these rows are directly comparable with
the published table cell for cell. Both phases measured: the palindrome halves
agree to **−0.30 %**, against the GPU substrate's −61 % and +86 %.

| quantity | ICASSP | exact | factor |
|---|---:|---:|---:|
| convex policy, offline [s] | 2430 | **82.8** | **29.3×** |
| convex policy, online setup [ms] | 696 | **27.0** | 25.8× |
| convex policy, per step [ms] | 0.453 | **0.105** | 4.3× |
| convex policy, offline peak RSS [MiB] | 4920 | **888** | 5.5× |
| frozen policy, synthesis [s] | 0.757 | **0.0238** | 31.8× |
| frozen policy, online setup [ms] | 737 | **38.3** | 19.2× |
| frozen policy, per step [ms] | 0.461 | **0.0975** | 4.7× |
| frozen policy, offline peak RSS [MiB] | 170 | **5.88** | 28.9× |

### The number the paper quotes does not survive

The cost table's headline reads *"the proposed controller trains **26× faster**
than the convex policy, and produces a control **11.4× faster**"*. Re-measured
against a convex policy solved as the structured problem it is:

| claim | as published | re-measured |
|---|---:|---:|
| proposed trains N× faster than the convex policy | 26.1× | **0.88× — inverted** |
| proposed produces a control N× faster | 11.4× | 2.6× |

**The convex policy now trains slightly faster than the proposed controller**
(82.8 s against 93.6 s), and the per-step advantage falls from an order of
magnitude to a factor of two and a half.

**Nothing about accuracy moves.** No controller's expected cost changes; what
changes is a *cost-of-computation* comparison that was, in part, a comparison of
implementations. This is the consequence the research proposal predicted before
any of it was built — *"a reviewer can already ask whether the baseline was
handicapped by tooling; answering that question with a measurement is a stronger
position than leaving it open, but it is a number that moves"* — and it has now
moved. The claim it leaves standing is narrower and better founded: the proposed
controller is competitive on cost with a well-implemented convex policy and
faster per step, rather than an order of magnitude cheaper than a baseline
running through a general-purpose solver.

**This is the author's call, not a correction to make silently.** The ICASSP
table and its numbers are untouched and remain correct for what they measured.

## Table 2 — the cost grid at `fig3_large_depth`'s instance and substrate

Table 1 is measured at n = 4 on CPU/float64. This is the same grid at n = 100,
m = 30 on **CUDA/float32**, at the same depth J = 3, produced by the same driver
and emitter. **The two are not comparable cell by cell**: different plant,
different substrate, different machine state. Table 1's own caveat already says
it — *the ratios transfer; the absolute numbers belong to this machine* — and
here even the ratios cross a substrate boundary, so only the within-table ones
mean anything.

| controller | offline time [s] | offline peak RSS [MiB] |
|---|---:|---:|
| Riccati (unconstrained) | 0.622 (0.534–0.709) | 350 |
| Truncated-Riccati | 0.555 (0.537–0.573) | 355 |
| Standard-PGD | 0.184 (0.182–0.186) | 142 |
| UF-α | 364 (355–373) | 1120 |
| **UF-αP (proposed)** | **422 (416–427)** | 1150 |
| UF-αPʲ | 400 (386–414) | 1150 |
| GRU | 474 (453–494) | 1130 |
| **COCP (exact)** | **5050 (4840–5260)** | 1270 |
| **Dual-frozen policy** | **7.58 (7.48–7.69)** | **91** |

Two readings. The convex policy trains at **12× the proposed controller's cost**
at this dimension — expensive, but *possible*, where before it was neither. And
the **dual-frozen policy synthesises in 7.58 s in 91 MiB**, the cheapest entry
in the table by an order of magnitude, for a controller that beats every fixed
unfolding and the clipped baseline.

The extrapolation checks out against the real thing: 5050 s extrapolated against
4786–5026 s actually measured when `fig3_large_depth` trained, an understatement of 3–8 %
— the same direction and size Table 1 records, because a per-epoch median
discards the slower first epoch.

| controller | online setup [ms] | per step [ms] | online peak RSS [MiB] |
|---|---:|---:|---:|
| Riccati (unconstrained) | 507 | 0.0272 | 540 |
| Truncated-Riccati | 568 | 0.0835 | 574 |
| Standard-PGD | 377 | 0.524 | 487 |
| UF-α | 402 | 0.402 | 488 |
| **UF-αP (proposed)** | 412 | **0.410** | 526 |
| UF-αPʲ | 412 | 0.557 | 523 |
| GRU | 343 | 0.175 | 510 |
| **COCP (exact)** | 968 | **1.17** | 732 |
| **Dual-frozen policy** | 7970 | 1.16 | 721 |

At inference the convex policy costs **2.9× the proposed controller per step**
(1.17 ms against 0.410 ms) and 1.4× its memory — the price of solving a QP at
every step rather than unrolling three fixed iterations. The frozen policy pays
the same per step and an 8-second setup, which is its one-time synthesis.

### The online columns nearly could not be measured, and why

#### The measurement, and three wrong answers before the right one

The palindrome gate refused these columns three times — **−61.0 %**, **+86.1 %**,
**+83.3 %** against a 60 % structural limit — and each refusal was right.

| hypothesis | verdict |
|---|---|
| thermal throttling | **refuted** — 50 °C and 52 W throughout |
| clock downshift | **refuted** — pinned at 2467 MHz *during* the degradation |
| host-side synchronisation cost | **refuted** — CUDA events show the **device** time itself moving 17× (15.5 → 294 µs) |
| **the GPU is not dedicated** | **confirmed** |

Under WSL2 the card also drives the **Windows desktop**. `nvidia-smi` reports
2860 MiB in use and 7 % utilisation beside *"No running processes found"*,
because it cannot see host processes. Our kernels contend for the SMs with a
compositor, and the *device* elapsed time genuinely stretches.

The mechanism gives the remedy. Contention can only make a call slower, so the
**cheapest observation** is the closest estimate of the uncontended cost — but
only if a clean one exists. Episodes last **about a second** while five blocks of
a fast policy span **fifty milliseconds**, so a whole cell can sit inside one
episode, see five uniformly slow blocks (`[215, 215, 215, 214, 211]` µs, agreeing
to 2 %) and settle on the contended cost. The cell now samples for a **minimum
wall-clock window** long enough to contain both states, reports the cheapest
block, and keeps the rest so the contention stays in the record.

| | before | after |
|---|---|---|
| repeatability across runs | 27.4 / 32.1 / 27.2 µs | **27.0 / 25.4 / 27.2 / 27.3 µs** |
| palindrome halves | −61 %, +86 %, +83 % | **−0.06 %** |

**The caveat this leaves is stronger than Table 1's.** That one says the absolute
numbers belong to one machine; these belong to one machine *whose GPU is shared
with a desktop*. Within-table ratios are what transfer.

## `fig5_stress_depth` — the paper's Figure 3, where the box binds nine times in ten

**`fig3_large_depth` is not edited, re-run or superseded by this.** Its study, its models,
its measurements and every number quoted above stand exactly as they are. This
is the same n = 100, m = 30 plant with **one declared scalar changed** — the
control box narrowed from `u_max = 0.1` to **0.02** — so `A`, `B`, `Q` and `R`
are bit-identical and `L = 1386.686968045241` carries over to the last digit.
The constraint is active on **89.6 %** of scalar control entries instead of
32.3 % — both read along the clipped loop at the `1e-6` tolerance the campaign's
earlier record used, so the two are a like-for-like comparison. It is the campaign's *fifth* figure because the cost grid was
already its fourth; in the paper it is **Figure 3**.

The regime is the point: a hundred-state plant under a constraint that binds
nine times in ten is what a box-aware method is *for*, and the campaign had
never measured it.

### The result, at J = 10, five seeds

Referenced to the unconstrained optimum (55.4296), which is the one quantity no
box touches and therefore the only honest yardstick across instances.

| contender | cost | × unconstrained | across-seed spread |
|---|---:|---:|---:|
| unconstrained Riccati (infeasible) | 55.4296 | ×1.000 | 0 |
| box-aware floor | 99.6420 | ×1.798 | — |
| **COCP (exact)** | **114.2114** | **×2.0605** | 0.0021 |
| UF-αPʲ | 114.2878 | ×2.0619 | 0.0026 |
| UF-αP (proposed) | 114.3096 | ×2.0622 | 0.0011 |
| dual-frozen policy (untrained, 26 s) | 114.4080 | ×2.0640 | 0 |
| recurrent baseline | 114.8257 | ×2.0716 | 0.0327 |
| UF-α | 131.2622 | ×2.3681 | 0.0052 |
| Standard-PGD (2/L) | 133.0119 | ×2.3997 | 0 |
| clipped Riccati | 150.8614 | ×2.7217 | 0 |

**What decides the outcome here is carrying a box-aware quadratic model**, and
the separation is architectural rather than about depth: it is worth **+16.4 %**
over the tuned fixed step, **+14.8 %** over learning the step alone and
**+32.0 %** over clipping, while depth over `J = 1..10` is worth **0.2 %**.

**The exact convex policy wins**, at 114.2114 against the best unfolded family's
114.2878 — 0.067 %, which is 29× the larger of the two across-seed σ (0.0026)
and therefore separable. `fig3_large_depth` already had it marginally ahead at the loose box; at this
one the margin has the same sign and is larger against the field.

**Training beats solving-and-freezing by a tenth of a per cent, and the tenth is
real.** Against the dual-frozen policy's deterministic 114.4080, COCP leads by
0.172 %, UF-αPʲ by 0.105 % and UF-αP by 0.086 % — one to two orders of magnitude
more than their own dispersion.

**Learning a step alone is the weakest contender, and that is all it is.** UF-α
reaches 131.2622 — ahead of the untuned fixed step by 1.32 %, and behind the
proposed contender by 14.8 %. Its across-seed spread is 0.0052 at the full
depth and never exceeds 0.0606, against 0.0011 to 0.0327 for the other
contenders: the largest in the cast, and of the same order.

**An earlier version of this document called it unstable, and that was our
optimiser rather than the contender.** Every unfolded family in this study had
inherited one learning rate and nothing had asked whether it suited them.
Sweeping it with everything else held — and with the incumbent rate reproducing
the tracked values bit-for-bit, so the comparison is inside this study rather
than across tiers — moved UF-α from 133.2149 with an across-seed spread of
1.0914 to the numbers above, a 210-fold collapse in dispersion. The claim that
survives is narrower and better founded: the step-only family depends on its
learning rate, the preconditioned families do not, and the step-only family is
still far behind once both are tuned.

**The honest cost of the figure**, accepted before it ran: the margin over the
recurrent baseline falls from **+3.41 %** at `u_max = 0.1` to **+0.45 %** here.
Both are ratios to the proposed contender — the convention this document's
+16.4 % / +14.8 % / +32.0 % also use — read off the drawn tables at `J = 10`:
62.1664 / 60.1170 and 114.8257 / 114.3096. (The pre-run ablation predicted
+3.43 % → +0.48 %, and those were the numbers this paragraph carried until they
were re-read off the publication rung. The prediction was good to two hundredths
of a point; it is still a prediction.) At this saturation every method converges
toward the constrained optimum. That is a property of the regime, not a defect
of the figure, and reporting it is worth more than a wide margin on a problem
where the constraint barely acts.

### The feasibility audit, which this instance made necessary

At a box this tight `max|u| − u_max` stops being a formality, so it is measured
per contender rather than asserted in prose — from the constraint-activity
statistic every measurement now retains, at a declared saturation tolerance of
`1e-3`.

| | |
|---|---|
| every feasible contender | `max\|u\| − u_max` = **−4.470e-10**, all eight |
| the unconstrained reference | `max\|u\|` = **0.611416**, **30.6×** the box |
| saturation, per contender | **82.93 % – 94.27 %** |

The audit counts at `1e-3`, where the clipped loop reads 89.67 % against the
89.65 % the `1e-6` convention gives it — a tolerance-insensitive contender. The
two readings are *not* interchangeable in general: across this cast the choice
moves the exact convex policy from 76.32 % to 87.82 %, and from last place to
third. Every saturation number is therefore quoted with the tolerance it was
counted at, and the store records that tolerance beside every fraction.

The identical margin is not eight measurements agreeing. It is
`0.02 − float32(0.02) = 4.470348362317633e-10` exactly: every feasible policy is
pinned at the **representable** bound, so the feasibility margin is a property of
the declared precision and not of any controller. Only a per-trajectory maximum
could have shown that; no aggregate could.

Saturation is genuinely a per-contender quantity — eleven points of spread across
policies flying one plant under one box — which is why the campaign's two
*conventions* dissolve here into two rows of one table: the "unconstrained-loop"
reading is simply the reference row, and it is the row expected to fail its own
feasibility check.

Regenerate the table with `tools/emit_feasibility_audit.py`; it reads the stored
measurements and re-rolls nothing.

## A limitation of the exact box-QP, measured rather than inferred

*Added 2026-08-23, prompted by a CI failure and settled by measurement. Nothing
below changes a published number; it states what those numbers rest on.*

`solve_box_qp` runs an active-set loop capped at 50 iterations and, on reaching
the cap, returns `converged = False` **rather than raising**. Two facts make
that worth writing down:

1. **No production code reads the flag.** Both call sites — the autograd wrapper
   and `ExactCOCPController.certificate_at` — take `solution.u` and discard
   `converged`. An uncertified solve therefore reaches a rollout silently.
2. **On this campaign's own tight-box instance, the flag is `False`.** Measured
   on the stored Figure-5 COCP model at its own evaluation states: every sampled
   time step hits the cap.

**What that does and does not mean**, because the two are easy to confuse. It is
*not* that the controls are wrong:

| check | result |
|---|---:|
| agreement with an independent interior-point solver (CLARABEL), 4 real states | **1.2e-7** against `u_max = 0.02` |
| mean \|shipped − well-converged reference\| over 122,880 entries | **1.29e-6** — 0.006 % of `u_max` |
| entries differing by more than 1 % of `u_max` | **58 of 122,880 — 0.047 %** |

So **99.95 % of control entries are right to about a part in ten thousand**, and
a few dozen are not. Three things explain the gap between that and a bare
`converged = False`:

* **The flag is a batch-wide maximum.** One unconverged entry out of 122,880
  condemns the whole batch, so at publication batch sizes the flag is close to
  useless as a summary — which is also why the CI test that asserts it is
  fragile.
* **It is not an iteration budget.** A float64 re-solve at **40× the cap** also
  returns `converged = False`, so raising the cap would not fix it; the residual
  is concentrated in a handful of near-degenerate states.
* **The tight box makes it worse.** At 88 % saturation, entries sitting a
  fraction outside the bound classify as *free* and their multipliers read as
  large stationarity residuals — the trap `certify_box_qp`'s own docstring
  names.

**The direction of the error is conservative for the paper's claim.** COCP wins
this figure. An under-solved QP can only make its cost *higher* than the exactly
solved policy would achieve, so the reported margin is a lower bound on the true
one, not an inflation of it.

**Nothing was changed in response.** Altering the solver would alter the
controls, hence the costs, hence every stored COCP number in this campaign and
in `fig3_large_depth` — for a defect confined to 0.05 % of entries whose direction does
not favour the result. The honest action is to record it here and to leave the
measurements as they are.

## Where the artifacts are

Each document renders into its own study directory under `store/studies/`, and
the figure ids carry `_exact` because `present.runner` refuses a figure id held
by two studies. Nothing under the ICASSP campaign's directories was written to.

## Related

* The reduced-form research proposal — the measurements that motivated this.
* The finite-horizon bound — the floor in the convention these costs use.
* The solver plan and the bounds plan.
* [The ICASSP campaign's own methodology](icassp_experimental_methodology.md) — the control this campaign is measured against.
