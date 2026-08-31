# The ICASSP experiments: what is measured, and what each number means

The methodology behind the paper's figures, written to be **transcribed into the notebook's
prose**. Every claim here is measured on the tracked studies rather than reasoned about, and each
one exists because getting it wrong would let a reader — or a reviewer — draw a conclusion the
experiment does not support.

> **Transcription rule.** The notebook's markdown carries **zero code references**: no function
> names, no file paths, no field names, no identifiers. That is a standing rule for every
> notebook in this project, recorded twice after being broken twice. This document names
> implementation surfaces so that the claims can be *checked*; the notebook states the same
> claims in prose. Where a paragraph below names a mechanism, the notebook says what it
> guarantees, not what it is called.

---

## 1. The plotted quantity is inference, never training loss

Every curve and every level in Figures 1–3 is the **expected cost of rolling out the frozen,
synthesised controller** on evaluation trajectories it never trained on. A training loss appears
in no figure.

Two distinct batches are involved, and confusing them is the easiest way to misread the results:

| | size | how it is drawn |
|---|---|---|
| **Training** | 8192 trajectories per step | a stream derived from the study's data seed and the replicate index — so it **differs between seeds** |
| **Evaluation** | 4096 × 8 = **32 768 trajectories** | one fixed stream, **identical for every contender, every depth and every seed** |

The evaluation set is therefore held out from training and common to the whole comparison.

---

## 2. The comparison is fair by construction, not by inspection

Two properties hold, and both were verified by resolving the study rather than by reading code.

**Every contender at a given seed trains on identical trajectories.** The routine that derives a
replicate's random streams takes only the study's data seed and the replicate index — there is
**no parameter through which a contender, a family or a recipe could arrive**. So it is not that
the contenders happen to receive the same batches; it is that the machinery cannot give them
different ones.

**Every point in the study shares one evaluation specification.** Verified across all 225 points
of Figure 1: one problem, one batch specification, one seed, one batch count. So no contender is
scored on trajectories another did not see.

This is what makes the differences between controllers *paired*: they are differences on the same
trajectories, not differences between two independent estimates.

---

## 3. What the error bar is, and why some controllers have none

The bar is the **across-seed standard deviation of the per-seed mean cost**. Because the
evaluation set is fixed and common, it isolates **training variability** — the training draw and
the parameter initialisation — scored on one common test set. It is not Monte-Carlo error of the
evaluation.

**Controllers that do not train have a spread of exactly zero, and that is correct.** Four of the
nine are closed-form or frozen: the unconstrained Riccati optimum, the truncated (clipped) Riccati
policy, the analytic projected-gradient controller, and the SDP-frozen convex-optimisation policy.
Nothing random happens to them, so all five seeds build the identical controller and the identical
evaluation returns the identical number five times.

**A caveat the caption must carry:** for those four, "five seeds" is **five identical copies, not
five independent samples**. It does not bias the mean, and it is why they are drawn without bars —
but the phrase "n = 5" means something different for them than for the learned models, and a
reader is entitled to be told which.

---

## 4. The tier ladder, and why two tiers' numbers must never be compared

Studies run at three efforts. Only the last is a result:

| tier | epochs | training batch | seeds | evaluation |
|---|---|---|---|---|
| smoke | 3 | reduced | 1 | truncated axis — **refused by the analysis by design** |
| standard | 50 | 2048 | 1 | 4 × 4096 |
| **publication** | **100** | **8192** | **5** | **8 × 4096** |

Because the tiers differ in the **evaluation** batch as well as the training effort, their
absolute levels are not comparable. This is measured, not feared: the truncated-Riccati policy is
closed-form and cannot change, yet it reads **9.127150** at `standard` and **9.150350** at
`publication` — a shift of **0.023**, which is larger than several of the effects the figures
discuss. Cross-tier comparison is therefore invalid; only within-tier comparison counts.

---

## 5. The baselines are not straw men

**The analytic projected-gradient controller's step is chosen the way every other family's is —
by ablation.** It learns nothing: it uses the exact Riccati matrices of the corresponding
*unconstrained* problem and a fixed step, and the only free quantity is that step's size, stated
as a multiple of 1/L where L is the Lipschitz constant of the per-step objective's gradient.

*This sentence used to read "a principled step size, not a convenient one", and the change is not
cosmetic.* On the small instance the campaign's law is 1/(2L), and that is defensible as a law.
On the large instance it is not: the baseline is still descending at the deepest depth the figure
draws, and a baseline that has not converged in the window is a baseline the comparison flatters.
So its step was swept there, over five exact binary multiples of the plant's own 1/L
(§5.4) — and the paper now reports a baseline whose step was *selected*, not merely declared.
That is a weaker claim about principle and a much stronger one about fairness.

This matters more than it sounds. The literal step every earlier study in this project declared
(0.05) **exceeds the classical stability limit 2/L on four of the seven problem instances** — by
factors of 5.0, 10.9, 6.0 and 23.9 at state dimensions 15, 20, 30 and 50 respectively, at every
one of the 100 time steps. At those dimensions the iteration limit-cycles instead of converging,
and the control constraint **hides it completely**: the projection keeps the iterate bounded while
the cost-versus-depth curve quietly goes non-monotone. Presenting that as the classical baseline
would have understated it by 5–8 %.

**The recurrent baseline's size was chosen by ablation, and the ablation is reported.** Five
widths were trained at the full publication budget. The finding is a **threshold, not an optimum**:
below 64 hidden units the fit is *bimodal* — seeds land either on the good solution or on a
distinctly worse one, two of five at both 16 and 32 — and the mean is then worse than the analytic
clipped baseline. At and above 64 the across-seed spread collapses by roughly two orders of
magnitude and the width stops mattering: three widths spanning 0.22 % against spreads of 0.01–0.02.
The chosen width is the cheapest reliable one.

A methodological note that belongs with it: **a wider interval must not earn a candidate the
choice.** Selecting "the cheapest width whose interval overlaps the best" would pick width 32,
whose interval overlaps *only because two seeds in five fail*. Reliability is a precondition for
being in the comparison, not a tie-break inside it.

### 5.1 The recurrent baseline at n = 100: how its three free parameters were fixed

Everything above was measured at the small instance, n = 4 with m = 2. Carried unexamined to
n = 100 with m = 30 — a state 25 times larger and a control 15 times larger — **it produced a
straw man, and the paper's own procedure is what found it.** The correction took three
measurements and moved the baseline's cost by a factor of 14.5. The sequence is reported here in
full, because the credibility of a comparison against a learned baseline rests on it.

**The symptom.** Re-ablating the width at n = 100 — widths 64, 128, 256, 512 at five seeds each —
returned 883.49, 806.69, 737.94 and 735.27 against the trivially attainable Truncated-Riccati's
64.166. A baseline eleven times worse than clipping the unconstrained optimum, at *every* capacity
it was given, is not a capacity curve. The stored per-epoch trajectories said what it actually
was: in **19 of those 20 arms the training loss reaches its minimum at epoch 1–5** and is then
1.1× to 4.2× *worse* by epoch 100 — the twentieth turns at epoch 22. That is an optimiser
stepping past its basin, at every width and every seed. **A capacity sweep run under a diverging
optimiser measures the divergence.**

**The learning rate, swept across three decades.** The incumbent 0.02 had been chosen by its own
ablation at n = 4. At n = 100, at the incumbent width, eight rates give:

| Adam learning rate | 0.0003 | 0.001 | 0.003 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean cost | 296.32 | **238.40** | 269.54 | 484.11 | 883.49 | 776.96 | 742.79 | 857.78 |
| across-seed spread | 7.35 | 8.75 | 82.63 | 197.25 | 377.80 | 134.10 | 243.60 | 415.28 |

The minimum is at 0.001 and **the incumbent is the worst point on the curve**. The three rates
*above* the incumbent were run deliberately, so that the sweep brackets its minimum on both sides
rather than stopping at the value under defence: they do not degrade monotonically but saturate
onto a broken plateau of 740–860, statistically indistinguishable from 0.02 given those spreads,
because past roughly 0.003 every arm has already left its basin within a few epochs. Instability,
not slow progress, is what the high rates buy.

**The budget, run rather than extrapolated.** With the rate fixed at a stable value, capacity
recovers its meaning: at 0.001 the width curve reads 238.40, 90.41, 78.60, 83.54 for widths 64,
128, 256, 512 — a 3.0× improvement from 64 to 256, where the diverging sweep had shown a flat
883 → 735. But every arm at the stable rates was **still descending when the pinned 100 epochs cut
it off**: 5 of 5 seeds in all eight cells, losing a further 5.8–11.9 over their final twenty
epochs. Fitting the tail to decide where it would settle was tried first and **refused by the
floor**: an `a + b·t^(-c)` fit over epochs 40–100 places the asymptote *below zero* in three of
six cells and predicts every cell under 55.43 at epoch 500 — the unconstrained Riccati optimum,
which no policy can beat in expectation. A window still inside the near-linear part of a descent
does not contain the location of its own knee, and any curve-fit that claims otherwise should be
checked against the problem's bounds before it is believed.

The budget was therefore measured. Because the epoch axis is **nested** — a long trajectory
contains every shorter one as a prefix — a single run to 1000 epochs answers the question at every
intermediate budget:

| epoch | 100 | 200 | 300 | 500 | 700 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| lr 0.001 | 74.72 | 64.37 | 62.14 | 61.45 | 61.22 | 61.07 |
| lr 0.0003 | 95.73 | 73.93 | 67.44 | 63.34 | 62.08 | 61.53 |

Reading a training trajectory as a cost curve is licensed here by measurement rather than by
assumption: on this family and plant the per-seed final training losses agree with the
independently evaluated costs to within 0.3 % over 163,840 held-out trajectories, so there is no
train-versus-inference gap to invalidate the reading. **Epoch 500 is within 0.6 % of the
1000-epoch value at the winning rate**, and the final hundred epochs move it by 0.03–0.3 %; 500 is
the declared budget and it is a measured choice, not a round number. Note that sufficiency is a
property of the rate and not of the problem — the same 500 leaves the slower rate 2.9 % short of
its own converged value.

**The capacity, re-ablated last — because it could only be asked last.** With the rate at 0.001
and the budget at 500 epochs, five seeds per width:

| hidden units | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| mean cost | 216.63 | 62.166 | **61.668** | 62.166 |
| across-seed spread | 75.54 | 0.354 | 0.487 | 1.268 |
| loss moved over the final 50 epochs | 6.1–9.4, one seed *rising* | 0.18–0.40 | 0.06–0.59 | 0.01–0.80 |

**The finding is a threshold, not an optimum — the same shape as at n = 4, at twice the width.**
At 128 and above the three widths span 0.50 (0.8 %), which is inside their own seed spreads, and
every arm has stopped moving: the final fifty epochs shift the loss by at most 1.3 %. Width 64 is
not merely the slowest point on a trend — at a budget five times the campaign's it has neither
converged nor stabilised, its five seeds sitting at 134–302 and still falling 6–9 per fifty epochs,
with one seed turning at epoch 305 and *rising* by 36.9 thereafter. Its spread of 75.5 is that
disorder, not a measurement of capacity. **The declared width is 128**, by §5's own rule: the
cheapest width that is reliable. 256 is nominally better by 0.50 but that gap sits inside the
spread and costs more to train, and 512 is no better than 128 while varying three times as much.

**The outcome.** Trained this way the recurrent baseline **beats the trivial contender**: 62.166
at the declared width 128 against Truncated-Riccati's 64.166, a margin of 3.1 %, with an
across-seed spread of 0.354 where the incumbent settings gave 377.8. Carried further — width 256
at 1000 epochs, two seeds — it reaches 61.098, 4.8 % better than Truncated-Riccati and within
10.2 % of the unconstrained floor. Those two numbers are stated with their budgets attached and
must not be quoted against each other. The two stable rates converge to within 0.7 %, so at a
sufficient budget the rate ceases to matter — it buys the speed of arrival, not the destination.

**Scope of the defect, measured rather than assumed.** The divergence is confined to the large
plant. Reading the stored curves of every GRU arm the campaign has trained, the dimension ladder
is clean at all six of its plants — n = 4, 7, 15, 20, 30 and 50 at m = 2 — with zero diverged
arms, median best-loss epochs of 77–98 out of 100, and final losses within 1.2 % of their minima.
The small-instance figure, whose GRU trains at the incumbent 0.02, is clean on the same test. No
result already measured on those plants is affected by this section; what is affected is any study
on the n = 100, m = 30 plant that declares the incumbent settings.

**Three rules this episode establishes, which apply to every learned contender in the paper.**
A hyper-parameter tuned at one instance size is not a hyper-parameter for another, and the burden
is on the study that transfers it. A capacity verdict taken under a truncated budget is not a
capacity verdict — before reading a width curve, check whether its arms had stopped improving.
And a baseline whose loss is an order of magnitude off the trivial contender should be assumed
broken rather than reported, because it usually is.

### 5.2 The 500-epoch budget is the large instance's, and every learned family gets it

The budget measured in §5.1 is a property of the **plant**, not of the contender that exposed it.
On the n = 100 instance **all four learned families train for 500 epochs** — the three unfolded
kinds and the recurrent baseline alike — and the study document writes that budget once, over
every trainable contender it declares. The small-instance figures keep 100 epochs, again for all
families. **Within a figure the budget is identical for everyone; between figures it differs, and
the two are never compared** (§4).

That the extra budget was needed by the unfolded families too, rather than merely granted to them
for symmetry's sake, is measured from the stored curves rather than assumed: across the completed
500-epoch unfolded trainings on this plant, **not one diverged** — final loss within 1.001–1.002
of each arm's own minimum, moving by less than ±0.08 on a value near 60 over the final fifty
epochs — and their minima fall at **epochs 271–498**, i.e. they were still improving when the run
ended. A 100-epoch budget would have truncated them as surely as it truncated the baseline.

**What is *not* symmetric, and is stated rather than removed, is the tuning.** The recurrent
baseline had its step size re-selected on this instance from an eight-point sweep spanning three
decades, and its width chosen from a four-point ablation at the full budget. The unfolded families
carried their step size (0.05) over from the small instance without re-selection. That asymmetry
favours the baseline, and it is left in place: a baseline that loses to a method whose own
hyper-parameters were never re-tuned for the instance loses on the merits. The transfer was
nevertheless *checked* rather than trusted — the divergence test above is exactly the check §5.1's
first rule demands of any transferred hyper-parameter, and the unfolded families pass it on this
plant.

### 5.3 Every learned family's step size was selected by its own ablation

The paper's fairness claim rests on this, so the whole grid is reported per family rather than
summarised as "tuned". Each sweep was run at the tier, batch and epoch budget its own figure is
drawn at, scored on held-out trajectories, and — where the incumbent value was already in the
store — the incumbent arm is the *same models the figure draws*, reused by identity rather than
re-measured.

**The unfolded families** (all three kinds, one rate for the family). Small instance, batch 16384,
100 epochs, expected cost of the proposed kind:

| Adam lr | J = 1 | J = 4 | J = 10 |
|---|---:|---:|---:|
| 0.01 (incumbent) | 8.30859 | 8.28372 | 8.27986 |
| **0.05 (selected)** | **8.27457** | **8.27206** | **8.27174** |
| 0.1 | 8.28428 | 8.27187 | 8.27154 |

**Selected 0.05, and 0.1 was rejected on the shallowest depth.** The top two rates tie at J ≥ 4 to
within 2e-4 — at or below the across-seed spread — but 0.1 is **0.0097 worse at J = 1**, where the
model has least capacity to absorb an overshoot, and a paper whose claim is that few iterations
suffice may not adopt a rate that degrades the shallowest one. The incumbent was separately shown
*under-trained* rather than merely slower: 300 epochs at 0.01 reaches 8.27530, still short of 100
epochs at 0.05 (8.27206), while 300 at 0.05 buys a further 0.0009 — converged. Consequence worth
carrying: the depth effect over J = 1 → 10 collapses from 0.0288 at the incumbent rate to 0.0028
at the selected one, so most of the depth curve's original shape was an optimisation artefact.

**The recurrent baseline, small instance**, batch 16384, 100 epochs, width 64, five seeds:

| Adam lr | 0.003 | 0.01 | **0.02** | 0.03 | 0.05 |
|---|---:|---:|---:|---:|---:|
| mean cost | 11.6942 | 8.3036 | **8.2892** | 8.2958 | 10.0344 |
| across-seed spread | 7.3716 | 0.0178 | **0.0088** | 0.0110 | 2.8988 |

**Selected 0.02 — the minimum of the curve and the tightest arm on it.** The cliff is sharp in
both directions and is a real architectural fact about this family: 0.05 diverges on this plant
(one seed at 8.36, others past 15), and 0.003 fails to fit within the budget. Note the optimum
sits far below the unfolded families' 0.05 and far above the same architecture's rate on the large
plant — which is the observation §5.1 is built on.

**The recurrent baseline, large instance**: eight rates over three decades, **0.001 selected**, the
incumbent 0.02 the worst point on the curve — the full table and its diagnosis are §5.1.

**The convex-optimisation policy**, batch 16384, 100 epochs, five seeds, MOREAU on CPU:

| Adam lr | 0.03 | **0.1** | 0.3 |
|---|---:|---:|---:|
| mean cost | 8.271152 | **8.270511** | 8.270554 |
| across-seed spread | 0.000130 | 0.000309 | 0.000316 |

**The incumbent 0.1 survives, and the entire axis lies inside the noise.** Worst to best across
the whole decade is 0.00064, roughly twice one arm's own across-seed spread; 0.1 and 0.3 differ by
4.3e-5, a seventh of either spread, and their per-seed values interleave (0.27026–0.27104 against
0.27034–0.27110). Only 0.03 is consistently worse, and by 0.00064. This is the outcome a fairness
ablation hopes for and cannot assume: **the convex baseline was neither advantaged nor handicapped
by an untuned step size**, so the margin the paper reports against it is not an artefact of one.

**The analytic projected-gradient baseline has no such ablation, and that is not an omission.**
Its step is fixed by the problem at 1/(2L) — a law, not a fitted quantity — and the one choice
available there, the factor, was itself measured rather than assumed: 1/L and 1/(2L) converge to
the *identical* limit 8.844852, so the difference between them is a rate of convergence and not a
stability property (§8). It learns nothing and has no optimiser.

### 5.4 The training budget is per family, and that is a deliberate deviation

The campaign's plan was one epoch count for every learned family in a figure, and the small
instance no longer follows it. The deviation is the author's, it is measured, and it is stated
here because a reader comparing two contenders is entitled to know they were not given the same
number of epochs.

**What was measured.** Figure 1's whole cast was retrained at twice its budget, at the deepest
depth, five seeds:

| family | 100 epochs | 200 epochs | gain | its own spread |
|---|---:|---:|---:|---:|
| **recurrent baseline** | 8.2892 | **8.2799** | **0.0093** | 0.0088 |
| per-iteration-P ablation | 8.2723 | 8.2717 | 0.0007 | 0.0001 |
| learned-step unfolded | 8.8115 | 8.8111 | 0.0004 | 0.0001 |
| proposed | 8.2716 | 8.2713 | 0.0002 | 0.0001 |
| convex policy | 8.27051 | 8.27087 | **−0.00036 (worse)** | 0.00031 |

**Exactly one family was truncated.** The convex policy moves the wrong way by about one spread;
the three unfolded kinds move by 1 % of the gap the figure argues about. Raising everyone to 200
for parity would have cost roughly 25 hours — two thirds of it retraining a convex policy just
measured as converged — to change no number that any claim depends on.

**So the rule is convergence, not parity: every learned family is trained to its own measured
convergence, verified per family.** Equal epochs is not equal treatment when one family converges
in a fifth of them; it is only equal *bookkeeping*. The asymmetry runs in the baseline's favour —
it gets twice the budget of the method being proposed — which is the direction a fairness
argument can afford to be wrong in.

**And the rate had to move with the budget**, which is the part that would have been easy to miss.
Re-ablating the baseline's step size *at* 200 epochs flips the winner: 0.01 reads 8.3036 at 100
epochs and **8.2766** at 200, while 0.02 reads 8.2892 and 8.2799. Carrying the 100-epoch winner
into a 200-epoch figure would have handed the baseline the rate that is second-best at its own
budget. §5.1's rule — *a verdict taken under a truncated budget is not a verdict* — applies to a
learning rate exactly as it applies to a capacity, and here it is measured rather than feared.
Note also what the next rate up does: at 0.03, **more epochs make it worse** (8.2958 → 10.5535,
one seed diverging), the same past-the-edge behaviour the large instance showed at 0.02.

**What it costs the headline, stated plainly.** The proposed controller's margin over the
recurrent baseline falls from 0.0176 to **0.0050** — still about three times the baseline's own
across-seed spread, with the convex policy at 8.2705 below both. A margin that survives giving
the baseline twice the budget and its own re-tuned rate is a margin worth reporting; the one that
preceded it was partly an artefact of an under-trained rival.

### 5.5 The analytic baseline's step, swept across the stability limit

The one family whose free parameter was still declared rather than measured, asked on the large
instance where it matters: Standard-PGD is at 114.68 at $J = 1$ and still falling at $J = 10$
(63.19), so a reader is entitled to ask whether the classical method was handicapped by a
conservative step. Five fixed steps, each an **exact binary multiple** of that plant's own 1/L —
so none is a hand-typed number — one seed, since the family trains nothing and is deterministic:

| step | J = 1 | J = 2 | J = 3 | J = 5 | J = 10 |
|---|---:|---:|---:|---:|---:|
| 1/(2L) — the declared law | 114.68 | 88.34 | 77.81 | 69.10 | 63.19 |
| 1/L | 84.30 | 71.00 | 66.34 | 62.97 | 61.21 |
| **2/L — the classical limit** | **73.44** | **65.98** | **63.36** | **61.72** | **60.98** |
| 4/L | 82.40 | 120.59 | 112.63 | 137.76 | 197.82 |
| 8/L | 92.79 | 191.33 | 156.39 | 186.19 | 277.55 |

**2/L is best at every depth, and past it the iteration does exactly what theory says**: 4/L and
8/L alternate between even and odd depths with growing amplitude — the classical divergence,
visible because the sweep brackets the limit instead of stopping at it. Figure 3 adopts 2/L.

*Why a sweep and not one larger step.* The proposal on the table was a single arm "about an order
of magnitude" up, which lands between the 4/L and 8/L rows. It would have measured a controller
oscillating and read as evidence that the baseline is weak — the same failure this section already
records from the other direction, where a declared 0.05 sat 5–24× above 2/L and the box hid it.
A lone arm past a limit measures the limit; five arms straddling it locate the limit.

**What adopting it costs the paper, and why it is worth paying.** At 2/L the analytic baseline
reaches 60.98 and now beats both the recurrent baseline (62.17) and Truncated-Riccati (64.17); the
proposed controller's margin over it falls from 3.07 to **0.86**. The claim that survives is the
better one: the unfolded families still win against a baseline whose step was selected by sweep,
while **their own initial step was not selected at all** — it is the campaign's law, 1/(2L), and
they are left there deliberately. UF-α learns its way to roughly 4.5× that initialisation, which
is 2/L: it discovers the swept answer unaided. Re-initialising the learned families to match would
have cost 150 trainings and hidden precisely that.

**And it is where the learned steps live that makes the point.** Eight of UF-α's ten learned steps
sit at or above 2/L — in the region where a *fixed* step provably oscillates. An unrolled network
trained for a finite $J$, with the projection bounding its iterates, can exploit steps a
fixed-step method cannot safely use. That is a mechanism, and the sweep is what turns it from a
hypothesis into a measurement.

**How to read this collection.** Four families, four independently chosen step sizes spanning
0.001 to 0.1 — a factor of a hundred — every one selected by a sweep at the budget its figure is
drawn at, and two of the four (COCP, and the GRU at the small instance) confirming the value
already in use rather than replacing it. A single shared learning rate would have been the
straw-man construction, and the spread between these four is the measurement that says so.

---

## 6. What is proposed, and what is an ablation of it

The **proposed** controller learns a step size and **one cost-to-go matrix shared across all
unrolling iterations**. The variant that learns an **independent matrix per iteration** is an
**ablation of our own method**, not a rival: it carries as many matrices as there are iterations
and exists to show that the extra parameters buy little. The recurrent network and the
convex-optimisation policy are the external baselines.

### 6.1 The extra matrices do not merely buy little — they cost, and the reason is provable

The per-iteration variant's hypothesis class **contains** the proposed controller's: set all its
matrices equal and it *is* the shared-matrix method, exactly, since both carry the same learned
step sizes. Its optimum is therefore bounded above by the proposed controller's, and it cannot be
worse for any reason except optimisation.

It is worse. On the small instance at depth 10, on the same data, seeds and budget:

| | training objective | evaluated cost |
|---|---:|---:|
| proposed — one shared matrix | **8.312297** | **8.271582** |
| ablation — ten matrices | 8.313232 | 8.272329 |

**It ends 0.000935 worse on its own training objective than a point it can represent exactly.**
That is not capacity and not a generalisation gap; it is the optimiser, and the containment
argument is what makes the conclusion airtight rather than suggestive.

It did not merely fail to reach the shared solution — it went elsewhere. Reading the ten learned
matrices out of the trained model, they differ from their own mean by **42 %** of the matrices'
size, in a U-shape: the first and last iterations' matrices deviate most, the middle ones least.
Ten matrices, each receiving gradient through only its own iteration, at the same rate and budget
as one matrix receiving all of it.

**This also explains the depth curves' shape**, which is otherwise the figure's most puzzling
feature. Measured with the Monte-Carlo error removed — the same trajectories per seed, so the
comparison is paired — from depth 4 to depth 10 the proposed controller **falls** by 0.000228
(consistent across all five seeds), while the per-iteration ablation **rises** by 0.000458 and the
learned-step-only family rises by 0.003915. The family whose parameter count does not grow with
depth is the one that stays monotone.

---

## 7. Bounds: which side each quantity is on

The naming here has misled readers, including during this work, so the notebook should state it
explicitly rather than rely on labels.

| quantity | side of the constrained optimum |
|---|---|
| unconstrained Riccati cost | **lower** — it violates the control constraint, which is what makes it a floor |
| box-aware semidefinite floor | **lower**, but see the units warning below |
| SDP-frozen convex policy | **upper** — an attained, feasible policy |
| convex-optimisation policy (trained) | **upper** |
| clipped Riccati policy | **upper** |

**The SDP-frozen policy is not a lower bound on anything**, despite a name that suggests it. It is
the same one-step convex policy with its cost-to-go frozen at the semidefinite relaxation's
solution — a real, feasible controller whose control never exceeds the bound. So the trained
convex policy attaining a *lower* cost than it is the expected direction: a trained cost-to-go
beating a frozen one. **The paper should not print the misleading name**; something like
"convex policy (SDP-frozen)" says what it is.

**The units warning, which is the one that could put a false claim on a figure.** The box-aware
semidefinite floor is an *infinite-horizon steady-state* average cost. The figures plot a
*finite-horizon* time average from a random initial state. Those are different quantities: on the
campaign's own instance the constrained loops have not reached steady state by the end of the
horizon, and their running average only rises past that floor around a third of the way through.
**The floor sits below the plotted numbers because the horizon is long enough, not because it
bounds them** — at a shorter horizon it would sit above. The figures therefore bracket with the
unconstrained Riccati cost, which is measured under exactly the plotted convention.

---

## 8. Reading depth: what a rising curve means

A controller unrolled to depth *J* can reproduce a controller of depth *J′ < J* by driving the
extra step sizes toward zero. The deeper model's hypothesis class therefore **contains** the
shallower one, and its trained cost should not be worse.

So a cost-versus-depth curve that **rises** is evidence about optimisation, not about
representation — and it is a claim the reader can check on the figure. This distinction is what
separates the two families in Figure 1: the controllers that keep the *unconstrained* cost-to-go
cannot use extra depth (they converge toward a per-step optimum that is not the closed-loop
optimum once the constraint is active on most of the trajectory), while the controllers that
**learn** the cost-to-go improve with depth and plateau.

**Each depth is a separately trained model, and the curve is not monotone by construction.** This
is worth stating because the containment argument above invites the opposite assumption. The model
at depth *J* is not the model at *J* − 1 with an iteration added: it is an independent fit, from
its own initialisation, with its own optimisation path, and nothing forces it to land at least as
well. Warm-starting each depth from the one below — initialising the new step size at zero, so the
deeper model *begins* at the shallower one's cost — would make the curve monotone in the training
objective by construction, and the machinery for it exists.

**It was not used, and the reason is that it would change what the depth axis means.** Layer-wise
training makes "100 epochs" ill-defined across a depth sweep: the total becomes *J*·warmup +
refinement, so holding the budget fixed across depths needs a different pair at every point, and
over *J* = 1…10 the schedule costs 2375 epochs against the 1000 a flat plan implies. **The axis
would then confound "more iterations" with "more training"** — the one thing this figure must not
do. Trained end to end at one budget, a rise is information: it says the deeper unroll is harder
to optimise at a fixed budget, which is itself part of the case for few iterations.

A single measurement makes the trade concrete. For the analytic controller — which trains nothing,
so its depth axis *is* one iteration sequence — the curve still turns up, from 8.844275 at depth 8
to 8.844852 at depth 50. Non-monotonicity in depth is not peculiar to learned models.

**The rise was then tested against the budget, and it split by family.** If a curve that turns
up is evidence about optimisation, the first thing to ask is whether more optimisation removes
it. The three unfolded families were retrained over the whole depth axis at twice the budget —
200 epochs against 100, everything else identical — and the second hundred epochs is worth
something at **every one of the thirty (family, depth) cells**, never nothing and never
negative:

| rise from the family's own minimum to J = 10 | 100 epochs | 200 epochs |
|---|---:|---:|
| UF-α (learned steps) | +0.003915 | +0.004249 |
| UF-αP (proposed) | monotone | monotone |
| UF-αPʲ (per-iteration P) | +0.000458 | +0.000091 |

So the per-iteration ablation's rise is **mostly under-training** — doubling the budget shrinks
it fivefold — while the learned step's rise does not move at all. The reading in this section
survives both: a rising curve is about optimisation, and here the *amount* of optimisation is
shown to matter for the family with the most parameters per iteration and not to matter for the
family with the fewest.

**A methodological correction rides with that measurement.** The budget probe that set the
campaign's per-family epochs tested one depth, J = 10, where the second hundred epochs is worth
0.00041 / 0.00025 / 0.00066 against across-seed spreads of 0.0002–0.0004 — one to two spreads,
correctly called noise at that point. Swept over the whole axis the same comparison is negative
in thirty cells of thirty, which no per-point test could resolve. The figure now draws all three
families at 200 epochs, and *the ablation claim got weaker in the process*: with both families
better trained, the proposal still sits below its per-iteration ablation at every J ≥ 2, but the
gaps (0.00012–0.00050) exceed both error bars at five depths of nine rather than at all nine —
and at **J = 3, the operating point, the two are not separated**. The claim this campaign can
make is the proposal against the recurrent and convex baselines, not against its own ablation
at one depth.

---

## 9. The mismatch experiment: what Figure 2 measures, and how to read it

**The governing principle is the practical offline/online separation.** Every controller has
an offline phase, run when the true plant A was known — training for the learned contenders,
precomputation for the analytic ones — and an online phase that computes controls step by
step. No offline computation is ever repeated online, analytic or not.

### 9.0 What the figure draws, as of 2026-08-14: three conditions against severity

Figure 2 is a **sweep over mismatch severity**, not three bars at one rotation. `A` is rotated
by 0, 5, 10, 20, 30 and 45 degrees with `B` left alone, and every controller is measured under
three conditions at every angle — **blind** (the rotation is unknown), **told** (the rotated
matrices are handed over and enter the online expressions only), and **world-trained** (the
learned families train on trajectories the rotated plant generated while their declared model
stays nominal). θ = 0 is the matched condition, so the three panels share an endpoint, and it
is *literally* Figure 1's models: 35 of 35 resolve to identical ModelIDs.

**Why one angle was not enough, in one measurement.** Plant difficulty is **not monotone** in
the rotation: blind costs rise to 10–20° and then fall, and 45° is markedly *easier* for a
controller that can absorb it while being catastrophic for one that cannot. At 30° alone the
picture is genuinely ambiguous; the ladder below is only legible across the axis.

**The value of information is monotone in how model-based the controller is.** Comparing each
controller against *itself* blind — which is the comparison the three-bar figure could not
make, because it had no blind column — at 45°:

| told against blind, 45° | blind | told | value of the news |
|---|---:|---:|---:|
| PGD | 14.983 | 7.506 | **−50 %** |
| Clipped-LQR | 13.358 | 7.833 | **−41 %** |
| UF-α | 14.321 | 8.697 | **−39 %** |
| UF-αPʲ | 16.643 | 13.371 | −20 % |
| UF-αP (proposed) | 16.622 | 13.792 | −17 % |
| COCP | 16.335 | 16.013 | −2 % |
| GRU | 16.082 | 16.082 | **0 %, structurally** |

The earlier reading of this section compared *told* against *matched*, which conflates two
different things — that the plant changed, and that information helped — and made the
learned families look as though the news hurt them. Against blind, the hypothesis this
campaign set out to test holds, and it holds with a gradient rather than as a binary.

**The discriminating family is UF-α, and it is why the split is not "model-based versus black
box".** UF-α *trains*, and in the world-trained condition it trains on the rotated plant's own
trajectories — yet at 45° it lands at 12.655 beside the analytic pair's 13.358 and 14.983,
while every family carrying a **learned cost-to-go** collapses to the floor (UF-αP 7.711,
UF-αPʲ 7.611, COCP 7.652, GRU 7.401). UF-α has learned step sizes and a *nominal Riccati*
cost-to-go, so it has nothing to fit the new world with. The dividing line is whether the
controller carries a cost-to-go it can fit to the world it will run in — not whether it is
trained, and not whether it is analytic.

**Two structural checks ride in the figure and both pass exactly.** The recurrent baseline
cannot read a matrix, so its *told* column must equal its *blind* column: measured
**0.00e+00 at every angle**, which is a check on the whole rehost path rather than a result.
And *told at 0°* is an independent recomputation of Figure 1's matched costs under a different
MeasurementID — all seven contenders agree to **0.00e+00**.

Everything below in this section describes the superseded three-bar figure at 30°. It is kept
because its mechanism analysis — the substitution experiment in particular — is still the best
account of *why* a learned cost-to-go is fragile, and because its numbers are the ones the
notebook's fuller treatment uses. **Its tier is `publication` where the figure above is
`publication_b16k`; the two must not be compared.**

**Two inference regimes are reported side by side, on two rotated plants.** *Blind*: the
rotation is unknown — the controllers still believe A (and B) while the world runs the rotated
plant. *Informed*: the rotated plant is also handed to the controllers, and it enters their
**online expressions only**: feedback gains re-form per step from the frozen offline cost-to-go
with the matrices in hand; the unfolded families' gradient coefficients form the same way; the
convex policy's per-step program is posed with the given matrices under its frozen trained
cost-to-go. Everything computed offline — Riccati cost-to-go stacks, the analytic
first-order method's step (the offline 1/L of the nominal plant, on every column), and every
learned parameter — stays exactly as the offline phase fixed it.

**The plotted quantity** is the percentage change of the per-seed mean cost against the same
contender's own matched nominal performance, paired per seed as everywhere in the campaign.

**The ordering under the genuine plant change (rot A), informed** (five seeds, publication
effort): the clipped Riccati controller gains from the news (−9.2 %) — its frozen cost-to-go
with live gains captures most of the adaptation; the analytic first-order method and its
learned-step variant sit near their nominal cost (−2.1 %, −1.9 %); the proposed controller and
its per-iteration ablation pay +13.9 %; the recurrent baseline +24.9 %; the convex policy
+32.8 %. A negative bar is meaningful: the denominator is the contender's own nominal cost,
and the rotated instance is genuinely cheaper for a controller whose gains adapt.

> **THOSE NUMBERS ARE THE `publication` TIER'S AND THE FIGURE IS NOW DRAWN AT
> `publication_b16k`.** They must not be compared with the ones below — §4's rule — and this
> section is being rewritten against the drawn data as the severity sweep lands. What the
> rendered figure actually measures, read from its own analysis table (five seeds, batch
> 16384, J = 3):
>
> | rot A, informed | told | vs matched |
> |---|---:|---:|
> | Clipped-LQR | 8.1370 | −11.1 % |
> | PGD | 8.1242 | −8.2 % |
> | UF-α | 8.6078 | −2.3 % |
> | UF-αPʲ | 10.3456 | +25.1 % |
> | UF-αP (proposed) | 10.4523 | +26.4 % |
> | GRU | 10.5371 | +27.3 % |
> | COCP | 10.9880 | +32.9 % |
>
> **One claim below does not survive the move and is withdrawn here rather than left
> standing:** the proposed controller does *not* move from behind the recurrent baseline to
> comfortably ahead of it. At this tier the two are 0.9 points apart (+26.4 against +27.3) —
> a tie, not a result. **What the drawn column does show is a split by mechanism**: the told
> costs fall into {8.12, 8.14, 8.61} and {10.35, 10.45, 10.54, 10.99}, and the boundary is not
> model-based against black-box but *carries a learned artifact fitted to the nominal plant*
> against *does not*. The recurrent baseline — which cannot read the matrices it is handed, so
> its told cell **is** its blind cell — sits inside the second group rather than behind it.
> Whether being told actively *hurts* a learned cost-to-go, or is merely useless to one,
> cannot be settled from a figure with no blind column, which is why the severity sweep
> measures blind at every angle.

**Why negative bars are the plant and not magic — the intrinsic-difficulty decomposition.**
Rotating A while B stays put is **not** a change of coordinates — only the A+B co-rotation
is — so the rot A instance is a genuinely different plant, and it happens to be an easier
one: a controller fully re-derived from the rotated matrices (the reference the figure
deliberately does not draw, because it violates the offline/online separation) costs
**8.137 against the nominal optimum's 9.150 — its own optimum sits 11.07 % below**. Against
that reference the information ladder is monotone in every cell and nothing ever beats full
adaptation:

| rot A | blind | informed | fully adapted (reference) |
|---|---:|---:|---:|
| clipped Riccati | +1.7 % | −9.2 % | −11.1 % |
| analytic first-order | +9.4 % | −2.1 % | −8.4 % |

On the control plant the same reference lands at −0.03 % and −0.09 % — zero to Monte-Carlo
resolution, as unitary equivalence demands. The informed columns recover *most* of what full
adaptation would buy while recomputing nothing offline; they never exceed it.

**Every drawn number of both halves of Figure 2 was re-derived independently before it was
believed.** A from-scratch reimplementation — own backward recursion, own projected-gradient
loop, own exact active-set solver for the per-step box program, sharing only the frozen
plant files, the stored parameters and the common evaluation draws — reproduces every
closed-form and unfolded cell to 1e-14 relative and every convex-policy cell to the QP
solver's own tolerance, per seed. The suspicious signs in this section are measured
properties of the controllers, not artifacts of the pipeline.

**The co-rotated plant (rot A+B) is the control condition, and its reading is exact.**
Rotating both matrices is the nominal problem in rotated coordinates — the plant's difficulty
does not change at all — so every bar there is pure mismatch cost, with no plant-difficulty
confound. Under the offline/online separation nobody fully adapts, and the column compares
every cost-to-go-carrying controller under the same handicap: a cost-to-go frozen in the old
coordinates. Measured: the analytic frozen-P controllers pay **+4.5 to +5.6 %** informed
(against +19.7 to +20.2 % blind), the learned-cost-to-go controllers pay **+9.1 to +9.2 %**
informed (+12.1 to +13.0 % blind) — the learned matrix is roughly twice as coordinate-fragile
as the analytic one under identical conditions — the convex policy +7.3 % (+10.5 % blind), and
the recurrent baseline +4.8 % in both regimes.

Statements the caption should carry:

* **The recurrent baseline's numbers are identical under blind and informed, to the last
  bit, on both plants.** It has no input through which a plant could arrive, so telling it is
  a no-op — measured as an exact equality, and it is the structural asymmetry the comparison
  exists to show. It is also strikingly robust to the pure coordinate change (+4.8 % where
  the blind analytic controllers pay ~20 %).
* **The proposed controller converts the news into roughly a nineteen-point recovery on the
  genuine plant change** — +33.2 % blind against +13.9 % informed — and moves from behind the
  recurrent baseline to ahead of it. The model-based hypothesis holds only once the controller
  is actually told; blind, the ranking inverts.
* **The convex policy is hurt by the news on the genuine plant change** (+27.7 % blind against
  +32.8 % informed) **and helped by it on the coordinate change** (+10.5 % against +7.3 %):
  posing the per-step program with a model inconsistent with the frozen trained cost-to-go is
  not monotone in knowledge. **The mechanism is the learned cost-to-go, and it is proven by
  substitution, not conjectured**: put the *true* unconstrained Riccati solution into the very
  same per-step program and the news helps exactly as it helps the analytic families (+9.4 %
  blind → −2.1 % informed on rot A, indistinguishable from the analytic first-order method).
  The trained matrix sits **3.4× away from the Riccati solution** in relative norm and its
  implied gains 40 % away from the Riccati gains — yet its matched nominal cost beats every
  analytic baseline. Training made it a **compensator co-adapted to the model it was trained
  with** (the box, the one-step truncation, the nominal matrices), not an estimate of any
  value function. Blind keeps the trained state-to-control map intact; informed pairs the true
  dynamics with a cost-to-go tuned to the wrong ones, and the co-adaptation is voided — the
  same reason the learned-matrix unfolded controllers stay at +13.9 % informed where the
  true-matrix families go negative. One coin, two faces: the co-adaptation is why the learned
  cost-to-go wins on the nominal plant (Figure 1) and why it is fragile under a plant change
  (Figure 2).
* **Informed-versus-blind is worth 14 to 15 points to the analytic frozen-P controllers on the
  control plant** and up to 11 points on the genuine one — the value of using the handed
  matrices in the online expressions alone, with nothing recomputed.

## 10. The training-world experiment: Figure 2's second half, and the unified mechanism

> **Superseded as a figure, confirmed as a mechanism (2026-08-14).** This experiment is now the
> third panel of the severity sweep (§9.0) and is measured at six angles rather than one. Its
> central claim survives that widening and is strengthened by it: *the win comes from learnable
> geometry, not from "learning" generically.* Swept, the learned-steps variant trains on the
> rotated plant's own trajectories and still lands at **12.655** at 45°, beside the analytic
> pair's 13.358 and 14.983, while every family carrying a learned cost-to-go collapses to
> **7.4–7.7**. The gap between "trained, with geometry to fit" and "trained, without it" widens
> monotonically with severity — which is the claim below, seen at five angles instead of one.
> The numbers in this section are the `publication` tier's and are not comparable with those.

**What is measured.** Each learned contender is *told* the nominal plant A — its internal
model, its gradient coefficients, its per-step program are all built from A — while the
trajectories it trains on are generated by a rotated plant, and the loss is computed on what
the data does, not on what the contender believes. Evaluation is on the same rotated plant,
in the same two inference regimes as the first half: blind (the controller still believes A)
and informed (the rotated matrices enter the online expressions; everything trained stays
frozen). The denominator is the same contender trained *and* evaluated nominally — the
matched case, bit-identical to Figure 1's models. The analytic contenders have no training
half; their cells are the first half's blind columns.

**The hypothesis holds, in the blind column** (five seeds, publication effort, mean across
seeds). Every contender with learnable geometry, fitted on the world's data under the wrong
model, essentially neutralises the mismatch: on the genuine plant change the proposed
controller lands at **−1.7 %**, its per-iteration ablation −1.9 %, the recurrent baseline
−2.2 %, the convex policy −2.1 % — all *below* their own matched nominal, cashing part of
the rotated plant's intrinsic easiness — where the classical baselines, which can consume no
data, stand at +1.7 % and +9.4 %. On the coordinate-change control the same four sit at
**+0.2 to +0.7 %** — zero to within their spreads — against ~+20 % for the blind analytic
column: the data channel is worth up to twenty points where the model channel is silent.
The learned-steps variant is the built-in inverse control: scalar step sizes carry no
geometry to relearn, so its blind cells (+5.8 %, +19.3 %) sit beside the analytic
first-order method's — the win comes from learnable *geometry*, not from "learning"
generically.

**Being told now hurts every learned-cost-to-go carrier, and this is the first half's
mechanism seen from the other side.** Trained under the mismatch, the compensation for the
rotation is already baked into the learned matrix; handing the true matrices to the online
expressions applies the correction twice. On the genuine change the proposed controller
moves −1.7 → +1.1 %, the ablation −1.9 → +3.6 %, the convex policy −2.1 → +4.4 %; on the
control plant the double-correction costs twelve to fourteen points (+0.7 → +13.7 %,
+0.4 → +12.5 %, +0.2 → +5.1 %) — and the doubly-corrected proposed controller lands almost
exactly where the *uncorrected* one stood in the first half (+13.7 % here against +13.0 %
there): correcting twice looks like not correcting at all. The learned-steps variant, whose
frozen matrix is the true Riccati one, is again the mirror image — the news helps it
(−3.8 % on the genuine change, its best cell in either experiment). The recurrent baseline
is bit-identical under both regimes, as always.

**The one-line statement spanning both experiments**, which the notebook should close on:
**information helps carriers of a true value function; data helps learners of a surrogate
one; a surrogate-learner given both channels applies the same correction twice.** The first
half shows the first clause (informed analytic −9.2/−2.1 against learned matrices stuck at
+13.9, the convex policy actively hurt); the second half shows the other two (surrogate
learners at ≈0 blind, hurt when told). Every rung is measured, including the substitution
proof that the same per-step program is helped by the news the moment it carries the true
cost-to-go.

## 11. The dimension scaling: what Figure 3 measures, and how to read it

**The plotted quantity is a ratio, and the floor is drawn at exactly 1.** Each contender's
per-seed mean cost is divided by the unconstrained Riccati optimum's on the **same** plant,
paired per training seed — the author's normalisation, because raw cost spans 1.85 to 71.0
across the grid purely from the objective summing over n state dimensions. The absolute costs
ride the table beside the figure. The x axis is logarithmic; each grid point is a genuinely
different frozen plant, so the curve's shape between points reflects instance difficulty as
well as dimension (the clipped baseline's own ratio — 4.9, 1.7, 2.4, 1.8, 1.8, 1.8 —
reproduces the campaign's independent feasibility probe at every n).

**The depth is the trained interval's deepest (J = 5), and every stepper's declared step is
the plant's own exact 1/L, gated.** The historical literal 0.05 exceeds the classical
stability limit 2/L on four of these six plants by 5–24×, where the iteration limit-cycles
while the box constraint hides it; every step literal in these documents sits behind a
parse-time gate that recomputes L from the plant and refuses anything but the exact value.

**The convex policy is absent from the grid — and the reason recorded here did not survive
measurement (corrected 2026-08-18).** The exclusion was attributed to the declared-solver
contract: on the frozen n = 50 plant the canary reported ‖du‖ = 1.43e-01 between the declared
backend and the reference, and the build was refused at the grid's own largest dimension.
Adjudicated afterwards against an exact interior-point solution of the same program, **the
declared backend is the accurate one**: MOREAU sits 2.96e-13 from the true control, while the
DIFFCP/SCS *reference* sits 1.43e-01 away and returns an **infeasible** control — |u| = 0.238
against a bound of 0.100. The 1.43e-01 this sentence once quoted as the candidate's failure is
the reference's own error, and the gate fired on it.

**Nothing drawn changes**: the contender is still absent from this grid. But the contract
clause must not be repeated — the defensible exclusion is **cost**. See
the reduced-form research document
§1.1 for the adjudication across n = 4 … 100.

**Statements the caption should carry:**

* **The proposed controller is best or tied at every dimension** — 5.1 % to 14.5 % below the
  clipped-Riccati baseline, the largest gap at n = 15, the grid's hardest instance.
* **The learned step sizes alone buy essentially nothing over the principled 1/L** (the
  analytic first-order method and its learned-step variant coincide to ~0.005 at every n):
  the learned cost-to-go matrix is the differentiator, at every dimension.
* **The per-iteration-matrix ablation stays within 0.003 of the shared-matrix controller
  grid-wide** — J times the parameters for no measurable gain, now at six dimensions rather
  than one.
* The recurrent baseline tracks the proposed controller closely and stays just behind it at
  nearly every n.

## 12. The cost grid: what Figure 4 measures, and what its numbers mean

**Each controller is measured as though it had its own dedicated machine, one per phase.**
Every {controller} × {offline, online} cell runs in a fresh process: no import, allocator or
Riccati result is shared, so a computation two controllers both need is charged in full to
each — that is what "its own machine" means. Offline for a trained family is the clean
per-epoch median times the declared epoch count (the extrapolation validated at 1.6 % error),
never the stored provenance time, which was taken in a long shared process under drifting
load. Offline for an analytic family is the full synthesis, timed whole.

**Online decomposes into setup and per-step, and per-step is at batch 1.** Setup is loading
the frozen artifact through the first control — for the convex policy that includes compiling
its program and validating its declared solver, and the split exists precisely so that cost
is visible rather than amortised away. Per-step is the median of 2000 single-state policy
calls; the batched throughput is reported separately, as a per-batch distribution, because
dividing it by the batch size would flatter the per-step number by exactly the vectorisation
factor (measured: ~500× for the unfolded families).

**The integrity check is a palindrome, and its readout is a number the table prints.** The
cast runs forwards then backwards; drift over a run is *systematic*, so it moves the **median
signed difference between the halves across the whole cast**, and that median must stay
within a declared bound or the pass fails. The published pass reports **+0.72, +1.89, +0.64,
+0.07 and +0.41 %** over its five passes — mixed in sign, no slope.

That statistic replaced a per-cell relative bound on 2026-08-13, and the reason is worth
stating because it is a measurement rather than a preference. A per-cell bound is wrong in
both directions here: four *adjacent* fresh processes measuring the same sub-millisecond cell
on an idle 16-core machine read 0.4526, 0.5143, 0.5720 and 0.6034 ms — a 33 % span, while the
median of the 2000 calls each of them reduces has a sampling error near 0.5 %, so the spread
is between processes and no extra calls shrink it — and *at the same time* a machine slowing
5 % across a run would sit inside any bound loose enough to survive that noise. A per-cell
test also fails more often the more passes you run, which is exactly backwards for a design
whose passes buy the dispersion.

**What is measured directly, and what is estimated.** Online setup and per-step are direct:
the artifact is loaded and 2000 real batch-1 policy calls are timed after 50 warm-ups. A
closed-form family's offline cost is direct too, and it is the **cold** synthesis — what a
deployment pays once — with the warm re-synthesis reported beside it, because the two differ
by 1.5–1.8× for the numerical families and by **9.4× for the SDP-frozen policy** (749 ms
against 79.6 ms), which is one-time solver compilation and a real cost. **A trained family's
offline total is the one estimated quantity**: five real epochs are timed, their median
multiplied by the declared epoch count, and the setup remainder added.

**That estimate has now been measured against the thing it estimates** (2026-08-14, the author's
instruction). Every contender's *full* declared training was run in its own fresh process,
forwards and backwards, and the same execution produced both numbers — so the comparison is of
one training run against its own extrapolation rather than against a different run:

| | measured total | extrapolated | error | peak RSS, full | from 5 epochs |
|---|---:|---:|---:|---:|---:|
| UF-α | 85.9 s | 85.3 s | −0.71 % | 1029 MiB | −0.32 % |
| UF-αP (proposed) | 93.1 s | 91.9 s | −1.21 % | 1222 MiB | −0.17 % |
| UF-αPʲ | 98.6 s | 97.7 s | −0.96 % | 1248 MiB | −0.50 % |
| GRU | 556.8 s | 555.7 s | −0.20 % | 8783 MiB | −0.92 % |
| COCP | 2497.3 s | 2494.5 s | −0.11 % | 4921 MiB | −0.09 % |

Worst case **1.21 %** on time and **0.92 %** on memory, and every error is *negative* for a
reason: the extrapolation multiplies the per-epoch **median**, which discards the slower first
epoch, and a peak taken over five epochs cannot see a later allocation. Both proxies understate,
slightly and predictably.

**The table keeps the extrapolated column, and the reason is dispersion rather than accuracy.**
A single measured pass gives two samples; the extrapolation gives ten, and the run-to-run
variation of a training time is a few percent — larger than the extrapolation's own error. So
the honest arrangement is the one now in place: the column is estimated by a route whose error
against a direct measurement is ≤1.21 %, and the quartiles beside it are real.

**Memory is peak resident set above the measuring process's own baseline**, taken after the
torch import so that ~700 MiB of interpreter is charged to nobody. It is a process-level
proxy, not a model size: it includes the allocator's arenas and every transient buffer, and
its resolution is quantised — readings differ by exact multiples of 256 KiB, about 3 % of the
9 MiB a Riccati synthesis costs.

**Threads are not pinned.** The harness sets no `torch.set_num_threads` and no
`OMP_NUM_THREADS`, so every controller is measured with all 32 hardware threads available to
it. That is coherent with "its own dedicated machine", and it means every absolute number
here is specific to this machine and its core count — an AMD Ryzen 9 9950X under WSL2. The
*ratios* are the transferable quantity.

**The numbers the paper's claim rests on** (n = 4, J = 3, one seed, five palindrome passes,
median (Q1–Q3) over ten fresh processes):

| | offline | online setup | per step | offline peak RSS |
|---|---:|---:|---:|---:|
| UF-αP (proposed) | 93.2 s | 24.3 ms | 0.0398 ms | 1.22 GiB |
| GRU | 536 s | 23.0 ms | 0.0300 ms | 8.70 GiB |
| COCP | 2430 s | 696 ms | 0.4530 ms | 4.92 GiB |
| SDP-frozen policy | 0.757 s | 737 ms | 0.4610 ms | 170 MiB |
| Clipped-LQR | 0.00403 s | 11.5 ms | 0.0062 ms | 8.75 MiB |

The proposed controller trains **26× faster** than the convex policy and produces a control
**11.4× faster**. The recurrent baseline is the cheapest per control (0.0300 against
0.0398 ms) and by far the most expensive to fit — 5.8× the time and **7.1× the memory**.

**The depth these are measured at is the depth the paper operates at.** An unrolled method's
per-step cost is essentially linear in J, so a cost table measured at a depth the paper does
not use would misreport it by that factor; this table was measured at J = 7 first, and
correcting it to J = 3 changed the convex-policy ratios from 33× and 5× to 26× and 11.4×.

## 13. Related

* `../architecture/03_analysis_and_visual_standard.md`
  — the figure grammar these plots obey, including the broken-axis and margin-label rules.
* `../planning/05_icassp_paper/four_figure_campaign/measured_setup_and_build_order.md`
  — the campaign plan, carrying every measurement quoted above with the method that produced it.
* [`cross_machine_numerical_reproducibility.md`](cross_machine_numerical_reproducibility.md)
  — why a step size is frozen as a literal rather than recomputed.
