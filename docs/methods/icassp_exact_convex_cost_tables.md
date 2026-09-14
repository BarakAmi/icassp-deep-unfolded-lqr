# The exact-convex campaign's cost tables

Both grids in full, every contender, exactly as `tools/emit_cost_grid_table.py`
wrote them. Committed for the same reason the ICASSP table is: **a number the
paper rests on belongs in a tracked document**, and the store is deliberately
untracked. Regenerate with `tools/run_cost_grid.py` then the emitter; the
notebook reads these same files rather than rebuilding them.


**On the tooling named above.** `tools/run_cost_grid.py`, the emitter and the campaign notebook are how these tables are MADE, in the research repository where the campaign lives. A paper's artifact repository carries the emitted table and not the machinery that emitted it — the same reason the projection's own authoring tools do not ship — so a reader of that tree will not find them, and does not need them: the numbers below are the measurement.

Both are measured at **J = 3**, the depth the paper operates at, so they sit
beside [the ICASSP table](icassp_cost_table.md) row for row.

## What is comparable with what

* **Within a table**, the ratios are the result.
* **Table 1 against the ICASSP table**: same instance, same substrate, same
  depth, same driver — comparable cell by cell. This is the A/B, **with one
  asymmetry that has to be stated rather than discovered**: every table here is
  the median of **2 fresh-process samples over 1 pass**, and the ICASSP table is
  the median of **10 over 5**. So the medians are comparable and the
  parenthesised quartiles are not — with two samples a "Q1–Q3" is min–max.

  That is a deliberate choice, not an oversight. A mixed pass of long and short
  phases put a 70 % disagreement into cells of a few tens of milliseconds on
  this host; run as separate per-phase invocations the same cells agree to
  3.27 %, and shortening the pass was preferred to widening the gate. The
  emitter composes per-phase invocations into one table by design, and the
  published ICASSP grid was itself produced that way. **Quote the medians and
  the ratios; do not quote these quartiles against the ICASSP table's.**
* **Table 1 against Table 2**: *not* comparable. Different plant (n = 4 against
  n = 100, m = 30) and different substrate (CPU/float64 against CUDA/float32).
* **Table 2 against Table 3**: comparable, and **this is the second A/B**. Same
  plant, same substrate, same depth, same driver, same machine — the two differ
  in **one declared scalar**, the control box. Read cell by cell, they say what a
  tighter constraint costs to synthesise and to run.
* **Any absolute number**: belongs to this machine — and for Table 2, to a
  machine **whose GPU also drives a Windows desktop**. See
  [the results document](icassp_exact_convex_results.md) for what that cost and
  how the measurement was made robust to it.

---

## Table 1 — n = 4, m = 2, CPU / float64

Each entry is the **median (Q1–Q3)** of 2 independent fresh-process samples: two palindrome halves per pass, 1 passes. Resident memory is the peak **above the measuring process's own baseline**. The last column is a *within-process* spread — the p10–p90 of 2000 timed calls — and is not comparable with the quartiles beside it.

| controller | offline time [s] | online setup [ms] | per step [ms] | offline peak RSS [MiB] | online peak RSS [MiB] | per-step p10–p90 [ms] |
|---|---:|---:|---:|---:|---:|---:|
| Riccati (unconstrained) | 0.00396 (0.00393–0.00398) | 12.3 (12–12.5) | 0.00323 (0.00321–0.00325) | 9 (9–9) | 375 (375–375) | 0.0032–0.00395 |
| Truncated-Riccati | 0.00546 (0.00542–0.0055) | 13.3 (13.1–13.4) | 0.00639 (0.00639–0.0064) | 9 (9–9) | 376 (376–376) | 0.0063–0.0067 |
| Standard-PGD | 0.00254 (0.00233–0.00274) | 11 (10.8–11.2) | 0.0399 (0.0398–0.0399) | 4.12 (4.06–4.19) | 373 (373–374) | 0.0395–0.0494 |
| UF-$\alpha$ (learned steps) | 89.9 (88.3–91.4) | 25.3 (24.8–25.7) | 0.0403 (0.0403–0.0404) | 1.03e+03 (1.03e+03–1.03e+03) | 385 (385–385) | 0.0398–0.0747 |
| UF-$\alpha$P (proposed) | 93.6 (93.1–94.1) | 25.5 (24.6–26.5) | 0.0401 (0.04–0.0401) | 1.22e+03 (1.22e+03–1.22e+03) | 385 (385–386) | 0.0396–0.0695 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 102 (101–104) | 28 (27.7–28.2) | 0.0509 (0.0492–0.0526) | 1.24e+03 (1.24e+03–1.24e+03) | 386 (386–386) | 0.047–0.133 |
| GRU | 583 (558–608) | 25.9 (24.4–27.4) | 0.0301 (0.03–0.0302) | 8.73e+03 (8.72e+03–8.75e+03) | 391 (390–391) | 0.0295–0.0423 |
| COCP (exact) | 82.8 (82.7–82.8) | 27 (26.9–27) | 0.105 (0.0968–0.113) | 888 (886–889) | 390 (390–390) | 0.0861–0.211 |
| Dual-frozen policy | 0.0238 (0.0232–0.0244) | 38.3 (37.5–39.1) | 0.0975 (0.0947–0.1) | 5.88 (5.81–5.94) | 379 (379–379) | 0.0838–0.202 |

### The closed-form families: one-time cost against steady state

The offline column above charges the **cold** synthesis, which is what a deployment pays once. A ratio near 1.5 is interpreter and cache warm-up; a ratio near 10 is a solver compiling its program, and that cost is real.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 3.96 (3.93–3.98) | 2.65 (2.57–2.73) | 1.49x |
| Truncated-Riccati | 5.46 (5.42–5.5) | 3.06 (2.99–3.13) | 1.78x |
| Standard-PGD | 2.54 (2.33–2.74) | 1.13 (1.12–1.14) | 2.24x |
| Dual-frozen policy | 23.8 (23.2–24.4) | 19.3 (19.2–19.4) | 1.23x |

### The extrapolated offline total, two routes to its spread

The total is a per-epoch median times a declared epoch count, never a timed whole. Both routes are reported because their disagreement would be a finding about the extrapolation.

| controller | total [s] | within-process route [s] | across-process route [s] |
|---|---:|---:|---:|
| UF-$\alpha$ (learned steps) | 89.9 | 26.5 | 4.38 |
| UF-$\alpha$P (proposed) | 93.6 | 33.5 | 1.48 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 102 | 30.2 | 5.34 |
| GRU | 583 | 136 | 71.2 |
| COCP (exact) | 82.8 | 13.7 | 0.0254 |

---

## Table 2 — n = 100, m = 30, CUDA / float32

Each entry is the **median (Q1–Q3)** of 2 independent fresh-process samples: two palindrome halves per pass, 1 passes. Resident memory is the peak **above the measuring process's own baseline**. The last column is a *within-process* spread — the p10–p90 of 2000 timed calls — and is not comparable with the quartiles beside it.

| controller | offline time [s] | online setup [ms] | per step [ms] | offline peak RSS [MiB] | online peak RSS [MiB] | per-step p10–p90 [ms] |
|---|---:|---:|---:|---:|---:|---:|
| Riccati (unconstrained) | 0.622 (0.534–0.709) | 507 (500–515) | 0.0272 (0.0271–0.0273) | 350 (350–350) | 540 (540–540) | 0.0242–0.0504 |
| Truncated-Riccati | 0.555 (0.537–0.573) | 568 (558–578) | 0.0835 (0.0833–0.0837) | 355 (352–358) | 574 (571–577) | 0.0705–0.136 |
| Standard-PGD ($2/L$) | 0.184 (0.182–0.186) | 377 (373–382) | 0.524 (0.453–0.595) | 142 (139–145) | 487 (487–488) | 0.427–0.761 |
| UF-$\alpha$ (learned steps) | 364 (355–373) | 402 (402–402) | 0.402 (0.394–0.409) | 1.12e+03 (1.12e+03–1.13e+03) | 488 (488–489) | 0.355–0.564 |
| UF-$\alpha$P (proposed) | 422 (416–427) | 412 (405–418) | 0.41 (0.401–0.419) | 1.15e+03 (1.14e+03–1.15e+03) | 526 (526–526) | 0.359–0.582 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 400 (386–414) | 412 (410–415) | 0.557 (0.496–0.617) | 1.15e+03 (1.14e+03–1.15e+03) | 523 (523–523) | 0.446–0.801 |
| GRU | 474 (453–494) | 343 (340–347) | 0.175 (0.169–0.181) | 1.13e+03 (1.13e+03–1.13e+03) | 510 (506–513) | 0.151–0.253 |
| COCP (exact) | 5.05e+03 (4.84e+03–5.26e+03) | 968 (961–975) | 1.17 (1.16–1.17) | 1.27e+03 (1.27e+03–1.27e+03) | 732 (729–735) | 1.08–1.35 |
| Dual-frozen policy | 7.58 (7.48–7.69) | 7.97e+03 (7.97e+03–7.98e+03) | 1.16 (1.16–1.16) | 91 (90.8–91.2) | 721 (718–725) | 1.07–1.47 |

### The closed-form families: one-time cost against steady state

The offline column above charges the **cold** synthesis, which is what a deployment pays once. A ratio near 1.5 is interpreter and cache warm-up; a ratio near 10 is a solver compiling its program, and that cost is real.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 622 (534–709) | 48.3 (48–48.7) | 12.87x |
| Truncated-Riccati | 555 (537–573) | 45.7 (45.6–45.8) | 12.14x |
| Standard-PGD ($2/L$) | 184 (182–186) | 18.3 (17.8–18.7) | 10.06x |
| Dual-frozen policy | 7.58e+03 (7.48e+03–7.69e+03) | 7.29e+03 (7.19e+03–7.39e+03) | 1.04x |

### The extrapolated offline total, two routes to its spread

The total is a per-epoch median times a declared epoch count, never a timed whole. Both routes are reported because their disagreement would be a finding about the extrapolation.

| controller | total [s] | within-process route [s] | across-process route [s] |
|---|---:|---:|---:|
| UF-$\alpha$ (learned steps) | 364 | 134 | 24.4 |
| UF-$\alpha$P (proposed) | 422 | 151 | 16.6 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 400 | 198 | 39.2 |
| GRU | 474 | 151 | 57.9 |
| COCP (exact) | 5.05e+03 | 2.08e+03 | 603 |

---

## Related

* [The ICASSP campaign's cost table](icassp_cost_table.md) — the control Table 1 is measured against.
* [The campaign's results](icassp_exact_convex_results.md) — the A/B, the inverted headline ratio, and the shared-GPU diagnosis.
* `notebooks/experiments/09_icassp_exact_convex_campaign.ipynb` — reads these same files.

---

## Table 3 — the same plant at `u_max = 0.02`, CUDA / float32

`fig5_stress_depth`'s instance — the paper's Figure 3. **Re-measured rather
than carried over from Table 2**,
because the convex policy's offline cost moves with the box and a table that is
right for one instance and quoted for another is the error the cross-tier rule
exists to prevent.

Measured as two passes, one per phase, after a 36-cell mixed pass was **refused
twice** — see the plan's §7g.
The halves agree at **+3.27 %** (offline) and **−6.61 %** (online).

Each entry is the **median (Q1–Q3)** of 2 independent fresh-process samples: two palindrome halves per pass, 1 passes. Resident memory is the peak **above the measuring process's own baseline**. The last column is a *within-process* spread — the p10–p90 of 2000 timed calls — and is not comparable with the quartiles beside it.

| controller | offline time [s] | online setup [ms] | per step [ms] | offline peak RSS [MiB] | online peak RSS [MiB] | per-step p10–p90 [ms] |
|---|---:|---:|---:|---:|---:|---:|
| Riccati (unconstrained) | 0.684 (0.6–0.768) | 560 (538–582) | 0.0292 (0.0285–0.0299) | 355 (352–357) | 545 (543–548) | 0.0259–0.0646 |
| Truncated-Riccati | 0.614 (0.545–0.683) | 670 (608–732) | 0.0838 (0.0813–0.0863) | 349 (349–349) | 574 (571–577) | 0.0708–0.131 |
| Standard-PGD ($2/L$) | 0.203 (0.203–0.204) | 427 (417–437) | 0.486 (0.451–0.521) | 142 (139–145) | 483 (483–483) | 0.376–2 |
| UF-$\alpha$ (learned steps) | 487 (469–504) | 426 (420–433) | 0.461 (0.446–0.477) | 1.11e+03 (1.11e+03–1.11e+03) | 495 (495–495) | 0.376–0.737 |
| UF-$\alpha$P (proposed) | 423 (420–426) | 455 (455–456) | 0.429 (0.423–0.435) | 1.15e+03 (1.15e+03–1.15e+03) | 537 (534–539) | 0.376–0.668 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 521 (442–600) | 450 (438–461) | 0.456 (0.449–0.462) | 1.14e+03 (1.14e+03–1.14e+03) | 529 (526–531) | 0.385–0.716 |
| GRU | 553 (534–572) | 387 (377–396) | 0.171 (0.17–0.173) | 1.12e+03 (1.12e+03–1.12e+03) | 518 (514–521) | 0.156–0.273 |
| COCP (exact) | 6.06e+03 (6.02e+03–6.1e+03) | 874 (827–920) | 1.26 (1.25–1.27) | 1.26e+03 (1.26e+03–1.26e+03) | 732 (728–735) | 1.09–1.7 |
| Dual-frozen policy | 25.9 (25.6–26.1) | 2.79e+04 (2.7e+04–2.88e+04) | 1.24 (1.21–1.26) | 105 (105–105) | 722 (719–725) | 1.11–1.57 |

## The closed-form families: one-time cost against steady state

The offline column above charges the **cold** synthesis, which is what a deployment pays once. A ratio near 1.5 is interpreter and cache warm-up; a ratio near 10 is a solver compiling its program, and that cost is real.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 684 (600–768) | 74 (60.8–87.1) | 9.25x |
| Truncated-Riccati | 614 (545–683) | 49.5 (49.4–49.6) | 12.41x |
| Standard-PGD ($2/L$) | 203 (203–204) | 23.8 (23.3–24.3) | 8.56x |
| Dual-frozen policy | 2.59e+04 (2.56e+04–2.61e+04) | 2.63e+04 (2.61e+04–2.65e+04) | 0.98x |

## The extrapolated offline total, two routes to its spread

The total is a per-epoch median times a declared epoch count, never a timed whole. Both routes are reported because their disagreement would be a finding about the extrapolation.

| controller | total [s] | within-process route [s] | across-process route [s] |
|---|---:|---:|---:|
| UF-$\alpha$ (learned steps) | 487 | 164 | 49.6 |
| UF-$\alpha$P (proposed) | 423 | 221 | 9 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 521 | 162 | 223 |
| GRU | 553 | 156 | 54 |
| COCP (exact) | 6.06e+03 | 2.14e+03 | 114 |

### What the tighter box actually costs

Cell by cell against Table 2, which differs from this one in the box alone.

**The analytic contenders are the control arm, and they are why the rest can be
read.** Their costs *cannot* depend on the box — a Riccati recursion is a Riccati
recursion — and all three moved by the same amount: **×1.10, ×1.11, ×1.10**
offline. That is the between-session offset of this machine, and every other
ratio below is only interesting to the extent it exceeds it.

| controller | offline | online setup | per step |
|---|---:|---:|---:|
| Riccati (unconstrained) | ×1.10 | ×1.10 | ×1.07 |
| Truncated-Riccati | ×1.11 | ×1.18 | ×1.00 |
| Standard-PGD | ×1.10 | ×1.13 | ×0.93 |
| UF-α | ×1.34 | ×1.06 | ×1.15 |
| UF-αP (proposed) | ×1.00 | ×1.10 | ×1.05 |
| UF-αPʲ | ×1.30 | ×1.09 | ×0.82 |
| GRU | ×1.17 | ×1.13 | ×0.98 |
| **COCP (exact)** | **×1.20** | ×0.90 | **×1.08** |
| **Dual-frozen policy** | **×3.42** | **×3.50** | ×1.07 |

**The dual-frozen policy is what the box hits, and it is not close.** Its
synthesis goes **7.58 s → 25.9 s** and its online setup **7.97 s → 27.9 s**, a
factor of three and a half against a ×1.10 baseline. This matters to how the
campaign describes it: at `fig3_large_depth`'s box it is *"a seven-second
synthesis that beats every fixed unfolding"*, and **that phrase does not
transfer to `fig5_stress_depth`**.
It is a twenty-six-second synthesis here — still trivial against 6060 s of COCP
training, and still untrained — but the number quoted must be the instance's own.

**The exact convex policy costs ×1.20 to synthesise**, about +9 % once the
session offset is removed. The plan budgeted +15 % on it and was right in
direction and slightly low in magnitude.

**And the per-step cost barely moves — which is the answer to the question this
table was re-measured to ask.** §6.3 said the per-step cost of the exact box-QP
at 89.6 % saturation was *"a number the paper does not have and cannot infer"*.
It is **1.26 ms against 1.17 ms**, ×1.08, essentially the session offset. **A
tighter active set does not make the solve materially more expensive per step.**
The unfolded families move in both directions (×0.82 to ×1.15) by less than their
own extrapolation spread, and the recurrent baseline does not move at all.

**Memory does not move anywhere**, which confirms on a second instrument what the
box measurements already found: the box changes what the numbers are and not what
the run costs.

### The caveat this table inherits, and one it adds

Everything Table 2's caveat says applies here: these absolutes belong to one
machine whose GPU also drives a Windows desktop, and within-table ratios are what
transfer.

The one it adds is about the **cheap cells**. The analytic contenders' offline
column is tens of milliseconds, and a 36-cell heterogeneous pass could not measure
it to 60 % agreement between its first and last cell — the refusal that forced the
per-phase split. Those three rows are the least trustworthy in the table and the
least consequential; nobody chooses a controller on a 47 ms synthesis. Every cell
the figure rests on is seconds to minutes and reproduces.
