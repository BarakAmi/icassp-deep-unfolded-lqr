# Figure 4 — the cost grid at depth J = 3

Each entry is the **median (Q1–Q3)** of 10 independent fresh-process samples: two palindrome halves per pass, 5 passes. Resident memory is the peak **above the measuring process's own baseline**. The last column is a *within-process* spread — the p10–p90 of 2000 timed calls — and is not comparable with the quartiles beside it.

| controller | offline time [s] | online setup [ms] | per step [ms] | offline peak RSS [MiB] | online peak RSS [MiB] | per-step p10–p90 [ms] |
|---|---:|---:|---:|---:|---:|---:|
| Riccati (unconstrained) | 0.00435 (0.00407–0.00472) | 11.6 (11.2–12.3) | 0.00303 (0.00298–0.00308) | 8.75 (8.75–9) | 375 (375–375) | 0.003–0.00314 |
| Truncated-Riccati | 0.00403 (0.00393–0.0041) | 11.5 (11.4–12.9) | 0.00622 (0.00615–0.00637) | 8.75 (8.75–9) | 375 (375–375) | 0.00611–0.00655 |
| Standard-PGD | 0.00203 (0.00197–0.00209) | 9.81 (9.33–11.1) | 0.04 (0.0399–0.0401) | 4 (3.81–4) | 374 (374–374) | 0.0395–0.0821 |
| UF-$\alpha$ (learned steps) | 88.8 (84.9–90.2) | 23.8 (22.8–24.1) | 0.0398 (0.0395–0.0401) | 1.03e+03 (1.03e+03–1.03e+03) | 385 (385–385) | 0.0392–0.057 |
| UF-$\alpha$P (proposed) | 93.2 (90–98.7) | 24.3 (23.1–25.6) | 0.0398 (0.0394–0.0403) | 1.22e+03 (1.22e+03–1.22e+03) | 386 (386–386) | 0.0393–0.0568 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 104 (98.6–113) | 27 (25.6–28.7) | 0.0476 (0.047–0.0484) | 1.24e+03 (1.24e+03–1.24e+03) | 386 (386–386) | 0.0468–0.0755 |
| GRU | 536 (530–553) | 23 (21.2–24.4) | 0.03 (0.0294–0.0304) | 8.7e+03 (8.69e+03–8.74e+03) | 391 (391–395) | 0.0293–0.0646 |
| COCP | 2.43e+03 (2.39e+03–2.43e+03) | 696 (680–714) | 0.453 (0.434–0.534) | 4.92e+03 (4.91e+03–4.92e+03) | 611 (604–614) | 0.364–0.623 |
| SDP-frozen policy | 0.757 (0.746–0.769) | 737 (721–763) | 0.461 (0.454–0.529) | 170 (169–170) | 602 (599–602) | 0.361–0.643 |

## The closed-form families: one-time cost against steady state

The offline column above charges the **cold** synthesis, which is what a deployment pays once. A ratio near 1.5 is interpreter and cache warm-up; a ratio near 10 is a solver compiling its program, and that cost is real.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 4.35 (4.07–4.72) | 2.57 (2.53–2.7) | 1.69x |
| Truncated-Riccati | 4.03 (3.93–4.1) | 2.58 (2.56–2.65) | 1.56x |
| Standard-PGD | 2.03 (1.97–2.09) | 1.15 (1.13–1.2) | 1.75x |
| SDP-frozen policy | 757 (746–769) | 78.1 (76.3–81.7) | 9.69x |

## The extrapolated offline total, two routes to its spread

The total is a per-epoch median times a declared epoch count, never a timed whole. Both routes are reported because their disagreement would be a finding about the extrapolation.

| controller | total [s] | within-process route [s] | across-process route [s] |
|---|---:|---:|---:|
| UF-$\alpha$ (learned steps) | 88.8 | 24.9 | 3.49 |
| UF-$\alpha$P (proposed) | 93.2 | 29.4 | 7.01 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 104 | 34.1 | 9.05 |
| GRU | 536 | 90.4 | 30.2 |
| COCP | 2.43e+03 | 164 | 32.6 |
