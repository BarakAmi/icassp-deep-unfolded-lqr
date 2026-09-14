# Figure 4 — the cost grid at depth J = 3

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

## The closed-form families: one-time cost against steady state

The offline column above charges the **cold** synthesis, which is what a deployment pays once. A ratio near 1.5 is interpreter and cache warm-up; a ratio near 10 is a solver compiling its program, and that cost is real.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 3.96 (3.93–3.98) | 2.65 (2.57–2.73) | 1.49x |
| Truncated-Riccati | 5.46 (5.42–5.5) | 3.06 (2.99–3.13) | 1.78x |
| Standard-PGD | 2.54 (2.33–2.74) | 1.13 (1.12–1.14) | 2.24x |
| Dual-frozen policy | 23.8 (23.2–24.4) | 19.3 (19.2–19.4) | 1.23x |

## The extrapolated offline total, two routes to its spread

The total is a per-epoch median times a declared epoch count, never a timed whole. Both routes are reported because their disagreement would be a finding about the extrapolation.

| controller | total [s] | within-process route [s] | across-process route [s] |
|---|---:|---:|---:|
| UF-$\alpha$ (learned steps) | 89.9 | 26.5 | 4.38 |
| UF-$\alpha$P (proposed) | 93.6 | 33.5 | 1.48 |
| UF-$\alpha$P$^{(j)}$ (per-iteration P) | 102 | 30.2 | 5.34 |
| GRU | 583 | 136 | 71.2 |
| COCP (exact) | 82.8 | 13.7 | 0.0254 |
