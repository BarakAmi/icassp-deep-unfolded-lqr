# The measured cost table, as the paper should quote it

The paper's runtime/memory table, committed. It exists in the store as
`store/benchmarks/fig4_cost_grid/cost_grid_table.{md,tex}`, but the store is
deliberately untracked — this project does not commit experiment artifacts —
so the measurement was not among the files shared with a co-author and could
not be. **A number the paper rests on belongs in a tracked document**, and this
is that document. Regenerate it with `tools/emit_cost_grid_table.py`; the
`.tex` below is what the paper pastes.

Everything here is measured at **J = 3** — the depth the paper operates at, and
the depth Figure 2 runs. Measuring it at another depth would misreport the
unrolled families by roughly the ratio of the depths, since their per-step cost
is close to linear in J.

## What each number is

* **median (Q1–Q3) over ten independent fresh-process samples** — two
  palindrome halves per pass, five passes. Every cell runs in its own
  interpreter, so a computation two controllers both need is charged in full
  to each.
* **Online setup and per step are measured directly**: the frozen artifact is
  loaded and 2000 single-state control computations are timed after 50
  warm-ups. Per step is at **batch 1** — the batched throughput must never be
  divided by the batch size, which would flatter it by the vectorisation
  factor (measured ≈500× for the unfolded families).
* **The offline total of a trained family is extrapolated**: five real epochs,
  their median times the declared epoch count, plus setup. **Its error against
  a direct measurement is now known** — see below.
* **The offline cost of a closed-form family is the cold synthesis**, what a
  deployment pays once, with the warm re-synthesis reported beside it.
* **Memory is peak resident set above the measuring process's own baseline**,
  taken after the torch import so ~700 MiB of interpreter is charged to
  nobody. It is a process-level proxy, quantised at 256 KiB.

## Caveats a caption must carry

* **The machine is not partitioned.** No thread count is pinned, so every
  controller is measured with all 32 hardware threads of an AMD Ryzen 9 9950X
  under WSL2 available to it. The **ratios** transfer; the absolute numbers
  belong to this machine.
* **The palindrome's readout is part of the result.** The cast runs forwards
  then backwards, and drift moves the median signed difference between the
  halves. The published pass reports **+0.72, +1.89, +0.64, +0.07, +0.41 %**
  across its five passes — mixed in sign, no slope.
* **Do not quote the "within-process route" as an uncertainty.** It multiplies
  a per-epoch standard deviation by the epoch count, which assumes every epoch
  deviates in the same direction; independent epochs give σ√N, not σN, and it
  overstates by roughly an order of magnitude. The defensible uncertainty on
  the offline total is the across-process route, and the defensible statement
  about the extrapolation's accuracy is the direct measurement below.

## The extrapolation, measured against what it estimates

Every contender's **full** declared training was run in its own fresh process,
forwards and backwards, and the same execution produced both numbers — so this
compares one training run against its own extrapolation rather than against a
different run.

| | measured total | extrapolated | error | peak RSS, full training | from 5 epochs |
|---|---:|---:|---:|---:|---:|
| UF-α | 85.9 s | 85.3 s | −0.71 % | 1029 MiB | −0.32 % |
| UF-αP (proposed) | 93.1 s | 91.9 s | −1.21 % | 1222 MiB | −0.17 % |
| UF-αPʲ | 98.6 s | 97.7 s | −0.96 % | 1248 MiB | −0.50 % |
| GRU | 556.8 s | 555.7 s | −0.20 % | 8783 MiB | −0.92 % |
| COCP | 2497.3 s | 2494.5 s | −0.11 % | 4921 MiB | −0.09 % |

**Worst case 1.21 % on time and 0.92 % on memory**, and every error is negative
for a reason: the extrapolation multiplies the per-epoch *median*, which
discards the slower first epoch, and a peak over five epochs cannot see a later
allocation. Both proxies understate, slightly and predictably.

The table keeps the extrapolated column for **dispersion** rather than
accuracy: a single measured pass gives two samples where the extrapolation
gives ten, and the run-to-run variation of a training time is a few percent —
larger than the extrapolation's own error.

## The ratios the paper's claim rests on

At the operating point, against the convex policy: the proposed controller
trains **26× faster** and produces a control **11.4× faster**. Against the
recurrent baseline: it trains **5.8× faster** and uses **7.1× less memory** to
train, while producing a control ~33 % slower (0.0398 against 0.0300 ms).

## The table

| controller | offline time [s] | online setup [ms] | per step [ms] | offline peak RSS [MiB] | online peak RSS [MiB] | per-step p10–p90 [ms] |
|---|---:|---:|---:|---:|---:|---:|
| Riccati (unconstrained) | 0.00435 (0.00407–0.00472) | 11.6 (11.2–12.3) | 0.00303 (0.00298–0.00308) | 8.75 (8.75–9) | 375 (375–375) | 0.003–0.00314 |
| Truncated-Riccati | 0.00403 (0.00393–0.0041) | 11.5 (11.4–12.9) | 0.00622 (0.00615–0.00637) | 8.75 (8.75–9) | 375 (375–375) | 0.00611–0.00655 |
| Standard-PGD | 0.00203 (0.00197–0.00209) | 9.81 (9.33–11.1) | 0.04 (0.0399–0.0401) | 4 (3.81–4) | 374 (374–374) | 0.0395–0.0821 |
| UF-α (learned steps) | 88.8 (84.9–90.2) | 23.8 (22.8–24.1) | 0.0398 (0.0395–0.0401) | 1030 (1030–1030) | 385 (385–385) | 0.0392–0.057 |
| UF-αP (proposed) | 93.2 (90–98.7) | 24.3 (23.1–25.6) | 0.0398 (0.0394–0.0403) | 1220 (1220–1220) | 386 (386–386) | 0.0393–0.0568 |
| UF-αPʲ (per-iteration P) | 104 (98.6–113) | 27 (25.6–28.7) | 0.0476 (0.047–0.0484) | 1240 (1240–1240) | 386 (386–386) | 0.0468–0.0755 |
| GRU | 536 (530–553) | 23 (21.2–24.4) | 0.03 (0.0294–0.0304) | 8700 (8690–8740) | 391 (391–395) | 0.0293–0.0646 |
| COCP | 2430 (2390–2430) | 696 (680–714) | 0.453 (0.434–0.534) | 4920 (4910–4920) | 611 (604–614) | 0.364–0.623 |
| SDP-frozen policy | 0.757 (0.746–0.769) | 737 (721–763) | 0.461 (0.454–0.529) | 170 (169–170) | 602 (599–602) | 0.361–0.643 |

### The closed-form families: one-time cost against steady state

A ratio near 1.5 is a cold interpreter; a ratio near 10 is a solver compiling
its program, and a deployment synthesises once.

| controller | cold [ms] | warm re-synthesis [ms] | ratio |
|---|---:|---:|---:|
| Riccati (unconstrained) | 4.35 (4.07–4.72) | 2.57 (2.53–2.70) | 1.69× |
| Truncated-Riccati | 4.03 (3.93–4.10) | 2.58 (2.56–2.65) | 1.56× |
| Standard-PGD | 2.03 (1.97–2.09) | 1.15 (1.13–1.20) | 1.75× |
| SDP-frozen policy | 757 (746–769) | 78.1 (76.3–81.7) | 9.69× |

## The LaTeX, as generated

```latex
\begin{tabular}{lrrrrr}
\toprule
controller & offline time [s] & online setup [ms] & per step [ms] & offline peak RSS [MiB] & online peak RSS [MiB] \\
\midrule
Riccati (unconstrained) & 0.00435 (0.00407--0.00472) & 11.6 (11.2--12.3) & 0.00303 (0.00298--0.00308) & 8.75 (8.75--9) & 375 (375--375) \\
Truncated-Riccati & 0.00403 (0.00393--0.0041) & 11.5 (11.4--12.9) & 0.00622 (0.00615--0.00637) & 8.75 (8.75--9) & 375 (375--375) \\
Standard-PGD & 0.00203 (0.00197--0.00209) & 9.81 (9.33--11.1) & 0.04 (0.0399--0.0401) & 4 (3.81--4) & 374 (374--374) \\
UF-$\alpha$ (learned steps) & 88.8 (84.9--90.2) & 23.8 (22.8--24.1) & 0.0398 (0.0395--0.0401) & 1.03e+03 (1.03e+03--1.03e+03) & 385 (385--385) \\
UF-$\alpha$P (proposed) & 93.2 (90--98.7) & 24.3 (23.1--25.6) & 0.0398 (0.0394--0.0403) & 1.22e+03 (1.22e+03--1.22e+03) & 386 (386--386) \\
UF-$\alpha$P$^{(j)}$ (per-iteration P) & 104 (98.6--113) & 27 (25.6--28.7) & 0.0476 (0.047--0.0484) & 1.24e+03 (1.24e+03--1.24e+03) & 386 (386--386) \\
GRU & 536 (530--553) & 23 (21.2--24.4) & 0.03 (0.0294--0.0304) & 8.7e+03 (8.69e+03--8.74e+03) & 391 (391--395) \\
COCP & 2.43e+03 (2.39e+03--2.43e+03) & 696 (680--714) & 0.453 (0.434--0.534) & 4.92e+03 (4.91e+03--4.92e+03) & 611 (604--614) \\
SDP-frozen policy & 0.757 (0.746--0.769) & 737 (721--763) & 0.461 (0.454--0.529) & 170 (169--170) & 602 (599--602) \\
\bottomrule
\end{tabular}
```

## Related

* [The experimental methodology](icassp_experimental_methodology.md) — §12 is this table's full account.
* The dispersion-table plan — why the table has quartiles at all.
