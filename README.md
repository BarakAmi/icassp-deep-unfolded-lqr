# Deep-unfolded control for box-constrained LQR — ICASSP artifact

Code, data and instructions for the experiments in the paper. Everything the
paper reports is here: the figures, the table, the trained controllers, the
inference experiments, and the logs of both.

You can read a figure's numbers in a minute, recompute every one of them in
about ten, or retrain the whole thing in about two days. Pick a rung.

---

## Before you install: one dependency is proprietary

The convex-policy baseline (COCP) solves its quadratic programs through
[`moreau`](https://pypi.org/project/moreau/), which is **not open source**. Its
licence reads:

> Copyright (c) 2024-2025 Optimal Intellect, Inc. This software is proprietary
> and confidential. Unauthorized copying, distribution, modification, or use of
> this software, via any medium, is strictly prohibited without prior written
> permission.

It installs from public PyPI without credentials, and **no licence key is
needed for anything in this paper** — every solver declaration here runs on the
CPU backend, and only the CUDA backend is key-gated. We state this before the
install command rather than after it, so the choice is yours to make knowingly.

Every other dependency is permissively licensed.

## Install

```bash
git clone <this repository>
cd <this repository>
uv sync
uv run pytest -q          # ~9 minutes, no GPU and no data needed
```

Linux x86-64 and macOS 14+. **Windows is not supported**: `moreau-cpu`
publishes no Windows wheel and no source distribution, so there is nothing to
install and nothing to compile.

---

## The four rungs

Each is self-contained. Stop wherever you have what you came for.

| | You get | You need | Cost |
|---|---|---|---|
| **0 — Install** | the code, and its own test suite green | nothing | ~9 min |
| **1 — Look** | every figure and the table redrawn from their stored data | nothing extra | seconds |
| **2 — Check** | **every number in the paper recomputed** from the per-trajectory costs behind it | the results bundle | ~1 min |
| **3 — Re-measure** | the published numbers re-derived from the published weights | the results bundle | ~1.1 h |
| **4 — Retrain** | the models rebuilt from their specifications | nothing extra | ~47 h |

**Rung 2 is the one that matters.** The bundle carries `samples.parquet` for
every measurement — the cost of each of 32,768 simulated trajectories — so
`mbl analyse` re-derives each figure's table in front of you rather than asking
you to trust ours. Measured: every plotted column reproduces at exactly
`0.000e+00`.

Rungs 2–4 need the results bundle. Fetch it from this repository's Releases
page; `manifest.lock` in the bundle records the sha256 it should have.

See [`docs/running_experiments.md`](docs/running_experiments.md) for the exact
commands, and [`docs/repository_layout.md`](docs/repository_layout.md) for what
lives where.

---

## What is here

The paper's four artifacts, and the studies that produced them:

| Artifact | Study | Tier |
|---|---|---|
| Figure 1 — cost against unfolding depth | `studies/icassp/fig1_depth.toml` | `publication_b16k` |
| Figure 2 — cost against mismatch severity | `fig2_angle_{blind,told,world}.toml` | `publication_b16k` |
| Figure 3 — the large instance | `studies/icassp/fig3_large_depth.toml` | `publication` |
| Figure 4 — the cost table | `store/benchmarks/fig4_cost_grid/` | — |

**The tier is part of the identity.** A study resolves to a different set of
records at each tier, so the commands in the run guide state it every time. Ask
for `--tier publication` on a document that was run at `publication_b16k` and
you will get a store that is 25/225 complete and a refusal that looks like a
corrupt download.

Also included, as **supplementary material the four-page limit excluded**:
twelve studies that tune the *baselines* — the recurrent controller against its
learning rate, its width and its epoch budget; the convex policy against its
learning rate; the analytic baseline against its step size; and the whole cast
at twice its budget. These are the evidence behind the paper's claim that the
comparison is fair by construction. They ship as a second release asset.

## What is not here

This repository is a projection of a larger research codebase, and it carries
only this paper's world. Other experiments, other campaigns, internal planning
records and the project's architecture documents are not published. Where a
shipped document cites one of them, the citation is kept as plain text and the
link removed, so nothing dangles.

## One check may fail, and here is exactly why

`tests/spec/test_loader.py` checks that every study document still loads. Two
of them freeze the analytic baseline's step size as the literal `1/L` of their
plant, and a gate refuses any other value by **exact** binary64 equality —
which is what stops one contender existing under two nearly-identical step
sizes.

That recomputed eigenvalue is not always bit-identical. A multithreaded linear
algebra library combines its partial sums in an order that depends on how many
threads it uses, floating-point addition is not associative, and the last bit
or two can move — enough to fail an exact comparison. On a hosted CI runner the
effective parallelism varies with whatever else is on the machine, and we
measured the consequence: **five runs on one runner image produced 3, 1, 1, 3,
3 failures, with no code change between some of them.**

We tried the obvious fix and it made things worse. Pinning the thread count to
one gave **ten** failures rather than three, because the frozen literals were
authored on a machine whose library ran multithreaded — forcing single-threaded
moves the recomputation *away* from the values they were authored against. So
the instability is real, that is not its cure, and we would rather tell you the
condition than promise you a colour.

We did not take either of the two easier routes. Loosening the gate to a
tolerance would re-admit the defect it was written for. Re-authoring the frozen
literal would orphan the stored models this paper reports. Both would make the
check green and one of the paper's guarantees weaker.

If you see it fail, nothing about the results is wrong: the affected study
loads and runs correctly on the machine its literal was authored on, and every
number in the paper comes from records whose identifiers `mbl store verify`
recomputes and matches.

## Licence and citation

MIT — see [`LICENSE`](LICENSE). Third-party attribution is in
[`NOTICE`](NOTICE): the semidefinite bound in
`src/mbl/models/constrained/lower_bound.py` is derived from the code released
with *Learning convex optimization control policies* (Agrawal, Barratt, Boyd
and Stellato, 2019), under the Apache License 2.0.

If you use this work, please cite the paper; see [`CITATION.cff`](CITATION.cff).
