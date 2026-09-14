# Deep-unfolded control for box-constrained LQR — exact-convex artifact

Code, data and instructions for the experiments in the paper. Everything the
paper reports is here: the three figures, the cost table, the trained
controllers, the inference experiments, and the logs of both.

You can read a figure's numbers in a minute, recompute every one of them in
about ten, or retrain the whole thing over a couple of days. Pick a rung.

---

## What is different about this version

The convex policy is solved **exactly**. Its per-step program is a
box-constrained quadratic program, and it is solved as that — through the
problem's own optimality conditions, and differentiated through the same
system — rather than through a general convex solver. The bound reported beside
it is computed from the same program.

Two consequences a reader should know before installing:

* **No result in this paper is produced by proprietary software.** Every solve
  runs in the same array library as the rest of the pipeline. See the next
  section for what that does and does not mean for the install.
* **The comparison gains a third instance.** Figure 3 repeats the depth
  comparison under a control bound tight enough that most control entries sit
  against it, where the first figure's bound is mostly slack.

## Before you install: one dependency is proprietary, and unused here

The package installs [`moreau`](https://pypi.org/project/moreau/), which is
**not open source**. Its licence reads:

> Copyright (c) 2024-2025 Optimal Intellect, Inc. This software is proprietary
> and confidential. Unauthorized copying, distribution, modification, or use of
> this software, via any medium, is strictly prohibited without prior written
> permission.

It installs from public PyPI without credentials, and **no licence key is
needed for anything here** — nor is the solver itself: not one contender in
this paper's studies routes through it. It remains a declared dependency
because the package ships whole rather than trimmed to one paper, and other
work in the same codebase uses it. We state this before the install command
rather than after it, so the choice is yours to make knowingly.

Every other dependency is permissively licensed.

## Install

```bash
git clone <this repository>
cd <this repository>
uv sync
uv run pytest -q          # no GPU and no data needed
```

Linux x86-64 and macOS 14+. **Windows is not supported**: `moreau-cpu`
publishes no Windows wheel and no source distribution, so there is nothing to
install and nothing to compile.

---

## The four rungs

Each is self-contained. Stop wherever you have what you came for.

| | You get | You need | Cost |
|---|---|---|---|
| **0 — Install** | the code, and its own test suite green | nothing | minutes |
| **1 — Look** | every figure redrawn from its stored data | nothing extra | seconds |
| **2 — Check** | **every number in the paper recomputed** from the per-trajectory costs behind it | the results bundle | ~1 min |
| **3 — Re-measure** | the published numbers re-derived from the published weights | the results bundle | ~1 h |
| **4 — Retrain** | the models rebuilt from their specifications | nothing extra | days |

**Rung 2 is the one that matters.** The bundle carries the cost of every
simulated trajectory, so the analysis re-derives each figure's table in front of
you rather than asking you to trust ours.

**Rung 1 is byte-exact.** Redrawing a figure reproduces the shipped file
exactly, so `git status` stays clean afterwards; nothing in a rendered artifact
records the clock or the renderer's version. If a redraw reports a modified
file, something has actually changed.

**These are the paper's figures, laid out differently.** The manuscript prints
the same figures at a different size, without the panel titles, and with the
legend made translucent so that it does not cover a curve's label — adjustments
made once, by hand, for the two-column page. Every series, every number and
every error bar is identical, and the artifacts here are what produced them. If
you are comparing a figure in the paper against one in this repository, expect
the layout to differ and the content not to.

Rungs 2–4 need the results bundle. Fetch it from this repository's Releases
page; the release notes give the sha256 it should have, and `manifest.lock`
inside it records every identifier it contains.

See [`docs/running_experiments.md`](docs/running_experiments.md) for the exact
commands, and [`docs/repository_layout.md`](docs/repository_layout.md) for what
lives where.

---

## What is here

The paper's four artifacts, and the studies that produced them:

| Artifact | Study | Tier |
|---|---|---|
| Figure 1 — cost against unfolding depth | `studies/icassp_exact_convex/fig1_depth.toml` | `publication_b16k` |
| Figure 2 — cost against mismatch severity | `fig2_angle_{blind,told,world}.toml` | `publication_b16k` |
| Figure 3 — the binding bound | `studies/icassp_exact_convex/fig5_stress_depth.toml` | `publication` |
| The cost table | `paper_artifacts/benchmarks/icassp_exact_convex_cost_grid_n4/` | — |

**Figure 3's study is named `fig5`.** That is our internal experiment
numbering and it is deliberately left alone: the identifiers under which the
results are stored are derived from the document, so renaming it to match the
paper would detach every record the paper reports. The figure the paper prints
third is the one this document produces.

**Figure 2 belongs to no single study.** It is composed of the three mismatch
conditions, and is filed under an identifier derived from all three — so
re-running any one of them moves it, and a figure assembled from a different
set is a different figure rather than a silent replacement.

**The tier is part of the identity.** A study resolves to a different set of
records at each tier, so the commands in the run guide state it every time. Ask
for `--tier publication` on a document that was run at `publication_b16k` and
you will get a store that is partly complete and a refusal that looks like a
corrupt download.

## What is not here

This repository is a projection of a larger research codebase, and it carries
only this paper's world. Other experiments, other campaigns, earlier versions
of this same work, internal planning records and the project's architecture
documents are not published. Where a shipped document cites one of them, the
citation is kept as plain text and the link removed, so nothing dangles.

The baseline-tuning studies that accompanied the earlier version of this work
are not carried here. The fairness argument they support is unchanged and is
described in the methods documents.

## One check may fail, and here is exactly why

`tests/models/constrained/test_box_qp.py` grades the exact solver against the
optimality conditions of the problem it solves, over eight thousand states per
plant. One case — the largest plant, in double precision — is sensitive to how
the host's linear algebra library sums its partial products.

Floating-point addition is not associative, so a multithreaded library
combining partial sums in a different order can move the last bit or two of a
residual, and the residual is compared against a threshold derived from the
problem's own scale. On a hosted runner the effective parallelism varies with
whatever else is on the machine, and we have watched the same commit produce
different outcomes on different runs.

We did not loosen the threshold. It is scaled to the problem rather than fitted
to the answer, and widening it until nothing fails would make the check
unable to refuse the thing it exists to catch. We would rather tell you the
condition than promise you a colour.

If you see it fail, nothing about the results is wrong: the certificate is
recomputed for every solve the paper reports, and every stored identifier is
recomputed and matched by `mbl store verify`.

## Licence and citation

MIT — see [`LICENSE`](LICENSE). Third-party attribution is in
[`NOTICE`](NOTICE): the semidefinite bound in
`src/mbl/models/constrained/lower_bound.py` is derived from the code released
with *Learning convex optimization control policies* (Agrawal, Barratt, Boyd
and Stellato, 2019), under the Apache License 2.0.

If you use this work, please cite the paper; see [`CITATION.cff`](CITATION.cff).
