# Running the experiments

Every command below was executed against this repository and its results
bundle. Where a number appears, it was measured rather than estimated.

## The one thing to get right: the tier

A study document is a *specification*, and a tier scales the effort it is
executed at. **The two together identify the result** — the same document at
two tiers resolves to two different sets of records, so every command here
names its tier and you should not drop it.

Three of the paper's four artifacts were run at `publication_b16k`; the large
instance was run at `publication`. Asking for the wrong one is not an error, it
is a different study: Figure 1 at `publication` resolves to a store that is
25/225 complete, and the refusal reads like a corrupt download.

| Tier | What it scales | Use |
|---|---|---|
| `smoke` | 5 epochs, 1 seed, a subset of each axis | wiring; minutes |
| `standard` | 50 epochs, 1 seed | exploration |
| `publication` | the paper's effort | reproduction |
| `publication_b16k` | the same at batch 16384 | reproduction |

---

## Rung 1 — redraw the paper's figures

Nothing to download. The figure artifacts ship with the repository.

```bash
uv run mbl --store paper_artifacts figure rebuild fig1_cost_vs_depth --style ieee-paper
uv run mbl --store paper_artifacts figure rebuild fig3_large_cost_vs_depth --style ieee-paper
uv run python tools/render_severity_figure.py --store paper_artifacts
```

Figure 2 is assembled by a script rather than by `mbl figure`, because it is a
composite of three studies and belongs to no single one of them.

**`--store` is a global option**: it goes before the subcommand, never after.
`mbl figure --store X ...` writes into `./store` instead.

## Rung 2 — recompute every number

Fetch the results bundle from the Releases page and extract it, then:

```bash
tar -xzf icassp-results-core.tar.gz            # ~386 MiB, ~1.5 GiB extracted
uv run mbl --store store store reindex          # rebuilds the local index
uv run mbl --store store store verify           # 1515 identifiers recomputed
```

`verify` recomputes every identifier from the record's own specification and
checks it against the name the record is filed under. All 1,515 match.

Then re-derive each figure's table:

```bash
uv run mbl --store store analyse studies/icassp/fig1_depth.toml       --tier publication_b16k
uv run mbl --store store analyse studies/icassp/fig2_angle_blind.toml --tier publication_b16k
uv run mbl --store store analyse studies/icassp/fig2_angle_told.toml  --tier publication_b16k
uv run mbl --store store analyse studies/icassp/fig2_angle_world.toml --tier publication_b16k
uv run mbl --store store analyse studies/icassp/fig3_large_depth.toml --tier publication
```

Every plotted column reproduces at exactly `0.000e+00` against the tables the
paper was written from — measured across all five studies, from a cold start.

Two columns will **not** match, and it is worth saying why rather than letting
you find it. `interval_low` and `interval_high` are a bootstrap confidence
interval, and the tables shipped here were computed before that bootstrap was
seeded; they differ by up to 2.8e-2. No figure in the paper draws them — every
figure declares `dispersion: none` — and the analysis is deterministic from now
on, so two of your own runs will agree exactly. If you diff the parquet files
column by column, expect those two and nothing else.

The analysis reads `samples.parquet` — the cost of each
of 32,768 simulated trajectories — so this is a recomputation and not a
comparison of two summaries.

## Rung 3 — re-measure from the published weights

```bash
uv run mbl --store store run studies/icassp/fig1_depth.toml --tier publication_b16k
```

With the bundle extracted, every model is already present, so this evaluates
rather than trains. Recorded cost across all five studies: **1.14 h**.

**Figure 3 needs an NVIDIA GPU for this rung and the next.** All 380 of its
records declare `cuda:0`, and the device is part of what a model *is* — it
cannot be overridden by a tier or a flag. On a machine without one, `mbl run`
refuses that document by design. Rungs 1 and 2 work on any host, Figure 3
included: reading, analysing and redrawing never touch a device.

## Rung 4 — retrain from the specifications

```bash
uv run mbl --store store run studies/icassp/fig1_depth.toml --tier publication_b16k
```

against an empty store. Recorded: **46.87 h** total — 30.3 h CPU and 16.5 h
GPU, dominated by the unfolded family (22.1 h) and the convex policy (20.1 h).

**Order does not matter.** Models are content-addressed, so whichever document
trains a shared model first, every later document resolves the identical
record; running the studies in any order costs the same total and changes no
published number. If you want whole figures early rather than all of them
partly done, run `fig1_depth` first.

Use `--dry-run` to see what a run would do. It counts *points*, not distinct
models, so it over-reports on an empty store; treat it as a plan, not a
forecast.

## Running something of your own

The studies are TOML and the vocabulary is small. To sweep a new depth range,
copy `studies/icassp/fig1_depth.toml`, change the axis values, and run it at
`smoke` first:

```bash
uv run mbl --store store run my_study.toml --tier smoke
uv run mbl --store store analyse my_study.toml --tier smoke
uv run mbl --store store figure render my_study.toml --tier smoke
```

`smoke` is the only tier permitted to subset a swept axis, which is why it is
for wiring rather than for results. `uv run mbl --help` lists every command.
