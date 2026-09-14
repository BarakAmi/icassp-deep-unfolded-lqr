# Running the experiments

Four rungs, from reading a number to rebuilding every model. Each is
self-contained; stop where you have what you came for.

## The one thing to get right: the tier

A study document describes an experiment at several **tiers** — a smoke tier, a
standard tier, and the publication tier the paper reports. The tier is part of
what identifies a record, not a speed setting, so the same document resolves to
a different set of results at each one.

Four of this paper's five documents were run at `publication_b16k` and one
at `publication`. Every command below states the tier. Asking for the wrong one
gives you a store that is partly complete and a refusal that reads like a
corrupt download.

| Document | Tier |
|---|---|
| `studies/icassp_exact_convex/fig1_depth.toml` | `publication_b16k` |
| `studies/icassp_exact_convex/fig2_angle_blind.toml` | `publication_b16k` |
| `studies/icassp_exact_convex/fig2_angle_told.toml` | `publication_b16k` |
| `studies/icassp_exact_convex/fig2_angle_world.toml` | `publication_b16k` |
| `studies/icassp_exact_convex/fig5_stress_depth.toml` | `publication` |

## Rung 1 — redraw the paper's figures

Nothing to download. The figure artifacts ship with the repository.

```bash
uv run mbl --store paper_artifacts figure rebuild fig1_cost_vs_depth_exact --style ieee-paper
uv run mbl --store paper_artifacts figure rebuild fig5_stress_cost_vs_depth --style ieee-paper
uv run python tools/render_severity_figure.py --store paper_artifacts \
    --panel-suffix _exact --figure-id fig2_mismatch_severity_exact \
    --rename cocp=cocp_exact --style ieee-paper
```

Figure 2 is assembled by a script rather than by `mbl figure`, because it is
composed of three studies and belongs to no single one of them. The script
computes nothing: every point it draws comes from a stored table.

**This is byte-exact.** Redrawing reproduces the shipped files exactly, so
`git status` is clean when it finishes. If it reports a modified figure,
something really did change.

**`--store` is a global option**: it goes before the subcommand, never after.
`mbl figure --store X ...` writes into `./store` instead.

## Rung 2 — recompute every number

Fetch the results bundle from the Releases page and extract it, then:

```bash
tar -xzf results-core.tar.gz
uv run mbl --store store store reindex          # rebuilds the local index
uv run mbl --store store store verify           # recomputes every identifier
```

`verify` recomputes each identifier from the record's own specification and
checks it against the name the record is filed under. This paper rests on
**575 models and 960 measurements — 1,535 identifiers in all**.

Then re-derive each figure's table:

```bash
uv run mbl --store store analyse studies/icassp_exact_convex/fig1_depth.toml        --tier publication_b16k
uv run mbl --store store analyse studies/icassp_exact_convex/fig2_angle_blind.toml  --tier publication_b16k
uv run mbl --store store analyse studies/icassp_exact_convex/fig2_angle_told.toml   --tier publication_b16k
uv run mbl --store store analyse studies/icassp_exact_convex/fig2_angle_world.toml  --tier publication_b16k
uv run mbl --store store analyse studies/icassp_exact_convex/fig5_stress_depth.toml --tier publication
```

Each writes a tidy table beside the study's records. The tables are derived
from the per-trajectory costs in the bundle, so what you get is a
recomputation rather than a copy of ours.

To compare against what the paper drew, the shipped `.data.parquet` beside each
figure holds exactly the plotted values.

## Rung 3 — re-measure from the published weights

The bundle carries the trained controllers, so their evaluation can be run
again without retraining anything:

```bash
uv run mbl --store store run studies/icassp_exact_convex/fig1_depth.toml --tier publication_b16k
```

Models already present are reused; only measurements are recomputed. Expect
about an hour for Figure 1 on one GPU, and longer for Figure 3, whose plant is
an order of magnitude larger.

## Rung 4 — retrain from the specifications

The same command against an empty store retrains everything it cannot find:

```bash
uv run mbl --store fresh-store run studies/icassp_exact_convex/fig1_depth.toml --tier publication_b16k
```

This is days of GPU time for the full set, and it is the only rung whose
results will not be bit-identical to ours — training is not reproducible across
different hardware, and the paper's claims are about aggregates over seeds
rather than about individual weights.

## Reading the notebook

```bash
uv run jupyter lab notebooks/icassp_exact_convex_reviewer.ipynb
```

It rebuilds every figure and table in the paper from the recorded results and
runs in seconds. It trains nothing. Against a store that cannot answer, each
cell hands you a refusal naming the command that would fill it, rather than a
traceback.

## Running something of your own

A study document is the whole interface. Copy one, change what you want, and
run it: a different plant, a different depth, a different contender set. The
identifiers are derived from the document, so a changed document resolves to
different records and cannot silently overwrite the paper's.
