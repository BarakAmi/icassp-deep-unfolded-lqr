# The `mbl` command line

**What this covers:** every command the `mbl` executable currently accepts, what each one does,
and where each one stops working. This is a `methods/` document — it describes what is
**implemented**, not what is planned. Commands the annexes specify but that do not yet exist are
listed once, in [§10](#10-what-does-not-exist-yet), so their absence is not mistaken for an
omission here.

**Every example is real output**, captured on 2026-08-03 from a store built by the commands
shown. Identifiers and timings will differ on your machine; the shapes will not.

> **Always `uv run`.** This project is managed by `uv`, and `mbl` is an entry point of the
> project's own environment. `uv run mbl …` resolves and activates that environment for the one
> command. A bare `mbl` either is not found or — worse — is a stale copy from some other
> environment. Note also that `uv run` needs the project: **`cd`-ing outside the repository
> breaks it**, reporting `Failed to spawn: mbl`.

---

## 1. The shape of the thing

Three verbs *produce*, four groups *inspect*.

```
uv run mbl run      <study.toml>     # document  -> models + measurements
uv run mbl analyse  <study.toml>     # measurements -> a tidy table
uv run mbl figure   render <study.toml>   # a table -> four figure artifacts

uv run mbl models        …           # what models the store holds
uv run mbl measurements  …           # what evaluations it holds
uv run mbl store         …           # size, integrity, index, collection
uv run mbl figure rebuild …          # re-style a stored figure, reading no model
```

The whole pipeline for a study, start to finish:

```bash
uv run mbl run     studies/box_lqr/depth_scaling.toml --tier standard
uv run mbl analyse studies/box_lqr/depth_scaling.toml --tier standard
uv run mbl figure render studies/box_lqr/depth_scaling.toml --tier standard
```

which leaves, under the study's own identifier:

```
<store>/studies/<StudyID>/analyses/cost_by_depth.{parquet,json}
<store>/studies/<StudyID>/figures/fig_cost_vs_depth.{pdf,png,data.parquet,spec.json}
```

### 1.1 Where the store lives

`--store` is a **global** option and belongs *before* the subcommand. This is the single most
common mistake with this CLI:

```bash
uv run mbl --store /data/mbl models list      # correct
uv run mbl models list --store /data/mbl      # error: unrecognized arguments
```

Resolution order: `--store`, else `$MBL_STORE`, else the `store/` of **the project the working
directory is inside**, else the one belonging to the repository the package was imported from. The
project is found by walking up for `pyproject.toml`. When the directory does not exist yet, that is
where a first run creates it — at the project root, not beside wherever you happened to be
standing.

The search exists because `./store` alone is wrong for every caller not standing at the project
root, and wrong *silently*: a store at the wrong path is **absent**, and absence reads as "not
produced yet". A notebook reported a complete 135-point study as `0/135 measured` for exactly this
reason — a Jupyter kernel's working directory is the notebook's own.

The anchor is the project marker and **not a directory named `store`**, which is a correction
rather than a refinement. This repository contains six directories called `store` or `studies`
that are not either of them — two Python packages of each — and, decisively, **every store
contains a `studies/` of its own**, since artifacts are filed under `store/studies/<StudyID>/`. A
name search run from inside any store resolves into that store.

Exporting the variable is still the clearest thing to do when several projects are in play:

```bash
export MBL_STORE=/data/mbl-store
uv run mbl models list
```

**Only `run` creates a store.** Every other command refuses a path that does not exist, so a
mistyped `--store` is reported rather than silently answered from a new empty store. That is a
flag on the command (`creates_store`) and not a check on its name, so a second writing command
cannot forget it.

> `./store` is in `.gitignore`, anchored as `/store/` — an unanchored `store/` would also match
> `src/mbl/store/` and `tests/store/`.

### 1.2 Exit codes

`0` on success, `1` on any specification failure, missing store, missing measurement or failed
gate. There is no third code. **Never read the status through a pipe** — `uv run mbl … | tail -2;
echo $?` reports `tail`'s status, not `mbl`'s.

---

## 2. `mbl run` — a study document, executed

```
uv run mbl run <study.toml> [--tier NAME] [--catalogue PATH] [--set PATH=VALUE] [--dry-run]
```

Loads the document, resolves it at a tier, materialises its points, and produces every model and
measurement the store does not already hold.

| Option | Default | Meaning |
|---|---|---|
| `--tier NAME` | `standard` | `smoke`, `standard`, `publication`, `comprehensive` |
| `--catalogue PATH` | the shipped `studies/_tiers.toml` | an alternative tier catalogue |
| `--set PATH=VALUE` | — | per-invocation override; repeatable; **`VALUE` is TOML** |
| `--dry-run` | off | report what would be trained versus reused, and stop |

```console
$ uv run mbl run studies/box_lqr/depth_scaling.toml --tier standard
box_lqr/depth_scaling at standard: 27 point(s), trained 27, reused 0

$ uv run mbl run studies/box_lqr/depth_scaling.toml --tier standard
box_lqr/depth_scaling at standard: 27 point(s), trained 0, reused 27
```

The second line is the point of the whole design: **a run is idempotent**, because a model is
identified by what it *is* rather than by when it was made. Re-running costs nothing and the
store is byte-identical afterwards.

### 2.1 `--dry-run` is instant, and that is a property rather than luck

```console
$ uv run mbl run studies/box_lqr/depth_scaling.toml --tier standard --dry-run
box_lqr/depth_scaling at standard: 27 point(s), would train 27, reused 0
```

The counts come from the identifiers `materialise()` derives and from `ModelStore.exists` —
**nothing is constructed**. Recipe construction is free (0.00 s), but `build_controller()` runs
the COCP solver canary and costs ~1.4 s per contender, so a preview that reached it would be a
different kind of operation. It does not.

### 2.2 Tiers scale effort and never content

| tier | seeds | epochs | batch | eval batches | NB04 points | NB04 wall clock |
|---|---|---|---|---|---|---|
| `smoke` | 1 | 5 | 128 | 1 | **9** (axis subsetted) | seconds |
| `standard` | 1 | 50 | 2,048 | 4 | 27 | **244 s** |
| `publication` | 5 | 200 | 8,192 | 8 × 4,096 | 135 | **3.43 h** |
| `comprehensive` | 10 | 400 | 8,192 | 16 × 4,096 | 270 | ≤ 13.7 h |

A tier may write only to a closed whitelist of *effort* knobs — seeds, epochs, batch size,
evaluation batch counts. It can never touch `problem.*`, a contender's `family`, sweep values or
gates, and attempting it fails when the catalogue is read rather than when the run reaches it.

**`smoke` is the one tier allowed to truncate a swept axis** (9 points instead of 27), and every
measurement it produces is stamped `axis_subset=true` and **refused by `mbl analyse`**. A
plumbing check cannot become a figure.

### 2.3 `--set` — the third editing level, and the trap in it

Values are parsed as **TOML**, so a string needs quoting inside the shell's quoting:

```bash
uv run mbl run studies/box_lqr/depth_scaling.toml --tier standard \
  --set training.seeds=3 \
  --set 'contenders.*.config.plan.epochs=20' \
  --set 'training.batch.effective_size=512'
```

`--set training.seeds=2` must reach the grammar as the integer `2`, not the string `"2"`; parsing
through `tomllib` makes the command line agree with the document surface by construction.

> ### ⚠ `--set` participates in identity — give `analyse` and `figure` the *same* overrides
>
> An override changes the effective specification, therefore the `ModelID`, therefore the
> `MeasurementID` and the `StudyID`. `mbl analyse` without the overrides `mbl run` was given looks
> for measurements that were never produced:
>
> ```console
> $ uv run mbl run     studies/box_lqr/depth_scaling.toml --tier standard --set training.batch.effective_size=128
> box_lqr/depth_scaling at standard: 27 point(s), trained 27, reused 0
>
> $ uv run mbl analyse studies/box_lqr/depth_scaling.toml --tier standard
> study 'box_lqr/depth_scaling' contender 'truncated_riccati' at seed 0 has no measurement
> cf58cfbad9fa4fe4 in the store; an analysis reads what was produced and never produces it
> -- run the study first
> ```
>
> This is correct behaviour, not a defect: the two commands were asked about two different
> studies. Put the overrides in a shell variable and pass it to all three commands, or — better
> for anything you intend to keep — put them in the document's own `[tier_overrides.<tier>]`
> table, where they are tracked and reviewable.

---

## 3. `mbl analyse` — measurements to a tidy table

```
uv run mbl analyse <study.toml> [--tier NAME] [--catalogue PATH] [--set PATH=VALUE] [--only ID]
```

Computes every analysis the study declares in its `[[analyses]]` tables and writes each as a
`.parquet` plus a JSON sidecar under `<store>/studies/<StudyID>/analyses/`.

```console
$ uv run mbl analyse studies/box_lqr/depth_scaling.toml --tier standard
box_lqr/depth_scaling at tier standard:
  cost_by_depth (cost_vs_axis): 27 rows
```

Three properties worth relying on:

- **It reads the store, never a run.** An analysis does not care whether the measurements were
  produced a second ago or a month ago, and re-running it costs no training.
- **It never reads a model.** Delete `<store>/models/` entirely and the table comes back
  byte-identical. That is what makes the store's expensive half disposable.
- **It replaces in place.** Re-running overwrites the table atomically rather than versioning it,
  because an analysis is meant to be recomputed. (This is why a figure keeps its *own copy* of the
  data — see §4.)

`--only ID` computes one declared analysis instead of all of them, and is repeatable. Naming an
analysis the study does not declare is an error, not a no-op.

### 3.1 What the sidecar declares

The `.json` beside the table records the statistics so a reader never has to infer them:

```
  aggregate: mean
  aggregation_order: ['evaluation_trajectories', 'training_seeds']
  interval: {'kind': None, 'level': None, 'over': 'training_seeds',
             'reason': 'fewer than two training seeds, so the across-seed interval §A.3
                        reports does not exist; the within-seed dispersion is reported
                        separately as `within_seed_spread` and is not a confidence interval'}
  n_evaluation_trajectories: 27648
  n_training_seeds: [1]
  within_seed_spread_kind: std
```

**At `smoke` and `standard` there are no error bars, and that is deliberate.** Both tiers use one
training seed, so the across-seed interval does not exist; the analysis emits a null interval with
a stated reason rather than substituting the across-trajectory spread, which is always available,
always narrower, and would produce a plausible and wrong figure that nothing downstream could
detect. Error bars appear at `publication`, which is the first tier with five seeds.

---

## 4. `mbl figure` — a table to four artifacts

### 4.1 `render`

```
uv run mbl figure render <study.toml> [--tier NAME] [--catalogue PATH] [--set PATH=VALUE]
                                      [--only ID] [--style PROFILE]
```

```console
$ uv run mbl figure render studies/box_lqr/depth_scaling.toml --tier standard
box_lqr/depth_scaling at style thesis:
  fig_cost_vs_depth (axis_scaling) -> <store>/studies/<StudyID>/figures/fig_cost_vs_depth.pdf
```

Each declared figure produces **four** files and never a pickle:

| file | what it is |
|---|---|
| `<id>.pdf` | vector, Type 42 fonts (Type 3 is rejected by IEEE and ACM checkers) |
| `<id>.png` | raster at 600 dpi |
| `<id>.data.parquet` | **a copy** of the analysis table |
| `<id>.spec.json` | the declaration that produced it |

The data file is a *copy* rather than a reference on purpose: an analysis replaces in place, so a
figure that pointed at the live table would silently change what it claims to have plotted the
next time the analysis ran.

### 4.2 `rebuild` — re-style without re-running anything

```
uv run mbl figure rebuild <figure-id> [--style PROFILE]
```

```console
$ uv run mbl figure rebuild fig_cost_vs_depth --style ieee-2col
fig_cost_vs_depth at style ieee-2col:
  fig_cost_vs_depth (axis_scaling) -> <store>/studies/<StudyID>/figures/fig_cost_vs_depth.pdf
```

This reads **only** the figure's own `.data.parquet` and `.spec.json`. It has been verified with
both `models/` and `analyses/` deleted from the store — which is the property that makes the
data-plus-spec pair strictly more capable than a saved figure object.

### 4.3 Style profiles

`thesis` (default), `ieee-1col`, `ieee-2col`, `neurips`, `talk`. A profile owns the physical
presentation — width, font family and size, dpi, line weights. A figure declaration owns what the
figure is *about* — its source table, its labels, its title — and **may not** carry a width or a
font size. That separation is exactly what makes `rebuild --style` a re-style rather than a
contradiction.

Every figure must also pass the **greyscale-separability gate**: two series must differ in
linestyle or marker, or else in WCAG relative luminance by at least 0.15. Colour alone is not a
distinction.

---

## 5. `mbl models` — what the store holds

### 5.1 `list`

```
uv run mbl models list [--problem P] [--family F] [--contender C] [--seed S] [--since DATE]
                       [--sort {contender,created,family,name,problem,seed,vram,wall}]
```

```console
$ uv run mbl models list
identifier  seed  created              name
----------  ----  -------------------  --------------------------------------------------------------
24226a64    0     2026-08-02T22:16:50  box_lqr-n4m2-N50-u0.5-s0/cocp/adam-lr0.1-ep2-b128-seed0#24226a64
8b7ec911    0     2026-08-02T22:16:51  box_lqr-n4m2-N50-u0.5-s0/cocp_lower_bound/seed0#8b7ec911
c91dfa8c    0     2026-08-02T22:16:48  box_lqr-n4m2-N50-u0.5-s0/truncated_riccati/seed0#c91dfa8c
e592e3bd    0     2026-08-02T22:16:49  box_lqr-n4m2-N50-u0.5-s0/unfolded-learned_step_size-J1/adam-lr1e-2-ep2-b128-seed0#e592e3bd
```

The **semantic name** is derived from the specification, not stored as a label:
`<problem>/<contender>-<variant>-J<depth>/<optimizer>-lr<rate>-ep<epochs>-b<batch>-seed<n>#<id>`.
Identifiers are abbreviated git-style to the shortest width that is still unique in this store, so
the column widens as the store grows — an earlier fixed 8-character truncation printed duplicates
that resolved to neither model.

### 5.2 `tree`

```console
$ uv run mbl models tree
box_lqr-n4m2-N50-u0.5-s0                                    (6 contenders, 27 models, 470 KB)
├── unfolded_warmstart-learned_step_size_and_matrix J=1    adam-lr1e-2-b128     seeds 0    ✓ 1/1
│   └── measurements: 1 nominal, 0 shifted
├── unfolded_warmstart-learned_step_size_and_matrix J=10   adam-lr1e-2-b128     seeds 0    ✓ 1/1
│   └── measurements: 1 nominal, 0 shifted
```

Grouped problem → contender → seed. `✓ 1/1` is seeds present over seeds expected. **`shifted`
counts measurements whose evaluation problem differs from the training problem** — distribution
shift is ordinary evaluation here, not a subsystem, so an out-of-distribution result is one more
measurement of the same model.

### 5.3 `show` and `diff`

```console
$ uv run mbl models show 24226a64
box_lqr-n4m2-N50-u0.5-s0/cocp/adam-lr0.1-ep2-b128-seed0#24226a64

  model         24226a64f45b530d
  problem       eea2ed781a200b41
  family        cocp
  seed          0
  dimensions    n=4 m=2 N=50
  created       2026-08-02T22:16:50+00:00
  stamp         model-based-learning-for-stochastic-control-0.1.0/schema-1
  wall time     0.9 s
  peak VRAM     —
  measurements  1 nominal, 0 shifted

specification
  contender.family                                   cocp
  contender.plan.epochs                              2
  …
```

`reference` accepts a full identifier, a semantic name, or **a unique prefix of either**.
`--json` gives the same content machine-readably.

```console
$ uv run mbl models diff e592e3bd 24226a64
field                     e592e3bd            24226a64
------------------------  ------------------  --------
contender.family          unfolded            cocp
contender.horizon         50                  None
contender.kind            learned_step_size   —
```

`diff` prints only the fields that differ, with `—` for absent. It is the fastest answer to *why
did this retrain?*

> `peak VRAM —` is not a gap in the record. NB04 declares `[compute] device = "cpu"`, and
> `compute.*` is not tier-writable, so the study genuinely uses no accelerator memory.

---

## 6. `mbl measurements`

```
uv run mbl measurements list [--model MODEL] [--shifted]
```

```console
$ uv run mbl measurements list
identifier  model     eval problem  shift    eval_expected_cost  eval_expected_cost_std
----------  --------  ------------  -------  ------------------  ----------------------
1e4e9c15    4f346191  eea2ed78      nominal  2.10184             0.0277142
2687d151    97c6aa60  eea2ed78      nominal  2.68635             0.0526376
```

`--shifted` restricts to measurements whose evaluation problem differs from the training problem.
Each measurement also stores one cost **per trajectory** in `samples.parquet`, keyed
`(batch_index, trajectory_index)` — the key is within-batch because the evaluation batches are
common random numbers shared by every contender, which is what makes a paired comparison paired.

### 6.1 What a record keeps about how it came to be

Four members hold the run's own diagnostics (Annex 02 §2.2). No command displays them and no
notebook section renders them; they are there to be opened when a number looks wrong.

| where | member | one row / unit is |
|---|---|---|
| `store/models/<ModelID>/` | `training.parquet` | one epoch: `epoch`, `phase`, `wall_time_s`, and the strategy's metrics |
| | `synthesis.log` | the offline phase's captured log, at `INFO` |
| `store/measurements/<MeasurementID>/` | `evaluation.parquet` | one evaluation batch: `batch_index`, `batch_cost`, `wall_time_s` |
| | `evaluation.log` | the online phase's captured log |

```python
import pandas as pd
pd.read_parquet("store/models/4f346191ab63d7c2/training.parquet")
```

Two things to know before you go looking.

**A record written before 2026-08-04 has none of them**, including every record produced by the
NB04 publication run. They are optional on read — always, by design: a payload that a record could
be *missing* would make the 162 existing records read as incomplete, and incomplete means the next
run retrains them.

**A family with no training phase has no training curve.** `truncated_riccati`, `standard_pgd` and
`cocp_lower_bound` never build an engine, so their records carry no `training.parquet` at all —
absence here means "nothing was trained", not "the recording failed".

The online wall-clock also reaches the notebook's provenance section, which reported *online
compute: not recorded* until these existed.

---

## 7. `mbl store` — size, integrity, index, collection

### 7.1 `stat`

```console
$ uv run mbl store stat
class         count  size
------------  -----  ------
models        27     74 KB
measurements  27     397 KB
index         1      127 KB

total  878 KB  (/data/mbl-store)

growth
month    models  measurements
-------  ------  ------------
2026-08  27      27
```

### 7.2 `verify`

```console
$ uv run mbl store verify
54 identifiers recomputed and matched; 0 unverifiable
```

Re-derives every identifier from the stored specification and compares it with the name the record
is filed under. **`0 unverifiable` is the number to watch.** It was 100 % across the retired legacy
archive — nothing in it could pin its own identity — and it is 0 for every record the producer
writes. A non-zero count means something entered the store that cannot prove what it is.

### 7.3 `reindex`

Rebuilds the SQLite index by walking the record trees. The index is a cache; the trees are the
truth. Safe at any time, and the queue table is preserved.

### 7.4 `gc`

```
uv run mbl store gc [--keep-studies S …] [--models] [--partials] [--older-than AGE]
                    [--dry-run] [--yes]
```

Drops what no live study references. Measurements only, unless `--models` is given —
**a model is the one expensive object in the store**, so collecting it is opt-in.

```console
$ uv run mbl store gc --dry-run
no live study references anything, so every measurement in the store would be collectable.
This is refused: studies arrive with Tier 5, so an empty study table means 'nothing can say
yet what is referenced', not 'nothing is referenced'.
Use --partials to collect abandoned resume artifacts, which belong to no study.
```

Note the refusal: an empty study table is treated as *ignorance*, not as *emptiness*. Always
`--dry-run` first; `--yes` skips the confirmation and is for scripts.

---

## 8. Recipes

**Check the cost before committing the machine.**

```bash
uv run mbl run studies/box_lqr/depth_scaling.toml --tier publication --dry-run
```

**Explore cheaply, then scale — the cheap models are never reused, by design.**

```bash
uv run mbl run studies/box_lqr/depth_scaling.toml --tier smoke        # plumbing, seconds
uv run mbl run studies/box_lqr/depth_scaling.toml --tier standard     # research, ~4 min
uv run mbl run studies/box_lqr/depth_scaling.toml --tier publication  # the paper, ~3.4 h
```

`smoke` cuts epochs and the batch, so its models differ in effective fields and `standard` will
not silently reuse them.

**Re-analyse and re-render without re-running.** The two are seconds; the run is hours.

```bash
uv run mbl analyse       studies/box_lqr/depth_scaling.toml --tier publication
uv run mbl figure render studies/box_lqr/depth_scaling.toml --tier publication
```

**Produce a camera-ready variant of a figure you already have.**

```bash
uv run mbl figure rebuild fig_cost_vs_depth --style ieee-2col
```

**Reclaim space without touching the models.**

```bash
uv run mbl store gc --dry-run
uv run mbl store gc --partials --yes
```

**Answer "why did this retrain?"**

```bash
uv run mbl models list --family unfolded --sort created
uv run mbl models diff <old> <new>
```

---

## 9. Failure modes you will actually hit

| Symptom | Cause | Fix |
|---|---|---|
| `unrecognized arguments: --store …` | `--store` after the subcommand | it is global: `uv run mbl --store P run …` |
| `Failed to spawn: mbl` | the working directory is outside the repository | `uv run` needs the project; `cd` back |
| `has no measurement <id> in the store` | `analyse` given different `--set`/`--tier` than `run` | pass the same overrides, or move them into `[tier_overrides]` |
| `…has not produced <analysis>; run \`mbl analyse\` first` | rendering before analysing | run `analyse` first |
| `this measurement is stamped axis_subset=true (tier 'smoke')` | analysing a plumbing run | re-run at a tier that does not subset |
| an exit code that looks wrong | the status came through a pipe | never `mbl … \| tail`; check `$?` directly |
| a store appears where you did not expect one | `run` without `--store`, with no store anywhere above the working directory | set `$MBL_STORE`, or run from the project root |
| a command reports a study as unmeasured that another command says is complete | the two resolved **different stores** | every refusal now prints the store it consulted — compare the two |

---

## 10. What does not exist yet

Specified in the annexes, not implemented, and deliberately not stubbed:

| Surface | Where it is specified | Why it is not here |
|---|---|---|
| `mbl queue add/list/…` | Annex 06 §5 | Justified by multi-day unattended batches. Measured: `publication` is 3.43 h, `comprehensive` ≤ 13.7 h — the latter earns it, and it is next |
| `--jobs`, `--jobs-gpu`, `--jobs-cpu` | Annex 06 §3 | Parallelism is worth 1.84×/2.42×/3.10× at 2/4/8 processes here — **but only with the per-process thread budget divided**; without it, four jobs are 232× *slower*. The flag and the budget must ship together |
| `--nice`, `--vram-fraction`, `.pause` | Annex 06 §6 | Fencing belongs with the queue |
| `mbl run --profile-only` | Annex 03 §A.5 | The dedicated online timing pass with warm-up does not exist; today's online numbers come from the evaluation pass |
| epoch-granular resume | Annex 06 §4.2 | The longest single node measured is 23.8 min, and the study level already resumes |

---

## See also

- Annex 01 — the experiment grammar: what a study
  document may say, and the three editing levels `--tier`/overlay/`--set` implement.
- Annex 02 — the result store: the layout,
  the index, semantic naming, and what `verify` and `gc` are checking.
- Annex 03 — analysis and visual standard:
  the analysis registry, the four-artifact figure contract, the greyscale gate.
- `tools/probes/`: the end-to-end slice check, which runs these
  commands and then disbelieves the result.
