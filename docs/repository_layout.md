# What lives where

```
src/mbl/            the package — the whole pipeline
├── spec/           study documents: the grammar, the tiers, identity
├── core/           problems, costs, constraints, the numerical kernels
├── models/         the controllers — analytic, unfolded, recurrent, convex
│   └── constrained/  the exact box-QP, its certificate and its adjoint
├── engine/         training
├── runner/         executing a study into stored records
├── store/          content-addressed storage and its index
├── analysis/       measurements -> tidy tables
├── present/        tables -> figures, and the venue style profiles
└── replay/         the one import a notebook needs

studies/
├── icassp_exact_convex/   the paper's five study documents
└── icassp/                the frozen plants they run on
tests/              the suite, which needs neither data nor a GPU
tools/              scripts that render the figures and author the plants
notebooks/          the reviewer notebook
paper_artifacts/    the figure data the repository redraws from
docs/methods/       how the experiments were designed and what they cost
```

**The plants are not in the campaign's own directory**, and that is deliberate
rather than untidy. This paper re-derives only its convex contenders and reuses
every other trained controller from the work it extends. A controller's identity
is derived in part from its plant, so identical controllers require identical
plants — the same frozen files, not equivalent ones. Moving them would move
every identifier that depends on them.

## The three levels of identity

Understanding these makes the rest of the repository obvious, and they are the
reason a result can be checked at all.

A **ProblemID** names a plant — the matrices, the horizon, the constraint.
Frozen into a `.npz` file and shipped, rather than regenerated, because two
machines do not agree bit-for-bit on a random draw.

A **ModelID** names a trained controller: its problem, its family, its
configuration, its training specification, its seed. It contains **no
evaluation term**, so the same model can be scored many ways without being
retrained.

A **MeasurementID** names one evaluation of one model under one protocol.

Records are stored under these names, so the same specification run twice
resolves to the same record rather than producing a second one. That is what
makes the reuse above checkable rather than asserted: a controller shared with
earlier work is the same record under the same name, and if any part of its
specification had drifted it would resolve elsewhere and be retrained.

## Where a figure comes from

```
study document  --run-->      models + measurements
                --analyse-->  a tidy table
                --figure-->   a .spec.json + .data.parquet pair, and a PDF
```

The last step stores **data plus a specification**, never a pickled figure: a
pickle does not survive a library upgrade and cannot be restyled, while the
pair can be redrawn at any venue's geometry. That is why rung 1 needs no store.

Nothing in a rendered artifact records the time it was made or the version that
made it, so redrawing a figure reproduces the shipped bytes exactly.

## A figure composed of several studies

Figure 2 has no study of its own: it is three mismatch conditions drawn as one
panelled figure. It is filed under an identifier **derived from the three
studies it composes**, in the same layout as every other result, so re-running
any one of them moves it and a figure built from a different set cannot quietly
take its place. The script that assembles it computes nothing — every drawn
point comes from one of the three stored tables.

## Adding a contender

A controller family is registered by name and referenced from a study document
by that name. `src/mbl/models/` holds the implementations and
`src/mbl/applications/recipes/` binds them to the specification grammar; the
registry maps one to the other. A new family is a module, a registration, and
a study document that names it — no change to the runner, the store or the
analysis.
