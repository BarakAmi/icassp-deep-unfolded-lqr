# What lives where

```
src/mbl/            the package — the whole pipeline
├── spec/           study documents: the grammar, the tiers, identity
├── core/           problems, costs, constraints, the numerical kernels
├── models/         the controllers — analytic, unfolded, recurrent, convex
├── engine/         training
├── runner/         executing a study into stored records
├── store/          content-addressed storage and its index
├── analysis/       measurements -> tidy tables
├── present/        tables -> figures, and the venue style profiles
└── replay/         the one import a notebook needs

studies/icassp/     the paper's five study documents, twelve supplementary
                    ones, and the frozen plants they run on
tests/              the suite, which needs neither data nor a GPU
tools/              scripts that render the figures and author the plants
notebooks/          the campaign notebook
paper_artifacts/    the figure data the repository redraws from
docs/methods/       how the experiments were designed and what they cost
```

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
resolves to the same record rather than producing a second one. That is why a
study's records can be scattered across the campaign that first computed them
— 135 of Figure 1's 225 measurements were first written by a study with a
different name — and why the manifest that pins this paper resolves its
contents from the specifications rather than matching on names.

## Where a figure comes from

```
study document  --run-->  models + measurements
                --analyse-->  a tidy table
                --figure-->  a .spec.json + .data.parquet pair, and a PDF
```

The last step stores **data plus a specification**, never a pickled figure: a
pickle does not survive a library upgrade and cannot be restyled, while the
pair can be redrawn at any venue's geometry. That is what `mbl figure rebuild`
does, and it is why rung 1 needs no store.

## Adding a contender

A controller family is registered by name and referenced from a study document
by that name. `src/mbl/models/` holds the implementations and
`src/mbl/applications/recipes/` binds them to the specification grammar; the
registry maps one to the other. A new family is a module, a registration, and
a study document that names it — no change to the runner, the store or the
analysis.
