"""Controller families and the lifecycle that produces them (Tier 2).

`base.py` declares what a controller is, `lifecycle.py` what a synthesizer is,
`registry.py` how a family is selected by name, and the subpackages hold the
families themselves — analytic, constrained, iterative, neural, open-loop and
unfolded.

**Deliberately empty of re-exports.** This file exists so the package is a
*regular* package rather than a namespace one, and for nothing else. Two
reasons, in order of importance:

1. `pkgutil.walk_packages` skips namespace packages silently. Measured before
   this file existed: it reported **0** modules under `mbl.models` and a
   repo-wide signature census of **53** where the true figure is 89 — a census
   that under-reports by a third while raising no error is worse than one that
   fails, and one of Stage 7's own surveys was misled by exactly this.
2. An `__init__` that imports is how this tree would acquire a cycle. Callers
   import the module they mean.
"""
