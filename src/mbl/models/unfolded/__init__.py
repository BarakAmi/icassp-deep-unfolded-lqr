"""Deep-unfolded controllers: a fixed number of projected-gradient iterations
with learnable per-iteration parameters.

`base.py` builds the unrolling, `parameters.py` holds the learnable step size
and matrix, `layerwise.py` the per-layer activation schedule, and
`iterative_refinement.py` the refinement pass.

Re-export-free by the same rule as `mbl.models` — see that package's docstring
for why the file exists at all.
"""
