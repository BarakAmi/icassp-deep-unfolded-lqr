"""Cost functionals (Tier 1).

`cost.py` declares the `Cost` ABC, `quadratic_cost.py` the linear-quadratic
instance every study in this project uses, and `quadratic_form.py` the reduction
kernels both share.

Re-export-free, and here the rule is stricter than convenience: `core`'s purity
is pinned by `tests/architecture/test_boundaries.py`, so an `__init__` that
imported anything would be the first place that pin could be quietly bent. See
`mbl.models` for why the file exists at all.
"""
