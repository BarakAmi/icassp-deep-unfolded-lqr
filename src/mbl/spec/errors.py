"""The one error the specification grammar raises.

Kept in its own module so every Tier-3 type can raise it without importing a
sibling: `contender.py` needs it and has no other reason to pull in `problem.py`
and, with it, NumPy and the whole `core` tree.

It subclasses `ValueError` because that is what a malformed value is, and
because callers written against the pre-grammar types already catch `ValueError`
there.
"""

from __future__ import annotations


class SpecificationError(ValueError):
    """A specification that cannot describe what it claims to.

    Raised at construction or at resolution rather than at solve time, and
    always naming the offending field: a shape mismatch discovered inside a
    solver, or a misspelled family discovered after an hour of training, is
    expensive to trace back to the declaration that caused it.
    """
