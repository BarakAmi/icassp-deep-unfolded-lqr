"""The failures a notebook is expected to hit, and how they report themselves."""

from __future__ import annotations


class UnknownArtifactError(Exception):
    """A notebook asked for an analysis or figure the study does not declare.

    A `KeyError` here would be technically correct and useless: it names the
    key that was missing and nothing about the alternatives, and a notebook
    author has no argv to experiment with. This names what does exist, which
    turns a mistyped id into a one-line fix.
    """


class StoreIncompleteError(Exception):
    """The store does not yet hold what the study declares.

    Not a `SpecificationError`: nothing is wrong with the document. The store
    is simply behind it, which is the *normal* state of a notebook opened
    before its study has been run, and the remedy is a command rather than an
    edit. Keeping the two exception types apart is what lets a notebook catch
    "not produced yet" without also swallowing "this document is malformed" —
    the trap of asserting an exception type that something downstream also
    raises, recorded four times in this project's verification log.

    The message is the deliverable. It names what is missing and the exact
    command that produces it, because a notebook is the one surface with no
    argv to correct.
    """


class GateFailedError(Exception):
    """A gate the study declares was evaluated and did not hold.

    Distinct from `StoreIncompleteError` on purpose, and the distinction is the
    one a reader acts on: an incomplete store is fixed by running a command,
    while a failed gate is a statement about the *science* and is fixed by
    changing the study or accepting the finding. Annex 04 §4 requires that a
    failed gate stop the notebook, which is why this is raised from `resolve`
    rather than reported — a section that renders a failure without stopping is
    the decoration this whole step exists to end.
    """
