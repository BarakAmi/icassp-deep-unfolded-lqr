"""The paper's Figure 1 declares no series as a bound (Phase A6).

`cocp_lower_bound` was declared `role = "bound"`, drawn, and printed in the
companion table as "SDP-frozen policy | 8.2725" -- while this repository's own
commits say the SDP floor "does not bound the quantity the figures draw" and
that "every use of it was the error". The relaxation bounds the infinite-horizon
AVERAGE cost; the figure draws a finite-horizon quantity. The display name had
been corrected; the machine-readable role and the key had not, so a reviewer
reading the shipped specification would have seen a lower bound the methodology
denies.

Two facts make the repair safe, and both are asserted below rather than
believed. `role` is excluded from every signature, so re-classifying the series
moves no identifier -- the StudyID is pinned here precisely because a change to
it would mean 225 models had been orphaned by a presentational edit. And the
`label` is the store's join key for five models, so it does NOT change however
misleading the word "lower_bound" inside it now reads: a name that is wrong is
cheaper to explain than five records that can no longer be found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mbl.replay import load_study
from mbl.spec.contender import Role

#: The paper's Figure 1, and the identity it must keep. A failure here is not a
#: style question: it says a change orphaned the 225 models behind the figure.
FIGURE_ONE = ("icassp/fig1_depth", "publication_b16k")
FIGURE_ONE_STUDY_ID = "5d4b628b96b2a3f6"

#: Era-05 study documents -- the ones the artifact repository publishes.
ERA_05 = Path(__file__).resolve().parents[2] / "studies" / "icassp"


class TestFigureOne:
    def test_the_frozen_policy_is_a_baseline_and_not_a_bound(self) -> None:
        resolved = load_study(*FIGURE_ONE[:1], tier=FIGURE_ONE[1])
        roles = {
            contender.label: contender.role for contender in resolved.study.contenders
        }
        assert "cocp_lower_bound" in roles, (
            "the series is gone; if that was deliberate, delete this test with "
            "the contender rather than loosening it"
        )
        assert roles["cocp_lower_bound"] is Role.BASELINE, (
            f"it is {roles['cocp_lower_bound']}; the SDP relaxation bounds the "
            "infinite-horizon average cost and this figure draws a "
            "finite-horizon quantity, so nothing here is a bound"
        )

    def test_the_label_still_joins_to_the_stored_models(self) -> None:
        """The name is wrong and stays. Renaming it orphans five records."""
        resolved = load_study(*FIGURE_ONE[:1], tier=FIGURE_ONE[1])
        labels = {contender.label for contender in resolved.study.contenders}
        assert "cocp_lower_bound" in labels

    def test_re_classifying_moved_no_identifier(self) -> None:
        """The identity-and-provenance checkpoint, kept as a test.

        Measured when the role was changed: the StudyID before and after was
        the same string. If this ever fails, a presentational edit has become a
        retrain of 225 models, and the fix is to revert it -- not to re-baseline
        the literal.
        """
        resolved = load_study(*FIGURE_ONE[:1], tier=FIGURE_ONE[1])
        assert resolved.study.study_id == FIGURE_ONE_STUDY_ID


class TestNoEra05DocumentDeclaresABound:
    @pytest.mark.parametrize("document", sorted(p.name for p in ERA_05.glob("*.toml")))
    def test_it_declares_no_bound_role(self, document: str) -> None:
        text = (ERA_05 / document).read_text()
        assert 'role = "bound"' not in text, (
            f"{document} declares a series as a bound. The only bound this "
            "campaign has is the one era 06 built against the convention the "
            "figures evaluate, and it is not shipped here."
        )

    def test_there_are_documents_to_check(self) -> None:
        """Anti-vacuity: a glob that stops matching turns the gate green."""
        assert len(list(ERA_05.glob("*.toml"))) >= 10
