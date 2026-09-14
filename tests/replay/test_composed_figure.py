"""A figure composed of several studies is still a figure a reviewer can see.

`render_figure` looks a figure up among the ones its study declares, so a figure
assembled from several studies is unreachable through it -- it is declared by
none of them. The exact-convex paper's Figure 2 is exactly that: three mismatch
conditions drawn as one panelled figure. Without this call its reviewer notebook
showed only the tables, which left a printed figure nobody could check.

**The address is derived, never typed.** A composed figure is filed under an
identifier computed from the studies it composes, so passing the resolved parts
means the figure found is the one built from exactly those studies at exactly
those tiers. A literal identifier in a notebook would go stale the first time
any condition was re-run, and would then display a different figure without
complaining -- which is worse than refusing.

These need the real store: a composed figure is written by a separate tool over
a whole campaign, and building one in a temporary directory would be
reconstructing the campaign. `store/` is untracked, so on CI there is nothing to
address and the file skips rather than pretending.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mbl.replay import (
    UnknownArtifactError,
    load_study,
    render_composed_figure,
    resolve,
)
from mbl.store.ids import composite_study_id

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("blind", "told", "world")
FIGURE = "fig2_mismatch_severity_exact"
TIER = "publication_b16k"

pytestmark = pytest.mark.skipif(
    not (REPO / "store" / "models").is_dir(),
    reason=(
        "a composed figure is addressed inside the real store, which is "
        "untracked by design and absent on CI. The notebook's own suite covers "
        "the structural half without it."
    ),
)


@pytest.fixture(scope="module")
def parts() -> list:
    """The three conditions the paper's Figure 2 is composed of."""
    return [
        resolve(load_study(f"icassp_exact_convex/fig2_angle_{name}", tier=TIER))
        for name in CONDITIONS
    ]


def test_it_renders_the_figure_with_its_image_embedded(parts: list) -> None:
    """Embedded rather than linked: an executed notebook carries its figures."""
    rendered = render_composed_figure(parts, FIGURE, title="Severity")
    assert "data:image/png;base64," in rendered
    assert "**Figure — Severity.**" in rendered


def test_the_caption_counts_what_was_drawn_rather_than_asserting_it(
    parts: list,
) -> None:
    """The numbers come from the stored table, so they cannot drift from it."""
    import pandas as pd

    from mbl.replay.resolution import _figure_artifacts
    from mbl.store.location import default_store
    from mbl.store.study_artifacts import StudyArtifactStore

    study_id = str(
        composite_study_id(
            sorted(str(p.loaded.study.study_id) for p in parts), kind=FIGURE
        )
    )
    root = StudyArtifactStore(default_store()).figures_root(study_id)
    table = pd.read_parquet(_figure_artifacts(root, FIGURE)["data.parquet"])

    rendered = render_composed_figure(parts, FIGURE)
    assert f"{len(table)} plotted point(s)" in rendered
    assert f"{len(set(table['contender']))} contender(s)" in rendered


def test_the_order_of_the_parts_does_not_move_the_figure(parts: list) -> None:
    """The parts are a set; the caller should not have to know an order."""
    forward = render_composed_figure(parts, FIGURE)
    backward = render_composed_figure(list(reversed(parts)), FIGURE)
    assert forward == backward


def test_a_wrong_set_of_parts_refuses_rather_than_showing_something_else(
    parts: list,
) -> None:
    """The whole point of deriving the address.

    Two conditions compose to a different identifier, and no figure is filed
    there. Showing the three-part figure anyway would be showing a reader a
    picture that is not of what they asked for.
    """
    with pytest.raises(UnknownArtifactError) as refusal:
        render_composed_figure(parts[:2], FIGURE)
    assert "composed figure" in str(refusal.value)
    for part in parts[:2]:
        assert str(part.loaded.study.study_id) in str(refusal.value)


def test_no_parts_at_all_refuses(parts: list) -> None:
    """A composed figure of nothing is a caller error, not an empty figure."""
    with pytest.raises(UnknownArtifactError, match="composed of nothing"):
        render_composed_figure([], FIGURE)


def test_an_unknown_figure_id_refuses(parts: list) -> None:
    """Same parts, different figure: a different address, and nothing there."""
    with pytest.raises(UnknownArtifactError):
        render_composed_figure(parts, "fig2_not_a_figure")
