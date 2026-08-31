"""Acceptance tests for the `mbl` command line's rendering (Annex 02 §5).

The annex prints `mbl models tree`'s output **verbatim**, so that block is the
acceptance test: `test_annex_tree_example_is_reproduced_verbatim` asserts the
renderer reproduces it character for character, exactly as the semantic-name
worked example was treated in `tests/store/test_naming.py`.

One number in that block was wrong and is corrected here and in the annex: the
header claimed 20 models while the four rows beneath it sum to 5+5+5+1 = 16.
"""

from __future__ import annotations

import pytest

from mbl.cli.render import (
    DASH,
    abbreviate,
    ContenderGroup,
    ProblemGroup,
    format_seeds,
    format_size,
    group_models,
    render_tree,
)
from mbl.store.index import MeasurementRow, ModelRow

# --------------------------------------------------------------------------
# The annex's verbatim block.
# --------------------------------------------------------------------------

ANNEX_TREE = (
    "boxlqr-n7m3-N100-u0.1-s0                                    "
    "(4 contenders, 16 models, 61 MB)\n"
    "├── unfolded-aP        J=10   adam-lr1e-2-ep200   seeds 0-4  ✓ 5/5\n"
    "├── unfolded-a         J=10   adam-lr1e-2-ep200   seeds 0-4  ✓ 5/5\n"
    "├── pgd-fixed          J=10   —                   seeds 0-4  ✓ 5/5\n"
    "└── riccati-trunc      —      —                   —          ✓ 1/1\n"
    "    └── measurements: 47 nominal, 213 shifted"
)

PROBLEM = "boxlqr-n7m3-N100-u0.1-s0"
PROBLEM_ID = "0123456789abcdef"


def _digest(n: int) -> str:
    """A distinct, well-formed 16-hex identifier."""
    return f"{n:016x}"


def _model(index: int, contender: str, training: str, seed: int) -> ModelRow:
    return ModelRow(
        model_id=_digest(index),
        semantic_name=f"{PROBLEM}/{contender}/{training}#{_digest(index)[:6]}",
        problem_id=PROBLEM_ID,
        family=contender.split("-")[0],
        contender_id=contender,
        seed=seed,
    )


def _annex_models() -> list[ModelRow]:
    """The sixteen models the annex's tree describes.

    The four rows differ in exactly the way that makes each column meaningful:
    `unfolded-*` carry a full training descriptor, `pgd-fixed` carries only a
    seed (so its training column is empty but its seed column is not), and
    `riccati-trunc` carries no training segment at all.
    """
    rows: list[ModelRow] = []
    counter = 0
    for contender in ("unfolded-aP-J10", "unfolded-a-J10"):
        for seed in range(5):
            rows.append(
                _model(counter, contender, f"adam-lr1e-2-ep200-seed{seed}", seed)
            )
            counter += 1
    for seed in range(5):
        rows.append(_model(counter, "pgd-fixed-J10", f"seed{seed}", seed))
        counter += 1
    rows.append(_model(counter, "riccati-trunc", "-", 0))
    return rows


def _annex_measurements(model_id: str) -> list[MeasurementRow]:
    """47 nominal and 213 shifted measurements of the analytic baseline."""
    rows = []
    for n in range(47):
        rows.append(
            MeasurementRow(
                measurement_id=_digest(1_000 + n),
                model_id=model_id,
                eval_problem_id=PROBLEM_ID,
                is_shifted=False,
            )
        )
    for n in range(213):
        rows.append(
            MeasurementRow(
                measurement_id=_digest(2_000 + n),
                model_id=model_id,
                eval_problem_id="fedcba9876543210",
                is_shifted=True,
            )
        )
    return rows


def test_annex_tree_example_is_reproduced_verbatim() -> None:
    """The acceptance gate: Annex 02 §5's block, character for character."""
    models = _annex_models()
    measurements = _annex_measurements(models[-1].model_id)
    groups = group_models(models, measurements, sizes={PROBLEM_ID: 61_000_000})
    assert render_tree(groups) == ANNEX_TREE


def test_annex_tree_rows_are_all_sixty_six_columns_wide() -> None:
    """Alignment is the whole point of the fixed widths, so it is asserted
    independently of the verbatim comparison -- a single stray space would
    otherwise only show up as an opaque string mismatch."""
    body = render_tree(
        group_models(_annex_models(), (), sizes={PROBLEM_ID: 61_000_000})
    ).splitlines()[1:]
    assert [len(line) for line in body] == [66, 66, 66, 66]


# --------------------------------------------------------------------------
# Grouping.
# --------------------------------------------------------------------------


def test_depth_and_seed_are_lifted_out_of_the_name_segments() -> None:
    """`unfolded-aP-J10` renders as `unfolded-aP` plus a `J=10` column, and
    `adam-...-seed3` as `adam-...` plus a seed column."""
    (problem,) = group_models([_model(0, "unfolded-aP-J10", "adam-ep200-seed3", 3)])
    (contender,) = problem.contenders
    assert contender.contender == "unfolded-aP"
    assert contender.depth == 10
    assert contender.training == "adam-ep200"
    assert contender.seeds == (3,)
    assert contender.has_seed


@pytest.mark.parametrize(
    "training",
    ["adam-lr1e-2-ep200-seed0", "adam-lr0.01-ep200-seed0", "adam-lr1e-2-b8192-seed0"],
)
def test_a_learning_rate_hyphen_is_not_mistaken_for_a_field_separator(
    training: str,
) -> None:
    """`lr1e-2` spends a hyphen on its own exponent, so a segment holds no fixed
    number of hyphen-separated fields.

    The three cases differ in exactly that count -- five tokens, four, and five
    again with the fifth being a batch size -- so an implementation that lifts
    fields by counting from the right mangles at least one of them, while one
    that matches the `seed<n>` suffix handles all three. A single case cannot
    tell those apart, which is why this is parametrised.
    """
    expected = training.removesuffix("-seed0")
    (problem,) = group_models([_model(0, "unfolded-a-J4", training, 0)])
    assert problem.contenders[0].training == expected


def test_batch_size_stays_in_the_training_column() -> None:
    """Only the seed is lifted out of the training segment.

    Two models that differ solely in batch size are genuinely different models.
    Dropping the field -- as the annex's tree row appears at a glance to do --
    would merge them into one row reporting `1/1` twice over, so the store would
    look like it holds half of what it holds.
    """
    models = [
        _model(0, "unfolded-a-J4", "adam-ep200-b8192-seed0", 0),
        _model(1, "unfolded-a-J4", "adam-ep200-b512-seed0", 0),
    ]
    (problem,) = group_models(models)
    assert [c.training for c in problem.contenders] == [
        "adam-ep200-b8192",
        "adam-ep200-b512",
    ]
    assert problem.model_count == 2


def test_a_contender_with_no_training_segment_has_no_seed_dimension() -> None:
    """`riccati-trunc` is analytic: no training, and no meaningful seed."""
    (problem,) = group_models([_model(0, "riccati-trunc", "-", 0)])
    (contender,) = problem.contenders
    assert contender.training is None
    assert contender.depth is None
    assert not contender.has_seed


def test_a_seed_only_training_segment_keeps_its_seed_dimension() -> None:
    """`pgd-fixed` has no training configuration but is still seeded -- the
    distinction the annex's third and fourth rows turn on."""
    (problem,) = group_models(
        [_model(i, "pgd-fixed-J10", f"seed{i}", i) for i in range(3)]
    )
    (contender,) = problem.contenders
    assert contender.training is None
    assert contender.has_seed
    assert contender.seeds == (0, 1, 2)


def test_contenders_are_ordered_by_population_then_descending_name() -> None:
    """The order the annex's example shows: the most-populated contender first,
    and within a tie the descending name, which puts the elaborate variant
    before the plain one and the learned families before the baselines."""
    models = [_model(i, "aaa", f"seed{i}", i) for i in range(3)]
    models += [_model(10 + i, "zzz", f"seed{i}", i) for i in range(3)]
    models += [_model(20, "mmm", "seed0", 0)]
    (problem,) = group_models(models)
    assert [c.contender for c in problem.contenders] == ["zzz", "aaa", "mmm"]


def test_problems_are_ordered_by_name() -> None:
    rows = [
        ModelRow(_digest(1), "zeta/c/-#aaa", "1" * 16, "f", "c", 0),
        ModelRow(_digest(2), "alpha/c/-#bbb", "2" * 16, "f", "c", 0),
    ]
    assert [g.display for g in group_models(rows)] == ["alpha", "zeta"]


def test_two_problems_sharing_a_display_name_are_disambiguated() -> None:
    """Distinct problems can render identical descriptors -- two box-LQR
    instances of the same shape differing only in their matrices. Merging them
    under one header would report one problem's models as another's, so the
    identifier is appended when, and only when, a name collides."""
    rows = [
        ModelRow(_digest(1), "boxlqr-n2m1/c/-#aaa", "1" * 16, "f", "c", 0),
        ModelRow(_digest(2), "boxlqr-n2m1/c/-#bbb", "2" * 16, "f", "c", 0),
    ]
    groups = group_models(rows)
    assert [g.display for g in groups] == [
        "boxlqr-n2m1 #11111111",
        "boxlqr-n2m1 #22222222",
    ]


def test_a_model_with_no_semantic_name_still_appears() -> None:
    """`reindex` writes rows whose semantic name is empty, because projecting
    one needs the grammar. Such a model must still be listed -- silently
    dropping it would make the tree under-report the store."""
    row = ModelRow(_digest(7), "", "abcdef0123456789", "riccati", "riccati-trunc", 2)
    (problem,) = group_models([row])
    assert problem.display == "abcdef0123456789"
    assert problem.contenders[0].contender == "riccati-trunc"
    assert problem.contenders[0].seeds == (2,)


# --------------------------------------------------------------------------
# Completeness.
# --------------------------------------------------------------------------


def test_a_gap_in_the_seed_sequence_is_reported_incomplete() -> None:
    """The `k/n` column exists to catch exactly this: four models where the
    seed range says there should be five."""
    models = [_model(i, "unfolded-a-J4", f"adam-seed{i}", i) for i in (0, 1, 2, 4)]
    (problem,) = group_models(models)
    (contender,) = problem.contenders
    assert (contender.present, contender.expected) == (4, 5)
    assert not contender.complete
    assert "✗ 4/5" in render_tree([problem])


def test_a_complete_seed_sequence_is_reported_complete() -> None:
    models = [_model(i, "unfolded-a-J4", f"adam-seed{i}", i) for i in range(5)]
    (contender,) = group_models(models)[0].contenders
    assert (contender.present, contender.expected) == (5, 5)
    assert contender.complete


def test_a_seedless_contender_counts_its_models() -> None:
    """With no seed dimension there is no range to check, so completeness is
    the trivially satisfied `n/n` -- but the count must still be the models."""
    (contender,) = group_models([_model(0, "riccati-trunc", "-", 0)])[0].contenders
    assert (contender.present, contender.expected) == (1, 1)


# --------------------------------------------------------------------------
# Measurements.
# --------------------------------------------------------------------------


def test_a_contender_with_no_measurements_has_no_measurement_line() -> None:
    """Omitting the line when there is nothing to report is what lets the
    annex's example show it under one contender only."""
    rendered = render_tree(group_models(_annex_models(), ()))
    assert "measurements:" not in rendered


def test_a_measurement_line_under_a_non_final_contender_continues_the_trunk() -> None:
    """The annex only shows the line under the last contender, where the trunk
    has ended. Under any earlier one the vertical must continue, or the tree
    reads as though the branch closed."""
    models = [_model(i, "zzz", f"seed{i}", i) for i in range(3)]
    models += [_model(9, "aaa", "seed0", 0)]
    measurements = [
        MeasurementRow(_digest(500), models[0].model_id, PROBLEM_ID, is_shifted=False)
    ]
    lines = render_tree(group_models(models, measurements)).splitlines()
    assert lines[2] == "│   └── measurements: 1 nominal, 0 shifted"


def test_measurements_are_attributed_to_the_contender_of_their_model() -> None:
    models = [_model(0, "aaa", "seed0", 0), _model(1, "bbb", "seed0", 0)]
    measurements = [
        MeasurementRow(_digest(500), models[1].model_id, PROBLEM_ID, is_shifted=True)
    ]
    by_name = {c.contender: c for c in group_models(models, measurements)[0].contenders}
    assert (by_name["aaa"].nominal, by_name["aaa"].shifted) == (0, 0)
    assert (by_name["bbb"].nominal, by_name["bbb"].shifted) == (0, 1)


def test_a_measurement_of_an_unlisted_model_is_ignored() -> None:
    """`mbl models tree --problem X` narrows the models; the measurements of a
    model outside that filter must not be attributed to a group inside it."""
    models = [_model(0, "aaa", "seed0", 0)]
    stray = [MeasurementRow(_digest(500), _digest(999), PROBLEM_ID, is_shifted=True)]
    (contender,) = group_models(models, stray)[0].contenders
    assert (contender.nominal, contender.shifted) == (0, 0)


# --------------------------------------------------------------------------
# Columns widen rather than overflow.
# --------------------------------------------------------------------------


def test_a_long_field_widens_every_row_instead_of_breaking_alignment() -> None:
    """A column narrower than its content would push the rest of that one row
    out of alignment, which is worse than a wider table."""
    models = [
        _model(0, "a-very-long-contender-name-indeed-J4", "adam-seed0", 0),
        _model(1, "short", "adam-seed0", 0),
    ]
    body = render_tree(group_models(models)).splitlines()[1:]
    assert len({len(line) for line in body}) == 1
    assert len(body[0]) > 66


def test_the_annex_widths_are_the_minimum_not_the_maximum() -> None:
    """Short content must not shrink the table below the annex's layout."""
    body = render_tree(group_models([_model(0, "x", "a-seed0", 0)])).splitlines()[1:]
    assert len(body[0]) == 66


# --------------------------------------------------------------------------
# Scalars.
# --------------------------------------------------------------------------


def test_the_empty_store_says_so_rather_than_rendering_nothing() -> None:
    assert render_tree(()) == "the store holds no models"


@pytest.mark.parametrize(
    ("seeds", "expected"),
    [
        ((0, 1, 2, 3, 4), "seeds 0-4"),
        ((3,), "seeds 3"),
        ((0, 1, 2, 4), "seeds 0-2,4"),
        ((0, 2, 4), "seeds 0,2,4"),
        ((), DASH),
    ],
)
def test_seed_runs_are_compacted(seeds: tuple[int, ...], expected: str) -> None:
    """A gap must be visible in the seeds column too, not only in the count --
    knowing seed 3 is the missing one is what tells you what to re-run."""
    assert format_seeds(seeds) == expected


def test_identifiers_abbreviate_to_a_unique_prefix() -> None:
    """The identifier column is what a user copies into the next command, so a
    fixed truncation is not enough: two records sharing their first eight
    characters would print as one repeated value that resolves to neither."""
    ids = ["3f9a1c00aaaaaaaa", "3f9a1c00bbbbbbbb", "a1b2c3d4e5f60718"]
    short = abbreviate(ids)
    assert len(set(short.values())) == 3
    assert short["3f9a1c00aaaaaaaa"] == "3f9a1c00a"


def test_abbreviation_widens_all_the_way_when_it_has_to() -> None:
    """Two identifiers differing only in their final character need the whole
    thing; stopping short would print the same value twice."""
    short = abbreviate(["3f9a1c0000000001", "3f9a1c0000000002"])
    assert set(short.values()) == {"3f9a1c0000000001", "3f9a1c0000000002"}


def test_abbreviation_uses_one_width_for_the_whole_column() -> None:
    """Per-identifier widths would ragged the column; git picks one length for
    the set and so does this."""
    short = abbreviate(["3f9a1c0000000001", "3f9a1c0000000002", "a1b2c3d4e5f60718"])
    assert len({len(value) for value in short.values()}) == 1


def test_abbreviation_does_not_shrink_below_the_minimum() -> None:
    """A single record still shows enough to be recognisable and pasteable."""
    assert abbreviate(["3f9a1c8e2b4d5f60"]) == {"3f9a1c8e2b4d5f60": "3f9a1c8e"}


def test_abbreviation_of_nothing_is_nothing() -> None:
    assert abbreviate([]) == {}


def test_abbreviation_never_exceeds_the_identifier() -> None:
    """Two identical identifiers are one record, not an unresolvable collision,
    so widening must terminate rather than run past the end."""
    assert abbreviate(["3f9a1c8e2b4d5f60", "3f9a1c8e2b4d5f60"]) == {
        "3f9a1c8e2b4d5f60": "3f9a1c8e"
    }


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (61_000_000, "61 MB"),
        (0, "0 B"),
        (999, "999 B"),
        (1_000, "1.0 KB"),
        (52_000_000_000, "52 GB"),
        (9_500_000, "9.5 MB"),
    ],
)
def test_sizes_are_rendered_compactly(size: int, expected: str) -> None:
    assert format_size(size) == expected


def test_a_problem_of_unknown_size_omits_the_size_rather_than_reporting_zero() -> None:
    """Reporting `0 B` for a size that was never measured would read as an
    empty problem."""
    (group,) = group_models([_model(0, "c", "-", 0)])
    assert group.size_bytes is None
    assert render_tree([group]).splitlines()[0].endswith("(1 contender, 1 model)")


def test_group_shapes_are_immutable() -> None:
    (group,) = group_models([_model(0, "c", "-", 0)])
    with pytest.raises(AttributeError):
        group.display = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        group.contenders[0].contender = "other"  # type: ignore[misc]
    assert isinstance(group, ProblemGroup)
    assert isinstance(group.contenders[0], ContenderGroup)


def test_the_header_counts_contenders_and_not_rows() -> None:
    """One contender at several depths is one contender, and several rows.

    Exposed by Stage 2 Phase F3: until the producer wrote real semantic names,
    `_decompose` fell back to the indexed columns and every depth of a contender
    collapsed into one row, so the count could not disagree with the noun. With
    real names NB04 at `smoke` prints six contenders across nine rows, and the
    header claimed nine contenders for a study that declares six.

    Annex 02 §3's own example is a case where the two coincide -- four
    contenders, one depth each -- which is why the verbatim block above cannot
    catch this.
    """
    rows = [
        _model(1, "unfolded-a-J1", "adam-lr1e-2-ep200-seed0", 0),
        _model(2, "unfolded-a-J20", "adam-lr1e-2-ep200-seed0", 0),
        _model(3, "riccati-trunc", "seed0", 0),
    ]
    (group,) = group_models(rows)
    assert len(group.contenders) == 3, "three rows, because depth splits them"
    assert group.contender_count == 2, "two contenders, because depth does not"
    assert "(2 contenders, 3 models" in render_tree([group])
