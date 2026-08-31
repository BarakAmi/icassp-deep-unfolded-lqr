"""The encoding vocabulary has room for the figures this project declares.

Annex 03 §B.4.2 fixes the *pairing*: slot $i$ takes palette entry $i$,
linestyle $i$ and marker $i$, "each cycle at least as long as the series
count". The clause is a capacity requirement and nothing asserted it.

The case that broke it is Figure 1 of the ICASSP campaign: eight non-bound
series and two bounds. It fits the eight markers with nothing to spare -- but
the marker was indexed within the study's **full declaration order**, which
the two bounds are part of and which they consume slots from without ever
drawing a marker. Two bounds declared before the last contender therefore
pushed two drawn series past the end of an eight-entry cycle, `% 8` wrapped
them onto markers already in use, and `require_greyscale_separable` refused
the figure before a single artifact was written.

**Measured, and the method stated with the figure** (`encode_series` over each
of the C(10,2) = 45 ways to place two bounds among ten declaration slots,
`greyscale_conflicts` as the predicate, roles as Phase C declares them):
**35 of 45 placements raise** against the pre-change rule.

The count is a property of the *cast*, not of the placement rule alone, and
that is worth knowing before a study is written: it is 35 when the GRU
contender keeps `NeuralRecipe`'s default label `"neural"`, which the colour
registry knows, and **41** if the study declares `label = "gru"` instead. An
unregistered label takes a spare palette slot rather than its own registered
colour, and two of the spares sit 0.017 apart in luminance. Renaming a
contender is not a presentation-neutral edit.

The remedy is a mechanism rather than a convention -- a rule that "bounds are
declared last" would be designed around a free parameter the author sets per
study, which this project has a standing rule against.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from matplotlib import pyplot as plt

from mbl.present.encodings import (
    LINESTYLES,
    MARKERS,
    EncodingCapacityError,
    encode_series,
)
from mbl.present.greyscale import (
    GreyscaleError,
    SeriesEncoding,
    greyscale_conflicts,
    require_greyscale_separable,
)
from mbl.spec.contender import Role

#: Phase C's cast, verbatim, with the roles that study declares. The GRU keeps
#: `NeuralRecipe`'s default label, because that is what a document declaring
#: `family = "neural"` and no label produces.
FIGURE_1_NON_BOUND: tuple[tuple[str, Role], ...] = (
    ("unfolded_alpha", Role.CONTENDER),
    ("unfolded_alpha_p", Role.CONTENDER),
    ("unfolded_alpha_p_periter", Role.CONTENDER),
    ("standard_pgd", Role.BASELINE),
    ("truncated_riccati", Role.BASELINE),
    ("riccati_unconstrained", Role.REFERENCE),
    ("neural", Role.CONTENDER),
    ("cocp", Role.CONTENDER),
)
FIGURE_1_BOUNDS = ("cocp_lower_bound", "sdp_floor")

#: NB04's tracked publication study, in its document's declaration order
#: (`studies/box_lqr/depth_scaling.toml`). Its bound is declared last, which
#: is why this study is the neutrality control: the two indexings agree
#: exactly when nothing is declared after a bound.
NB04_ORDER = (
    "truncated_riccati",
    "standard_pgd",
    "unfolded_alpha",
    "unfolded_alpha_p",
    "cocp",
    "cocp_lower_bound",
)
NB04_ROLES = {
    "truncated_riccati": Role.BASELINE,
    "standard_pgd": Role.BASELINE,
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "cocp": Role.CONTENDER,
    "cocp_lower_bound": Role.BOUND,
}

#: Read off the RENDERED tracked figure, not off the encoder: the six series
#: `store/studies/8d4349a97cad1fae`'s frozen frame actually carries, captured
#: from the live matplotlib artists.
#:
#: **RE-BASELINED BY HAND 2026-08-07, after §B.3.1 was corrected against
#: §B.4.2**, and the diff was read before it was written rather than accepted
#: wholesale. What moved, and only this:
#:
#: | series | was | is | why |
#: |---|---|---|---|
#: | `truncated_riccati` | `-.` | `-` | slot 0 |
#: | `standard_pgd` | `-.` | `--` | slot 1 |
#: | `unfolded_alpha` | `-` | `-.` | slot 2 |
#: | `unfolded_alpha_p` | `-` | `:` | slot 3 |
#: | `cocp` | `-` | `(0, (3, 1, 1, 1))` | slot 4 |
#: | `cocp_lower_bound` | `--` | `--` | a BOUND takes no slot |
#:
#: **Every colour and every marker is unchanged**, which is the property that
#: makes this a linestyle change and not a repaint: §B.3.1's "a figure that
#: drops a contender must not repaint the survivors" is about the identity
#: channel, and the identity channel did not move. Three of the six were solid
#: before and one is now.
NB04_RENDERED = {
    "truncated_riccati": ("#2a78d6", "-", "o"),
    "standard_pgd": ("#eda100", "--", "s"),
    "unfolded_alpha": ("#1baf7a", "-.", "^"),
    "unfolded_alpha_p": ("#e34948", ":", "D"),
    "cocp": ("#eb6834", (0, (3, 1, 1, 1)), "v"),
    "cocp_lower_bound": ("0.45", "--", ""),
}

#: What each of those linestyles becomes once matplotlib has drawn it, captured
#: from the same live artists. Held separately because `get_linestyle()` is
#: LOSSY -- it reports every custom dash tuple as the name `'--'` -- so a
#: golden written in the encoder's vocabulary alone cannot tell whether the
#: figure drew `cocp`'s pattern or an ordinary dashed line.
NB04_DASHES = {
    "truncated_riccati": (0, None),
    "standard_pgd": (0.0, (3.7, 1.6)),
    "unfolded_alpha": (0.0, (6.4, 1.6, 1.0, 1.6)),
    "unfolded_alpha_p": (0.0, (1.0, 1.65)),
    "cocp": (0, (3, 1, 1, 1)),
    "cocp_lower_bound": (0.0, (3.7, 1.6)),
}


def _placement(slots: tuple[int, ...]) -> tuple[tuple[str, ...], dict[str, Role]]:
    """Figure 1's cast with the two bounds at `slots` of the declaration."""
    order: list[str] = []
    roles: dict[str, Role] = {}
    bound = other = 0
    for position in range(len(FIGURE_1_NON_BOUND) + len(FIGURE_1_BOUNDS)):
        if position in slots:
            order.append(FIGURE_1_BOUNDS[bound])
            roles[FIGURE_1_BOUNDS[bound]] = Role.BOUND
            bound += 1
        else:
            label, role = FIGURE_1_NON_BOUND[other]
            order.append(label)
            roles[label] = role
            other += 1
    return tuple(order), roles


def _conflicts(order: tuple[str, ...], roles: dict[str, Role]) -> tuple[object, ...]:
    styles = encode_series(order, roles)
    return greyscale_conflicts(
        tuple(
            SeriesEncoding(label, style.color, style.linestyle, style.marker)
            for label, style in styles.items()
        )
    )


ALL_PLACEMENTS = tuple(combinations(range(10), 2))


class TestEveryTwoBoundPlacementIsDrawable:
    """The property test Phase B's acceptance clause asks for."""

    def test_there_are_forty_five_of_them(self) -> None:
        # Guards the parametrisation itself: a generator that silently
        # produced three cases would make every assertion below vacuous, and
        # a property test over an unstated population is not a property test.
        assert len(ALL_PLACEMENTS) == 45

    @pytest.mark.parametrize("slots", ALL_PLACEMENTS)
    def test_the_placement_encodes_separably(self, slots: tuple[int, ...]) -> None:
        order, roles = _placement(slots)
        assert _conflicts(order, roles) == (), f"bounds at {slots}"

    @pytest.mark.parametrize("slots", ALL_PLACEMENTS)
    def test_the_placement_survives_the_gate_on_a_real_figure(
        self, slots: tuple[int, ...]
    ) -> None:
        """Not the predicate in isolation -- the drawn figure.

        `greyscale_conflicts` over an encoding is what the encoder produces;
        `require_greyscale_separable` is what a render is refused by. Only the
        second is the claim "every one must render", and the two are separate
        functions that could disagree.
        """
        order, roles = _placement(slots)
        styles = encode_series(order, roles)
        figure, axes = plt.subplots()
        for label in order:
            style = styles[label]
            axes.plot(
                [1, 2],
                [1, 2],
                label=label,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
            )
        try:
            require_greyscale_separable(figure, f"placement{slots}")
        finally:
            plt.close(figure)

    def test_no_two_drawn_series_share_a_marker_in_any_placement(self) -> None:
        """The mechanism, stated directly rather than through its symptom.

        A pair that shares a marker can still pass §B.4.1 on colour alone, so
        the conflict count above can be zero while the defect survives in a
        cast whose palette happens to separate. This asserts the rule itself.
        """
        for slots in ALL_PLACEMENTS:
            order, roles = _placement(slots)
            styles = encode_series(order, roles)
            markers = [
                styles[label].marker
                for label in order
                if roles[label] is not Role.BOUND
            ]
            assert len(set(markers)) == len(markers), f"bounds at {slots}: {markers}"

    def test_a_bound_consumes_no_marker_slot(self) -> None:
        """Bounds-first and bounds-last must give the drawn series one answer.

        This is the whole change in one assertion: the marker is indexed
        within the DRAWN series, so where the bounds sit cannot move it.
        """
        first = encode_series(*_placement((0, 1)))
        last = encode_series(*_placement((8, 9)))
        for label, _ in FIGURE_1_NON_BOUND:
            assert first[label].marker == last[label].marker, label


class TestTheNegativeControlStillFails:
    """Headroom must not have been bought by weakening the gate."""

    def test_an_identically_encoded_pair_is_still_refused(self) -> None:
        figure, axes = plt.subplots()
        for label in ("alpha", "beta"):
            axes.plot(
                [1, 2], [1, 2], label=label, color="#2a78d6", linestyle="-", marker="o"
            )
        with pytest.raises(GreyscaleError, match="alpha"):
            require_greyscale_separable(figure, "probe")
        plt.close(figure)

    def test_the_encoder_cannot_be_asked_for_more_than_it_has(self) -> None:
        """Exhaustion is refused by name, never wrapped.

        `% len(MARKERS)` turned "I have run out of shapes" into two series
        drawn identically -- a silent answer to a question with no answer,
        which the gate then reported as a figure defect several frames away
        from the study that caused it.
        """
        order = tuple(f"c{index}" for index in range(len(MARKERS) + 1))
        roles = {label: Role.CONTENDER for label in order}
        with pytest.raises(EncodingCapacityError) as raised:
            encode_series(order, roles)
        message = str(raised.value)
        assert str(len(MARKERS) + 1) in message and str(len(MARKERS)) in message

    def test_exactly_the_vocabulary_size_is_accepted(self) -> None:
        # The boundary, from the passing side: an off-by-one in the refusal
        # would reject a study that fits, and a test only of the failing side
        # cannot tell the two errors apart.
        order = tuple(f"c{index}" for index in range(len(MARKERS)))
        roles = {label: Role.CONTENDER for label in order}
        styles = encode_series(order, roles)
        assert len({style.marker for style in styles.values()}) == len(MARKERS)


class TestTheVocabularyItself:
    def test_there_are_twelve_distinct_markers(self) -> None:
        # Twelve rather than eight is headroom for Figure 1's ten-series cast
        # plus the two-series margin a fourth figure would need. Distinctness
        # is asserted because a duplicated entry would silently halve it.
        assert len(MARKERS) == 12
        assert len(set(MARKERS)) == 12

    def test_the_first_eight_slots_are_unchanged(self) -> None:
        """§B.4.2 pairs slot *i* across three channels, so extending the
        marker cycle must APPEND. Renumbering it would recolour -- reshape --
        every figure already rendered in the thesis."""
        assert MARKERS[:8] == ("o", "s", "^", "D", "v", "p", "*", "P")

    def test_every_marker_is_one_matplotlib_accepts(self) -> None:
        # A typo in a marker string is not an error at plot time: matplotlib
        # raises only when the artist is drawn, which in a study is after the
        # training has already been paid for.
        from matplotlib.markers import MarkerStyle

        for marker in MARKERS:
            assert MarkerStyle(marker).get_path() is not None, marker


class TestTheTrackedStudyIsUntouched:
    """Neutrality, against the rendered artifact rather than against a rerun
    of the code being changed."""

    def test_nb04s_six_encodings_are_bit_identical(self) -> None:
        styles = encode_series(NB04_ORDER, NB04_ROLES)
        actual = {
            label: (style.color, style.linestyle, style.marker)
            for label, style in styles.items()
        }
        assert actual == NB04_RENDERED

    def test_the_colour_and_the_marker_did_not_move_when_the_linestyle_did(
        self,
    ) -> None:
        """The 2026-08-07 re-baseline's own guard.

        A golden re-baselined wholesale records whatever the code now does. The
        change was to ONE channel, so the other two are asserted against the
        values from before it — written out here rather than referenced, so
        this test cannot be satisfied by the same edit that satisfies the one
        above.
        """
        before = {
            "truncated_riccati": ("#2a78d6", "o"),
            "standard_pgd": ("#eda100", "s"),
            "unfolded_alpha": ("#1baf7a", "^"),
            "unfolded_alpha_p": ("#e34948", "D"),
            "cocp": ("#eb6834", "v"),
            "cocp_lower_bound": ("0.45", ""),
        }
        styles = encode_series(NB04_ORDER, NB04_ROLES)
        assert {
            label: (style.color, style.marker) for label, style in styles.items()
        } == before

    def test_each_encoding_draws_the_dash_pattern_it_names(self) -> None:
        """The encoder's vocabulary against what matplotlib actually draws.

        `get_linestyle()` reports every custom dash tuple as `'--'`, so the
        golden above cannot by itself distinguish `cocp`'s pattern from an
        ordinary dashed line. This renders each style and compares the pattern.
        """
        from matplotlib import pyplot as plt

        from mbl.present.greyscale import dash_pattern_of

        styles = encode_series(NB04_ORDER, NB04_ROLES)
        figure, axes = plt.subplots()
        try:
            for label, style in styles.items():
                (line,) = axes.plot([0, 1], [0, 1], linestyle=style.linestyle)
                assert dash_pattern_of(line) == NB04_DASHES[label], label
        finally:
            plt.close(figure)

    def test_the_six_dash_patterns_are_distinct_where_the_names_are_not(self) -> None:
        """Five of the six draw five different patterns, and the sixth is the
        BOUND's reserved dashed line — which shares a pattern with slot 1 and
        is separated from it by colour and marker, as §B.4.1 permits."""
        assert len(set(NB04_DASHES.values())) == 5
        assert NB04_DASHES["cocp_lower_bound"] == NB04_DASHES["standard_pgd"]
        assert _conflicts(NB04_ORDER, NB04_ROLES) == ()

    def test_the_tracked_study_still_passes_the_gate(self) -> None:
        assert _conflicts(NB04_ORDER, NB04_ROLES) == ()


class TestTheLinestyleIsTheSlotsAndNotTheRoles:
    """§B.3.1 corrected against §B.4.2 on 2026-08-07, on the author's review.

    Read as one linestyle per role, §B.3.1's table gives every contender a
    solid line — fine for the two-series figure it was written against, and
    measured on ICASSP Figure 1 it left **six of nine** series solid and the
    **four that actually vary** carrying **two** linestyles between them. After
    colour removal the marker was the only channel separating three curves,
    which is what §B.4 exists to prevent. §B.4.2's own sentence — "slot *i*
    takes palette entry *i*, linestyle *i* and marker *i*" — was already the
    rule.
    """

    def test_the_cycle_is_as_long_as_the_marker_cycle(self) -> None:
        """§B.4.2: "each cycle at least as long as the series count". Two
        cycles of different lengths would make slot *i* mean two things and
        the capacity guard answer only one of them."""
        assert len(LINESTYLES) == len(MARKERS)

    def test_every_entry_is_distinct(self) -> None:
        assert len(set(LINESTYLES)) == len(LINESTYLES)

    def test_every_entry_draws_a_distinct_dash_pattern(self) -> None:
        """Distinctness as declared is not distinctness as drawn: matplotlib
        reports every custom tuple as the name `'--'`, so two entries could be
        different objects and one pattern."""
        from matplotlib import pyplot as plt

        from mbl.present.greyscale import dash_pattern_of

        figure, axes = plt.subplots()
        try:
            patterns = []
            for style in LINESTYLES:
                (line,) = axes.plot([0, 1], [0, 1], linestyle=style)
                patterns.append(dash_pattern_of(line))
        finally:
            plt.close(figure)
        assert len(set(patterns)) == len(LINESTYLES), patterns

    def test_matplotlib_accepts_every_one(self) -> None:
        """A bad dash tuple raises when the artist is DRAWN, which in a study
        is after the training has been paid for."""
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots()
        try:
            for style in LINESTYLES:
                axes.plot([0, 1], [0, 1], linestyle=style)
            figure.canvas.draw()
        finally:
            plt.close(figure)

    def test_the_first_slot_is_solid(self) -> None:
        assert LINESTYLES[0] == "-"

    def test_a_nine_series_cast_takes_nine_distinct_shapes(self) -> None:
        """The measured complaint, as a property. Figure 1's own cast."""
        order = (
            "riccati_unconstrained",
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "unfolded_alpha_pj",
            "neural",
            "cocp",
            "cocp_lower_bound",
        )
        roles = dict.fromkeys(order, Role.CONTENDER)
        roles["riccati_unconstrained"] = Role.REFERENCE
        roles["cocp_lower_bound"] = Role.BOUND
        roles["truncated_riccati"] = Role.BASELINE
        roles["standard_pgd"] = Role.BASELINE
        styles = encode_series(order, roles)
        assert len({(s.linestyle, s.marker) for s in styles.values()}) == len(order)

    def test_the_four_that_vary_no_longer_share_a_linestyle(self) -> None:
        """The four series a reader of Figure 1 actually compares. Measured
        before the change: two distinct linestyles between them."""
        order = (
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "unfolded_alpha_pj",
        )
        roles = dict.fromkeys(order, Role.CONTENDER)
        roles["truncated_riccati"] = Role.BASELINE
        roles["standard_pgd"] = Role.BASELINE
        styles = encode_series(order, roles)
        varying = (
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "unfolded_alpha_pj",
        )
        assert len({styles[key].linestyle for key in varying}) == 4

    def test_a_reserved_role_takes_no_slot_and_keeps_its_own_style(self) -> None:
        """Role still decides MEMBERSHIP: a reference and a bound are not among
        the things being compared, so a bound declared early cannot push a
        contender's linestyle along."""
        early = encode_series(
            ("cocp_lower_bound", "a", "b"),
            {"cocp_lower_bound": Role.BOUND, "a": Role.CONTENDER, "b": Role.CONTENDER},
        )
        late = encode_series(
            ("a", "b", "cocp_lower_bound"),
            {"cocp_lower_bound": Role.BOUND, "a": Role.CONTENDER, "b": Role.CONTENDER},
        )
        assert early["a"].linestyle == late["a"].linestyle == LINESTYLES[0]
        assert early["b"].linestyle == late["b"].linestyle == LINESTYLES[1]
        assert early["cocp_lower_bound"].linestyle == late["cocp_lower_bound"].linestyle
