"""Acceptance tests for §1 and §5 — Stage 6, Phase B (plan §B1).

Phase A delivered §3 and §7, B0 delivered §4. Annex 04 §1.2 marks two more
sections **generated** and nothing generated them: §1's parameter table and §5's
figures. These are those two.

The failure mode both are written against is the same one, and it is why the
tests here are almost all differential: a renderer that emits a plausible
constant passes every `in` assertion anyone would think to write. §1 is the
worse case of the two, because the quantities it reports — "time-averaged", "no
terminal cost" — are *true of this study*, so a hard-coded string is right until
the day a second study is written. So the conventions are tested by **flipping
the problem they are read off** and requiring the section to move.

§5's caption carries the one number Annex 04 §1.2 requires of it, $n$, and it
takes it from the **figure's own** `.data.parquet` rather than from the analysis
that produced it. That is Annex 03 §B.1.1: an analysis *replaces in place* and a
rendered figure does not, so a caption sourced from the analysis would silently
restate what the figure would say if it were re-rendered today. Tested by
rewriting the stored analysis under a rendered figure and requiring the caption
not to move.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.replay import load_study, render_figure, render_problem, resolve
from mbl.replay.errors import UnknownArtifactError
from mbl.spec.problem import ProblemData, ProblemSpec

from ..runner.test_producer import _problem_file, _write
from .test_resolution import REPORTING

#: The figure the shared fixture declares.
FIGURE = "fig_cost"

#: The analysis it is rendered from.
ANALYSIS = "cost_by_depth"


def _document(directory: Path, /, **overrides: Any) -> Path:
    path = _write(directory, **overrides)
    path.write_text(path.read_text() + REPORTING)
    return path


def _produce(document: Path, store: Path, *, tier: str = "standard") -> None:
    from mbl.cli.app import main

    for argv in (
        ["run", str(document), "--tier", tier],
        ["analyse", str(document), "--tier", tier],
        ["figure", "render", str(document), "--tier", tier],
    ):
        assert main(["--store", str(store), *argv]) == 0


def _resolved(tmp_path: Path, /, **overrides: Any):
    document = _document(tmp_path / "doc", **overrides)
    store = tmp_path / "store"
    _produce(document, store)
    return resolve(load_study(document, tier="standard"), store=store)


#: A second, *different* instance bound along an axis, with the labels Annex 01
#: §2.5.1 requires of a spec-valued one. Appended rather than templated: the
#: shared fixture sweeps a scalar, and a spec-valued axis is the case where an
#: unlabelled section prints a repr where a table promised a category.
SHIFTED_AXIS = """
[[sweep]]
path = "evaluation.problem"
values = ["problem.npz", "shifted.npz"]
labels = ["nominal", "shifted"]
"""

#: The world axis: the plant the *trajectories* come from, bound positionwise
#: to the scored one, while the contender's declared model stays nominal. It
#: is the one shape under which reading a point's own problem reports the
#: nominal plant as every world the study trained in.
WORLD_AXIS = """
[[sweep]]
path = "training.problem"
values = ["", "shifted.npz"]
applies_to = ["unfolded_a"]
compose = "zip"
"""


def _with_a_second_instance(
    directory: Path, *, reporting: bool = True, world: bool = False
) -> Path:
    """The fixture's study, scored on its own plant *and* on a shifted one.

    `reporting` drops the declared analysis, which sweeps one axis and refuses
    two — a real refusal, and not one the sections under test are about.
    `world` additionally trains on the shifted plant, zipped to the scored one.
    """
    document = _document(directory) if reporting else _write(directory)
    frozen = ProblemSpec.load(directory / "problem.npz")
    shifted = np.asarray(frozen.data.system["A"]) * 0.5
    ProblemSpec(
        ProblemData(
            system={"A": shifted, "B": np.asarray(frozen.data.system["B"])},
            cost=dict(frozen.data.cost),
            horizon=frozen.data.horizon,
            control_bound=frozen.data.control_bound,
        )
    ).save(directory / "shifted.npz")
    document.write_text(
        document.read_text() + SHIFTED_AXIS + (WORLD_AXIS if world else "")
    )
    return document


def _instance_row(text: str, loaded: Any, *, nominal: bool) -> str:
    """One row of §1's instance table, chosen by which instance it is about."""
    declared = str(loaded.study.problem.problem_id)
    # Scoped to the section, not searched across the whole page: the parameter
    # table above carries a `ProblemID` row that matches every filter a reader
    # would think to write, and picking it up makes the assertions vacuous.
    section = text.split("### The instances it runs on", 1)[1]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "`" in line and "$" not in line
    ]
    rows = [line for line in rows if (declared in line) == nominal]
    assert len(rows) == 1, f"expected one row, found {len(rows)}"
    return rows[0]


def _large_problem(directory: Path, *, state_dim: int, scale: float):
    """A study whose plant is too large for §1 to state entry by entry.

    `scale` moves `A` without moving its shape, which is what lets the summary
    be tested differentially: two plants of one shape must not render alike.
    """
    document = _document(directory)
    frozen = ProblemSpec.load(directory / "problem.npz")
    rng = np.random.default_rng(0)
    horizon = frozen.data.horizon
    control_dim = frozen.data.control_dim
    ProblemSpec(
        ProblemData(
            system={
                "A": scale * rng.normal(size=(state_dim, state_dim)),
                "B": rng.normal(size=(state_dim, control_dim)),
            },
            cost={
                "Q": np.repeat(np.eye(state_dim)[None], horizon + 1, axis=0),
                "R": np.asarray(frozen.data.cost["R"]),
            },
            horizon=horizon,
            control_bound=frozen.data.control_bound,
        )
    ).save(directory / "problem.npz")
    return load_study(document, tier="standard")


class TestProblemSection:
    """§1's parameter table — the concrete instance, not the name."""

    def test_it_reports_the_dimensions_the_frozen_matrices_have(
        self, tmp_path: Path
    ) -> None:
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        problem = loaded.study.problem

        text = render_problem(loaded)

        assert str(problem.state_dim) in text
        assert str(problem.control_dim) in text
        assert str(problem.horizon) in text
        assert str(problem.problem_id) in text

    def test_it_carries_the_matrices_and_not_only_their_shapes(
        self, tmp_path: Path
    ) -> None:
        """Annex 04 §1.2: the named family *and* the concrete instance, because
        "a name alone is not a specification".

        Asserted against the entries themselves. A section reporting `A: (4, 4)`
        satisfies every assertion about dimensions above and states nothing a
        reader could reproduce the study from.
        """
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        a = np.asarray(loaded.study.problem.data.system["A"])

        text = render_problem(loaded)

        for entry in a.reshape(-1)[:4]:
            assert f"{entry:.4f}" in text

    def test_two_problem_instances_render_differently(self, tmp_path: Path) -> None:
        """Differential, and the property that decides whether §1 is generated
        at all: the fixture's two problems share every dimension, every bound
        and both cost matrices, and differ only in `A` and `B`. A section built
        from the dimensions renders them identically."""
        one = load_study(_document(tmp_path / "one"), tier="standard")
        other_dir = tmp_path / "two"
        _document(other_dir)
        _problem_file(other_dir, "problem.npz", seed=7)
        two = load_study(other_dir / "study.toml", tier="standard")

        assert one.study.problem.state_dim == two.study.problem.state_dim
        assert render_problem(one) != render_problem(two)

    @pytest.mark.parametrize(
        "convention", ["include_terminal_cost", "is_time_averaged"]
    )
    def test_each_cost_convention_is_read_off_the_built_problem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, convention: str
    ) -> None:
        """Annex 04 §1.3 requires the terminal/averaging conventions stated
        explicitly rather than inherited silently — and this is the assertion
        that is unfailable if written the obvious way.

        Every problem this grammar can express today builds with
        `include_terminal_cost=False, is_time_averaged=True`, so asserting that
        "time-averaged" appears holds against a renderer that prints the words
        unconditionally. The conventions are therefore **flipped at the source**
        and the section is required to move with them.

        Written first flipping **both at once**, and mutation testing killed
        that version twice over: hard-coding either convention alone left the
        other one still moving, so one assertion covered two claims and neither
        of them individually. A negative control must cover every instance the
        fix claims to cover — the fifth recorded occurrence — so the flip is
        parametrised and each convention is now its own failing direction.
        """
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        as_built = render_problem(loaded)
        original = ProblemSpec.build
        flags = {"include_terminal_cost": False, "is_time_averaged": True}
        flags[convention] = not flags[convention]

        def flipped(self: ProblemSpec) -> OptimalControlProblem:
            problem = original(self)
            return OptimalControlProblem(
                system=problem.system,
                cost=QuadraticCost(
                    Q=np.asarray(self.data.cost["Q"]),
                    R=np.asarray(self.data.cost["R"]),
                    **flags,
                ),
                constraints=list(problem.constraints),
            )

        monkeypatch.setattr(ProblemSpec, "build", flipped)

        assert render_problem(loaded) != as_built

    def test_it_says_the_terminal_matrix_is_stored_and_not_scored(
        self, tmp_path: Path
    ) -> None:
        """The concrete drift this section exists to make impossible.

        `ProblemData` requires `Q` over `N + 1` slices and the problem builds
        with `include_terminal_cost=False`, so `Q[N]` is present in the file and
        absent from every number the study reports. Both readings are plausible
        in prose; only one is true.
        """
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        q = np.asarray(loaded.study.problem.data.cost["Q"])
        assert q.shape[0] == loaded.study.problem.horizon + 1

        text = render_problem(loaded).lower()

        assert "not scored" in text

    def test_an_unconstrained_problem_is_said_to_be_unconstrained(
        self, tmp_path: Path
    ) -> None:
        """Degeneracy. `control_bound=None` is a real case — every unconstrained
        study is one — and the failure to guard it renders `None` or `0` into a
        row a reader takes for a bound of zero."""
        directory = tmp_path / "doc"
        _document(directory)
        bounded = ProblemSpec.load(directory / "problem.npz")
        ProblemSpec(
            ProblemData(
                system=dict(bounded.data.system),
                cost=dict(bounded.data.cost),
                horizon=bounded.data.horizon,
                control_bound=None,
            )
        ).save(directory / "problem.npz")

        text = render_problem(load_study(directory / "study.toml", tier="standard"))

        assert "unconstrained" in text.lower()
        assert "None" not in text

    def test_time_variation_is_measured_and_not_assumed(self, tmp_path: Path) -> None:
        """`A` may be `(n, n)` or `(N, n, n)`, and NB05 is the second. A section
        that called every problem time-invariant would be right today and wrong
        for the next study to be written."""
        directory = tmp_path / "doc"
        _document(directory)
        lti = ProblemSpec.load(directory / "problem.npz")
        invariant = render_problem(
            load_study(directory / "study.toml", tier="standard")
        )

        a = np.asarray(lti.data.system["A"])
        b = np.asarray(lti.data.system["B"])
        varying = np.repeat(a[None], lti.data.horizon, axis=0).copy()
        varying[0] *= 0.5  # genuinely time-varying, not a stacked copy
        ProblemSpec(
            ProblemData(
                system={
                    "A": varying,
                    "B": np.repeat(b[None], lti.data.horizon, axis=0),
                },
                cost=dict(lti.data.cost),
                horizon=lti.data.horizon,
                control_bound=lti.data.control_bound,
            )
        ).save(directory / "problem.npz")

        text = render_problem(load_study(directory / "study.toml", tier="standard"))

        assert text != invariant
        assert "time-varying" in text.lower()
        assert "time-invariant" in invariant.lower()

    def test_a_scalar_multiple_of_the_identity_is_stated_in_closed_form(
        self, tmp_path: Path
    ) -> None:
        """Annex 04 §1.2 as corrected 2026-08-21: the *most specific form that
        is still readable*, and for `c I` that form is exact at every size.

        The fixture's cost matrices are the identity, so a renderer that lost
        this branch would still be correct — just unreadable at n = 100. The
        assertion is therefore that the closed form is used **and** that the
        entries are gone, which is the half a weaker test would miss."""
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        problem = loaded.study.problem.data

        text = render_problem(loaded)

        assert f"I_{{{problem.state_dim}}}" in text
        assert f"I_{{{problem.control_dim}}}" in text
        assert text.count("begin{bmatrix}") == 2  # A and B only; Q and R closed

    def test_a_matrix_that_is_only_nearly_the_identity_is_not_called_one(
        self, tmp_path: Path
    ) -> None:
        """The closed form is a claim of exactness, so the check must be exact.

        One ULP is the whole test: `allclose` would state `Q = I` of a matrix
        that is not, and §1 is the section a reader reproduces the study from."""
        directory = tmp_path / "doc"
        _document(directory)
        frozen = ProblemSpec.load(directory / "problem.npz")
        q = np.asarray(frozen.data.cost["Q"]).copy()
        q[0, 0, 0] = np.nextafter(q[0, 0, 0], 2.0)
        ProblemSpec(
            ProblemData(
                system=dict(frozen.data.system),
                cost={"Q": q, "R": frozen.data.cost["R"]},
                horizon=frozen.data.horizon,
                control_bound=frozen.data.control_bound,
            )
        ).save(directory / "problem.npz")

        text = render_problem(load_study(directory / "study.toml", tier="standard"))

        # Counted, not searched: a tolerant check renders `1 \, I_4`, which
        # contains neither the exact closed form nor the entries, so an `in`
        # assertion against the closed form passes for the wrong reason. Found
        # by mutating the exact comparison to `allclose` and watching the first
        # version of this test survive.
        assert "I_" not in text.split("$$Q_0")[1].split("$$")[0]
        assert text.count("begin{bmatrix}") == 3  # A, B and now Q; R is still I

    def test_a_matrix_too_large_to_read_is_stated_by_its_invariants(
        self, tmp_path: Path
    ) -> None:
        """The correction's own case, and the reason it was written: the
        campaign's n = 100 plant renders 222,806 characters entry by entry.

        Differential on the invariants, not only on the shape. A summary that
        emitted `100 x 100` and stopped would be short, readable and would
        describe every plant of that size equally — which is exactly the "a
        name alone is not a specification" failure §1 exists to prevent."""
        one = _large_problem(tmp_path / "one", state_dim=12, scale=1.0)
        two = _large_problem(tmp_path / "two", state_dim=12, scale=0.5)
        first = render_problem(one)
        second = render_problem(two)

        entries = np.asarray(one.study.problem.data.system["A"]).reshape(-1)
        assert all(f"{value:.4f}" not in first for value in entries[:4])
        assert "12 \\times 12" in first
        assert "spectral radius" in first
        assert first != second

    def test_the_instances_an_axis_binds_are_stated_too(self, tmp_path: Path) -> None:
        """A study that sweeps its problem has more than one instance, and §1
        stating only the declared one states the plant the sweep departs
        *from* — never the plants it measures. That is the whole of the
        mismatch experiments' setup, and it was absent."""
        document = _with_a_second_instance(tmp_path / "doc")
        loaded = load_study(document, tier="standard")
        bound = {
            str(point.evaluation.problem.problem_id)
            for point in loaded.study.materialise()
        }

        text = render_problem(loaded)

        assert len(bound) == 2, "the fixture no longer binds a second instance"
        assert all(identifier in text for identifier in bound)
        assert "nominal" in text and "shifted" in text

    def test_how_far_each_bound_instance_is_from_the_declared_one_is_measured(
        self, tmp_path: Path
    ) -> None:
        """The distances are the point, not decoration: the campaign's mismatch
        moves `A` and leaves `B` alone, which is what separates a real mismatch
        from a similarity transform. Stated as a measurement, a reader can see
        which matrix moved; stated as prose, they must be told."""
        document = _with_a_second_instance(tmp_path / "doc")
        loaded = load_study(document, tier="standard")
        nominal = np.asarray(loaded.study.problem.data.system["A"])
        shifted = next(
            np.asarray(point.evaluation.problem.data.system["A"])
            for point in loaded.study.materialise()
            if point.evaluation.problem.problem_id != loaded.study.problem.problem_id
        )

        text = render_problem(loaded)

        row = _instance_row(text, loaded, nominal=False)
        assert f"{np.linalg.norm(shifted - nominal):.4f}" in row
        assert row.rstrip().endswith("0.0000 |")  # B did not move, and it says so

    def test_the_world_a_contender_trains_in_is_read_off_the_training_axis(
        self, tmp_path: Path
    ) -> None:
        """The world axis moves the plant the *trajectories* come from while
        the contender's declared model stays nominal, so the training world is
        the training specification's problem and not the point's own. Reading
        the point's problem reports the nominal plant as every world the study
        ever trained in — which is what the campaign's world-trained document
        rendered, six labels crowded into one row."""
        document = _with_a_second_instance(
            tmp_path / "doc", reporting=False, world=True
        )
        loaded = load_study(document, tier="standard")

        text = render_problem(loaded)

        shifted = _instance_row(text, loaded, nominal=False)
        assert "trained on" in shifted and "scored on" in shifted
        assert _instance_row(text, loaded, nominal=True).count("|") == shifted.count(
            "|"
        )

    def test_an_unlabelled_axis_is_not_named_by_its_identifier(
        self, tmp_path: Path
    ) -> None:
        """The world axis above declares no labels, and a hash is not a
        category. The identifier has its own column; repeating it under "axis
        value" would promise a name the study never gave."""
        document = _with_a_second_instance(
            tmp_path / "doc", reporting=False, world=True
        )
        loaded = load_study(document, tier="standard")

        row = _instance_row(render_problem(loaded), loaded, nominal=False)

        identifier = row.split("|")[3].strip().strip("`")
        assert identifier and row.split("|")[1].strip() == "shifted"

    def test_a_study_binding_one_instance_states_no_table_of_instances(
        self, tmp_path: Path
    ) -> None:
        """Degeneracy, and the one that matters: a one-row table of "the
        instances it runs on" implies a sweep that is not there."""
        text = render_problem(load_study(_document(tmp_path / "doc"), tier="standard"))
        assert "instances it runs on" not in text

    def test_it_is_deterministic(self, tmp_path: Path) -> None:
        loaded = load_study(_document(tmp_path / "doc"), tier="standard")
        assert render_problem(loaded) == render_problem(loaded)


def _caption(text: str) -> str:
    """The section with its embedded image removed.

    The base64 payload is the overwhelming majority of the string and it moves
    whenever anything about the data does, so any differential assertion made
    over the whole section is carried by the picture and says nothing at all
    about the caption. Found by mutation: a hard-coded `n` survived a test that
    compared two whole sections.
    """
    return re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", text)


def _embedded(text: str) -> bytes:
    """The bytes the section actually carries, decoded back out of it."""
    match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", text)
    assert match is not None, "the section embeds no image"
    return base64.b64decode(match.group(1))


class TestFigureSection:
    """§5 — the stored artifact and a caption reporting $n$."""

    def test_it_carries_the_bytes_of_the_stored_artifact(self, tmp_path: Path) -> None:
        """Asserted against the *content* of the file the resolution names.

        A section embedding some image passes any `<img` assertion; this one
        fails unless the bytes are the ones the figure tier wrote, which is the
        only way a reader is looking at the study's own figure rather than at a
        stale one from another store.
        """
        resolution = _resolved(tmp_path)

        text = render_figure(resolution, FIGURE)

        assert _embedded(text) == resolution.figures[FIGURE]["png"].read_bytes()

    def test_the_caption_reports_n_from_the_stored_table(self, tmp_path: Path) -> None:
        """Independent recomputation: the caption's $n$ is compared against the
        parquet, not against a second call to the renderer."""
        resolution = _resolved(tmp_path)
        table = pd.read_parquet(resolution.figures[FIGURE]["data.parquet"])
        seeds = int(table["n_seeds"].max())
        trajectories = int(table["n_trajectories"].max())
        assert table["n_seeds"].nunique() == 1, "the fixture is homogeneous"
        assert table["n_trajectories"].nunique() == 1

        caption = render_figure(resolution, FIGURE)

        assert str(trajectories) in caption
        assert f"{seeds} training seed" in caption

    def test_n_moves_when_the_study_does(self, tmp_path: Path) -> None:
        """Differential, **on the caption alone**.

        Written first over the whole section, and mutation testing showed it
        could not fail: two studies with different evaluation batches render
        different *images*, so the section differs whatever the caption says,
        and a hard-coded $n$ sailed through. The image is what dominates the
        string, so it has to be removed before the comparison means anything.
        """
        one = _caption(render_figure(_resolved(tmp_path / "a"), FIGURE))
        two = _caption(render_figure(_resolved(tmp_path / "b", eval_batch=64), FIGURE))
        assert one != two

    def test_the_caption_reports_a_range_when_the_rows_disagree(
        self, tmp_path: Path
    ) -> None:
        """The failing direction for the summary, which no homogeneous fixture
        can reach.

        Every study this grammar expresses today scores every contender on the
        same seeds, so `max` and the true span agree on every row and a mutant
        replacing one with the other survives — the single-specimen trap, and
        the reason the fixture has to be made inhomogeneous on purpose. A
        caption reading `5` over a table in which one contender was scored on
        two seeds is the one way a caption can actively mislead.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        figures = store / "studies" / str(loaded.study.study_id) / "figures"
        data = figures / f"{FIGURE}.data.parquet"

        table = pd.read_parquet(data)
        assert len(table) > 1, "the fixture needs two rows to disagree"
        table.loc[table.index[0], "n_seeds"] = 7
        table.loc[table.index[0], "n_trajectories"] = 3
        table.to_parquet(data)
        # Asserted as the whole rendered span and never as the two numbers
        # separately: the first version checked that "3" appeared, which it did
        # -- inside "3 plotted point(s)". Fourth occurrence of a substring that
        # appears anyway, and it left the summarising mutant alive.
        seeds = f"{table['n_seeds'].min()}–{table['n_seeds'].max()}"
        trajectories = (
            f"{table['n_trajectories'].min()}–{table['n_trajectories'].max()}"
        )

        caption = _caption(render_figure(resolve(loaded, store=store), FIGURE))

        assert f"{seeds} training seed" in caption
        assert f"{trajectories} evaluation trajectories" in caption

    def test_the_caption_follows_the_figure_and_not_the_analysis(
        self, tmp_path: Path
    ) -> None:
        """Annex 03 §B.1.1, and the reason the source is the figure's own data.

        An analysis **replaces in place**; a rendered figure does not. A caption
        read from the analysis would report what the figure *would* say if it
        were re-rendered today, under an image showing something else. Driven
        from the failing side: the stored analysis is rewritten under the
        already-rendered figure, and the caption must not move.

        **It must be re-resolved after the rewrite, and the first version of
        this test was not.** `resolve` loads every declared analysis into the
        `Resolution`, so a caption reading `resolution.analyses` reads a table
        held in memory since before the file changed — the mutant that did
        exactly that survived, and the test could not have seen it whatever the
        implementation was. The rewrite has to happen where the reader's would:
        between two resolutions.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _produce(document, store)
        loaded = load_study(document, tier="standard")
        before = render_figure(resolve(loaded, store=store), FIGURE)

        analysis = (
            store
            / "studies"
            / str(loaded.study.study_id)
            / "analyses"
            / f"{ANALYSIS}.parquet"
        )
        table = pd.read_parquet(analysis)
        table["n_trajectories"] = table["n_trajectories"] * 13
        table["n_seeds"] = table["n_seeds"] + 41
        table.to_parquet(analysis)

        after = render_figure(resolve(loaded, store=store), FIGURE)

        assert after == before

    def test_an_authored_claim_is_carried_into_the_caption(
        self, tmp_path: Path
    ) -> None:
        """Annex 04 §1.2: "each figure caption states the claim it supports and
        reports $n$". The claim is the author's; $n$ is not."""
        resolution = _resolved(tmp_path)
        claim = "the deeper unrolling closes the gap"

        with_claim = render_figure(resolution, FIGURE, claim=claim)

        assert claim in with_claim
        assert claim not in render_figure(resolution, FIGURE)

    def test_an_undeclared_figure_is_refused_by_name(self, tmp_path: Path) -> None:
        """A `KeyError` names the key and not the alternatives. A notebook
        author who mistypes a figure id is one line from the answer, and the
        refusal is where it belongs."""
        resolution = _resolved(tmp_path)

        with pytest.raises(UnknownArtifactError) as raised:
            render_figure(resolution, "fig_that_was_never_declared")

        assert "fig_that_was_never_declared" in str(raised.value)
        assert FIGURE in str(raised.value), "the refusal must name what does exist"

    def test_it_is_deterministic(self, tmp_path: Path) -> None:
        resolution = _resolved(tmp_path)
        assert render_figure(resolution, FIGURE) == render_figure(resolution, FIGURE)
