"""Acceptance tests for `AnalysisSpec` and `FigureSpec` — slice Phase B, B1.

Written before the implementation. Annex 01 §2.6 defines both as
registry-resolved names plus configuration, exactly like contenders, and
Annex 03 §A.6/§B.1 makes the analysis's parquet and the figure's
`.data.parquet` **the same object**. That identity is why a figure carries a
`source` naming the analysis it renders, and why a `source` naming nothing is
the first thing this suite refuses.

**The property that carries the phase is a negation: neither may reach
`StudyID`.** An analysis is a derivation *from* a study's measurements, so
declaring one must not invalidate them. If adding a figure moved the
identifier, `store/studies/<StudyID>/` would become a new directory, its
manifest would be orphaned, and `mbl run` would decide the whole study needs
re-running — an author would learn that adding a plot costs a retraining. The
same rule already applies one level down to a contender's `label` (G-1) and to
its `role`.

`TestNeitherReachesIdentity` is the discharge of that, and its second test is
the sharper one: not "an absent field is absent" but "a declared field changed
and no identifier moved".
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.applications.factories import GaussianBatchSpec
from mbl.experiments import EvaluationProtocol
from mbl.spec.analysis import AnalysisSpec, require_unique_analysis_ids
from mbl.spec.contender import ContenderSpec
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.figure import (
    FigureSpec,
    require_resolvable_sources,
    require_unique_figure_ids,
)
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import StudySpec
from mbl.spec.training import BatchPlan, TrainingSpec
from mbl.store.ids import STUDY_KEYS_EXCLUDED_FROM_STUDY_ID

from .test_contender import FAMILIES, REGISTRY

N, M, HORIZON = 4, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)

ANALYSIS = AnalysisSpec(id="cost_by_depth", kind="cost_vs_axis", config={"axis": "J"})
FIGURE = FigureSpec(
    id="fig_cost_vs_depth",
    kind="axis_scaling",
    source="cost_by_depth",
    config={"style": "thesis"},
)


def _problem(seed: int = 0) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    return ProblemSpec(
        ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
        )
    )


def _study(**overrides: Any) -> StudySpec:
    fields: dict[str, Any] = {
        "id": "probe/reporting",
        "problem": _problem(),
        "contenders": (
            ContenderSpec(
                family="unfolded",
                config=FAMILIES["unfolded"].config,
                label="unfolded_a",
                registry=REGISTRY,
            ),
        ),
        "training": TrainingSpec(
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
            plan=TrainingPlan(
                optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=3
            ),
            batch=BatchPlan(effective_size=256),
            ctx=CTX,
        ),
        "evaluation": EvaluationSpec(
            problem=_problem(),
            protocol=EvaluationProtocol(
                batch_spec=GaussianBatchSpec(
                    state_dim=N,
                    horizon=HORIZON,
                    batch_size=64,
                    seed=0,
                    process_noise_std=0.5,
                ),
                n_batches=2,
            ),
            ctx=CTX,
        ),
    }
    fields.update(overrides)
    return StudySpec(**fields)


class TestTheDeclarationsThemselves:
    def test_an_analysis_needs_an_id(self) -> None:
        with pytest.raises(SpecificationError, match="id"):
            AnalysisSpec(id="", kind="cost_vs_axis")

    def test_an_analysis_needs_a_kind(self) -> None:
        with pytest.raises(SpecificationError, match="kind"):
            AnalysisSpec(id="a", kind="")

    def test_a_figure_needs_a_source(self) -> None:
        with pytest.raises(SpecificationError, match="source"):
            FigureSpec(id="f", kind="axis_scaling", source="")

    def test_a_figure_needs_an_id(self) -> None:
        with pytest.raises(SpecificationError, match="id"):
            FigureSpec(id="", kind="axis_scaling", source="a")

    def test_a_figure_needs_a_kind(self) -> None:
        with pytest.raises(SpecificationError, match="kind"):
            FigureSpec(id="f", kind="", source="a")

    def test_a_config_that_cannot_be_written_down_is_refused(self) -> None:
        # Identity is derived from these -- not the study's, but the figure's
        # own `.spec.json`, which has to survive being written down. The same
        # rule `DataSpec.params` already applies to the same class of open map.
        with pytest.raises(SpecificationError, match="serialisable"):
            AnalysisSpec(id="a", kind="cost_vs_axis", config={"fn": object()})
        with pytest.raises(SpecificationError, match="serialisable"):
            FigureSpec(id="f", kind="k", source="a", config={"fn": object()})

    def test_the_signature_sorts_its_config(self) -> None:
        """Two authors writing the same declaration in a different order must
        collide rather than diverge, exactly as `metrics` and `gates` do.

        **The obvious assertion here cannot fail, and mutation testing said
        so.** Comparing the two signatures with `==` compares two dicts, and
        dict equality ignores key order — so an unsorted implementation passes.
        What the sort actually decides is the *serialised* order, which is the
        bytes of the analysis sidecar and of a figure's `.spec.json`, so the
        order is what this asserts.
        """
        first = AnalysisSpec(id="a", kind="k", config={"x": 1, "y": 2})
        second = AnalysisSpec(id="a", kind="k", config={"y": 2, "x": 1})

        assert list(first.get_signature()["config"]) == ["x", "y"]
        assert list(second.get_signature()["config"]) == ["x", "y"]
        assert json.dumps(first.get_signature()) == json.dumps(second.get_signature())

    def test_a_figure_signature_sorts_its_config_too(self) -> None:
        one = FigureSpec(id="f", kind="k", source="a", config={"x": 1, "y": 2})
        other = FigureSpec(id="f", kind="k", source="a", config={"y": 2, "x": 1})

        assert list(one.get_signature()["config"]) == ["x", "y"]
        assert json.dumps(one.get_signature()) == json.dumps(other.get_signature())

    def test_the_signature_carries_the_figure_s_source(self) -> None:
        # `source` is what makes the analysis parquet and the figure's
        # `.data.parquet` the same object; a signature omitting it would let
        # two figures over different tables record identical specifications.
        one = FigureSpec(id="f", kind="k", source="a")
        other = FigureSpec(id="f", kind="k", source="b")
        assert one.get_signature() != other.get_signature()


class TestTheValidators:
    def test_two_analyses_may_not_share_an_id(self) -> None:
        with pytest.raises(SpecificationError, match="cost_by_depth"):
            require_unique_analysis_ids(
                (ANALYSIS, AnalysisSpec(id="cost_by_depth", kind="other")),
                "probe/reporting",
            )

    def test_one_analysis_per_id_is_accepted(self) -> None:
        require_unique_analysis_ids(
            (ANALYSIS, AnalysisSpec(id="other", kind="other")), "probe/reporting"
        )

    def test_two_figures_may_not_share_an_id(self) -> None:
        with pytest.raises(SpecificationError, match="fig_cost_vs_depth"):
            require_unique_figure_ids((FIGURE, FIGURE), "probe/reporting")

    def test_a_figure_naming_no_declared_analysis_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="typo"):
            require_resolvable_sources(
                (FigureSpec(id="f", kind="axis_scaling", source="typo"),),
                ("cost_by_depth",),
                "probe/reporting",
            )

    def test_a_figure_naming_a_declared_analysis_is_accepted(self) -> None:
        require_resolvable_sources((FIGURE,), ("cost_by_depth",), "probe/reporting")

    def test_the_refusal_names_what_is_available(self) -> None:
        # The surface an author is editing: "typo is not declared" is half a
        # message without the list of ids that are.
        with pytest.raises(SpecificationError, match="cost_by_depth"):
            require_resolvable_sources(
                (FigureSpec(id="f", kind="axis_scaling", source="typo"),),
                ("cost_by_depth",),
                "probe/reporting",
            )


class TestTheStudyCarriesThem:
    def test_a_study_declares_none_by_default(self) -> None:
        study = _study()
        assert study.analyses == ()
        assert study.figures == ()

    def test_a_study_refuses_a_figure_whose_source_is_absent(self) -> None:
        with pytest.raises(SpecificationError, match="cost_by_depth"):
            _study(figures=(FIGURE,))

    def test_a_study_accepts_the_pair(self) -> None:
        study = _study(analyses=(ANALYSIS,), figures=(FIGURE,))
        assert study.analyses == (ANALYSIS,)
        assert study.figures == (FIGURE,)

    def test_a_study_refuses_duplicate_figure_ids(self) -> None:
        with pytest.raises(SpecificationError, match="fig_cost_vs_depth"):
            _study(analyses=(ANALYSIS,), figures=(FIGURE, FIGURE))

    def test_a_study_refuses_duplicate_analysis_ids(self) -> None:
        with pytest.raises(SpecificationError, match="cost_by_depth"):
            _study(analyses=(ANALYSIS, ANALYSIS))


class TestNeitherReachesIdentity:
    def test_declaring_an_analysis_moves_no_identifier(self) -> None:
        bare = _study()
        reported = _study(analyses=(ANALYSIS,), figures=(FIGURE,))

        assert reported.analyses != bare.analyses, (
            "the two studies are identical, so the equalities below would hold "
            "for the wrong reason"
        )
        assert reported.study_id == bare.study_id
        assert [str(point.model_id) for point in reported.materialise()] == [
            str(point.model_id) for point in bare.materialise()
        ]
        assert [str(point.measurement_id) for point in reported.materialise()] == [
            str(point.measurement_id) for point in bare.materialise()
        ]

    def test_changing_an_analysis_config_moves_no_identifier(self) -> None:
        # The sharper case: a DECLARED field changed and the identifier did
        # not. An author switching the aggregate from mean to median must not
        # retrain anything.
        mean = _study(analyses=(AnalysisSpec(id="a", kind="cost_vs_axis"),))
        median = _study(
            analyses=(
                AnalysisSpec(
                    id="a", kind="cost_vs_axis", config={"aggregate": "median"}
                ),
            )
        )

        assert mean.analyses[0] != median.analyses[0]
        assert mean.study_id == median.study_id

    def test_the_signature_still_carries_them(self) -> None:
        # Excluded from the IDENTIFIER, not from the signature: the signature
        # is a faithful dump and Tier 4 strips by name, which is where every
        # exclusion in this project is auditable in one place.
        signature = _study(analyses=(ANALYSIS,), figures=(FIGURE,)).get_signature()
        assert signature["analyses"] == [ANALYSIS.get_signature()]
        assert signature["figures"] == [FIGURE.get_signature()]

    def test_the_exclusion_is_a_by_name_contract_with_tier_four(self) -> None:
        signature = _study(analyses=(ANALYSIS,), figures=(FIGURE,)).get_signature()
        assert set(STUDY_KEYS_EXCLUDED_FROM_STUDY_ID) == {"id", "analyses", "figures"}
        for key in STUDY_KEYS_EXCLUDED_FROM_STUDY_ID:
            assert key in signature, (
                f"{key!r} is stripped by name from a signature that never "
                "emits it, so the exclusion is silently a no-op"
            )
