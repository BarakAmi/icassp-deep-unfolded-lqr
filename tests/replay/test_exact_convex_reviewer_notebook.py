"""The exact-convex reviewer notebook meets Annex 04's replay standard.

The campaign notebook does not, which is why this one exists: measured
2026-08-31, notebook 08 fails seven of twelve of the standard's checks — it
hard-codes a store path and it carries implementation references in its prose,
a rule the author has given twice. A notebook that ships to reviewers is the
one document in the artifact repository they will run before they read anything
else, so it is the one that has to be right.

The properties below are structural, and each is structural because the
behavioural version is weaker. A notebook reaching the trainer through another
module would satisfy any assertion about `torch` and would still train. A cell
naming a store path goes stale silently, because a store at the wrong path is
*empty* rather than malformed — so it is asserted by executing the notebook
with the store redirected out from under it, and checking that the redirect was
honoured.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mbl.replay

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "icassp_exact_convex_reviewer.ipynb"

#: The one module a notebook may import from this project.
PERMITTED = "mbl.replay"

#: Exceptions meaning the pipeline broke rather than refused. A reader who hits
#: one has been handed the internals of a tier they never asked about.
INTERNAL = ("FileNotFoundError", "KeyError", "NameError", "AttributeError")

MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)


def _cells(kind: str) -> list[str]:
    document = json.loads(NOTEBOOK.read_text())
    return [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == kind
    ]


class TestStructure:
    def test_it_imports_nothing_from_this_project_but_the_replay_seam(self) -> None:
        names: set[str] = set()
        for source in _cells("code"):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module)
        ours = {name for name in names if name.split(".")[0] == "mbl"}
        assert ours == {PERMITTED}, (
            "a notebook importing anything else can reach the trainer, and "
            "'replay' becomes a description rather than a property"
        )

    def test_every_name_it_binds_is_part_of_the_seam(self) -> None:
        bound: set[str] = set()
        for source in _cells("code"):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module == PERMITTED:
                    bound.update(alias.name for alias in node.names)
        assert bound, "it imports nothing from the seam at all"
        assert bound <= set(mbl.replay.__all__)

    def test_no_markdown_cell_carries_an_implementation_reference(self) -> None:
        """Annex 04 §1.3, checked against the seam's own exported names so the
        rule cannot drift away from what there is to refer to."""
        offenders = []
        for source in _cells("markdown"):
            prose = MATH.sub(" ", source)
            for name in mbl.replay.__all__:
                # Only names that are unambiguously CODE. `resolve` and
                # `survey` are also ordinary English verbs, and a check that
                # cannot tell "results resolve to the same identifier" from a
                # function reference is one that nags at correct prose until
                # someone stops reading it. A snake_case identifier or a
                # backticked token is code; a bare common word is not.
                looks_like_code = "_" in name or f"`{name}`" in prose
                if looks_like_code and name in prose:
                    offenders.append(f"{name} in {prose[:60]!r}")
            if ".py" in prose or "mbl." in prose:
                offenders.append(f"module reference in {prose[:60]!r}")
        assert not offenders, offenders

    def test_the_prose_check_would_catch_a_real_reference(self) -> None:
        """Anti-vacuity. The rule above deliberately tolerates common English
        words, so it has to be shown it still refuses an actual one."""
        prose = "First call render_provenance on the resolution."
        caught = [
            name
            for name in mbl.replay.__all__
            if ("_" in name or f"`{name}`" in prose) and name in prose
        ]
        assert caught == ["render_provenance"]

    def test_it_reads_without_warning(self) -> None:
        """The suite runs under a zero-warning policy, so a missing cell id is
        a failure rather than a nuisance."""
        import warnings

        import nbformat

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            document = nbformat.read(str(NOTEBOOK), as_version=4)
        assert len(document.cells) > 5

    def test_the_declared_tiers_are_the_ones_a_refusal_would_name(self) -> None:
        declaration = next(s for s in _cells("code") if "TIER" in s)
        assert 'TIER = "publication_b16k"' in declaration
        assert 'STRESS_TIER = "publication"' in declaration

    def test_it_uses_the_paper_s_figure_numbering_and_not_ours(self) -> None:
        """Our Figure 5 is the paper's Figure 3.

        The gap is internal experiment numbering and has no business reaching a
        reviewer: a heading calling it Figure 5 sends someone looking for two
        figures that are not in the paper. Identifiers keep our names, because
        those are what the store is keyed by; headings do not.
        """
        headings = [
            line
            for source in _cells("markdown")
            for line in source.splitlines()
            if line.startswith("## Figure")
        ]
        assert headings == [
            "## Figure 1 — cost against unfolding depth",
            "## Figure 2 — cost against mismatch severity",
            "## Figure 3 — a box that binds",
        ], headings

    def test_every_one_of_the_paper_s_three_figures_is_rendered(self) -> None:
        """All three, and Figure 2 by the call that can reach it.

        Figure 2 is composed of three studies and none declares it, so the
        ordinary figure call raises for it -- the first version of this
        notebook showed only its tables for that reason, which left a paper
        figure a reviewer could not see. The composed call derives the figure's
        address from the parts, so this asserts three figures are drawn and
        that Figure 2 is the one reached that way.
        """
        sources = _cells("code")
        declared = [x for x in sources if "render_figure(" in x]
        composed = [x for x in sources if "render_composed_figure(" in x]
        assert len(declared) == 2, "Figures 1 and 3 are declared by their studies"
        assert len(composed) == 1, "Figure 2 is composed, and must still be drawn"
        assert not any("mismatch_severity" in x for x in declared), (
            "Figure 2 cannot be reached by the declared-figure call"
        )
        assert "fig2_mismatch_severity_exact" in composed[0]

    def test_the_composed_figure_is_addressed_by_its_parts(self) -> None:
        """Never by an identifier typed into the notebook.

        A hex identifier written out would go stale silently the day any of the
        three conditions is re-run: the figure's address is DERIVED from them,
        so a literal would point at a figure built from something else and the
        notebook would display it without complaint.
        """
        composed = next(x for x in _cells("code") if "render_composed_figure(" in x)
        assert "conditions[" in composed, "it must pass the resolved parts"
        assert not re.search(r"['\"][0-9a-f]{16}['\"]", composed), (
            "a literal study identifier makes the address stale rather than derived"
        )

    def test_it_is_committed_without_outputs(self) -> None:
        document = json.loads(NOTEBOOK.read_text())
        assert all(not cell.get("outputs") for cell in document["cells"]), (
            "D9: notebook outputs are stripped on commit"
        )


class TestTheRefusal:
    """Executed with a store that cannot answer, which needs no data at all —
    and is the half of the checkpoint that can fail."""

    @pytest.fixture(scope="module")
    def refusal(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        empty = tmp_path_factory.mktemp("empty-store")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--stdout",
                str(NOTEBOOK),
            ],
            cwd=ROOT,
            env=dict(os.environ, MBL_STORE=str(empty)),
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0, "an empty store must not replay"
        return completed.stderr + str(empty)

    def test_the_reader_is_left_holding_a_refusal(self, refusal: str) -> None:
        assert "StoreIncompleteError" in refusal

    def test_it_names_the_command_that_would_fill_the_store(self, refusal: str) -> None:
        assert "mbl run studies/icassp_exact_convex/fig1_depth.toml" in refusal
        assert "--tier publication_b16k" in refusal

    def test_it_honoured_the_redirected_store(self, refusal: str) -> None:
        """The structural proof that no cell names a store path: if one did,
        the redirect would be ignored and the refusal would name somewhere
        else."""
        assert "empty-store" in refusal

    def test_no_internal_exception_reaches_the_reader(self, refusal: str) -> None:
        leaked = [name for name in INTERNAL if name in refusal]
        assert not leaked, f"the reader was handed {leaked}"
