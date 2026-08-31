"""The mutation harness's own contract (`tools/mutation_harness.py`).

The harness is the instrument D18 requires every acceptance suite to be checked
with, and it has been rebuilt from scratch in at least five sessions — each
rebuild rediscovering the same traps, two of which produced confident and
entirely false results before they were understood. Committing it only helps if
the traps stay closed, so the four clauses that encode them are asserted here.

**What is NOT tested here, and why.** Running the harness end to end means
copying the tree and running pytest inside it, several times. That is minutes
per mutant and is what the harness is for; these tests cover the pure decisions
it makes — how a pytest run is classified, whether a substitution applied, and
what gets copied — which is exactly where the false results came from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from mutation_harness import (  # noqa: E402
    INVALID,
    KILLED,
    MANIFEST,
    SURVIVED,
    Mutant,
    apply_mutant,
    environment,
    verdict,
)


class TestAPytestErrorIsNotAKill:
    """Trap 4. A mutant that is a `SyntaxError` "fails" too, and so does one
    that breaks a module-level fixture — neither has been TESTED."""

    def test_a_failure_is_a_kill(self) -> None:
        assert verdict("1 failed, 40 passed in 3s", 1)[0] == KILLED

    def test_a_pass_is_a_survival(self) -> None:
        assert verdict("41 passed in 3s", 0)[0] == SURVIVED

    def test_a_collection_error_is_invalid(self) -> None:
        out = "ERROR tests/x.py\n!!! Interrupted: 1 error during collection !!!"
        assert verdict(out, 2)[0] == INVALID

    def test_the_last_line_is_carried_for_the_report(self) -> None:
        """So a survivor can be read without re-running it."""
        assert verdict("lots\nof\n41 passed in 3s", 0)[1] == "41 passed in 3s"

    def test_empty_output_does_not_crash_the_classifier(self) -> None:
        assert verdict("", 1) == (KILLED, "")


class TestASubstitutionThatDoesNotApplyIsLoud:
    """Trap 3, and `ruff format` will do this to you routinely: after a
    reformat, a mutant's `old` text stops matching and — without this — the
    mutant reports a false KILLED having changed nothing."""

    def test_it_reports_zero_matches_rather_than_patching_nothing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("value = 1\n")
        mutant = Mutant("m", "src/m.py", "absent = 2", "absent = 3", "why")
        assert apply_mutant(tmp_path, mutant) == 0
        assert (tmp_path / "src" / "m.py").read_text() == "value = 1\n", "it patched"

    def test_it_reports_the_match_count_so_an_ambiguous_pattern_shows(
        self, tmp_path: Path
    ) -> None:
        """Two matches means the mutant is patching the first of two places
        and the author probably meant one specific one."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("x = 1\ny = 1\n")
        assert apply_mutant(tmp_path, Mutant("m", "src/m.py", "= 1", "= 2", "")) == 2

    def test_it_replaces_only_the_first_occurrence(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("x = 1\ny = 1\n")
        apply_mutant(tmp_path, Mutant("m", "src/m.py", "= 1", "= 2", ""))
        assert (tmp_path / "src" / "m.py").read_text() == "x = 2\ny = 1\n"


class TestTheCopyRunsAgainstItsOwnSource:
    """Trap 1, the one that reported 0 killed / 10 survived on a mutant that
    deleted the feature outright: `uv run --project <root>` inside the copy
    resolved `mbl` through the ROOT's editable install."""

    def test_pythonpath_points_at_the_copy(self, tmp_path: Path) -> None:
        assert environment(tmp_path)["PYTHONPATH"] == str(tmp_path / "src")

    def test_it_does_not_point_at_the_repository(self, tmp_path: Path) -> None:
        repo = Path(__file__).resolve().parents[2]
        assert environment(tmp_path)["PYTHONPATH"] != str(repo / "src")

    def test_the_rest_of_the_environment_survives(self, tmp_path: Path) -> None:
        """`uv` needs PATH and HOME; an env built from scratch cannot run it."""
        assert "PATH" in environment(tmp_path)


class TestTheCopyManifest:
    """Trap 2's cause. The baseline failed twice before `studies/` and
    `notebooks/` were in the manifest: tests load real study documents, and a
    tree that cannot find them fails for reasons no mutant caused."""

    def test_it_carries_the_source_and_the_tests(self) -> None:
        assert "src" in MANIFEST and "tests" in MANIFEST

    def test_it_carries_the_study_documents_and_the_notebooks(self) -> None:
        assert "studies" in MANIFEST and "notebooks" in MANIFEST

    def test_it_carries_the_project_file(self) -> None:
        """Without it `uv run --project` has no project to resolve."""
        assert "pyproject.toml" in MANIFEST

    @pytest.mark.parametrize("entry", MANIFEST)
    def test_every_entry_exists_in_this_repository(self, entry: str) -> None:
        """A manifest naming something that has been renamed copies nothing and
        fails the baseline with an error about the wrong thing."""
        assert (Path(__file__).resolve().parents[2] / entry).exists(), entry


class TestTheSpecItReads:
    def test_a_mutant_round_trips_through_json(self, tmp_path: Path) -> None:
        """The file format is the surface an author writes by hand."""
        spec = {
            "suite": ["tests/x.py"],
            "mutants": [
                {
                    "name": "n",
                    "file": "src/m.py",
                    "old": "a",
                    "new": "b",
                    "why": "because",
                }
            ],
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(spec))
        loaded = json.loads(path.read_text())
        mutant = Mutant(**loaded["mutants"][0])
        assert (mutant.name, mutant.old, mutant.new, mutant.why) == (
            "n",
            "a",
            "b",
            "because",
        )

    def test_why_is_optional_but_name_and_the_patch_are_not(self) -> None:
        assert Mutant("n", "f", "a", "b").why == ""
        with pytest.raises(TypeError):
            Mutant("n", "f", "a")  # type: ignore[call-arg]
