"""Mutation testing for one acceptance suite — the contract, not a new idea.

D18 requires every acceptance suite to be mutation-tested: reintroduce the
defect the suite exists to prevent and confirm it fails. This has been rebuilt
from scratch in at least five sessions, and each rebuild rediscovered the same
four traps. They are baked in here so the sixth does not.

    uv run python tools/mutation_harness.py mutants.json /tmp/work

`mutants.json` is::

    {
      "suite": ["tests/present/test_margin_labels.py"],
      "mutants": [
        {"name": "no-margin-labels", "file": "src/mbl/present/axis_scaling.py",
         "old": "panels.label_flat_series(...)", "new": "pass",
         "why": "the feature deleted: a flat series is named nowhere"}
      ]
    }

`old` must appear in `file` **exactly once** where it matters; the harness
reports how many times it matched so a pattern that has become ambiguous is
visible rather than silently applied to the first hit.

THE FOUR TRAPS, each of which produced a confident and entirely false result
before it was understood:

1. **A PASSING BASELINE DOES NOT PROVE THE HARNESS RUNS THE COPY.** A run
   reported 0 killed / 10 survived — including a mutant that deleted the
   feature outright — because `uv run --project <root>` inside the copied tree
   resolved `mbl` through the ROOT's editable install. The copied TESTS ran
   against the ORIGINAL source and no mutant was ever executed. A tree
   importing unmutated source passes exactly as a correct baseline does, so the
   baseline check is structurally blind to this. `imports_from_tree` runs
   FIRST and asserts `mbl.__file__` is under the copy.
2. **The baseline must pass SECOND.** Otherwise every mutant reports a false
   KILLED, because a tree that cannot import exits non-zero for reasons that
   have nothing to do with the defect. It failed twice historically because the
   copy manifest was missing `studies/` and `notebooks/`.
3. **A substitution that does not apply must be LOUD.** After a fix changed the
   line one mutant targeted, the substitution stopped matching and the mutant
   reported a false KILLED. `ruff format` will do this to you routinely.
4. **A pytest ERROR is INVALID, not a kill.** A mutant that is a `SyntaxError`
   "fails" too, and so does one that breaks a module-level fixture. Only a
   FAILED line is an assertion doing its job.

And the reading rule, which is the point of the exercise: **do not tune mutants
until they die.** Read each survivor as a question about the code or the test.
Across this project's passes, survivors have been roughly half real gaps in the
tests, and the rest split between genuinely redundant code (delete it) and
deliberately equivalent controls (keep them, and say so).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: What a mutant needs a copy of. `studies/` and `notebooks/` are in it because
#: the baseline failed without them: tests load real study documents, and a
#: tree that cannot find them fails for reasons no mutant caused.
MANIFEST = ("src", "tests", "studies", "notebooks", "pyproject.toml")

KILLED, SURVIVED, INVALID, NOT_FOUND = "KILLED", "SURVIVED", "INVALID", "NOT FOUND"


@dataclass(frozen=True)
class Mutant:
    """One defect a reader could plausibly commit.

    Attributes:
        name: Short slug, printed in the report.
        file: Repo-relative path to patch.
        old: The exact text to replace.
        new: What to replace it with.
        why: Why this is a defect — printed beside the verdict, so a survivor
            is read as a question rather than as a number.
    """

    name: str
    file: str
    old: str
    new: str
    why: str = ""


def copy_tree(destination: Path) -> None:
    """A working copy of the manifest, replacing whatever was there."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for entry in MANIFEST:
        source, target = REPO / entry, destination / entry
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def environment(tree: Path) -> dict[str, str]:
    """`PYTHONPATH` pointing at the COPY's `src` — see trap 1."""
    return {**os.environ, "PYTHONPATH": str(tree / "src")}


def imports_from_tree(tree: Path) -> bool:
    """Does `mbl` resolve inside `tree`? The positive control (trap 1)."""
    done = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPO),
            "python",
            "-c",
            "import mbl, sys; sys.stdout.write(mbl.__file__)",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=300,
        env=environment(tree),
    )
    return done.stdout.startswith(str(tree))


def verdict(stdout: str, returncode: int) -> tuple[str, str]:
    """Classify one pytest run. See traps 2 and 4.

    Returns:
        `(verdict, the last line of output)`. `INVALID` when pytest ERRORed —
        a mutant that cannot even be collected has not been tested.
    """
    last = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if "ERROR" in stdout and " error" in stdout.lower():
        return INVALID, last
    return (SURVIVED if returncode == 0 else KILLED), last


def run_suite(tree: Path, suite: list[str]) -> tuple[str, str]:
    """The suite, inside `tree`, against `tree`'s own source."""
    done = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPO),
            "python",
            "-m",
            "pytest",
            *suite,
            "-q",
            "-p",
            "no:randomly",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment(tree),
    )
    return verdict(done.stdout + done.stderr, done.returncode)


def apply_mutant(tree: Path, mutant: Mutant) -> int:
    """Patch `tree`, returning how many times `old` matched.

    Zero matches is reported by the caller as NOT FOUND rather than run: a
    substitution that silently did nothing is trap 3, and it reports a false
    KILLED every time.
    """
    path = tree / mutant.file
    text = path.read_text(encoding="utf-8")
    found = text.count(mutant.old)
    if found:
        path.write_text(text.replace(mutant.old, mutant.new, 1), encoding="utf-8")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON with 'suite' and 'mutants'")
    parser.add_argument("work", type=Path, help="scratch directory for the copies")
    args = parser.parse_args(argv)

    declared = json.loads(args.spec.read_text())
    suite = list(declared["suite"])
    mutants = [Mutant(**entry) for entry in declared["mutants"]]

    baseline = args.work / "baseline"
    copy_tree(baseline)

    if not imports_from_tree(baseline):
        print("HARNESS BROKEN: `mbl` does not import from the copied tree. Stop.")
        print("Every mutant below would report SURVIVED whatever it did.")
        return 1
    print("positive control: `mbl` imports from the copied tree.")

    outcome, line = run_suite(baseline, suite)
    print(f"BASELINE: {outcome:9s} | {line}")
    if outcome != SURVIVED:
        print("BASELINE DOES NOT PASS -- every mutant below is a false KILLED. Stop.")
        return 1

    tally = {KILLED: 0, SURVIVED: 0, INVALID: 0, NOT_FOUND: 0}
    for mutant in mutants:
        tree = args.work / f"m_{mutant.name}"
        copy_tree(tree)
        found = apply_mutant(tree, mutant)
        if not found:
            print(f"{mutant.name:40s} {NOT_FOUND} in {mutant.file} -- never applied")
            tally[NOT_FOUND] += 1
            continue
        outcome, line = run_suite(tree, suite)
        note = f" (matched {found}x)" if found > 1 else ""
        print(f"{mutant.name:40s} {outcome:9s}{note} | {mutant.why} | {line}")
        tally[outcome] += 1

    print(
        f"\n{tally[KILLED]} killed / {tally[SURVIVED]} survived / "
        f"{tally[INVALID]} invalid / {tally[NOT_FOUND]} not found, of {len(mutants)}"
    )
    if tally[SURVIVED]:
        print("Read every survivor as a question about the code or the test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
