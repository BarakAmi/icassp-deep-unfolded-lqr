"""A study document, resolved, and able to say the command that produces it.

The first of Annex 04 §1.4's two code cells. Deliberately thin over
`spec.loader` plus the tier catalogue: `replay` must not become a second,
laxer editing surface, so an override that the tier whitelist forbids is
refused by the grammar here exactly as it is on the command line.

`LoadedStudy` keeps the document path, the tier and the overrides *beside* the
resolved study, and that is the whole reason it exists as a type. A refusal
that says "run the study" is useless; one that prints the exact invocation is
the deliverable, and reconstructing it needs the three things a bare
`StudySpec` has already forgotten.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..locate import PACKAGE_ROOT, candidates, project_directory
from ..spec.errors import SpecificationError
from ..spec.loader import load_study as _load_document
from ..spec.study import StudySpec
from ..spec.tiers import DEFAULT_TIER_CATALOGUE

#: Annex 01 §2.3's default, and `mbl run`'s. Spelled here rather than imported
#: from `cli.run`, because the top tier and the command line are peers over one
#: pipeline; an import would make a notebook depend on argparse.
DEFAULT_TIER = "standard"

#: Every example in this project's tooling is prefixed, and a command a reader
#: can paste has to be too: a bare `mbl` is either absent or a stale copy from
#: another environment.
LAUNCHER = "uv run mbl"

#: Environment variable naming the directory study documents live under.
STUDIES_ENV = "MBL_STUDIES"

#: Annex 04 §1.6's own name for it, and what is searched for when nothing is
#: declared.
STUDIES_DIRNAME = "studies"

#: What a study document is called on disk.
STUDY_SUFFIX = ".toml"

#: The `studies/` of the repository this package was imported from, which for
#: an editable install is the checkout the author is working in. Searched last,
#: and named here so a refusal can point at it.
PACKAGE_STUDIES = PACKAGE_ROOT / STUDIES_DIRNAME


def studies_root() -> Path:
    """Where study documents live.

    `$MBL_STUDIES` if it is set; else the `studies/` of the project the working
    directory is inside; else the one belonging to the repository this package
    was imported from. **Discovered rather than defaulted**, by the same
    mechanism that finds the store.

    Anchored on the project's marker rather than on a directory *named*
    `studies`, which is a correction rather than a refinement: every store files
    its artifacts under `store/studies/<StudyID>/`, so a name search run from
    inside any store resolves into that store.

    Returns:
        The directory to resolve a study id against. The fallback is the
        package's own, so that a refusal always has somewhere to point.
    """
    return project_directory(STUDIES_DIRNAME, env=STUDIES_ENV)


def _locate(document: Path | str) -> Path:
    """A study document, from either a path or a study id.

    Annex 04 §1.4 writes `load_study("box_lqr/depth_scaling")` — an **id**, and
    plan §2 says why: "a cell containing a directory name is a cell that will be
    stale". Phase A implemented the path half only, and the annex outranks it.

    A path is anything already carrying the document suffix, or anything that
    names an existing file; everything else is an id.

    Raises:
        SpecificationError: If neither reading finds a document. The message
            names both, because a notebook author cannot experiment with argv.
    """
    path = Path(document)
    if path.suffix == STUDY_SUFFIX or path.is_file():
        return path
    declared = os.environ.get(STUDIES_ENV)
    roots = [Path(declared)] if declared else candidates(STUDIES_DIRNAME)
    for root in roots:
        resolved = root / f"{document}{STUDY_SUFFIX}"
        if resolved.is_file():
            return resolved
    # Every location, not the last one tried. A refusal that names a single
    # directory cannot be acted on when the caller's wrong assumption is about
    # *which* directory was consulted -- which is the report that produced this.
    looked = "\n".join(f"  {root / f'{document}{STUDY_SUFFIX}'}" for root in roots)
    raise SpecificationError(
        f"no study {str(document)!r}: it is not a document path, and no "
        f"document is filed under that id. Looked in:\n{looked}\n"
        f"Set ${STUDIES_ENV} to name the directory studies live in, or pass "
        "the document path itself"
    )


def _pasteable(path: Path) -> str:
    """A document path as a command should name it.

    Relative to the repository, when it is inside one. Not cosmetic: `uv run`
    fails outright outside the project, so the only working directory a printed
    command can be pasted into is the root — and naming the document relative
    to it is the form that is correct there.
    """
    root = studies_root().parent
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _render_value(value: Any) -> str:
    """A `--set` value as the TOML the command line will parse back.

    `json.dumps` rather than `repr`: TOML and JSON agree on numbers, booleans
    and double-quoted strings, and `repr` does not — it emits `True` and
    single quotes, both of which `tomllib` refuses. The round trip is asserted
    rather than assumed (`test_the_command_it_prints_reproduces_the_same_study`).
    """
    return json.dumps(value)


@dataclass(frozen=True)
class LoadedStudy:
    """One study document, resolved at a tier, with its provenance intact.

    Attributes:
        study: The resolved study — what varies, and what it is compared at.
        document: The `.toml` it was read from. Kept so a refusal can name it.
        tier: The tier it was resolved at.
        overrides: The per-invocation edits, exactly as given.
        provenance: What the tier stamped, including `axis_subset`.
    """

    study: StudySpec
    document: Path
    tier: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def command(self, verb: str = "run") -> str:
        """The invocation that produces this study's `verb` stage.

        Args:
            verb: `run`, `analyse`, or `figure render`.

        Returns:
            A single line a reader can paste, with the tier and every override
            reproduced. The overrides are quoted with `shlex`, because a sweep
            path contains `*` and an unquoted one would be glob-expanded by the
            shell into whatever happens to be in the working directory.
        """
        parts = [
            LAUNCHER,
            verb,
            shlex.quote(_pasteable(self.document)),
            "--tier",
            self.tier,
        ]
        for path, value in self.overrides.items():
            parts += ["--set", shlex.quote(f"{path}={_render_value(value)}")]
        return " ".join(parts)

    def render_problem(self) -> str:
        """Annex 04 §1's parameter table, generated. See
        `replay.rendering.render_problem`."""
        from .rendering import render_problem

        return render_problem(self)

    def render_protocol(self) -> str:
        """Annex 04 §3, generated. See `replay.rendering.render_protocol`."""
        from .rendering import render_protocol

        return render_protocol(self)


def load_study(
    document: Path | str,
    *,
    tier: str = DEFAULT_TIER,
    catalogue: Path | str | None = None,
    **overrides: Any,
) -> LoadedStudy:
    """Read a study document and resolve it at a tier.

    Args:
        document: The study `.toml`, or the study id it is filed under —
            `"box_lqr/depth_scaling"`, resolved against `studies_root()`.
        tier: The execution tier. Defaults to Annex 01 §2.3's own default.
        catalogue: A tier catalogue, or `None` for the shipped one.
        overrides: The third editing level, as keyword arguments. Written
            `**{"training.seeds": 4}` because the paths are dotted; a notebook
            author reads that as data rather than as syntax.

    Returns:
        The resolved study, with everything a refusal needs to name a command.

    Raises:
        SpecificationError: On a malformed document, an unknown tier, or an
            override outside the tier whitelist — the grammar's own refusals,
            reached rather than reimplemented.
    """
    from ..experiments.bindings import DEFAULT_SPEC_BINDINGS
    from ..spec.loader import load_tier_catalogue

    path = _locate(document)
    parsed = _load_document(path, bindings=DEFAULT_SPEC_BINDINGS)
    tiers = (
        DEFAULT_TIER_CATALOGUE if catalogue is None else load_tier_catalogue(catalogue)
    )
    resolved = parsed.resolve(tiers, tier, overrides=dict(overrides))
    return LoadedStudy(
        study=resolved.study,
        document=path,
        tier=tier,
        overrides=dict(overrides),
        provenance=dict(resolved.provenance),
    )
