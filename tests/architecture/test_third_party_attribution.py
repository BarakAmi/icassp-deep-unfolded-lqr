"""Derived work carries its notice, and the repository carries a licence.

Two obligations that are invisible until the moment they are not. The
repository's own research record states that
`src/mbl/models/constrained/lower_bound.py` is "ported from the COCP paper's
own code"; that code is released under the Apache License 2.0, whose section 4
requires a derivative work to retain the attribution notices **and** to state
prominently that changes were made. Neither was present. Separately, the
repository had no `LICENSE` at all, which makes it legally all-rights-reserved
— the opposite of what a reviewer-facing artifact needs.

Both are the kind of thing that is added once and then quietly deleted by a
future tidy-up, which is why they are asserted rather than trusted. The checks
name the *tokens* an attribution must contain rather than matching the prose
verbatim, so the wording can be improved without the test objecting, while
removing the upstream, the licence or the modification statement fails.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Files the repository's own record names as derived from third-party work,
#: and the tokens each one's notice must carry. A file is added here when its
#: provenance is established, never speculatively: an attribution naming the
#: wrong upstream is worse than none.
DERIVED: dict[str, tuple[str, ...]] = {
    "src/mbl/models/constrained/lower_bound.py": (
        "github.com/cvxgrp/cocp",
        "Apache License 2.0",
        "Agrawal",
        "Boyd",
        "Changes were made",
    ),
}


def _missing(text: str, tokens: tuple[str, ...]) -> list[str]:
    """Which required tokens the text does not carry."""
    return [token for token in tokens if token not in text]


class TestTheRepositoryIsLicensed:
    def test_a_licence_file_exists_and_names_a_licence(self) -> None:
        licence = REPO / "LICENSE"
        assert licence.is_file(), (
            "there is no LICENSE; a public repository without one is legally "
            "all-rights-reserved, so a reviewer may not run it"
        )
        assert "MIT License" in licence.read_text()

    def test_the_notice_file_records_the_derived_work(self) -> None:
        notice = (REPO / "NOTICE").read_text()
        assert "cvxgrp/cocp" in notice
        assert "Apache License" in notice

    def test_the_package_metadata_declares_the_same_licence(self) -> None:
        """A LICENSE file nobody's metadata points at is decoration."""
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        assert project.get("license") == "MIT"
        assert "LICENSE" in (project.get("license-files") or [])

    def test_declaring_urls_did_not_swallow_the_dependencies(self) -> None:
        """The TOML trap that produces a package with no dependencies.

        `[project.urls]` placed among the bare keys of `[project]` absorbs
        every key after it into the urls table. It still parses. It yields a
        distribution that installs nothing and fails at first import, and the
        only symptom is a number nobody counts.
        """
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        assert project.get("requires-python"), "requires-python was absorbed"
        assert len(project.get("dependencies", [])) > 10, "dependencies absorbed"
        assert isinstance(project.get("urls"), dict)


class TestDerivedFilesCarryTheirNotice:
    @pytest.mark.parametrize("relative", sorted(DERIVED))
    def test_the_notice_is_present_and_complete(self, relative: str) -> None:
        path = REPO / relative
        assert path.is_file(), f"{relative} is named as derived but is absent"
        missing = _missing(path.read_text(), DERIVED[relative])
        assert not missing, (
            f"{relative} is derived from third-party work and its notice is "
            f"missing {missing}. Apache-2.0 §4 requires the attribution and a "
            "statement that changes were made."
        )

    def test_the_check_can_fail(self) -> None:
        """The anti-vacuity control.

        `_missing` returning `[]` for everything would make the suite above
        green forever. Hand it a text that carries none of the tokens and it
        must report all of them.
        """
        tokens = DERIVED["src/mbl/models/constrained/lower_bound.py"]
        assert _missing("a module with no attribution at all", tokens) == list(tokens)
        assert _missing(" ".join(tokens), tokens) == []
