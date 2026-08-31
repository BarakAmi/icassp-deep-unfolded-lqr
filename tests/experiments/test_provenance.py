"""Stage-S4 tests for the code-provenance stamp (T3.e's v3 extension)."""

from mbl.experiments.provenance import (
    PACKAGE_NAME,
    RESULTS_SCHEMA_VERSION,
    code_provenance_stamp,
    package_version,
)


def test_package_version_resolves_the_governed_single_source():
    version = package_version()
    assert version  # never empty, never guessed
    parts = version.split(".")
    assert len(parts) >= 2 and all(part.isdigit() for part in parts)


def test_stamp_composes_version_and_schema_constant():
    stamp = code_provenance_stamp()
    assert stamp == (
        f"{PACKAGE_NAME}-{package_version()}/schema-{RESULTS_SCHEMA_VERSION}"
    )


def test_stamp_is_deterministic_within_a_process():
    assert code_provenance_stamp() == code_provenance_stamp()
