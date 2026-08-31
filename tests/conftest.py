"""Session-wide test configuration.

Forces the non-interactive Agg backend before any test module imports
pyplot: backend selection is an entry-point decision (REFACTOR_PLAN v3, N9)
-- `src.viz` deliberately has no import-time backend side effect, and a GUI
backend would hang or error in a headless CI/WSL run.
"""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _no_fall_through_to_the_real_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Stop any test reaching the developer's own store or studies directory.

    `mbl.locate` resolves an undeclared `store/` or `studies/` to the project
    the working directory is inside, and failing that to **the repository the
    package was imported from** — which during a test run is this checkout,
    holding a real multi-gigabyte store. A test that forgot to build its own
    would read it, and would mostly *pass*, because the tracked study really is
    complete there.

    One such test existed: `test_the_default_is_relative_to_the_working_directory`
    built a store in `tmp_path` without a project marker and silently consulted
    the real one instead. It failed loudly, which was luck — the assertion
    happened to name a fixture-specific model. This makes the fall-through
    unreachable instead, so the next one fails by construction.

    A test that genuinely exercises the package-root candidate substitutes
    `PACKAGE_ROOT` itself, which overrides this.
    """
    from mbl import locate

    monkeypatch.setattr(
        locate, "PACKAGE_ROOT", tmp_path_factory.mktemp("not_the_real_repository")
    )
