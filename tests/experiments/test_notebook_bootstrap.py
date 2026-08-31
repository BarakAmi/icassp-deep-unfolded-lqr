"""The one-line notebook bootstrap (NB03 overhaul, Phase 1.2): project-root
discovery, the inline-backend no-op outside IPython, and the console-log
format that drops the logger name."""

import sys

import pytest

from mbl.experiments.notebook_bootstrap import (
    NOTEBOOK_LOG_FORMAT,
    bootstrap_experiment_notebook,
    enable_inline_backend,
    find_project_root,
)


def test_find_project_root_walks_up_to_the_pyproject_marker(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = tmp_path / "src" / "pkg" / "deep"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_returns_the_marker_dir_itself(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_find_project_root_raises_when_no_marker_is_found(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="project root"):
        find_project_root(tmp_path)


def test_find_project_root_locates_this_repos_root_by_default() -> None:
    root = find_project_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "mbl" / "experiments" / "notebook_bootstrap.py").is_file()


def test_enable_inline_backend_is_a_noop_without_an_ipython_shell() -> None:
    """Under pytest there is no IPython shell, so the inline registration must
    quietly report it did nothing rather than raise."""
    assert enable_inline_backend() is False


def test_log_format_omits_the_logger_name() -> None:
    """The console format carries the message but not ``%(name)s`` -- a
    training row reads ``Epoch: ...``, never ``src.engine.callbacks: ...``."""
    assert "%(message)s" in NOTEBOOK_LOG_FORMAT
    assert "%(name)s" not in NOTEBOOK_LOG_FORMAT


def test_bootstrap_returns_the_root_and_puts_it_on_the_path(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    start = tmp_path / "notebooks"
    start.mkdir()

    # configure_logging=False: leave pytest's own log capture untouched.
    root = bootstrap_experiment_notebook(start=start, configure_logging=False)

    assert root == tmp_path.resolve()
    assert str(tmp_path.resolve()) in sys.path
