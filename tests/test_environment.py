"""Is the environment the one the program expects?

Run in both environments: the lab one (Python 3.10, with pysmu) and the development one (newer
Python, without pysmu). Importing pysmu does not open the instrument; nothing here creates a session.
"""

import importlib
import importlib.machinery
import importlib.util
import sys

import pytest

import phycovolt


def test_version() -> None:
    assert phycovolt.__version__ == "0.1.0"


def test_python_is_at_least_3_10() -> None:
    assert sys.version_info >= (3, 10)


@pytest.mark.parametrize("name", ["numpy", "scipy", "matplotlib", "rich", "serial"])
def test_approved_dependency_is_installed(name: str) -> None:
    importlib.import_module(name)


def test_matplotlib_has_a_headless_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.close(figure)


def test_qt_is_available_for_the_plot_window() -> None:
    # Only checks that the module exists: no window and no QApplication (they need a display).
    assert importlib.util.find_spec("PySide6") is not None


def test_pysmu_only_in_the_lab_environment() -> None:
    # PathFinder looks at the installed files, past the guard that conftest.py puts in sys.modules.
    has_pysmu = importlib.machinery.PathFinder.find_spec("pysmu") is not None
    if sys.version_info[:2] == (3, 10):
        assert has_pysmu, "the lab environment (Python 3.10) must have pysmu from conda-forge's libsmu"
    else:
        assert not has_pysmu, "the development environment must not have pysmu"


def test_the_real_pysmu_cannot_be_used_by_tests() -> None:
    import pysmu

    with pytest.raises(RuntimeError, match="never use the real pysmu"):
        pysmu.Session  # noqa: B018 - the attribute access is the test
