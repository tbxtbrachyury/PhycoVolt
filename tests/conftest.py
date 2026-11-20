"""Shared test setup, and the guard that keeps every test away from the real instrument.

The lab environment has the real pysmu. A test that imported it and created a `pysmu.Session()`
would open the M1K, and a mode change would apply voltage or current to whatever is wired to it
(CLAUDE.md, rule 1). So, before any test module is imported, `pysmu` is replaced in `sys.modules`
by a module that refuses every use. Tests that need a pysmu pass the fake one (tests/fake_pysmu.py)
explicitly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


class _ForbiddenPysmu(types.ModuleType):
    """Stands in for pysmu during the tests: any attribute access fails loudly."""

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"tests must never use the real pysmu (pysmu.{name}); pass the fake one")


sys.modules["pysmu"] = _ForbiddenPysmu("pysmu")


@pytest.fixture
def project_calib_text() -> str:
    """The calibration stored in the M1K on 2026-11-18 (read only)."""
    return (PROJECT / "calib.txt").read_text(encoding="utf-8")
