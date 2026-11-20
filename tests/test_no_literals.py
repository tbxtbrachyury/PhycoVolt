"""Are instrument numbers only in hardware.py (and unit.ini), and thresholds only in checks.py?

The test reads the numbers written in the code (not in comments or docstrings) of every module and
looks for a curated list of values (spec 9.1, NF-08). It also checks the portability rules that a
search can check: no literal Linux path and no shell command.
"""

from __future__ import annotations

import io
import math
import re
import tokenize
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "phycovolt"

# Values that are the M1K's or this unit's, and belong in hardware.py or unit.ini.
INSTRUMENT_NUMBERS = [
    2.4457, 2.4451, 4.9286, 4.93, 0.34, 26_214, 32_768, 65_535, 65_536, 76.29e-6, 7.629e-6, 9.537e-6,
    0.24375, 0.195, 100_000, 1_000_000, 10_000, 1.6, 1e6,
]
# Values that are thresholds or decision constants, and belong in checks.py.
# (Only distinctive values: common ones such as 0.25 or 0.02 also appear as waveform defaults.)
THRESHOLDS = [500, 1.4826, 2.2, 2.7, 4.4, 5.3, 2.4, 4.65, 0.125, 50e3, 75e-3, 60e-6, 0.5e-3]


def numbers_in_code(path: Path) -> list[tuple[int, float]]:
    found = []
    source = path.read_text(encoding="utf-8")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NUMBER:
            text = token.string.replace("_", "")
            try:
                found.append((token.start[0], float(text)))
            except ValueError:            # complex or other unusual literals
                continue
    return found


def offending(path: Path, values: list[float]) -> list[str]:
    return [f"{path.name}:{line} {value}" for line, value in numbers_in_code(path)
            if any(math.isclose(value, wanted, rel_tol=1e-9) for wanted in values)]


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_instrument_numbers_only_in_hardware(path: Path) -> None:
    if path.name == "hardware.py":
        pytest.skip("the place for them")
    if path.name == "units.py":
        pytest.skip("the SI prefixes (mega = 1e6) are not instrument numbers")
    assert offending(path, INSTRUMENT_NUMBERS) == []


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_thresholds_only_in_checks(path: Path) -> None:
    if path.name == "checks.py":
        pytest.skip("the place for them")
    assert offending(path, THRESHOLDS) == []


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_linux_paths_and_no_shell(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    strings = [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
               if token.type == tokenize.STRING]
    linux_paths = [s for s in strings if re.search(r"['\"](/dev/|/home/|/tmp/|/etc/|/usr/)", s)]
    assert linux_paths == []
    assert "shell=True" not in source
    assert "os.system(" not in source
