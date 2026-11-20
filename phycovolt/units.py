"""Values with units: reading what the operator types, and writing values for people to read.

Questions it answers: what number does "200 mA", "4,7 k", "20 µA", "1 kHz" or "2,4457 V" mean, with
what resolution was it typed, and how is 0.00004752 A written back ("47.52 µA" for people, "47.52uA"
for a command that must read back to exactly the same number)?

Main choices: one place for this, because signal parameters, circuit parameters and the multimeter's
typed entry all read values. A decimal point or a decimal comma; SI prefixes from pico to
mega ("u" for micro too); a number typed without a unit is in the field's display unit (a field
shown in mA reads "50" as 50 mA). Fractions are typed in percent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

PREFIXES = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3,
            "K": 1e3, "M": 1e6}
# The unit a field holds, and the spellings accepted for it.
SPELLINGS = {
    "V": ("v",), "A": ("a",), "Hz": ("hz",), "s": ("s",), "W": ("w",),
    "Ω": ("ω", "ohm", "ohms"), "%": ("%",),
}
FRACTION_SCALE = 0.01                 # "50 %" is 0.5

_NUMBER = r"[+\-−]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+\-]?\d+)?"
_TYPED = re.compile(rf"^\s*({_NUMBER})\s*([pnuµμmkKM]?)\s*([^\s\d.,+\-−]*)\s*$")


class UnitError(ValueError):
    """A typed value that cannot be read: `code` is "not_a_number" or "wrong_unit"."""

    def __init__(self, code: str, text: str, unit: str) -> None:
        super().__init__(f"{code}: {text!r} as {unit}")
        self.code = code
        self.text = text
        self.unit = unit


@dataclass(frozen=True)
class Typed:
    """A typed value in base units, and the resolution of the typed digits (half of nothing: one
    unit of the last digit)."""

    value: float
    resolution: float


def _scale(prefix: str, unit: str) -> float:
    return PREFIXES[prefix] * (FRACTION_SCALE if unit == "%" else 1.0)


def read(text: str, unit: str, display_prefix: str = "") -> Typed:
    """Read a typed value of `unit` ("V", "A", "Hz", "s", "Ω", "W", "%"), in base units.

    `display_prefix` is the prefix of the field's display unit ("m" for a field shown in mA): a
    number without a unit is read in it.
    """
    match = _TYPED.match(text)
    if not match:
        raise UnitError("not_a_number", text, unit)
    number, prefix, written_unit = match.groups()
    if written_unit and written_unit.lower() not in SPELLINGS[unit]:
        raise UnitError("wrong_unit", text, unit)          # "5 mV" in a current field
    if not written_unit and not prefix:
        prefix = display_prefix
    clean = number.replace(",", ".").replace("−", "-")
    mantissa = clean.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(clean.lower().split("e")[1]) if "e" in clean.lower() else 0
    scale = _scale(prefix, unit)
    return Typed(float(clean) * scale, 10.0 ** (exponent - decimals) * scale)


def value(text: str, unit: str, display_prefix: str = "") -> float:
    """The value of a typed text, in base units (see `read`)."""
    return read(text, unit, display_prefix).value


_ENGINEERING = [(1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")]


def show(number: float, unit: str, digits: int = 4) -> str:
    """A value for people: engineering prefix and `digits` significant digits ("47.52 µA")."""
    if unit == "%":
        return f"{number / FRACTION_SCALE:.{max(digits - 2, 0)}g} %"
    if number == 0 or not math.isfinite(number):
        return f"{number:g} {unit}"
    magnitude = abs(number)
    for factor, prefix in _ENGINEERING:
        if magnitude >= factor * 0.9995 or factor == 1e-12:
            scaled = number / factor
            integer_digits = len(str(int(abs(scaled)))) if abs(scaled) >= 1 else 1
            decimals = max(digits - integer_digits, 0)
            return f"{scaled:.{decimals}f} {prefix}{unit}"
    return f"{number:g} {unit}"                        # not reached


_EXACT_PREFIXES = ["", "m", "u", "n", "p", "k", "M"]     # ASCII only: a command is typed or pasted
_ASCII_UNIT = {"Ω": "ohm"}


def _numbers(value: float) -> list[str]:
    """`value` with 1 to 17 significant digits, without an exponent, shortest first."""
    texts = []
    for digits in range(1, 18):
        text = f"{value:.{digits}g}"
        if "e" not in text and text not in texts:
            texts.append(text)
    return texts or [repr(value)]


def exact(number: float, unit: str, display_prefix: str = "") -> str:
    """The shortest text that `value(text, unit, display_prefix)` reads back as exactly `number`
    ("2mA", "0.5V", "50%", "100Hz", "0.1ms"), for commands written out in full (spec 3.1). On a
    tie, the prefix that gives 1 to 999 wins."""
    number = float(number)
    written = _ASCII_UNIT.get(unit, unit)
    if not unit:
        scales = [("", 1.0)]
    elif unit == "%":
        scales = [("%", FRACTION_SCALE)]
    else:
        scales = [(p + written, PREFIXES[p]) for p in _EXACT_PREFIXES]
    found = []
    for suffix, scale in scales:
        for text in _numbers(number / scale) if number else ["0"]:
            candidate = text + suffix
            try:
                read_back = float(candidate) if not unit else value(candidate, unit, display_prefix)
            except (UnitError, ValueError):
                continue
            if read_back == number:
                in_range = number == 0 or 1 <= abs(number / scale) < 1000
                found.append((len(candidate), not in_range, candidate))
                break
    if not found:
        raise ValueError(f"{number!r} {unit} has no exact text")       # not reached: 17 digits read back
    return min(found)[2]
