"""Are typed values read as the operator means them, and written back readably? (spec 3.3.3)"""

from __future__ import annotations

import pytest

from phycovolt import units


@pytest.mark.parametrize("text, unit, prefix, expected", [
    ("200 mA", "A", "", 0.2),
    ("4,7 k", "Ω", "", 4700.0),
    ("4.7kΩ", "Ω", "", 4700.0),
    ("10 ohm", "Ω", "", 10.0),
    ("20 µA", "A", "", 20e-6),
    ("20 uA", "A", "", 20e-6),
    ("1 kHz", "Hz", "", 1000.0),
    ("100Hz", "Hz", "", 100.0),
    ("2mA", "A", "", 2e-3),
    ("0", "A", "m", 0.0),
    ("50", "A", "m", 50e-3),             # a field shown in mA
    ("-0,393 mA", "A", "", -0.393e-3),
    ("−20 mA", "A", "", -20e-3),         # the typographic minus
    ("1 ms", "s", "", 1e-3),
    ("0.1", "s", "m", 0.1e-3),
    ("50 %", "%", "", 0.5),
    ("2.5V", "V", "", 2.5),
    ("1e-3 A", "A", "", 1e-3),
])
def test_values_are_read(text: str, unit: str, prefix: str, expected: float) -> None:
    assert units.value(text, unit, prefix) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("text, unit, prefix, resolution", [
    ("2.44570", "V", "", 1e-5),
    ("2,4457 V", "V", "", 1e-4),
    ("2445.70 mV", "V", "", 1e-5),
    ("63.930", "A", "m", 1e-6),
    ("-0,393 mA", "A", "", 1e-6),
])
def test_typed_resolution(text: str, unit: str, prefix: str, resolution: float) -> None:
    assert units.read(text, unit, prefix).resolution == pytest.approx(resolution)


@pytest.mark.parametrize("text, unit, code", [
    ("abc", "V", "not_a_number"),
    ("", "V", "not_a_number"),
    ("5 mV", "A", "wrong_unit"),
    ("3 Hz", "s", "wrong_unit"),
])
def test_unreadable_values(text: str, unit: str, code: str) -> None:
    with pytest.raises(units.UnitError) as error:
        units.value(text, unit)
    assert error.value.code == code


@pytest.mark.parametrize("number, unit, expected", [
    (47.52e-6, "A", "47.52 µA"),
    (2.4457, "V", "2.446 V"),
    (0.2, "A", "200.0 mA"),
    (-0.0003932, "A", "-393.2 µA"),
    (4700.0, "Ω", "4.700 kΩ"),
    (0.0, "V", "0 V"),
    (0.5, "%", "50 %"),
])
def test_values_are_shown(number: float, unit: str, expected: str) -> None:
    assert units.show(number, unit) == expected
