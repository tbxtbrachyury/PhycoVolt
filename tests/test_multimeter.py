"""Profiles, ranges and the SCPI dialogue of the multimeter (spec 3.8, B.3; a fake serial port)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from phycovolt import multimeter
from phycovolt.multimeter import Function, MultimeterError, ProfileError

PROFILE = multimeter.load_profile(multimeter.PROFILES / "agilent_34401a.ini")


def test_the_34401a_profile() -> None:
    assert PROFILE.name == "Agilent 34401A"
    assert PROFILE.terminals == {"voltage": "HI", "common": "LO", "current": "I"}
    ten_volts = PROFILE.ranges[Function.DC_VOLTAGE][10.0]
    assert (ten_volts.of_reading, ten_volts.of_range) == pytest.approx((0.000035, 0.000005))
    tenth_amp = PROFILE.ranges[Function.DC_CURRENT][0.1]
    assert (tenth_amp.of_reading, tenth_amp.of_range) == pytest.approx((0.0005, 0.00005))
    assert PROFILE.scpi["remote"] == "SYST:REM" and not PROFILE.typed


def test_the_template_is_refused_until_filled() -> None:
    with pytest.raises(ProfileError, match="FILL_IN"):
        multimeter.load_profile(multimeter.PROFILES / "template.ini")


def test_a_profile_with_a_section_missing(tmp_path: Path) -> None:
    text = (multimeter.PROFILES / "agilent_34401a.ini").read_text(encoding="utf-8")
    broken = tmp_path / "broken.ini"
    broken.write_text(text.split("[dc_current]")[0], encoding="utf-8")
    with pytest.raises(ProfileError):
        multimeter.load_profile(broken)


@pytest.mark.parametrize("function, value, expected", [
    (Function.DC_VOLTAGE, 2.5, 10.0), (Function.DC_VOLTAGE, 5.0, 10.0), (Function.DC_VOLTAGE, 0.0, 0.1),
    (Function.DC_VOLTAGE, 0.11, 0.1), (Function.DC_VOLTAGE, 0.13, 1.0),
    (Function.DC_CURRENT, 0.1, 0.1), (Function.DC_CURRENT, -0.1, 0.1), (Function.DC_CURRENT, 0.01, 0.01),
])
def test_ranges_with_overrange(function: Function, value: float, expected: float) -> None:
    assert multimeter.choose_range(PROFILE, function, value) == expected


def test_each_range_is_two_named_sources() -> None:
    errors = multimeter.range_errors(PROFILE, Function.DC_VOLTAGE, 10.0)
    assert errors.gain_name == "multimeter dc_voltage 10 V range gain"
    assert errors.gain_u == pytest.approx(0.000035 / math.sqrt(3))
    assert errors.zero_u == pytest.approx(0.000005 * 10 / math.sqrt(3))


class FakeSerial:
    """A 34401A on the other end of a cable: it answers what it is asked."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.sent: list[str] = []
        self.pending: list[str] = []

    def write_line(self, line: str) -> None:
        self.sent.append(line)
        if line in self.answers:
            self.pending.append(self.answers[line].pop(0))

    def read_line(self) -> str:
        if not self.pending:
            raise MultimeterError("no answer before the timeout")
        return self.pending.pop(0)

    def close(self) -> None:
        self.sent.append("<closed>")


def test_the_scpi_dialogue() -> None:
    port = FakeSerial({"*IDN?": ["HEWLETT-PACKARD,34401A,0,10-5-2"], "READ?": ["+2.44570000E+00"]})
    meter = multimeter.ScpiMultimeter(PROFILE, port, clock=lambda: 12.5)
    assert meter.identify().startswith("HEWLETT-PACKARD,34401A")
    meter.configure(Function.DC_VOLTAGE, 10.0)
    reading = meter.read()
    assert (reading.value, reading.time) == (2.4457, 12.5)
    assert port.sent == ["SYST:REM", "*IDN?", "CONF:VOLT:DC 10", "READ?"]    # remote mode first


def test_current_configuration() -> None:
    port = FakeSerial({})
    multimeter.ScpiMultimeter(PROFILE, port).configure(Function.DC_CURRENT, 0.1)
    assert port.sent == ["SYST:REM", "CONF:CURR:DC 0.1"]


def test_overload_is_an_answer_and_garbage_an_error() -> None:
    meter = multimeter.ScpiMultimeter(PROFILE, FakeSerial({"READ?": ["+9.90000000E+37", "garbage"]}))
    with pytest.raises(multimeter.MultimeterOverload):
        meter.read()
    with pytest.raises(MultimeterError):
        meter.read()


def test_no_answer_is_an_error() -> None:
    meter = multimeter.ScpiMultimeter(PROFILE, FakeSerial({}))
    with pytest.raises(MultimeterError):
        meter.read()


def test_only_serial_or_typed() -> None:
    typed = multimeter.Profile(**{**PROFILE.__dict__, "connection": {"transport": "usbtmc"}})
    with pytest.raises(ProfileError):
        multimeter.open_multimeter(typed)
