"""The bench multimeter: its profile, its accuracy, and how to read it.

Questions it answers: how accurate is the multimeter on each range (and so, which named error
sources does each reading carry), and how is it identified, configured and read?

Main choices (spec 3.8, B.3):

- A profile is an INI file (`profiles/*.ini`): name, specification source, the accuracy table per
  function and range, the distribution, terminal names, connection settings and the SCPI commands.
  A profile with any `FILL_IN` left is refused.
- The range used is the smallest whose full scale x `overrange` covers the value; autorange is off.
  Every range is two named error sources (its gain and its zero), rectangular (limit / sqrt 3) or
  normal (limit / 2), so readings on the same range share them and their correlations take care of
  themselves (C.6).
- One small interface, `Multimeter`: identify, configure (function and range), read (value and
  time stamp). The SCPI dialogue is separate from its transport (RS-232 now; USBTMC later). The
  virtual multimeter of the simulator follows the same interface (simulator.py).
- Typed entry has no link: the calibration asks the operator through the operator's interface.
"""

from __future__ import annotations

import configparser
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

PROFILES = Path(__file__).with_name("profiles")
FILL_IN = "FILL_IN"


class Function(Enum):
    DC_VOLTAGE = "dc_voltage"
    DC_CURRENT = "dc_current"


UNIT = {Function.DC_VOLTAGE: "V", Function.DC_CURRENT: "A"}


class MultimeterError(Exception):
    """The multimeter did not answer, or answered something that is not a reading."""


class MultimeterOverload(MultimeterError):
    """The multimeter answered "overload": the value is beyond its range (a wiring mistake)."""


@dataclass(frozen=True)
class Accuracy:
    """One range's specification: limit = of_reading x |value| + of_range x range (fractions)."""

    of_reading: float
    of_range: float


@dataclass(frozen=True)
class Profile:
    name: str
    source: str
    overrange: float
    distribution: str                 # "rectangular" or "normal-95"
    resolution: float                 # displayed resolution, as a fraction of the range
    terminals: dict[str, str]         # voltage, common, current
    connection: dict[str, str]
    scpi: dict[str, str]
    ranges: dict[Function, dict[float, Accuracy]] = field(default_factory=dict)

    @property
    def typed(self) -> bool:
        return self.connection.get("transport", "typed") == "typed"


class ProfileError(ValueError):
    """A profile that cannot be used; the message says why."""


def _accuracy(text: str) -> Accuracy:
    """"0.0035 + 0.0005" (percent of reading + percent of range) -> fractions."""
    parts = [part.strip() for part in text.split("+")]
    if len(parts) != 2:
        raise ProfileError(f"accuracy {text!r} is not 'percent of reading + percent of range'")
    return Accuracy(float(parts[0]) / 100, float(parts[1]) / 100)


def load_profile(path: Path) -> Profile:
    """Read a profile; refuse one with FILL_IN left, or with a section missing."""
    text = Path(path).read_text(encoding="utf-8")
    parser = configparser.ConfigParser(inline_comment_prefixes=(";",))
    parser.read_string(text)
    left = [f"[{section}] {key}" for section in parser.sections() for key, value in parser[section].items()
            if FILL_IN in key or FILL_IN in value]
    if left:
        raise ProfileError(f"{Path(path).name} still has FILL_IN in {', '.join(left)}")
    try:
        meter = parser["multimeter"]
        ranges = {function: {float(key): _accuracy(value) for key, value in parser[function.value].items()}
                  for function in Function}
        distribution = meter["distribution"]
        if distribution not in ("rectangular", "normal-95"):
            raise ProfileError(f"unknown distribution {distribution!r}")
        return Profile(
            name=meter["name"], source=meter["source"], overrange=meter.getfloat("overrange"),
            distribution=distribution, resolution=meter.getfloat("resolution"),
            terminals={"voltage": meter["terminal_voltage"], "common": meter["terminal_common"],
                       "current": meter["terminal_current"]},
            connection=dict(parser["connection"]), scpi=dict(parser["scpi"]) if parser.has_section("scpi") else {},
            ranges=ranges,
        )
    except (KeyError, ValueError) as error:
        raise ProfileError(f"{Path(path).name}: {error}") from error


# --- Ranges and error sources ---------------------------------------------------------------------


def choose_range(profile: Profile, function: Function, value: float) -> float:
    """The smallest range whose full scale x overrange covers |value|."""
    for full_scale in sorted(profile.ranges[function]):
        if abs(value) <= full_scale * profile.overrange:
            return full_scale
    raise ProfileError(f"{value} is beyond every {function.value} range of the {profile.name}")


def standard(profile: Profile, limit: float) -> float:
    """A specification limit as a standard uncertainty (spec 8)."""
    return limit / math.sqrt(3) if profile.distribution == "rectangular" else limit / 2


@dataclass(frozen=True)
class RangeErrors:
    """The two named error sources of one range: gain (relative) and zero (absolute)."""

    gain_name: str
    gain_u: float                     # relative: a reading of x carries x * gain error
    zero_name: str
    zero_u: float                     # absolute (V or A)


def range_errors(profile: Profile, function: Function, full_scale: float) -> RangeErrors:
    accuracy = profile.ranges[function][full_scale]
    label = f"multimeter {function.value} {full_scale:g} {UNIT[function]} range"
    return RangeErrors(f"{label} gain", standard(profile, accuracy.of_reading),
                       f"{label} zero", standard(profile, accuracy.of_range * full_scale))


def display_resolution(profile: Profile, full_scale: float) -> float:
    return profile.resolution * full_scale


# --- The interface --------------------------------------------------------------------------------


@dataclass(frozen=True)
class DmmReading:
    value: float                      # V or A
    time: float                       # s, on the calibration's clock


class Multimeter(Protocol):
    profile: Profile

    def identify(self) -> str: ...

    def configure(self, function: Function, full_scale: float) -> None: ...

    def read(self) -> DmmReading: ...

    def close(self) -> None: ...


# --- SCPI over a transport ------------------------------------------------------------------------


class Transport(Protocol):
    """A line-based link: RS-232 now, USBTMC later."""

    def write_line(self, line: str) -> None: ...

    def read_line(self) -> str: ...

    def close(self) -> None: ...


class SerialTransport:
    """RS-232 through pyserial, with the profile's settings."""

    def __init__(self, connection: dict[str, str]) -> None:
        import serial                                  # only for a real multimeter
        parity = {"none": serial.PARITY_NONE, "even": serial.PARITY_EVEN, "odd": serial.PARITY_ODD}
        try:
            self._port = serial.Serial(
                port=connection["port"], baudrate=int(connection["baud"]),
                parity=parity[connection.get("parity", "none")], bytesize=int(connection.get("data_bits", 8)),
                stopbits=int(connection.get("stop_bits", 1)), timeout=float(connection.get("timeout_s", 5)),
                dsrdtr=connection.get("handshake", "none") == "dtr_dsr")
        except (serial.SerialException, KeyError, ValueError) as error:
            raise MultimeterError(f"cannot open {connection.get('port')}: {error}") from error

    def write_line(self, line: str) -> None:
        try:
            self._port.write((line + "\n").encode("ascii"))
        except Exception as error:                     # pyserial raises several kinds
            raise MultimeterError(f"writing {line!r}: {error}") from error

    def read_line(self) -> str:
        try:
            answer = self._port.readline().decode("ascii", errors="replace")
        except Exception as error:
            raise MultimeterError(f"reading: {error}") from error
        if not answer.endswith("\n"):
            raise MultimeterError("no answer before the timeout")
        return answer.strip()

    def close(self) -> None:
        self._port.close()


class ScpiMultimeter:
    """A multimeter that speaks SCPI (the 34401A), over any line transport."""

    def __init__(self, profile: Profile, transport: Transport,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.profile = profile
        self._transport = transport
        self._clock = clock
        self._remote = False

    def _remote_mode(self) -> None:
        if not self._remote and self.profile.scpi.get("remote"):
            self._transport.write_line(self.profile.scpi["remote"])   # RS-232 needs it (34401A)
            self._remote = True

    def identify(self) -> str:
        self._remote_mode()
        self._transport.write_line(self.profile.scpi["identify"])
        return self._transport.read_line()

    def configure(self, function: Function, full_scale: float) -> None:
        self._remote_mode()
        command = self.profile.scpi[f"configure_{function.value}"].format(range=f"{full_scale:g}")
        self._transport.write_line(command)

    def read(self) -> DmmReading:
        self._transport.write_line(self.profile.scpi["read"])
        answer = self._transport.read_line()
        try:
            value = float(answer)
        except ValueError as error:
            raise MultimeterError(f"not a reading: {answer!r}") from error
        if not math.isfinite(value) or abs(value) > 1e30:            # 9.9e37: overload
            raise MultimeterOverload(f"overload: {answer!r}")
        return DmmReading(value, self._clock())

    def close(self) -> None:
        self._transport.close()


def open_multimeter(profile: Profile, clock: Callable[[], float] = time.monotonic) -> ScpiMultimeter:
    """The real multimeter of a profile (serial transport)."""
    transport = profile.connection.get("transport")
    if transport != "serial":
        raise ProfileError(f"{profile.name}: the transport {transport!r} is not available (serial or typed)")
    return ScpiMultimeter(profile, SerialTransport(profile.connection), clock)
