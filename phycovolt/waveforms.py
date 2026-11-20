"""Waveform parameters -> one period of samples, and the time base of a burst.

Questions it answers: which samples does a requested waveform become, how long is the burst, and
which fields does each waveform have (one definition for parsing and for writing a channel out)?

Main choices:

- Every waveform is one function that returns ONE period at 100 kS/s (DC: one 100 ms block); the
  channel repeats it cyclically (L-08). A period has N = round(100 000 / f) samples, so the real
  frequency is 100 000 / N; edges fall on sample boundaries and widths are whole samples; the real
  values are reported (I-15). A sine starts at its center (I-15). A biphasic pulse is
  charge-balanced by construction (equal phases around its center).
- This file only refuses what cannot be built (a duty cycle that rounds to no sample, a pulse
  longer than its period). The limits of what may be switched on (5 kHz, 10 s, at least 0.3 s of
  DC, at least 2 periods) are before-checks in checks.py, with their reasons.
- Defaults depend on the role and the return (spec 3.3.3): accepting every default never gives a
  waveform the before-checks refuse (S19).
- Values are in base units: V for source-v and A for source-i, Hz, s, fractions (0.5 = 50 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from phycovolt import hardware as hw
from phycovolt import units
from phycovolt.hardware import Role


class Shape(Enum):
    DC = "dc"
    SINE = "sine"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PULSE = "pulse"
    BIPHASIC = "biphasic"


@dataclass(frozen=True)
class Field:
    """One parameter of a waveform. `quantity`: "signal" (V or A, by the role), "frequency",
    "time", "fraction", "count" (periods) or "seconds" (the duration of DC)."""

    key: str
    quantity: str


FIELDS: dict[Shape, tuple[Field, ...]] = {
    Shape.DC: (Field("level", "signal"), Field("seconds", "seconds")),
    Shape.SINE: (Field("freq", "frequency"), Field("pp", "signal"), Field("center", "signal"),
                 Field("periods", "count")),
    Shape.SQUARE: (Field("freq", "frequency"), Field("low", "signal"), Field("high", "signal"),
                   Field("duty", "fraction"), Field("periods", "count")),
    Shape.TRIANGLE: (Field("freq", "frequency"), Field("pp", "signal"), Field("center", "signal"),
                     Field("periods", "count")),
    Shape.PULSE: (Field("freq", "frequency"), Field("baseline", "signal"), Field("pulse", "signal"),
                  Field("width", "time"), Field("periods", "count")),
    Shape.BIPHASIC: (Field("freq", "frequency"), Field("center", "signal"),
                     Field("amplitude", "signal"), Field("width", "time"), Field("gap", "time"),
                     Field("periods", "count")),
}


def field_unit(item: Field, role: Role) -> tuple[str, str]:
    """(unit, display prefix) of a field: a current is shown in mA, a width in ms."""
    if item.quantity == "signal":
        return ("A", "m") if role is Role.SOURCE_I else ("V", "")
    return {"frequency": ("Hz", ""), "time": ("s", "m"), "fraction": ("%", ""), "count": ("", ""),
            "seconds": ("s", "")}[item.quantity]


@dataclass(frozen=True)
class Waveform:
    """A waveform and its parameters, by field key, in base units."""

    shape: Shape
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = [item.key for item in FIELDS[self.shape]]
        if sorted(self.values) != sorted(keys):
            raise ValueError(f"{self.shape.value} needs {keys}, not {sorted(self.values)}")
        object.__setattr__(self, "values", {key: float(self.values[key]) for key in keys})

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    @property
    def periodic(self) -> bool:
        return self.shape is not Shape.DC


# --- Defaults (spec 3.3.3) ------------------------------------------------------------------------

_CURRENT_AROUND_2V5 = {
    Shape.DC: {"level": 10e-3, "seconds": 1.0},
    Shape.SINE: {"freq": 100, "pp": 200e-3, "center": 0.0, "periods": 100},
    Shape.SQUARE: {"freq": 100, "low": -50e-3, "high": 50e-3, "duty": 0.5, "periods": 100},
    Shape.TRIANGLE: {"freq": 100, "pp": 100e-3, "center": 0.0, "periods": 100},
    Shape.PULSE: {"freq": 10, "baseline": 0.0, "pulse": 50e-3, "width": 1e-3, "periods": 10},
    Shape.BIPHASIC: {"freq": 10, "center": 0.0, "amplitude": 50e-3, "width": 1e-3, "gap": 0.1e-3,
                     "periods": 10},
}
_CURRENT_INTO_5V = {           # the 5V return only lets current into the channel
    Shape.DC: {"level": -20e-3, "seconds": 1.0},
    Shape.SINE: {"freq": 100, "pp": 80e-3, "center": -60e-3, "periods": 100},
    Shape.SQUARE: {"freq": 100, "low": -80e-3, "high": -20e-3, "duty": 0.5, "periods": 100},
    Shape.TRIANGLE: {"freq": 100, "pp": 60e-3, "center": -50e-3, "periods": 100},
    Shape.PULSE: {"freq": 10, "baseline": -20e-3, "pulse": -80e-3, "width": 1e-3, "periods": 10},
}
_CURRENT_INTO_BOARD = {        # a few mA at the M1K: tens of uA at the culture with a scale ~0.01
    Shape.DC: {"level": 2e-3, "seconds": 1.0},
    Shape.SINE: {"freq": 100, "pp": 4e-3, "center": 0.0, "periods": 100},
    Shape.SQUARE: {"freq": 100, "low": -2e-3, "high": 2e-3, "duty": 0.5, "periods": 100},
    Shape.TRIANGLE: {"freq": 100, "pp": 4e-3, "center": 0.0, "periods": 100},
    Shape.PULSE: {"freq": 10, "baseline": 0.0, "pulse": 2e-3, "width": 1e-3, "periods": 10},
    Shape.BIPHASIC: {"freq": 10, "center": 0.0, "amplitude": 2e-3, "width": 1e-3, "gap": 0.1e-3,
                     "periods": 10},
}
_VOLTAGE = {                   # the same for every return
    Shape.DC: {"level": 1.0, "seconds": 1.0},
    Shape.SINE: {"freq": 100, "pp": 2.0, "center": 2.5, "periods": 100},
    Shape.SQUARE: {"freq": 100, "low": 1.5, "high": 3.5, "duty": 0.5, "periods": 100},
    Shape.TRIANGLE: {"freq": 100, "pp": 2.0, "center": 2.5, "periods": 100},
    Shape.PULSE: {"freq": 10, "baseline": 0.5, "pulse": 3.0, "width": 1e-3, "periods": 10},
    Shape.BIPHASIC: {"freq": 10, "center": 2.5, "amplitude": 1.0, "width": 1e-3, "gap": 0.1e-3,
                     "periods": 10},
}
# Into the 5V return the 0.5 V baseline would hold ~77 mA through 57 ohm (0.28 W in the load): the
# pulse sits near the pin instead.
_VOLTAGE_INTO_5V = {**_VOLTAGE, Shape.PULSE: {"freq": 10, "baseline": 4.5, "pulse": 2.5, "width": 1e-3,
                                              "periods": 10}}
BOARD = "board"                # the return of the ivi-board circuit


def _defaults_table(role: Role, return_pin: str) -> dict[Shape, dict[str, float]]:
    if role is Role.SOURCE_V:
        return _VOLTAGE_INTO_5V if return_pin == "5V" else _VOLTAGE
    if role is Role.SOURCE_I:
        return {"5V": _CURRENT_INTO_5V, BOARD: _CURRENT_INTO_BOARD}.get(return_pin, _CURRENT_AROUND_2V5)
    raise ValueError("a hi-z channel has no waveform")


def offered_shapes(role: Role, return_pin: str) -> list[Shape]:
    """The waveforms offered for a role and return (no biphasic into 5V: one direction only)."""
    return list(_defaults_table(role, return_pin))


def default_waveform(role: Role, shape: Shape, return_pin: str) -> Waveform:
    table = _defaults_table(role, return_pin)
    if shape not in table:
        raise ValueError(f"{shape.value} is not offered for {role.value} with the {return_pin} return")
    return Waveform(shape, dict(table[shape]))


def waveform_from_text(role: Role, shape: Shape, typed: dict[str, str], return_pin: str,
                       base: Waveform | None = None) -> Waveform:
    """A waveform from typed fields ("freq": "100Hz", "pp": "2 mA"), starting from `base` when it
    has the same shape (typed values are kept) or else from the defaults of the role and return."""
    start = base if base is not None and base.shape is shape else default_waveform(role, shape, return_pin)
    known = {item.key: item for item in FIELDS[shape]}
    values = dict(start.values)
    for key, text in typed.items():
        if key not in known:
            raise ValueError(f"{shape.value} has no field {key!r} (fields: {list(known)})")
        unit, prefix = field_unit(known[key], role)
        values[key] = float(text) if not unit else units.value(text, unit, prefix)
    return Waveform(shape, values)


# --- The time base and one period -----------------------------------------------------------------


@dataclass(frozen=True)
class Timing:
    """How a waveform falls on the 100 kS/s grid, and how long its burst is."""

    samples_per_period: int           # N; for DC the 100 ms block
    frequency: float | None           # the real frequency, 100 000 / N (None for DC)
    chunks: int                       # 100 ms chunks of the burst
    settling_samples: int             # discarded by the analysis: one period, or one block
    real: dict[str, float]            # what rounding to whole samples gave: duty, width, gap

    @property
    def samples(self) -> int:
        return self.chunks * hw.CHUNK_SAMPLES

    @property
    def seconds(self) -> float:
        return self.chunks * hw.CHUNK_SECONDS


def samples_per_period(frequency: float) -> int:
    if not frequency > 0:
        raise ValueError("the frequency must be above 0 Hz")
    samples = round(hw.SAMPLE_RATE / frequency)
    if not 1 <= samples <= hw.QUEUE_SAMPLES:
        raise ValueError(f"{frequency} Hz gives {samples} samples per period")
    return samples


def _whole_samples(seconds: float) -> int:
    return round(seconds * hw.SAMPLE_RATE)


def timing(waveform: Waveform) -> Timing:
    """The time base of a waveform; raises ValueError for what cannot be built."""
    if waveform.shape is Shape.DC:
        seconds = waveform["seconds"]
        if not seconds > 0:
            raise ValueError("the duration must be above 0 s")
        chunks = max(round(seconds / hw.CHUNK_SECONDS), 1)
        return Timing(hw.CHUNK_SAMPLES, None, chunks, hw.CHUNK_SAMPLES, {})
    n = samples_per_period(waveform["freq"])
    periods = waveform["periods"]
    if periods < 1 or periods != int(periods):
        raise ValueError("the number of periods must be a whole number, at least 1")
    real: dict[str, float] = {}
    if waveform.shape is Shape.SQUARE:
        high = round(waveform["duty"] * n)
        if not 1 <= high <= n - 1:
            raise ValueError(f"a duty cycle of {waveform['duty']:.4g} leaves no sample in one phase")
        real["duty"] = high / n
    elif waveform.shape is Shape.PULSE:
        width = _whole_samples(waveform["width"])
        if not 1 <= width <= n - 1:
            raise ValueError("the pulse must last at least one sample and less than a period")
        real["width"] = width / hw.SAMPLE_RATE
    elif waveform.shape is Shape.BIPHASIC:
        width, gap = _whole_samples(waveform["width"]), _whole_samples(waveform["gap"])
        if width < 1 or gap < 0 or 2 * width + gap > n:
            raise ValueError("the two phases and the gap must fit in one period")
        real["width"] = width / hw.SAMPLE_RATE
        real["gap"] = gap / hw.SAMPLE_RATE
    chunks = math.ceil(int(periods) * n / hw.CHUNK_SAMPLES)
    return Timing(n, hw.SAMPLE_RATE / n, chunks, n, real)


def period(waveform: Waveform) -> np.ndarray:
    """One period of the waveform (DC: one 100 ms block), at 100 kS/s, in V or A."""
    shape = waveform.shape
    if shape is Shape.DC:
        return np.full(hw.CHUNK_SAMPLES, waveform["level"])
    n = timing(waveform).samples_per_period
    k = np.arange(n)
    if shape is Shape.SINE:
        return waveform["center"] + waveform["pp"] / 2 * np.sin(2 * np.pi * k / n)
    if shape is Shape.TRIANGLE:
        x = k / n                                  # starts at the center, rising, like the sine
        unit_triangle = np.where(x < 0.25, 4 * x, np.where(x < 0.75, 2 - 4 * x, 4 * x - 4))
        return waveform["center"] + waveform["pp"] / 2 * unit_triangle
    if shape is Shape.SQUARE:
        high = round(waveform["duty"] * n)
        return np.where(k < high, waveform["high"], waveform["low"]).astype(float)
    if shape is Shape.PULSE:                       # each period starts with the pulse
        width = _whole_samples(waveform["width"])
        return np.where(k < width, waveform["pulse"], waveform["baseline"]).astype(float)
    width, gap = _whole_samples(waveform["width"]), _whole_samples(waveform["gap"])
    center, amplitude = waveform["center"], waveform["amplitude"]
    block = np.full(n, center)
    block[:width] = center + amplitude
    block[width + gap: 2 * width + gap] = center - amplitude
    return block


def burst(waveform: Waveform) -> np.ndarray:
    """Every requested sample of the burst: the period repeated over the whole number of chunks."""
    return np.resize(period(waveform), timing(waveform).samples)
