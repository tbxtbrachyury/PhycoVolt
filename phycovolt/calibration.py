"""The calibration procedure: the stages, the readings, the pairs, the records, their uncertainty,
and the verification (spec 3.7, C.1, C.6).

Questions it answers: how far off is each of the M1K's eight functions against the multimeter, which
calib.txt corrects it, how sure are we of every number, and does the result hold at a point that
did not enter the computation?

Main choices:

- Raw values (I-09): every M1K sample has the stored correction undone, and every request is
  pre-distorted so that libsmu sends exactly the wanted DAC code; so the EEPROM need not be reset.
- Acquisition and mathematics are separate. The stage runners (instrument + multimeter + operator)
  turn each point into an `Observation`; `solve` is a pure function of the observations. The Monte
  Carlo feeds `solve` with simulated observations, so it validates this very code.
- Only "fit" points make pairs (I-10); verification points are measured through the new record. A
  source pair's request is the value of the DAC code sent (I-11), written with 7 decimals for
  current; the zero pair stays "0.0000" (B.1).
- The written record is computed from the pairs' text in float32, as `smu -w` does (C.1). The
  uncertainty is propagated by perturbing every named error source by +-u through the same
  computation, in double precision (C.6): each multimeter range is two sources (gain and zero),
  shared by every reading on it; each point adds the multimeter's spread and the M1K's block
  statistics (the standard deviation of its block means, `analysis.uncertainty_of_mean`). U = 2u.
- "Predicted error ± U" compares what libsmu will give with the new record against the true
  behavior, which is the line through the points minus their errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from phycovolt import analysis, checks, libsmu_math, multimeter, units
from phycovolt import hardware as hw
from phycovolt.checks import Check, Level
from phycovolt.hardware import ChannelSetup, Instrument, InstrumentError, Kind, Record, Role
from phycovolt.libsmu_math import CalibBlock
from phycovolt.multimeter import Function, Multimeter, MultimeterError, Profile

STAGES = ("measure-v", "source-v", "measure-i", "source-i")
KIND = {"measure-v": Kind.MEASURE_V, "source-v": Kind.SOURCE_V, "measure-i": Kind.MEASURE_I,
        "source-i": Kind.SOURCE_I}
# The effect table: where the error before and after is shown (spec 3.7). "reference" is the 2.5V
# pin as the multimeter read it.
EFFECT_VALUES = {
    "measure-v": (0.0, 0.1, 1.0, "reference", 5.0),
    "source-v": (0.0, 0.1, 1.0, 2.5, 4.5, 5.0),
    "measure-i": (-0.2, -0.1, -0.01, -0.1e-3, 0.0, 0.1e-3, 0.01, 0.1, 0.2),
    "source-i": (-0.15, -0.1, -0.01, -1e-3, 0.0, 1e-3, 0.01, 0.1, 0.15),
}
MAIN_POINT = {"measure-v": "reference", "source-v": 2.5, "measure-i": 0.1, "source-i": 0.1}
STEP = {Kind.MEASURE_V: hw.V_STEP, Kind.MEASURE_I: hw.ADC_I_STEP, Kind.SOURCE_V: hw.V_STEP,
        Kind.SOURCE_I: hw.DAC_I_STEP}
UNIT = {Kind.MEASURE_V: "V", Kind.SOURCE_V: "V", Kind.MEASURE_I: "A", Kind.SOURCE_I: "A"}
DECIMALS = {"V": 6, "A": 7}


class StageStopped(Exception):
    """A stage cannot go on: `check` says why (V/I swap, the operator, too many failed attempts)."""

    def __init__(self, check: Check) -> None:
        super().__init__(check.text)
        self.check = check


# --- The operator ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class WiringStep:
    """What the operator mounts before a group of points (texts: checks.WIRING_TEXTS)."""

    stage: str
    kind: str                         # gnd, pin, dmm_voltage, open, current_loop, dmm_current
    channel: str | None = None
    pin: str | None = None
    full_scale: float | None = None   # dmm_current: the range to set (typed mode)


@dataclass(frozen=True)
class TypedRequest:
    """Typed mode: what the operator reads on the multimeter, and when."""

    stage: str
    point: str
    attempt: int
    function: Function
    full_scale: float
    window: tuple[float, float]       # on the calibration's clock: the 40 to 60 % of the acquisition
    unit: str = "V"                   # what the display shows: "V" or "mA"
    decimals: int = 0                 # how many decimals it shows on this range at full resolution


class Operator(Protocol):
    def wire(self, step: WiringStep) -> bool:
        """Show the wiring and wait until it is mounted; False stops the stage."""

    def confirm_warm_up(self) -> bool:
        """The M1K on for 15 min and the multimeter for 1 h (H-11)?"""

    def problem(self, check: Check) -> bool:
        """A reading failed (wiring, the multimeter): True to try again, False to stop the stage."""

    def accept_warning(self, check: Check) -> bool:
        """Drift or noise above the margin: True keeps the reading, False redoes it."""

    def typed_value(self, request: TypedRequest) -> str:
        """Typed mode: the value the multimeter showed during the window."""

    def acquiring(self, stage: str, point: str, attempt: int, seconds: float) -> None:
        """An acquisition starts now (typed mode: read the display from 40 to 60 % of it)."""


# --- Observations: what the mathematics needs from each point -------------------------------------


@dataclass(frozen=True)
class Observation:
    """One accepted point. M1K values are raw (stored correction undone): {"A V": (mean, u)}."""

    point: str
    role: str                         # fit, verification, auxiliary, probe
    m1k: dict[str, tuple[float, float]] = field(default_factory=dict)
    dmm: tuple[float, float] | None = None       # (mean, u)
    dmm_function: Function | None = None
    dmm_range: float | None = None
    codes: dict[str, int] = field(default_factory=dict)   # channel -> DAC code sent (raw)


@dataclass(frozen=True)
class StageData:
    """Everything a stage measured: the input of `solve`."""

    stage: str
    channels: tuple[str, ...]
    observations: dict[str, Observation]
    stored: list[Record]              # the EEPROM's records when the stage ran
    profile: Profile
    meta: dict[str, float] = field(default_factory=dict)   # probe results, currents, ...


# --- Rows of the files ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadingRow:                     # readings.csv (B.2): one per channel and quantity per attempt
    time: float
    stage: str
    point: str
    attempt: int
    status: str                       # accepted, or why it was redone
    channel: str
    quantity: str
    mean: float
    u: float
    sigma: float
    drift: float
    r1: float | None
    blocks: int
    jumps: int
    adc_limit_fraction: float
    dmm_mean: float | None
    dmm_u: float | None
    dmm_readings: int
    vi_swaps: int
    halved_words: int = 0             # F-02: samples of this channel (V and I) with a halved ADC word


@dataclass(frozen=True)
class DmmRow:                         # dmm.csv (B.2): every multimeter reading
    time: float
    stage: str
    point: str
    attempt: int
    function: str
    range: float
    value: float


# --- Acquisition ----------------------------------------------------------------------------------


@dataclass
class Context:
    """The bench as the procedure sees it."""

    instrument: Instrument
    dmm: Multimeter | None            # None: typed mode
    profile: Profile
    operator: Operator
    clock: Callable[[], float]
    stored: list[Record]
    skip_optional: bool = False
    readings: list[ReadingRow] = field(default_factory=list)
    dmm_rows: list[DmmRow] = field(default_factory=list)
    warnings: list[Check] = field(default_factory=list)

    @property
    def typed(self) -> bool:
        return self.dmm is None


@dataclass(frozen=True)
class PointSpec:
    stage: str
    point: str
    role: str
    seconds: float
    quantities: tuple[str, ...]                       # "A V", "B I", …: the M1K values kept
    drive: dict[str, tuple[Role, int]] = field(default_factory=dict)   # channel -> (role, DAC code)
    dmm_function: Function | None = None
    dmm_expected: float = 0.0                         # chooses the range
    physics: Callable[[Observation], list[Check]] | None = None


def raw_request_for_code(code: int, kind: Kind, stored: list[Record], channel: str) -> float:
    """The request libsmu turns into exactly `code`, with the stored calibration (I-09)."""
    return libsmu_math.request_for_code(code, kind, stored[hw.record_index(channel, kind)])


def code_for_raw(value: float, kind: Kind) -> int:
    """The DAC code whose value is the largest not above `value` (no calibration applied)."""
    return int(libsmu_math.dac_code(value, kind, Record()))


def _setups(ctx: Context, spec: PointSpec) -> tuple[ChannelSetup, ChannelSetup]:
    setups = []
    for channel in hw.CHANNELS:
        if channel not in spec.drive:
            setups.append(ChannelSetup())
            continue
        role, code = spec.drive[channel]
        request = raw_request_for_code(code, hw.SOURCE_KIND[role], ctx.stored, channel)
        setups.append(ChannelSetup(role, np.full(hw.CHUNK_SAMPLES, request)))
    return setups[0], setups[1]


def _raw(ctx: Context, channel: str, voltage: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = libsmu_math.raw_voltage(voltage, ctx.stored[hw.record_index(channel, Kind.MEASURE_V)])
    i = libsmu_math.raw_current(current, ctx.stored[hw.record_index(channel, Kind.MEASURE_I)])
    return v, i


def _margin(unit: str, value: float) -> float:
    absolute, relative = checks.MARGIN[unit]
    return absolute + relative * abs(value)


@dataclass
class _Acquired:
    summaries: dict[str, analysis.Summary]
    window: dict[str, np.ndarray]                  # typed mode: the raw samples of the 40-60 % window
    dmm: list[tuple[float, float]]                 # (time, value) after the settling
    swaps: dict[str, int]
    overcurrent: bool
    dmm_error: str | None
    full_scale: float | None                       # the multimeter's range for this point
    dmm_overload: bool = False                     # it showed "overload": the wiring, not the cable
    halved: dict[str, int] = field(default_factory=dict)   # F-02: "A I" -> samples with a halved word


def _acquire_once(ctx: Context, spec: PointSpec, attempt: int) -> _Acquired:
    a, b = _setups(ctx, spec)
    chunks = round(spec.seconds / hw.CHUNK_SECONDS)
    automatic = spec.dmm_function is not None and not ctx.typed
    full_scale = None
    dmm_error = None
    if spec.dmm_function is not None:
        full_scale = multimeter.choose_range(ctx.profile, spec.dmm_function, spec.dmm_expected)
        if automatic:
            try:
                ctx.dmm.configure(spec.dmm_function, full_scale)
            except MultimeterError as error:
                dmm_error = str(error)
    collected = []
    readings: list[tuple[float, float]] = []
    overload = False
    ctx.operator.acquiring(spec.stage, spec.point, attempt, spec.seconds)
    with ctx.instrument.stream(a, b, chunks) as stream:
        start = ctx.clock()
        for chunk in stream:
            collected.append(chunk)
            if automatic and dmm_error is None and not overload and len(collected) % checks.DMM_EVERY_CHUNKS == 0:
                try:
                    reading = ctx.dmm.read()
                    readings.append((reading.time, reading.value))
                    ctx.dmm_rows.append(DmmRow(reading.time, spec.stage, spec.point, attempt,
                                               spec.dmm_function.value, full_scale, reading.value))
                except multimeter.MultimeterOverload:
                    overload = True
                except MultimeterError as error:
                    dmm_error = str(error)
        overcurrent = stream.overcurrent
    settling = round(checks.CAL_SETTLING_S * hw.SAMPLE_RATE)
    summaries, window, swaps, halved = {}, {}, {}, {}
    low, high = (round(f * chunks * hw.CHUNK_SAMPLES) for f in checks.TYPED_WINDOW)
    for channel in hw.CHANNELS:
        voltage = np.concatenate([c.channel(channel)[0] for c in collected])
        current = np.concatenate([c.channel(channel)[1] for c in collected])
        swaps[channel] = analysis.count_vi_swaps(voltage, current, ctx.stored, channel) or 0
        for quantity, (count, mean_shift, block_shift) in (analysis.count_halved(voltage, current, ctx.stored,
                                                                                 channel) or {}).items():
            halved[f"{channel} {quantity}"] = count
            if count:
                ctx.warnings.append(checks.halved_word_check(channel, quantity, count, mean_shift, block_shift))
        v_raw, i_raw = _raw(ctx, channel, voltage, current)
        v_word, i_word = libsmu_math.adc_words(v_raw, i_raw)
        for quantity, raw, words, step in (("V", v_raw, v_word, hw.V_STEP), ("I", i_raw, i_word, hw.ADC_I_STEP)):
            key = f"{channel} {quantity}"
            if key in spec.quantities:
                summaries[key] = analysis.summarize(raw[settling:], step, words[settling:])
                window[key] = raw[low:high]
    kept = [(t, v) for t, v in readings if t >= start + checks.CAL_SETTLING_S - 1e-9]
    return _Acquired(summaries, window, kept, swaps, overcurrent, dmm_error, full_scale, overload, halved)


def _validity(spec: PointSpec, got: _Acquired, automatic: bool) -> list[Check]:
    """What invalidates a reading (spec 3.7): it is redone whole."""
    problems = []
    for key, summary in got.summaries.items():
        if summary.jumps:
            problems.append(Check(Level.WARNING, "reading", None, "jump", {"jumps_count": summary.jumps, "quantity": key}))
        if summary.adc_limit_fraction > checks.ADC_LIMIT_FRACTION:
            problems.append(Check(Level.WARNING, "reading", None, "adc_limit",
                                  {"fraction_frac": summary.adc_limit_fraction, "quantity": key}))
    if got.overcurrent:
        problems.append(Check(Level.WARNING, "reading", None, "overcurrent"))
    if automatic and got.dmm_overload:
        unit = multimeter.UNIT[spec.dmm_function]
        problems.append(Check(Level.ERROR, "point", None, "dmm_overload",
                              {f"range_{unit.lower()}": got.full_scale, "expected": units.show(spec.dmm_expected, unit)}))
    elif automatic and got.dmm_error is not None:
        problems.append(Check(Level.WARNING, "reading", None, "dmm_failed", {"detail": got.dmm_error}))
    elif automatic and not got.dmm:
        problems.append(Check(Level.WARNING, "reading", None, "dmm_missing"))
    return problems


def _noisy(got: _Acquired) -> list[Check]:
    """Drift or noise above the margin: a warning; the operator decides."""
    found = []
    for key, summary in got.summaries.items():
        unit = "V" if key.endswith("V") else "A"
        margin = _margin(unit, summary.value)
        if abs(summary.drift) > margin or summary.u > margin:
            suffix = "v" if unit == "V" else "a"
            found.append(Check(Level.WARNING, "reading", None, "noisy",
                               {"quantity": key, f"drift_{suffix}": summary.drift, f"u_{suffix}": summary.u,
                                f"margin_{suffix}": margin}))
    return found


def _log(ctx: Context, spec: PointSpec, attempt: int, status: str, got: _Acquired,
         dmm: tuple[float, float] | None, count: int) -> None:
    for key, summary in got.summaries.items():
        channel, quantity = key.split()
        ctx.readings.append(ReadingRow(
            ctx.clock(), spec.stage, spec.point, attempt, status, channel, quantity, summary.value, summary.u,
            summary.sigma, summary.drift, summary.r1, summary.blocks, summary.jumps, summary.adc_limit_fraction,
            dmm[0] if dmm else None, dmm[1] if dmm else None, count, got.swaps.get(channel, 0),
            got.halved.get(f"{channel} V", 0) + got.halved.get(f"{channel} I", 0)))


def _stop_if_swapped(got: _Acquired) -> None:
    for channel, count in got.swaps.items():
        if count:
            raise StageStopped(Check(Level.ERROR, "reading", channel, "vi_swap", {"swaps_count": count}))


def acquire(ctx: Context, spec: PointSpec) -> Observation:
    """Measure one point until a reading is valid (spec 3.7); the operator decides on warnings."""
    if ctx.typed and spec.dmm_function is not None:
        return _acquire_typed(ctx, spec)
    automatic = spec.dmm_function is not None
    dmm_failures = 0
    for attempt in range(1, checks.MAX_ATTEMPTS + 1):
        got = _acquire_once(ctx, spec, attempt)
        _stop_if_swapped(got)
        values = [v for _, v in got.dmm]
        dmm = (float(np.mean(values)), analysis.uncertainty_of_mean(np.array(values))) if len(values) > 1 else None
        if dmm is not None and not math.isfinite(dmm[1]):
            dmm = (dmm[0], 0.0)
        problems = _validity(spec, got, automatic)
        observation = _observation(spec, got, dmm)
        if not problems and spec.physics is not None:
            problems = [c for c in spec.physics(observation) if c.level is Level.ERROR]
            ctx.warnings.extend(c for c in spec.physics(observation) if c.level is Level.WARNING)
        if not problems:
            for warning in _noisy(got):
                if not ctx.operator.accept_warning(warning):
                    problems.append(warning)
                    break
                ctx.warnings.append(warning)
        status = "accepted" if not problems else problems[0].code + ":" + problems[0].variant
        _log(ctx, spec, attempt, status, got, dmm, len(values))
        if not problems:
            return observation
        dmm_failures = dmm_failures + 1 if problems[0].variant in ("dmm_failed", "dmm_missing") else 0
        if dmm_failures >= checks.DMM_FAILURES:
            pause = Check(Level.ERROR, "stage", None, "dmm_pause", {"failures_count": dmm_failures})
            if not ctx.operator.problem(pause):
                raise StageStopped(Check(Level.ERROR, "stage", None, "stopped"))
            dmm_failures = 0
        elif problems[0].code == "point" and not ctx.operator.problem(problems[0]):
            raise StageStopped(Check(Level.ERROR, "stage", None, "stopped"))
    raise StageStopped(Check(Level.ERROR, "stage", None, "attempts",
                             {"point": spec.point, "attempts_count": checks.MAX_ATTEMPTS}))


def _observation(spec: PointSpec, got: _Acquired, dmm: tuple[float, float] | None) -> Observation:
    m1k = {key: (s.value, s.u) for key, s in got.summaries.items() if s is not None}
    return Observation(spec.point, spec.role, m1k, dmm, spec.dmm_function, got.full_scale,
                       {channel: code for channel, (_, code) in spec.drive.items()})


def _acquire_typed(ctx: Context, spec: PointSpec) -> Observation:
    """Typed mode: three acquisitions; the operator types what the display showed in the 40-60 %
    window; an odd value (a typing error) is redone (spec 3.7, C.4)."""
    full_scale = multimeter.choose_range(ctx.profile, spec.dmm_function, spec.dmm_expected)
    unit = multimeter.UNIT[spec.dmm_function]
    display_prefix = "m" if unit == "A" else ""
    typed: list[units.Typed] = []
    windows: list[dict[str, np.ndarray]] = []
    attempt = 0
    while len(typed) < checks.TYPED_READINGS or _odd(typed, unit) is not None:
        attempt += 1
        if attempt > checks.MAX_ATTEMPTS + checks.TYPED_READINGS:
            raise StageStopped(Check(Level.ERROR, "stage", None, "attempts",
                                     {"point": spec.point, "attempts_count": attempt - 1}))
        odd = _odd(typed, unit) if len(typed) >= checks.TYPED_READINGS else None
        if odd is not None:
            check = Check(Level.WARNING, "reading", None, "odd_typed",
                          {f"value_{unit.lower()}": typed[odd].value,
                           f"median_{unit.lower()}": float(np.median([t.value for t in typed]))})
            ctx.warnings.append(check)
            del typed[odd], windows[odd]
        start = ctx.clock()
        got = _acquire_once(ctx, spec, attempt)
        _stop_if_swapped(got)
        problems = _validity(spec, got, automatic=False)
        if problems:
            _log(ctx, spec, attempt, problems[0].code + ":" + problems[0].variant, got, None, 0)
            continue
        duration = spec.seconds
        window = (start + checks.TYPED_WINDOW[0] * duration, start + checks.TYPED_WINDOW[1] * duration)
        shown_unit, shown = display_decimals(ctx.profile, spec.dmm_function, full_scale)
        text = ctx.operator.typed_value(TypedRequest(spec.stage, spec.point, attempt, spec.dmm_function,
                                                     full_scale, window, shown_unit, shown))
        try:
            reading = units.read(text, unit, display_prefix)
        except units.UnitError:
            _log(ctx, spec, attempt, "typed:unreadable", got, None, 0)
            continue
        if typed_decimals(reading, shown_unit) < shown:       # its coarser resolution enters u (C.4)
            ctx.warnings.append(Check(Level.WARNING, "reading", None, "few_decimals",
                                      {"point": spec.point, "typed_count": typed_decimals(reading, shown_unit),
                                       "shown_count": shown, "unit": shown_unit}))
        ctx.dmm_rows.append(DmmRow(window[0], spec.stage, spec.point, attempt, spec.dmm_function.value,
                                   full_scale, reading.value))
        typed.append(reading)
        windows.append(got.window)
        _log(ctx, spec, attempt, "typed", got, (reading.value, reading.resolution / math.sqrt(12)), 1)
    values = np.array([t.value for t in typed])
    u_each = [t.resolution / math.sqrt(12) for t in typed]
    u = max(math.sqrt(sum(x * x for x in u_each)) / len(typed), float(np.std(values, ddof=1)) / math.sqrt(len(typed)))
    m1k = {}
    for key in spec.quantities:
        samples = np.concatenate([w[key] for w in windows])
        step = hw.V_STEP if key.endswith("V") else hw.ADC_I_STEP
        summary = analysis.summarize(samples, step)
        m1k[key] = (summary.value, summary.u)
    observation = Observation(spec.point, spec.role, m1k, (float(np.mean(values)), u), spec.dmm_function,
                              full_scale, {channel: code for channel, (_, code) in spec.drive.items()})
    if spec.physics is not None:
        found = spec.physics(observation)
        errors = [c for c in found if c.level is Level.ERROR]
        ctx.warnings.extend(c for c in found if c.level is Level.WARNING)
        if errors:
            if not ctx.operator.problem(errors[0]):
                raise StageStopped(Check(Level.ERROR, "stage", None, "stopped"))
            return _acquire_typed(ctx, spec)
    return observation


def display_decimals(profile: Profile, function: Function, full_scale: float) -> tuple[str, int]:
    """(unit, decimals) of the display on a range at full resolution: 5 decimals in V on the 34401A's
    10 V range at 6 1/2 digits, 4 decimals in mA on its 100 mA range."""
    step = multimeter.display_resolution(profile, full_scale)
    unit, scale = ("mA", 1e3) if function is Function.DC_CURRENT else ("V", 1.0)
    return unit, max(0, round(-math.log10(step * scale)))


def typed_decimals(reading: units.Typed, unit: str) -> int:
    """How many decimals the operator typed, in the display's unit."""
    scale = 1e3 if unit == "mA" else 1.0
    return max(0, round(-math.log10(reading.resolution * scale)))


def _odd(typed: list[units.Typed], unit: str) -> int | None:
    """The typed reading farther than max(3 sqrt2 u, margin) from the median of three, if any."""
    if len(typed) < checks.TYPED_READINGS:
        return None
    values = np.array([t.value for t in typed])
    median = float(np.median(values))
    worst = int(np.argmax(np.abs(values - median)))
    u = typed[worst].resolution / math.sqrt(12)
    if abs(values[worst] - median) > max(3 * math.sqrt(2) * u, _margin(unit, median)):
        return worst
    return None


# --- The physics checks of each point (spec 3.7, A.2) ---------------------------------------------


def _limit(unit: str, value: float) -> float:
    relative, absolute = checks.PLAUSIBLE_AGREEMENT[unit]
    return relative * abs(value) + absolute


def _on_gnd(channels: tuple[str, ...]) -> Callable[[Observation], list[Check]]:
    def physics(observation: Observation) -> list[Check]:
        return [Check(Level.ERROR, "point", None, "gnd_high", {"channel": f"CH {c}", "value_v": v, "limit_v": checks.GND_MAX_V})
                for c in channels for v in [observation.m1k[f"{c} V"][0]] if abs(v) > checks.GND_MAX_V]
    return physics


def _agree(channels: tuple[str, ...], quantity: str, terminal: str) -> Callable[[Observation], list[Check]]:
    """The M1K and the multimeter on the same node (or current) agree within 5 % + 20 mV / 2 mA: the multimeter on the
    wrong channel or terminal is caught."""
    unit = "V" if quantity == "V" else "A"

    def physics(observation: Observation) -> list[Check]:
        found = []
        dmm = observation.dmm[0]
        for channel in channels:
            m1k = observation.m1k[f"{channel} {quantity}"][0]
            if abs(m1k - dmm) > _limit(unit, dmm):
                suffix = unit.lower()
                found.append(Check(Level.ERROR, "point", None, "disagree",
                                   {f"m1k_{suffix}": m1k, f"dmm_{suffix}": dmm, f"limit_{suffix}": _limit(unit, dmm),
                                    "channel": f"CH {channel}", "terminal": terminal}))
        return found
    return physics


def _zero_request(channel: str, terminal: str) -> Callable[[Observation], list[Check]]:
    def physics(observation: Observation) -> list[Check]:
        dmm = observation.dmm[0]
        if abs(dmm) > checks.ZERO_REQUEST_MAX_V:
            return [Check(Level.ERROR, "point", None, "zero_request_high",
                          {"dmm_v": dmm, "limit_v": checks.ZERO_REQUEST_MAX_V, "channel": f"CH {channel}"})]
        if dmm < checks.ZERO_REQUEST_WARNING_V:
            return [Check(Level.WARNING, "point", channel, "zero_request_low", {"dmm_v": dmm})]
        return _agree((channel,), "V", terminal)(observation)
    return physics


def _open(channels: tuple[str, ...]) -> Callable[[Observation], list[Check]]:
    def physics(observation: Observation) -> list[Check]:
        return [Check(Level.ERROR, "point", None, "open_high", {"channel": f"CH {c}", "value_a": i, "limit_a": checks.OPEN_MAX_A})
                for c in channels for i in [observation.m1k[f"{c} I"][0]] if abs(i) > checks.OPEN_MAX_A]
    return physics


def _current_point(channel: str, expected: float, terminal: str) -> Callable[[Observation], list[Check]]:
    def physics(observation: Observation) -> list[Check]:
        dmm = observation.dmm[0]
        if dmm * expected <= 0 or abs(dmm - expected) > checks.EXPECTED_CURRENT_FRACTION * abs(expected):
            return [Check(Level.ERROR, "point", None, "wrong_current", {"dmm_a": dmm, "expected_a": expected})]
        return _agree((channel,), "I", terminal)(observation)
    return physics


def _source_current(channel: str, request: float) -> Callable[[Observation], list[Check]]:
    def physics(observation: Observation) -> list[Check]:
        voltage = observation.m1k[f"{channel} V"][0]
        low, high = checks.POINT_RAILS_V
        if not low <= voltage <= high:
            return [Check(Level.ERROR, "point", None, "saturated",
                          {"channel": f"CH {channel}", "value_v": voltage, "low_v": low, "high_v": high})]
        dmm = observation.dmm[0]
        if abs(dmm - request) > _limit("A", request):
            return [Check(Level.ERROR, "point", None, "request_off",
                          {"dmm_a": dmm, "request_a": request, "limit_a": _limit("A", request)})]
        return []
    return physics


# --- The four stages ------------------------------------------------------------------------------


def _wire(ctx: Context, step: WiringStep) -> None:
    if not ctx.operator.wire(step):
        raise StageStopped(Check(Level.ERROR, "stage", None, "stopped"))


def run_measure_v(ctx: Context) -> StageData:
    """measure-v: zero on GND, the 2.5V pin as the reference, the 5V pin to verify (both channels)."""
    both = hw.CHANNELS
    voltages = tuple(f"{c} V" for c in both)
    terminal = ctx.profile.terminals["voltage"]
    observations = {}
    _wire(ctx, WiringStep("measure-v", "gnd"))
    observations["zero"] = acquire(ctx, PointSpec("measure-v", "zero", "fit", checks.CAL_ZERO_S, voltages,
                                                  physics=_on_gnd(both)))
    _wire(ctx, WiringStep("measure-v", "pin", pin="2.5V"))
    observations["reference"] = acquire(ctx, PointSpec(
        "measure-v", "reference", "fit", checks.CAL_READING_S, voltages, dmm_function=Function.DC_VOLTAGE,
        dmm_expected=hw.PIN_NOMINAL["2.5V"], physics=_agree(both, "V", terminal)))
    if not ctx.skip_optional:
        _wire(ctx, WiringStep("measure-v", "pin", pin=checks.VERIFY_V_PIN))
        observations["verification"] = acquire(ctx, PointSpec(
            "measure-v", "verification", "verification", checks.CAL_READING_S, voltages,
            dmm_function=Function.DC_VOLTAGE, dmm_expected=hw.PIN_NOMINAL[checks.VERIFY_V_PIN],
            physics=_agree(both, "V", terminal)))
    return StageData("measure-v", both, observations, ctx.stored, ctx.profile)


def run_source_v(ctx: Context) -> StageData:
    """source-v, channel by channel: 0 V and 2.5 V (exact codes) to fit, 4.5 V through the new record."""
    terminal = ctx.profile.terminals["voltage"]
    observations: dict[str, Observation] = {}
    for channel in hw.CHANNELS:
        _wire(ctx, WiringStep("source-v", "dmm_voltage", channel=channel))
        keep = (f"{channel} V", f"{channel} I")
        for volts, physics in zip(checks.SOURCE_V_FIT_V, (_zero_request(channel, terminal),
                                                           _agree((channel,), "V", terminal))):
            name = f"{channel} {volts:g}V"
            observations[name] = acquire(ctx, PointSpec(
                "source-v", name, "fit", checks.CAL_READING_S, keep,
                {channel: (Role.SOURCE_V, code_for_raw(volts, Kind.SOURCE_V))},
                Function.DC_VOLTAGE, volts, physics))
        if not ctx.skip_optional:
            new = provisional_record("source-v", channel, observations, ctx.stored, ctx.profile)
            code = int(libsmu_math.dac_code(checks.VERIFY_SOURCE_V, Kind.SOURCE_V, new))
            name = f"{channel} {checks.VERIFY_SOURCE_V:g}V"
            observations[name] = acquire(ctx, PointSpec(
                "source-v", name, "verification", checks.CAL_READING_S, keep, {channel: (Role.SOURCE_V, code)},
                Function.DC_VOLTAGE, checks.VERIFY_SOURCE_V, _agree((channel,), "V", terminal)))
    return StageData("source-v", hw.CHANNELS, observations, ctx.stored, ctx.profile)


def _probe(ctx: Context, stage: str, channel: str, observations: dict[str, Observation],
           meta: dict[str, float]) -> float:
    """The load probe: the channel at 3 V and 2 V gives R and the pin's real voltage (H-04);
    returns the fit current, min(100 mA, (4.9 - pin)/R, (pin - 0.1)/R), rounded to 0.1 mA.

    The voltages are what the DAC delivers and the currents what libsmu reports, both with the stored
    calibration: on one scale, so that the pin is not off by the uncalibrated 2 % (55 mV)."""
    keep = (f"{channel} V", f"{channel} I")
    measure_i = ctx.stored[hw.record_index(channel, Kind.MEASURE_I)]
    source_v = ctx.stored[hw.record_index(channel, Kind.SOURCE_V)]
    terminal = ctx.profile.terminals["current"]
    for _ in range(checks.MAX_ATTEMPTS):
        _current_range_first(ctx, stage)
        found = []
        for volts in checks.PROBE_V:
            name = f"{channel} probe {volts:g}V"
            code = int(libsmu_math.dac_code(volts, Kind.SOURCE_V, source_v))
            observations[name] = acquire(ctx, PointSpec(stage, name, "probe", checks.CAL_PROBE_S, keep,
                                                        {channel: (Role.SOURCE_V, code)}))
            found.append((float(libsmu_math.source_output(code, Kind.SOURCE_V, source_v)),
                          corrected(Kind.MEASURE_I, observations[name].m1k[f"{channel} I"][0], measure_i)))
        (v_high, i_high), (v_low, i_low) = found
        problem = None
        if min(abs(i_high), abs(i_low)) < checks.PROBE_MIN_A or i_high == i_low:
            problem = Check(Level.ERROR, "point", None, "probe_open",
                            {"current_a": min(abs(i_high), abs(i_low)), "terminal": terminal})
        else:
            ohms = (v_high - v_low) / (i_high - i_low)
            pin = v_high - i_high * ohms
            low, high = checks.PROBE_OHMS
            if not low <= ohms <= high:
                problem = Check(Level.ERROR, "point", None, "probe_load",
                                {"ohms_ohm": ohms, "low_ohm": low, "high_ohm": high, "terminal": terminal})
            elif not checks.PROBE_PIN_V[0] <= pin <= checks.PROBE_PIN_V[1]:
                problem = Check(Level.ERROR, "point", None, "probe_pin",
                                {"pin_v": pin, "low_v": checks.PROBE_PIN_V[0], "high_v": checks.PROBE_PIN_V[1]})
        if problem is None:
            meta[f"{channel} ohms"], meta[f"{channel} pin"] = ohms, pin
            return point_current(pin, ohms)
        if not ctx.operator.problem(problem):
            raise StageStopped(Check(Level.ERROR, "stage", None, "stopped"))
    raise StageStopped(Check(Level.ERROR, "stage", None, "attempts",
                             {"point": f"{channel} probe", "attempts_count": checks.MAX_ATTEMPTS}))


def _current_range_first(ctx: Context, stage: str) -> None:
    """Before the probe, the multimeter goes to DC current on the fit points' range, so that the probe
    measures the loop the fit points will use (its shunt included) and the loop is not open because
    the meter was left in another function. Typed mode: the operator sets it."""
    full_scale = multimeter.choose_range(ctx.profile, Function.DC_CURRENT, checks.POINT_A)
    if ctx.typed:
        _wire(ctx, WiringStep(stage, "dmm_current", full_scale=full_scale))
        return
    for failures in range(1, checks.DMM_FAILURES + 1):
        try:
            ctx.dmm.configure(Function.DC_CURRENT, full_scale)
            return
        except MultimeterError as error:
            if failures == checks.DMM_FAILURES:
                pause = Check(Level.ERROR, "stage", None, "dmm_pause", {"failures_count": failures, "detail": str(error)})
                if not ctx.operator.problem(pause):
                    raise StageStopped(Check(Level.ERROR, "stage", None, "stopped")) from error
                return _current_range_first(ctx, stage)


def point_current(pin: float, ohms: float) -> float:
    low, high = checks.POINT_RAILS_V
    current = min(checks.POINT_A, (high - pin) / ohms, (pin - low) / ohms)
    return round(current / checks.POINT_ROUNDING_A) * checks.POINT_ROUNDING_A


def run_measure_i(ctx: Context) -> StageData:
    """measure-i: the zeros with the channels open (at 2.5 V, and at 0 and 4.9 V for the slope), then
    per channel the probe and +-100 mA through a resistor and the multimeter, +-10 mA to verify."""
    both = hw.CHANNELS
    keep = tuple(f"{c} {q}" for c in both for q in ("I", "V"))
    observations: dict[str, Observation] = {}
    meta: dict[str, float] = {}
    _wire(ctx, WiringStep("measure-i", "open"))
    zeros = [(f"zero {checks.ZERO_V:g}V", checks.ZERO_V, "fit")]
    if not ctx.skip_optional:
        zeros += [(f"zero {v:g}V", v, "auxiliary") for v in checks.AUX_ZERO_V]
    for name, volts, role in zeros:
        drive = {c: (Role.SOURCE_V, code_for_raw(volts, Kind.SOURCE_V)) for c in both}
        observations[name] = acquire(ctx, PointSpec("measure-i", name, role, checks.CAL_ZERO_S, keep, drive,
                                                    physics=_open(both)))
    terminal = ctx.profile.terminals["current"]
    for channel in both:
        _wire(ctx, WiringStep("measure-i", "current_loop", channel=channel))
        current = _probe(ctx, "measure-i", channel, observations, meta)
        meta[f"{channel} current"] = current
        ohms, pin = meta[f"{channel} ohms"], meta[f"{channel} pin"]
        points = [("+fit", current, "fit"), ("-fit", -current, "fit")]
        if not ctx.skip_optional:
            points += [("+verify", checks.VERIFY_A, "verification"), ("-verify", -checks.VERIFY_A, "verification")]
        source_v = ctx.stored[hw.record_index(channel, Kind.SOURCE_V)]
        for label, amps, role in points:
            code = int(libsmu_math.dac_code(pin + amps * ohms, Kind.SOURCE_V, source_v))
            name = f"{channel} {label}"
            observations[name] = acquire(ctx, PointSpec(
                "measure-i", name, role, checks.CAL_READING_S, (f"{channel} I", f"{channel} V"),
                {channel: (Role.SOURCE_V, code)}, Function.DC_CURRENT, amps,
                _current_point(channel, amps, terminal)))
    return StageData("measure-i", both, observations, ctx.stored, ctx.profile, meta)


def run_source_i(ctx: Context) -> StageData:
    """source-i, channel by channel: the probe; 0, +100 and -100 mA (DAC codes) to fit; +-10 mA through
    the new record to verify."""
    observations: dict[str, Observation] = {}
    meta: dict[str, float] = {}
    for channel in hw.CHANNELS:
        _wire(ctx, WiringStep("source-i", "current_loop", channel=channel))
        current = _probe(ctx, "source-i", channel, observations, meta)
        meta[f"{channel} current"] = current
        keep = (f"{channel} V", f"{channel} I")
        for label, amps in (("zero", 0.0), ("+fit", current), ("-fit", -current)):
            code = code_for_raw(amps, Kind.SOURCE_I)
            request = float(libsmu_math.code_value(code, Kind.SOURCE_I))
            name = f"{channel} {label}"
            observations[name] = acquire(ctx, PointSpec(
                "source-i", name, "fit", checks.CAL_READING_S, keep, {channel: (Role.SOURCE_I, code)},
                Function.DC_CURRENT, amps, _source_current(channel, request)))
        if not ctx.skip_optional:
            new = provisional_record("source-i", channel, observations, ctx.stored, ctx.profile)
            for label, amps in (("+verify", checks.VERIFY_A), ("-verify", -checks.VERIFY_A)):
                code = int(libsmu_math.dac_code(amps, Kind.SOURCE_I, new))
                name = f"{channel} {label}"
                observations[name] = acquire(ctx, PointSpec(
                    "source-i", name, "verification", checks.CAL_READING_S, keep, {channel: (Role.SOURCE_I, code)},
                    Function.DC_CURRENT, amps, _source_current(channel, amps)))
    return StageData("source-i", hw.CHANNELS, observations, ctx.stored, ctx.profile, meta)


RUNNERS = {"measure-v": run_measure_v, "source-v": run_source_v, "measure-i": run_measure_i,
           "source-i": run_source_i}


# --- The mathematics ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A named error source: measured = true + effect x error, error of standard deviation u."""

    name: str
    u: float
    effect: dict[str, float]


@dataclass(frozen=True)
class _Verification:
    name: str
    dmm_key: str
    m1k_key: str | None = None        # measure: the M1K reading of the point
    volts: float = checks.ZERO_V      # measure-i: the channel's voltage then (the zero's slope)
    code: int | None = None           # source: the DAC code sent


@dataclass(frozen=True)
class _Fit:
    """One channel of one stage: how its pairs follow from the inputs."""

    channel: str
    kind: Kind
    pairs: Callable[[dict[str, float]], tuple[list[float], list[float]]]
    texts: tuple[tuple[str, str], ...]
    slope: Callable[[dict[str, float]], float]
    verifications: tuple[_Verification, ...]


def _text(value: float, unit: str) -> str:
    return f"{value:.{DECIMALS[unit]}f}"


def _values(data: StageData) -> dict[str, float]:
    values = {}
    for name, observation in data.observations.items():
        for key, (value, _) in observation.m1k.items():
            values[f"{name}|{key}"] = value
        if observation.dmm is not None:
            values[f"{name}|dmm"] = observation.dmm[0]
    return values


def _sources(data: StageData) -> list[Source]:
    """The named error sources of a stage (C.6): the M1K's block statistics and the multimeter's
    spread at each point; each multimeter range's gain and zero, shared by its readings."""
    found = []
    by_range: dict[tuple[Function, float], list[str]] = {}
    for name, observation in data.observations.items():
        if observation.role == "probe":
            continue
        for key, (_, u) in observation.m1k.items():
            found.append(Source(f"M1K {key} at {name}", u, {f"{name}|{key}": 1.0}))
        if observation.dmm is not None:
            key = f"{name}|dmm"
            found.append(Source(f"multimeter spread at {name}", observation.dmm[1], {key: 1.0}))
            by_range.setdefault((observation.dmm_function, observation.dmm_range), []).append(key)
    values = _values(data)
    for (function, full_scale), keys in by_range.items():
        errors = multimeter.range_errors(data.profile, function, full_scale)
        found.append(Source(errors.gain_name, errors.gain_u, {key: values[key] for key in keys}))
        found.append(Source(errors.zero_name, errors.zero_u, {key: 1.0 for key in keys}))
    return found


def _fits(data: StageData) -> list[_Fit]:
    """The pairs of every channel of a stage, as functions of the inputs (true or measured)."""
    measured = _values(data)
    observations = data.observations
    fits = []
    for channel in data.channels:
        kind = KIND[data.stage]
        no_slope: Callable[[dict[str, float]], float] = lambda values: 0.0
        verifications: list[_Verification] = []
        if data.stage == "measure-v":
            def pairs(v: dict[str, float], c: str = channel) -> tuple[list[float], list[float]]:
                return [0.0, v["reference|dmm"]], [v[f"zero|{c} V"], v[f"reference|{c} V"]]
            texts = (("0.0000", _text(measured[f"zero|{channel} V"], "V")),
                     (_text(measured["reference|dmm"], "V"), _text(measured[f"reference|{channel} V"], "V")))
            slope = no_slope
            if "verification" in observations:
                verifications.append(_Verification(f"{checks.VERIFY_V_PIN} pin", "verification|dmm",
                                                   f"verification|{channel} V"))
        elif data.stage == "source-v":
            names = [f"{channel} {v:g}V" for v in checks.SOURCE_V_FIT_V]
            refs = [float(libsmu_math.code_value(observations[n].codes[channel], kind)) for n in names]

            def pairs(v: dict[str, float], r: list[float] = refs, n: list[str] = names) -> tuple[list[float], list[float]]:
                return list(r), [v[f"{name}|dmm"] for name in n]
            texts = (("0.0000", _text(measured[f"{names[0]}|dmm"], "V")),
                     (f"{refs[1]:.4f}", _text(measured[f"{names[1]}|dmm"], "V")))
            slope = no_slope
            name = f"{channel} {checks.VERIFY_SOURCE_V:g}V"
            if name in observations:
                verifications.append(_Verification(f"{checks.VERIFY_SOURCE_V:g} V", f"{name}|dmm",
                                                   code=observations[name].codes[channel]))
        elif data.stage == "measure-i":
            zero, low, high = (f"zero {v:g}V" for v in (checks.ZERO_V, *checks.AUX_ZERO_V))
            has_slope = low in observations and high in observations
            span = checks.AUX_ZERO_V[1] - checks.AUX_ZERO_V[0]

            def slope(v: dict[str, float], c: str = channel, ok: bool = has_slope, a: str = low, b: str = high) -> float:
                return (v[f"{b}|{c} I"] - v[f"{a}|{c} I"]) / span if ok else 0.0
            plus, minus = f"{channel} +fit", f"{channel} -fit"
            measure_v = data.stored[hw.record_index(channel, Kind.MEASURE_V)]
            volts = {name: corrected(Kind.MEASURE_V, measured[f"{name}|{channel} V"], measure_v)
                     for name in (plus, minus)}

            def pairs(v: dict[str, float], c: str = channel, p: str = plus, m: str = minus, z: str = zero,
                      s: Callable = slope, vv: dict = volts) -> tuple[list[float], list[float]]:
                k = s(v)
                return ([0.0, v[f"{p}|dmm"], v[f"{m}|dmm"]],
                        [v[f"{z}|{c} I"]] + [v[f"{n}|{c} I"] - k * (vv[n] - checks.ZERO_V) for n in (p, m)])
            refs, vals = pairs(measured)
            texts = (("0.0000", _text(vals[0], "A")),
                     (_text(refs[1], "A"), _text(vals[1], "A")), (_text(refs[2], "A"), _text(vals[2], "A")))
            for label in ("+verify", "-verify"):
                name = f"{channel} {label}"
                if name in observations:
                    verifications.append(_Verification(
                        f"{label[0]}{checks.VERIFY_A * 1e3:g} mA", f"{name}|dmm", f"{name}|{channel} I",
                        corrected(Kind.MEASURE_V, measured[f"{name}|{channel} V"], measure_v)))
        else:                                               # source-i
            names = [f"{channel} {label}" for label in ("zero", "+fit", "-fit")]
            refs = [float(libsmu_math.code_value(observations[n].codes[channel], kind)) for n in names]

            def pairs(v: dict[str, float], r: list[float] = refs, n: list[str] = names) -> tuple[list[float], list[float]]:
                return list(r), [v[f"{name}|dmm"] for name in n]
            texts = (("0.0000", _text(measured[f"{names[0]}|dmm"], "A")),
                     (_text(refs[1], "A"), _text(measured[f"{names[1]}|dmm"], "A")),
                     (_text(refs[2], "A"), _text(measured[f"{names[2]}|dmm"], "A")))
            slope = no_slope
            for label in ("+verify", "-verify"):
                name = f"{channel} {label}"
                if name in observations:
                    verifications.append(_Verification(f"{label[0]}{checks.VERIFY_A * 1e3:g} mA", f"{name}|dmm",
                                                       code=observations[name].codes[channel]))
        fits.append(_Fit(channel, kind, pairs, texts, slope, tuple(verifications)))
    return fits


def provisional_record(stage: str, channel: str, observations: dict[str, Observation], stored: list[Record],
                       profile: Profile) -> Record:
    """The record the fit points give so far: what libsmu will use, for the verification points."""
    fit_only = {name: o for name, o in observations.items() if o.role == "fit"}
    data = StageData(stage, (channel,), fit_only, stored, profile)
    return CalibBlock(_fits(data)[0].texts).record()


def corrected(kind: Kind, raw: float, record: Record) -> float:
    """What libsmu gives for a raw measured value (C.2; the current gain by the value's sign)."""
    difference = raw - record.offset
    if kind is Kind.MEASURE_V:
        return difference * record.gain_pos
    return difference * (record.gain_pos if difference > 0 else record.gain_neg)


def predicted_error(kind: Kind, x: float, true: Record, used: Record, slope: float = 0.0,
                    volts: float = checks.ZERO_V, code: int | None = None) -> float:
    """The error at `x` of an M1K whose true behavior is `true`, used with the record `used`.

    Measure: the raw reading of x is true.offset + x / gain (+ the zero's slope), corrected by `used`.
    Source: libsmu sends the code of x with `used`; the output is true.offset + value / gain (C.3).
    """
    if kind in (Kind.MEASURE_V, Kind.MEASURE_I):
        gain = true.gain_pos if kind is Kind.MEASURE_V or x > 0 else true.gain_neg
        raw = true.offset + x / gain + slope * (volts - checks.ZERO_V)
        return corrected(kind, raw, used) - x
    if code is None:
        code = int(libsmu_math.dac_code(x, kind, used))
    return float(libsmu_math.source_output(code, kind, true)) - x


@dataclass(frozen=True)
class EffectRow:
    x: float
    before: float                     # the error without calibration, or with the stored record
    after: float                      # predicted error with the new record
    u: float                          # its standard uncertainty (U = 2u)


@dataclass(frozen=True)
class VerificationRow:
    name: str
    residual: float
    u: float
    limit: float                      # U + one M1K step
    passed: bool


@dataclass(frozen=True)
class ChannelCalibration:
    channel: str
    kind: Kind
    block: CalibBlock                 # the pairs as written
    record: Record                    # what smu -w computes from them (float32)
    estimate: Record                  # the true record's estimate (double precision)
    record_u: tuple[float, float, float]     # standard uncertainty of offset, gain+, gain-
    before_label: str                 # "no calibration" or "stored"
    effect: tuple[EffectRow, ...]
    verifications: tuple[VerificationRow, ...]
    budget: tuple[tuple[str, float], ...]    # (source, share of the variance) at the main point
    slope: float | None = None        # measure-i: the zero's slope (A/V)


@dataclass(frozen=True)
class StageResult:
    stage: str
    channels: dict[str, ChannelCalibration]
    notes: tuple[Check, ...] = ()


def _effect_x(stage: str, values: dict[str, float]) -> list[float]:
    return [values["reference|dmm"] if x == "reference" else float(x) for x in EFFECT_VALUES[stage]]


def solve(data: StageData) -> StageResult:
    """Pairs, records, predicted errors +- U, verifications and the budget of a stage (pure)."""
    values = _values(data)
    sources = _sources(data)
    channels = {}
    notes = []
    if data.stage == "measure-i" and f"zero {checks.AUX_ZERO_V[0]:g}V" not in data.observations:
        notes.append(Check(Level.WARNING, "stage", None, "aux_skipped"))
    for fit in _fits(data):
        channels[fit.channel] = _solve_channel(data, fit, values, sources)
    return StageResult(data.stage, channels, tuple(notes))


def _solve_channel(data: StageData, fit: _Fit, values: dict[str, float], sources: list[Source]) -> ChannelCalibration:
    kind = fit.kind
    block = CalibBlock(fit.texts)
    record = block.record()
    stored = data.stored[hw.record_index(fit.channel, kind)]
    xs = _effect_x(data.stage, values)
    source = kind in (Kind.SOURCE_V, Kind.SOURCE_I)
    codes = [int(libsmu_math.dac_code(x, kind, record)) for x in xs] if source else [None] * len(xs)
    before_codes = [int(libsmu_math.dac_code(x, kind, stored)) for x in xs] if source else [None] * len(xs)
    nominal_true = libsmu_math.record_from_values(*fit.pairs(values), float32=False)

    def outputs(v: dict[str, float]) -> dict[str, float]:
        true = libsmu_math.record_from_values(*fit.pairs(v), float32=False)
        k = fit.slope(v)
        out = {"offset": true.offset, "gain+": true.gain_pos, "gain-": true.gain_neg}
        for x, code in zip(xs, codes):
            out[f"error {x!r}"] = predicted_error(kind, x, true, record, k, code=code)
        for check in fit.verifications:
            out[f"verification {check.name}"] = _expected_residual(kind, check, v, values, true, record, k,
                                                                   nominal_true, fit)
        return out

    nominal, u, contributions = propagate(outputs, values, sources)
    effect = tuple(EffectRow(x, predicted_error(kind, x, nominal_true, stored, fit.slope(values), code=c),
                             nominal[f"error {x!r}"], u[f"error {x!r}"]) for x, c in zip(xs, before_codes))
    verifications = []
    for check in fit.verifications:
        observed = _observed_residual(kind, check, values, record, nominal_true)
        key = f"verification {check.name}"
        residual = observed - nominal[key]
        limit = checks.COVERAGE * u[key] + STEP[kind]
        verifications.append(VerificationRow(check.name, residual, u[key], limit, abs(residual) <= limit))
    main = MAIN_POINT[data.stage]
    main_x = values["reference|dmm"] if main == "reference" else float(main)
    main_key = f"error {main_x!r}"
    total = u[main_key] ** 2 or 1.0
    budget = sorted(((name, parts[main_key] ** 2 / total) for name, parts in contributions.items()
                     if parts[main_key] ** 2 / total >= 1e-4), key=lambda item: -item[1])
    return ChannelCalibration(
        fit.channel, kind, block, record, nominal_true, (u["offset"], u["gain+"], u["gain-"]),
        "no calibration" if stored == Record() else "stored", effect, tuple(verifications), tuple(budget),
        fit.slope(values) if data.stage == "measure-i" else None)


def _expected_residual(kind: Kind, check: _Verification, v: dict[str, float], measured: dict[str, float],
                       true: Record, record: Record, slope: float, nominal_true: Record, fit: _Fit) -> float:
    """What the verification's residual should be if the calibration is right, given the inputs `v`
    (true values): the error of the new record there, plus the verification point's own errors."""
    dmm_true, dmm_measured = v[check.dmm_key], measured[check.dmm_key]
    if check.code is not None:                                         # source
        out_true = float(libsmu_math.source_output(check.code, kind, true))
        out_nominal = float(libsmu_math.source_output(check.code, kind, nominal_true))
        return out_true + (dmm_measured - dmm_true) - out_nominal
    m1k_true, m1k_measured = v[check.m1k_key], measured[check.m1k_key]
    error = predicted_error(kind, dmm_true, true, record, slope, check.volts)
    return error + (corrected(kind, m1k_measured, record) - corrected(kind, m1k_true, record)) - (dmm_measured - dmm_true)


def _observed_residual(kind: Kind, check: _Verification, measured: dict[str, float], record: Record,
                       nominal_true: Record) -> float:
    if check.code is not None:
        return measured[check.dmm_key] - float(libsmu_math.source_output(check.code, kind, nominal_true))
    return corrected(kind, measured[check.m1k_key], record) - measured[check.dmm_key]


def propagate(function: Callable[[dict[str, float]], dict[str, float]], values: dict[str, float],
              sources: list[Source]) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """C.6: every output's nominal value, its standard uncertainty, and each source's contribution
    (c x u), by moving each source by +-u through the same computation."""
    nominal = function(values)
    contributions = {}
    for source in sources:
        if source.u == 0 or not math.isfinite(source.u):
            continue
        plus, minus = dict(values), dict(values)
        for key, effect in source.effect.items():
            plus[key] = values[key] - effect * source.u
            minus[key] = values[key] + effect * source.u
        high, low = function(plus), function(minus)
        contributions[source.name] = {key: (high[key] - low[key]) / 2 for key in nominal}
    u = {key: math.sqrt(sum(parts[key] ** 2 for parts in contributions.values())) for key in nominal}
    return nominal, u, contributions


# --- The whole procedure --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRun:
    stages: dict[str, StageResult]
    data: dict[str, StageData]
    readings: list[ReadingRow]
    dmm_rows: list[DmmRow]
    warnings: list[Check]
    stopped: Check | None             # why the run stopped early, if it did
    multimeter_id: str
    stored: list[Record]
    interrupted: bool = False         # Ctrl+C or a signal: exit code 130, what finished is kept


def calibrate(ctx: Context, stages: list[str]) -> CalibrationRun:
    """Run the stages in order (spec 3.7). A stage that stops ends the run, and so does Ctrl+C or a
    signal (the instrument is left in HI_Z by the stream); in every case what finished is kept, with
    every reading taken so far."""
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}: {', '.join(STAGES)}")
    results: dict[str, StageResult] = {}
    datas: dict[str, StageData] = {}
    stopped = None
    interrupted = False
    identity = "typed by the operator" if ctx.typed else ""
    try:
        if not ctx.typed:
            identity = ctx.dmm.identify()
        if not ctx.operator.confirm_warm_up():
            stopped = Check(Level.ERROR, "stage", None, "stopped")
        else:
            for stage in stages:
                try:
                    data = RUNNERS[stage](ctx)
                except StageStopped as stop:
                    stopped = stop.check
                    break
                result = solve(data)
                datas[stage], results[stage] = data, result
    except KeyboardInterrupt:
        interrupted = True
        stopped = Check(Level.ERROR, "stage", None, "interrupted")
    return CalibrationRun(results, datas, ctx.readings, ctx.dmm_rows, ctx.warnings, stopped, identity,
                          ctx.stored, interrupted)


def new_blocks(run: CalibrationRun) -> dict[int, CalibBlock]:
    """This run's blocks, by their position in calib.txt."""
    return {hw.record_index(channel, result.channels[channel].kind): result.channels[channel].block
            for result in run.stages.values() for channel in result.channels}


def complete_calib(run: CalibrationRun, base: list[CalibBlock] | None) -> list[CalibBlock] | None:
    """The 8 blocks for smu -w: this run's, and the base's for the rest; None if that is not possible."""
    blocks = new_blocks(run)
    if len(blocks) == len(hw.RECORD_ORDER):
        return [blocks[index] for index in range(len(hw.RECORD_ORDER))]
    if base is None:
        return None
    return libsmu_math.merge_blocks(base, blocks)
