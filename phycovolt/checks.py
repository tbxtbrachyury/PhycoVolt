"""Every threshold of the program with its reason (spec A.2), the checks, and their English text.

Questions it answers: where is the line between "fine" and "something is wrong", and why there;
and, before a signal is switched on, is anything requested that the instrument cannot do, or that
the wiring at rest contradicts (spec 3.3.4)?

Main choices:

- No threshold lives anywhere else in the code; a test searches the other files for them.
- A check is a level (error blocks switching on; warning informs and the operator decides; note
  explains; ok), a code, the channel, a text variant and the numbers behind it. The numbers carry
  their unit in their name (`_v`, `_a`, `_ohm`, `_w`, `_hz`, `_s`, `_frac`, `_count`), so they can
  be stored as they are and written for people by `describe`. The text lives only here.
- The before-checks work on the actual block of samples, what the DAC will deliver for it, the
  circuit and the rest reading, so they hold for any waveform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from phycovolt import circuits, libsmu_math, units, waveforms
from phycovolt import hardware as hw
from phycovolt.circuits import ChannelConfig, CircuitKind, Prediction
from phycovolt.hardware import Chunk, Kind, Record, Role
from phycovolt.waveforms import BOARD, Shape

# --- Robust spread (spec 8, open item 3: provisional) ---------------------------------------------

# 1.4826 x MAD estimates the standard deviation of normal noise while ignoring outliers.
MAD_TO_SIGMA = 1.4826

# --- The V/I swap (fault F-01, spec C.5) ----------------------------------------------------------

# A swapped sample carries the other quantity's ADC word: it lands within this many codes of the
# other quantity's level (the noise is a few codes; the two levels are thousands of codes apart).
VI_SWAP_NEAR_CODES = 500
# ... and it leaves its own level by more than this many robust deviations (noise is never that far).
VI_SWAP_ROBUST_DEVIATIONS = 8
# A periodic signal's level is the median of each phase across periods: it needs at least 3 periods.
VI_SWAP_MIN_PERIODS = 3
# A constant level is the median of 1 sample in 16 (fast, and the same answer).
VI_SWAP_LEVEL_DECIMATION = 16
# At opening: 2 s with both channels at rest (spec 3.2): F-01 showed 4 to 5 swaps per second.
BOARD_CHECK_SECONDS = 2.0

# --- The halved word (fault F-02, spec 3.3.7) -----------------------------------------------------

# A halved word (one bit shifted right) sits on half its level: within this many codes of it (the
# noise of half a word is a few codes) ...
HALVED_WORD_NEAR_CODES = 32
# ... and only where half the level is far from the level itself (near zero codes it cannot be told).
HALVED_WORD_MIN_CODES = 500

# --- Readings (spec C.4) --------------------------------------------------------------------------

# A block mean more than 8 robust deviations from the median is a jump (with 5, about 0.4 % false
# alarms on current readings); the robust deviation is never taken below a tenth of an ADC step.
JUMP_ROBUST_DEVIATIONS = 8
JUMP_FLOOR_STEPS = 0.1

# --- Calibration readings (spec 3.7, A.2) ---------------------------------------------------------

CAL_SETTLING_S = 0.5                  # the start of every reading: acquisition starts, output settles
CAL_ZERO_S = 5.0                      # zeros: the multimeter is not read
CAL_READING_S = 10.0                  # with the multimeter: time for it, and the M1K's share small
CAL_PROBE_S = 2.0                     # the load probe
DMM_EVERY_CHUNKS = 3                  # the multimeter is read every 300 ms during an acquisition
TYPED_READINGS = 3                    # typed mode: the median of three points at the odd one
TYPED_WINDOW = (0.4, 0.6)             # typed mode: both instruments at the same moment
ADC_LIMIT_FRACTION = 0.01             # more than 1 % of samples at the ADC's edges biases the mean
MARGIN = {"V": (30e-6, 20e-6), "A": (5e-6, 200e-6)}   # (absolute, relative): below it nothing changes
COVERAGE = 2.0                        # U = 2 u: about 95 % for a normal distribution
AUX_ZERO_V = (0.0, 4.9)               # measure-i: the zero's slope with the channel voltage (H-07)
ZERO_V = 2.5                          # measure-i: the zero itself, mid-scale
PROBE_V = (3.0, 2.0)                  # the load probe: R and the pin's real voltage
PROBE_OHMS = (2.0, 40.0)              # a load outside is a wiring mistake
PROBE_PIN_V = (2.2, 2.7)
PROBE_MIN_A = 1e-3                    # below it the loop is open
POINT_A = 100e-3                      # the fit current, limited so the channel stays 0.1 V from the
POINT_RAILS_V = (0.1, 4.9)            # rails, rounded to 0.1 mA
POINT_ROUNDING_A = 0.1e-3
VERIFY_V_PIN = "5V"                   # measure-v verification (outside the fit, on its own pin)
VERIFY_SOURCE_V = 4.5                 # source-v verification
VERIFY_A = 10e-3                      # measure-i and source-i verifications, +-10 mA
SOURCE_V_FIT_V = (0.0, 2.5)           # source-v fit points: both are exact DAC codes (0 and 32 768)
PLAUSIBLE_AGREEMENT = {"V": (0.05, 20e-3), "A": (0.05, 2e-3)}  # M1K x multimeter: relative + absolute
GND_MAX_V = 20e-3                     # a channel on GND
OPEN_MAX_A = 5e-3                     # a channel with nothing connected
EXPECTED_CURRENT_FRACTION = 0.40      # a current point within 40 % of what the probe predicts
ZERO_REQUEST_MAX_V = 20e-3            # 0 V asked: more on the multimeter is a wiring mistake
ZERO_REQUEST_WARNING_V = 0.3e-3       # less: a negative offset would be cut by code 0 (H-05)
MAX_ATTEMPTS = 5                      # a point that fails this often stops the stage
DMM_FAILURES = 3                      # consecutive communication failures pause the stage
MONTE_CARLO_RUNS = 1000
MONTE_CARLO_FAIL = 65                 # misses per value: coverage clearly too low (~0.1 % false alarm)
MONTE_CARLO_WARN = 25                 # U exaggerated

# --- Calibration records --------------------------------------------------------------------------

# A record outside these is not from an M1K (a typing error, a misread calib.txt).
PLAUSIBLE_GAIN = (0.9, 1.1)
PLAUSIBLE_OFFSET = {"V": 50e-3, "A": 5e-3}

# --- Signals: the waveform and the burst ----------------------------------------------------------

# Bursts only (L-11): libsmu's queue holds 10 s, and every non-continuous read ends in HI_Z.
MAX_BURST_SECONDS = hw.MAX_STREAM_CHUNKS * hw.CHUNK_SECONDS
# 20 samples per period is 5 kHz, the highest frequency generated; the longest period fits the queue.
MIN_SAMPLES_PER_PERIOD = 20
# Below 100 samples per period the waveform is coarse, and the response at that frequency is not
# calibrated (the calibration is static).
FEW_SAMPLES_PER_PERIOD = 100
# DC (and a measure-only burst): one block to settle and two for the statistics.
MIN_DC_SECONDS = 0.3
# Periodic: the first period is discarded while the channel leaves HI_Z; one more is needed.
MIN_PERIODS = 2
# The real frequency (100 000 / N) may differ from the request: above 0.1 % it is a warning.
FREQUENCY_TOLERANCE = 0.001
# The mains frequency: its hum adds to a signal at the same frequency.
MAINS_HZ = 60.0
# Fewer DAC levels than this per period make a visible staircase (uA): warning. The plot draws the
# staircase below 200 levels.
DAC_STAIRCASE_LEVELS = 30
DAC_STAIRCASE_PLOT_LEVELS = 200
# DC: the DAC delivering more than 2 % away from the request (uA levels) is a warning.
DAC_LEVEL_TOLERANCE = 0.02

# --- Signals: the circuit -------------------------------------------------------------------------

# Where each return's rest voltage may fall: the pin yields under load and USB varies (H-04).
PIN_WINDOWS = {"GND": (-0.15, 0.15), "2.5V": (2.2, 2.7), "5V": (4.4, 5.3)}
# The rest reading lasts 100 ms.
REST_SECONDS = 0.1
# Above 50 kOhm the 1 MOhm input pulls the rest reading too much to judge the return.
HIGH_IMPEDANCE_OHMS = 50e3
# Compliance of a current source on a resistor: R x I must keep the channel 0.1 V from 0 and 5 V,
# with a margin on 5V for a USB supply at 4.75 V.
COMPLIANCE_DROP_V = {"2.5V": (-2.4, 2.4), "5V": (-4.65, -0.1)}
# The channel's limiter acts at 200 mA; above 150 mA the calibration's correction gets close to it.
CURRENT_ERROR_A = hw.CHANNEL_CURRENT_LIMIT
CURRENT_WARNING_A = 0.150
# Resistors heat: warn above 0.25 W, and suggest the commercial rating that holds twice the power.
POWER_WARNING_W = 0.25
POWER_RATINGS_W = (0.125, 0.25, 0.5, 1.0, 2.0, 5.0)
POWER_MARGIN = 2.0
# Below 0.5 mA of peak current the DAC step and the measured-current zero weigh (H-08).
MICRO_AMPS_A = 0.5e-3
# With the 2.5V return in SVMI, a center within 1 mV of the pin read at rest zeroes the mean current.
CHARGE_BALANCE_CENTER_V = 1e-3
# The few milliseconds at the parked code before the first sample: warn if the current exceeds the
# limiter's (L-04).
SWITCH_ON_CURRENT_A = hw.CHANNEL_CURRENT_LIMIT

# --- Signals: after the measurement --------------------------------------------------------------

SETTLING_PERIODS = 1                  # discarded: the channel leaving HI_Z
SETTLING_BLOCKS = 1
RAIL_MARGIN_V = 20e-3                 # a sample within 20 mV of 0 or 5 V touches a rail ...
RAIL_FRACTION = 0.01                  # ... and more than 1 % of them is saturation (SIMV)
AMPLITUDE_TOLERANCE = {Role.SOURCE_I: 0.05, Role.SOURCE_V: 0.02}
LEVEL_TOLERANCE = {Role.SOURCE_I: (60e-6, 0.01), Role.SOURCE_V: (2e-3, 0.002)}   # (absolute, relative)
RESISTANCE_TOLERANCE = (1.0, 0.10)    # ohm, relative: bad jumpers add 15 to 30 ohm
RESISTANCE_MIN_CURRENT_A = 20e-6      # below it the ratio of amplitudes means nothing
OPEN_CIRCUIT = (0.5, 1e-3, 0.10)      # V from the return, A expected, fraction measured (DC, SVMI)
DRIFT_LIMIT = (10.0, 0.005)           # x the uncertainty, and fraction of the excursion (heating)
INVISIBLE_RETURN_V = 75e-3            # SVMI DC: R x 60 uA above this hides the return
SHUNT_AGREEMENT = (60e-6, 0.01)       # A, relative (H-08)

# --- The simulator --------------------------------------------------------------------------------

SIM_WIRE_OHMS = 0.4                   # added to the informed circuit: realism of the rehearsals


# --- The Check type -------------------------------------------------------------------------------


class Level(Enum):
    OK = "ok"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


LEVEL_ORDER = {Level.OK: 0, Level.NOTE: 1, Level.WARNING: 2, Level.ERROR: 3}
SYMBOL = {Level.OK: "✓", Level.NOTE: "◆", Level.WARNING: "⚠", Level.ERROR: "✗"}


@dataclass(frozen=True)
class Check:
    """One check: its level, code, channel ("A", "B" or None), text variant and numbers."""

    level: Level
    code: str
    channel: str | None = None
    variant: str = "ok"
    numbers: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return describe(self)


def worst(checks: list[Check]) -> Level:
    return max((check.level for check in checks), key=LEVEL_ORDER.get, default=Level.OK)


def blocks_switch_on(checks: list[Check]) -> bool:
    return any(check.level is Level.ERROR for check in checks)


# --- Texts ----------------------------------------------------------------------------------------

TEXTS = {
    ("parameters", "invalid"): "The waveform cannot be built: {problem}.",
    ("frequency", "ok"): "{samples} samples per period: exactly {real}.",
    ("frequency", "rounded"): "{requested} needs a whole number of samples per period ({samples}): "
                              "the real frequency is {real}, {error} off.",
    ("mains", "at_mains"): "The signal is at the mains frequency ({real}): the hum adds to it and "
                           "cannot be told apart from it.",
    ("mains", "submultiple"): "The mains frequency is {ratio} times the signal's: the hum falls on one "
                              "of its harmonics.",
    ("samples_per_period", "too_few"): "{samples} samples per period: the program generates at most "
                                       "5 kHz ({minimum} samples per period).",
    ("samples_per_period", "few"): "Only {samples} samples per period: the waveform is coarse, and the "
                                   "M1K's response at this frequency is not calibrated.",
    ("duration", "too_long"): "The burst would last {seconds}; the most is {maximum} (bursts only, L-11).",
    ("duration", "short_dc"): "A DC measurement needs at least {minimum}: one block to settle and two "
                              "for the statistics.",
    ("duration", "short_measure"): "A measurement needs at least {minimum}: one block to settle and "
                                   "two for the statistics.",
    ("duration", "too_long_measure"): "The measurement would last {seconds}; the most is {maximum} "
                                      "(bursts only, L-11).",
    ("duration", "few_periods"): "At least {minimum} periods are needed: the first one is discarded "
                                 "while the channel settles.",
    ("dac_staircase", "ok"): "The DAC delivers {levels} levels per period.",
    ("dac_staircase", "staircase"): "The DAC delivers only {levels} levels per period: the output is a "
                                    "staircase, and its mean moves by {mean_shift}.",
    ("dac_level", "ok"): "The DAC delivers {delivered} for the {requested} requested.",
    ("dac_level", "off"): "The DAC delivers {delivered} for the {requested} requested ({error} off): "
                          "its step is large at this level.",
    ("calibration", "ok"): "The {kind} record is calibrated.",
    ("calibration", "default"): "The EEPROM holds no {kind} calibration for this channel: values can be "
                                "off by a few percent (H-02).",
    ("calibration", "implausible"): "The stored {kind} record is not plausible (offset {offset}, "
                                    "gain+ {gain_pos}, gain- {gain_neg}).",
    ("wiring", "ok"): "At rest the channel reads {rest}: the {chosen} return is there.",
    ("wiring", "other_pin"): "At rest the channel reads {rest}, the {seen} pin, not the {chosen} return "
                             "chosen: check the return wire.",
    ("wiring", "far"): "At rest the channel reads {rest}, far from every pin: the channel may be open, "
                       "the circuit may be on the other channel, or electrodes may hold a voltage.",
    ("wiring", "far_other"): "At rest the channel reads {rest}, far from every pin, while CH {other} "
                             "is at the {chosen} return: the circuit may be on CH {other}.",
    ("wiring", "high_impedance"): "The circuit has {ohms}: the 1 MΩ input pulls the rest reading, "
                                  "and the return cannot be judged.",
    ("pin_limit", "ok"): "The channel stays between {v_min} and {v_max}, inside 0-5 V.",
    ("pin_limit", "out"): "R × I would span {drop_min} to {drop_max}: with the {chosen} return and "
                          "{ohms}, the current must stay between {i_min} and {i_max}.",
    ("pin_limit", "out_periodic"): "R × I would span {drop_min} to {drop_max}: with the {chosen} return "
                                   "and {ohms}, the current must stay between {i_min} and {i_max} "
                                   "(a center of {center} with at most {amplitude} of amplitude).",
    ("range", "ok"): "The waveform stays inside 0-5 V.",
    ("range", "out"): "The waveform goes from {low} to {high}; a channel sources 0 to 5 V.",
    ("dac_floor", "below"): "The waveform asks for {low}, but code 0 already gives {floor} (the "
                            "output's offset, H-05): anything lower comes out as {floor}.",
    ("dac_ceiling", "above"): "The waveform asks for {high}, but the top code only gives {ceiling} (H-16): "
                              "anything higher comes out as {ceiling}.",
    ("current_limit", "ok"): "The current peaks at {peak}.",
    ("current_limit", "near"): "The current would peak at {peak}, close to the 200 mA limit.",
    ("current_limit", "over"): "The current would peak at {peak}: the limiter acts above 200 mA.",
    ("power", "ok"): "Every resistor dissipates less than {limit}.",
    ("power", "hot"): "The {part} would dissipate {power}: use one rated {rating} or more.",
    ("power", "too_hot"): "The {part} would dissipate {power}: more than any rating offered; reduce "
                          "the current.",
    ("switch_on", "parked"): "Before the first sample the DAC is parked at the zero code, which gives "
                             "{current} on this channel (its source offset).",
    ("switch_on", "svmi"): "For a few milliseconds before the first sample the channel sits at "
                           "{voltage}: {current} through the circuit.",
    ("micro_amps", "small"): "The current peaks at {peak}: at this level the DAC step ({step}) and the "
                             "measured-current zero weigh (H-08).",
    ("charge_balance", "ok"): "The mean current is about zero.",
    ("charge_balance", "mean"): "The mean current is {mean}: a net charge on the electrodes.",
    ("charge_balance", "center"): "The mean current is {mean}; a center of {center} (the return read "
                                  "at rest) zeroes it.",
    ("charge_balance", "board"): "The mean current is {mean} at the M1K, {culture} at the culture.",
    ("hi_z_range", "ok"): "The node stays inside 0-5 V.",
    ("hi_z_range", "out"): "The node would reach {low} to {high}; a channel measures 0 to 5 V.",
}

_SUFFIX_UNITS = {"_v": "V", "_a": "A", "_ohm": "Ω", "_w": "W", "_hz": "Hz", "_s": "s", "_frac": "%"}


def _shown(key: str, value: object) -> tuple[str, str]:
    """(name without the unit suffix, the value written for people)."""
    if isinstance(value, str):
        return key, value
    for suffix, unit in _SUFFIX_UNITS.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], units.show(float(value), unit)
    if key.endswith("_count"):
        return key[: -len("_count")], f"{value:g}"
    return key, f"{value:.6g}"


def describe(check: Check) -> str:
    """The English text of a check, with its numbers written for people."""
    numbers = dict(_shown(key, value) for key, value in check.numbers.items())
    channel = f"CH {check.channel}: " if check.channel else ""
    return channel + TEXTS[(check.code, check.variant)].format(**numbers)


# --- Records --------------------------------------------------------------------------------------


def record_problem(record: Record, kind: Kind) -> str | None:
    """None, "default" (offset 0, gains 1: nothing stored) or "implausible"."""
    if record == Record():
        return "default"
    gain_low, gain_high = PLAUSIBLE_GAIN
    unit = "A" if kind in (Kind.MEASURE_I, Kind.SOURCE_I) else "V"
    gains_ok = all(gain_low <= gain <= gain_high for gain in (record.gain_pos, record.gain_neg))
    if not gains_ok or abs(record.offset) > PLAUSIBLE_OFFSET[unit]:
        return "implausible"
    return None


def _calibration_check(channel: str, kind: Kind, record: Record) -> Check:
    problem = record_problem(record, kind)
    numbers: dict[str, float | int | str] = {"kind": kind.value}
    if problem == "implausible":
        unit = "a" if kind in (Kind.MEASURE_I, Kind.SOURCE_I) else "v"
        numbers.update({f"offset_{unit}": record.offset, "gain_pos": record.gain_pos,
                        "gain_neg": record.gain_neg})
    return Check(Level.WARNING if problem else Level.OK, "calibration", channel, problem or "ok", numbers)


def stored_calibration_checks(records: list[Record]) -> list[Check]:
    """The 8 stored records (`phycovolt status`): warning at the default, error when implausible."""
    checks = []
    for (channel, kind), record in zip(hw.RECORD_ORDER, records):
        check = _calibration_check(channel, kind, record)
        if check.variant == "implausible":
            check = Check(Level.ERROR, check.code, check.channel, check.variant, check.numbers)
        checks.append(check)
    return checks


# --- Before switching on (spec 3.3.4) -------------------------------------------------------------


def rest_voltages(rest: Chunk) -> dict[str, float]:
    """The median voltage of each channel in the rest reading."""
    return {"A": float(np.median(rest.v_a)), "B": float(np.median(rest.v_b))}


def _window(return_name: str, board_input: float) -> tuple[float, float]:
    if return_name == BOARD:                  # the board holds its input like the 2.5V pin
        low, high = PIN_WINDOWS["2.5V"]
        shift = board_input - hw.PIN_NOMINAL["2.5V"]
        return low + shift, high + shift
    return PIN_WINDOWS[return_name]


def _inside(value: float, window: tuple[float, float]) -> bool:
    return window[0] <= value <= window[1]


def return_voltage(config: ChannelConfig, rest_voltage: float, open_input: float) -> float:
    """The return's real voltage: the rest reading (divider undone) when it falls in the chosen
    return's window; otherwise the nominal value (and the wiring check says why)."""
    circuit = config.circuit
    if circuit.ret is None:
        raise ValueError(f"{circuit.kind.value} has no return")
    if circuit.ohms <= HIGH_IMPEDANCE_OHMS:
        pin = circuits.pin_from_rest(rest_voltage, circuit.ohms, open_input)
        if _inside(pin, _window(circuit.ret, circuit.board_input)):
            return pin
    return float(circuit.nominal_return)


def before_checks(a: ChannelConfig, b: ChannelConfig, rest: Chunk, records: list[Record],
                  open_input: float | None = None, measure_seconds: float | None = None) -> list[Check]:
    """Every before-check of both channels, from the rest reading and the stored calibration.

    `open_input` is the voltage of an open HI_Z input (unit.ini), used to undo the 1 MOhm divider.
    `measure_seconds` is the duration of a measure-only burst (both channels hi-z).
    """
    circuits.check_pair(a, b)
    if open_input is None:
        open_input = hw.load_unit().open_input
    configs = {"A": a, "B": b}
    resting = rest_voltages(rest)
    predictions: dict[str, Prediction] = {}
    checks: list[Check] = []
    for name, config in configs.items():
        if config.role is Role.HI_Z:
            continue
        try:
            predictions[name] = circuits.predict(config, name, records,
                                                 return_voltage(config, resting[name], open_input))
        except ValueError as error:
            checks.append(Check(Level.ERROR, "parameters", name, "invalid", {"problem": str(error)}))
    for name, config in configs.items():
        other = "B" if name == "A" else "A"
        checks += _wiring(name, config, other, configs[other], resting, open_input)
        if name in predictions:
            checks += _source_checks(name, config, predictions[name], records)
        elif config.circuit.kind is CircuitKind.SHUNT_NODE and other in predictions:
            checks.append(_hi_z_range(name, configs[other], predictions[other]))
    if a.role is Role.HI_Z and b.role is Role.HI_Z and measure_seconds is not None:
        if measure_seconds < MIN_DC_SECONDS - 1e-9:
            checks.append(Check(Level.ERROR, "duration", None, "short_measure", {"minimum_s": MIN_DC_SECONDS}))
        elif measure_seconds > MAX_BURST_SECONDS + 1e-9:
            checks.append(Check(Level.ERROR, "duration", None, "too_long_measure",
                                {"seconds_s": measure_seconds, "maximum_s": MAX_BURST_SECONDS}))
    return checks


def _wiring(name: str, config: ChannelConfig, other: str, other_config: ChannelConfig,
            resting: dict[str, float], open_input: float) -> list[Check]:
    circuit = config.circuit
    if circuit.ret is None:
        return []
    if circuit.ohms > HIGH_IMPEDANCE_OHMS:
        return [Check(Level.NOTE, "wiring", name, "high_impedance", {"ohms_ohm": circuit.ohms})]
    pin = circuits.pin_from_rest(resting[name], circuit.ohms, open_input)
    chosen = circuit.ret
    numbers: dict[str, float | int | str] = {"rest_v": resting[name], "chosen": chosen}
    if _inside(pin, _window(chosen, circuit.board_input)):
        return [Check(Level.OK, "wiring", name, "ok", numbers)]
    for seen in hw.PINS:
        if seen != chosen and _inside(pin, PIN_WINDOWS[seen]):
            level = Level.ERROR if config.role is Role.SOURCE_V else Level.WARNING
            return [Check(level, "wiring", name, "other_pin", {**numbers, "seen": seen})]
    if _inside(resting[other], _window(chosen, circuit.board_input)):
        return [Check(Level.WARNING, "wiring", name, "far_other", {**numbers, "other": other})]
    return [Check(Level.WARNING, "wiring", name, "far", numbers)]


def _source_checks(name: str, config: ChannelConfig, prediction: Prediction,
                   records: list[Record]) -> list[Check]:
    waveform = config.waveform
    timing = prediction.timing
    unit = "a" if config.role is Role.SOURCE_I else "v"
    checks: list[Check] = []
    if waveform.periodic:
        checks += _frequency_checks(name, waveform["freq"], timing)
        levels = int(np.unique(prediction.codes).size)
        requested_levels = int(np.unique(prediction.requested).size)
        shift = float(np.mean(prediction.delivered) - np.mean(prediction.requested))
        staircase = levels < min(DAC_STAIRCASE_LEVELS, requested_levels)
        checks.append(Check(Level.WARNING if staircase else Level.OK, "dac_staircase", name,
                            "staircase" if staircase else "ok",
                            {"levels_count": levels, f"mean_shift_{unit}": shift}))
    checks += _duration_checks(name, waveform, timing)
    if not waveform.periodic:
        requested, delivered = float(prediction.requested[0]), float(prediction.delivered[0])
        error = abs(delivered - requested) / abs(requested) if requested else 0.0
        off = error > DAC_LEVEL_TOLERANCE
        checks.append(Check(Level.WARNING if off else Level.OK, "dac_level", name, "off" if off else "ok",
                            {f"requested_{unit}": requested, f"delivered_{unit}": delivered,
                             "error_frac": error}))
    kind = hw.SOURCE_KIND[config.role]
    checks.append(_calibration_check(name, kind, records[hw.record_index(name, kind)]))
    if config.role is Role.SOURCE_I:
        checks.append(_pin_limit(name, config, prediction))
        peak = float(np.max(np.abs(prediction.delivered)))
        if peak < MICRO_AMPS_A:
            checks.append(Check(Level.NOTE, "micro_amps", name, "small", {"peak_a": peak, "step_a": hw.DAC_I_STEP}))
    else:
        low, high = float(np.min(prediction.requested)), float(np.max(prediction.requested))
        out = low < hw.V_MIN or high > hw.V_MAX
        checks.append(Check(Level.ERROR if out else Level.OK, "range", name, "out" if out else "ok",
                            {"low_v": low, "high_v": high}))
        record = records[hw.record_index(name, Kind.SOURCE_V)]
        floor = float(libsmu_math.source_output(0, Kind.SOURCE_V, record))
        if low < floor:
            checks.append(Check(Level.WARNING, "dac_floor", name, "below", {"low_v": low, "floor_v": floor}))
        ceiling = float(libsmu_math.source_output(hw.CODE_MAX, Kind.SOURCE_V, record))
        if high > ceiling:                                    # H-16: the top code is not 5 V
            checks.append(Check(Level.WARNING, "dac_ceiling", name, "above", {"high_v": high, "ceiling_v": ceiling}))
    checks.append(_current_limit(name, prediction))
    checks.append(_power(name, config, prediction))
    checks.append(_switch_on(name, config, prediction, records))
    checks.append(_charge_balance(name, config, prediction))
    return checks


def _frequency_checks(name: str, requested: float, timing: waveforms.Timing) -> list[Check]:
    real = float(timing.frequency)
    samples = timing.samples_per_period
    error = abs(real - requested) / requested
    numbers = {"requested_hz": requested, "real_hz": real, "samples_count": samples, "error_frac": error}
    if error > FREQUENCY_TOLERANCE:
        checks = [Check(Level.WARNING, "frequency", name, "rounded", numbers)]
    elif error > 0:
        checks = [Check(Level.NOTE, "frequency", name, "rounded", numbers)]
    else:
        checks = [Check(Level.OK, "frequency", name, "ok", numbers)]
    # 60 Hz itself is 1666.67 samples: the real 59.988 Hz is still the mains for a 10 s burst.
    ratio = MAINS_HZ / real
    if math.isclose(real, MAINS_HZ, rel_tol=FREQUENCY_TOLERANCE):
        checks.append(Check(Level.WARNING, "mains", name, "at_mains", {"real_hz": real}))
    elif ratio > 1.5 and math.isclose(ratio, round(ratio), rel_tol=FREQUENCY_TOLERANCE):
        checks.append(Check(Level.NOTE, "mains", name, "submultiple", {"ratio_count": round(ratio)}))
    if samples < MIN_SAMPLES_PER_PERIOD:
        checks.append(Check(Level.ERROR, "samples_per_period", name, "too_few",
                            {"samples_count": samples, "minimum_count": MIN_SAMPLES_PER_PERIOD}))
    elif samples < FEW_SAMPLES_PER_PERIOD:
        checks.append(Check(Level.WARNING, "samples_per_period", name, "few", {"samples_count": samples}))
    return checks


def _duration_checks(name: str, waveform: waveforms.Waveform, timing: waveforms.Timing) -> list[Check]:
    checks = []
    if timing.seconds > MAX_BURST_SECONDS + 1e-9:
        checks.append(Check(Level.ERROR, "duration", name, "too_long",
                            {"seconds_s": timing.seconds, "maximum_s": MAX_BURST_SECONDS}))
    if not waveform.periodic and timing.seconds < MIN_DC_SECONDS - 1e-9:
        checks.append(Check(Level.ERROR, "duration", name, "short_dc", {"minimum_s": MIN_DC_SECONDS}))
    if waveform.periodic and waveform["periods"] < MIN_PERIODS:
        checks.append(Check(Level.ERROR, "duration", name, "few_periods", {"minimum_count": MIN_PERIODS}))
    return checks


def _pin_limit(name: str, config: ChannelConfig, prediction: Prediction) -> Check:
    circuit = config.circuit
    drop = prediction.delivered * circuit.ohms
    voltage = prediction.voltage
    ok_numbers = {"v_min_v": float(np.min(voltage)), "v_max_v": float(np.max(voltage))}
    if circuit.ret not in COMPLIANCE_DROP_V:               # the board holds its input: no drop
        return Check(Level.OK, "pin_limit", name, "ok", ok_numbers)
    low, high = COMPLIANCE_DROP_V[circuit.ret]
    drop_min, drop_max = float(np.min(drop)), float(np.max(drop))
    if low <= drop_min and drop_max <= high:
        return Check(Level.OK, "pin_limit", name, "ok", ok_numbers)
    numbers = {"drop_min_v": drop_min, "drop_max_v": drop_max, "chosen": circuit.ret,
               "ohms_ohm": circuit.ohms, "i_min_a": low / circuit.ohms, "i_max_a": high / circuit.ohms}
    if config.waveform.periodic:
        numbers.update({"center_a": (low + high) / 2 / circuit.ohms,
                        "amplitude_a": (high - low) / 2 / circuit.ohms})
        return Check(Level.ERROR, "pin_limit", name, "out_periodic", numbers)
    return Check(Level.ERROR, "pin_limit", name, "out", numbers)


def _current_limit(name: str, prediction: Prediction) -> Check:
    peak = float(np.max(np.abs(prediction.current)))
    if peak > CURRENT_ERROR_A:
        return Check(Level.ERROR, "current_limit", name, "over", {"peak_a": peak})
    if peak > CURRENT_WARNING_A:
        return Check(Level.WARNING, "current_limit", name, "near", {"peak_a": peak})
    return Check(Level.OK, "current_limit", name, "ok", {"peak_a": peak})


def _power(name: str, config: ChannelConfig, prediction: Prediction) -> Check:
    hottest = None
    for part, power in circuits.resistor_powers(config, prediction.current).items():
        if power > POWER_WARNING_W and (hottest is None or power > hottest[1]):
            hottest = (part, power)
    if hottest is None:
        return Check(Level.OK, "power", name, "ok", {"limit_w": POWER_WARNING_W})
    part, power = hottest
    ratings = [rating for rating in POWER_RATINGS_W if rating >= POWER_MARGIN * power]
    if not ratings:
        return Check(Level.WARNING, "power", name, "too_hot", {"part": part, "power_w": power})
    return Check(Level.WARNING, "power", name, "hot", {"part": part, "power_w": power, "rating_w": ratings[0]})


def _switch_on(name: str, config: ChannelConfig, prediction: Prediction, records: list[Record]) -> Check:
    kind = hw.SOURCE_KIND[config.role]
    rest_output = float(libsmu_math.source_output(hw.DAC_REST_CODE, kind, records[hw.record_index(name, kind)]))
    if config.role is Role.SOURCE_I:
        return Check(Level.NOTE, "switch_on", name, "parked", {"current_a": rest_output})
    current = (rest_output - prediction.return_voltage) / config.circuit.ohms
    level = Level.WARNING if abs(current) > SWITCH_ON_CURRENT_A else Level.NOTE
    return Check(level, "switch_on", name, "svmi", {"voltage_v": rest_output, "current_a": current})


def _charge_balance(name: str, config: ChannelConfig, prediction: Prediction) -> Check:
    mean_current = float(np.mean(prediction.current))
    if config.role is Role.SOURCE_V:
        offset = float(np.mean(prediction.delivered)) - prediction.return_voltage
        if abs(offset) <= CHARGE_BALANCE_CENTER_V:
            return Check(Level.OK, "charge_balance", name, "ok", {"mean_a": mean_current})
        if config.circuit.ret == "2.5V" and "center" in config.waveform.values:
            center = config.waveform["center"] - offset
            return Check(Level.NOTE, "charge_balance", name, "center", {"mean_a": mean_current, "center_v": center})
        return Check(Level.NOTE, "charge_balance", name, "mean", {"mean_a": mean_current})
    if abs(mean_current) < hw.DAC_I_STEP:
        return Check(Level.OK, "charge_balance", name, "ok", {"mean_a": mean_current})
    if config.circuit.kind is CircuitKind.IVI_BOARD:
        culture = float(np.mean(circuits.culture_current(config, prediction.current)))
        return Check(Level.NOTE, "charge_balance", name, "board", {"mean_a": mean_current, "culture_a": culture})
    return Check(Level.NOTE, "charge_balance", name, "mean", {"mean_a": mean_current})


def _hi_z_range(name: str, other_config: ChannelConfig, other: Prediction) -> Check:
    node = circuits.shunt_node_voltage(other_config, other.current, other.return_voltage)
    low, high = float(np.min(node)), float(np.max(node))
    out = low < hw.V_MIN or high > hw.V_MAX
    return Check(Level.ERROR if out else Level.OK, "hi_z_range", name, "out" if out else "ok",
                 {"low_v": low, "high_v": high})


# --- After the measurement (spec 3.3.7) -----------------------------------------------------------


@dataclass(frozen=True)
class Observed:
    """What the analysis found on one channel, as plain numbers for the after-checks (V, A).

    Levels and amplitudes are the sine fit's for a sine, the cycle average's for other periodic
    waves (amplitude = half the peak-to-peak), the reading summary's for DC (amplitude None).
    """

    vi_swaps: int | None              # None: could not be told (fewer than 3 periods)
    v_level: float | None
    i_level: float | None
    v_amplitude: float | None = None
    i_amplitude: float | None = None
    requested_amplitude: float | None = None     # of the imposed quantity
    delivered_level: float | None = None         # DC: what the DAC delivers
    rail_fraction: float = 0.0                   # samples within 20 mV of a rail
    other_drift: float | None = None             # DC: the measured (not imposed) quantity
    other_u: float | None = None
    return_voltage: float | None = None          # the return read at rest
    shunt_level: float | None = None             # hi-z shunt-node: the shunt's current
    shunt_amplitude: float | None = None
    halved: dict[str, tuple[int, float, float]] | None = None   # F-02 by quantity ("V", "I"): (count,
                                                                # shift of the level, of a 100 ms mean)


TEXTS.update({
    ("analysis", "too_short"): "Too little was measured to analyze (the first period, or block, is "
                               "discarded while the channel settles).",
    ("board", "ok"): "No V/I swap in the samples.",
    ("board", "swapped"): "{swaps} samples carry the V/I swap (fault F-01): the measurement is not "
                          "valid. Unplug the M1K's USB cable, wait a few seconds and plug it in again.",
    ("board", "unknown"): "With fewer than 3 periods the V/I swap cannot be looked for.",
    ("halved_word", "ok"): "No halved ADC word in the samples.",
    ("halved_word", "found"): "{count} {quantity} samples carry a halved ADC word (fault F-02): they move a "
                              "100 ms mean by up to {block_shift} and the whole level by {mean_shift}. If "
                              "they keep appearing, unplug the M1K's USB cable, wait and plug it in again.",
    ("halved_word", "found_one"): "1 {quantity} sample carries a halved ADC word (fault F-02): it moves a "
                                  "100 ms mean by {block_shift} and the whole level by {mean_shift}. If it "
                                  "keeps appearing, unplug the M1K's USB cable, wait and plug it in again.",
    ("amplitude", "ok"): "Amplitude {measured} for {requested} requested ({error} off).",
    ("amplitude", "off"): "Amplitude {measured} for {requested} requested ({error} off): {cause}.",
    ("level", "ok"): "Level {measured} for {delivered} delivered by the DAC.",
    ("level", "off"): "Level {measured} for {delivered} delivered by the DAC ({difference} off).",
    ("pin_limit", "saturated"): "{fraction} of the samples touch 0 V or 5 V: the current source saturated "
                                "and the current fell short.",
    ("pin_limit", "clear"): "The channel stayed away from 0 V and 5 V.",
    ("overcurrent", "tripped"): "The limiter acted (overcurrent recorded by the M1K).",
    ("resistance", "ok"): "Resistance seen {seen} for {informed} informed.",
    ("resistance", "off"): "Resistance seen {seen} for {informed} informed: check the jumpers and "
                           "measure the resistors (bad jumpers add 15 to 30 Ω).",
    ("resistance", "open"): "Almost no current ({measured} for {expected} expected): the load seems off "
                            "the channel.",
    ("return", "ok"): "The other end of the circuit is at {estimate}: the {chosen} return.",
    ("return", "other"): "The other end of the circuit looks like {estimate}, the {seen} pin, not the "
                         "{chosen} return (or the resistance is not the one informed).",
    ("return", "far"): "The other end of the circuit looks like {estimate}, far from the {chosen} return "
                       "(or the resistance is not the one informed).",
    ("return", "invisible"): "With {ohms} the measured-current zero hides the return: it cannot be seen.",
    ("drift", "ok"): "The measured quantity drifted by {drift}.",
    ("drift", "drifting"): "The measured quantity drifted by {drift} ({excursion} excursion): resistor "
                           "heating? A higher-wattage resistor helps.",
    ("shunt_agreement", "ok"): "The shunt's current agrees with CH {other}'s ({difference} apart).",
    ("shunt_agreement", "off"): "The shunt's current differs from CH {other}'s by {difference}.",
})


def _seen_pin(voltage: float) -> str | None:
    for pin in hw.PINS:
        if _inside(voltage, PIN_WINDOWS[pin]):
            return pin
    return None


def after_checks(a: ChannelConfig, b: ChannelConfig, observed: dict[str, Observed],
                 overcurrent: bool) -> list[Check]:
    """Every after-check of both channels (spec 3.3.7)."""
    configs = {"A": a, "B": b}
    checks: list[Check] = []
    for name, config in configs.items():
        seen = observed[name]
        other = "B" if name == "A" else "A"
        if seen.vi_swaps is None:
            checks.append(Check(Level.NOTE, "board", name, "unknown"))
        elif seen.vi_swaps:
            checks.append(Check(Level.ERROR, "board", name, "swapped", {"swaps_count": seen.vi_swaps}))
        else:
            checks.append(Check(Level.OK, "board", name, "ok"))
        checks.extend(halved_word_checks(name, seen.halved))
        if seen.v_level is None or seen.i_level is None:
            checks.append(Check(Level.NOTE, "analysis", name, "too_short"))
            continue
        if config.role is Role.HI_Z:
            if config.circuit.kind is CircuitKind.SHUNT_NODE and observed[other].i_level is not None \
                    and seen.shunt_level is not None:
                checks.append(_shunt_agreement(name, other, seen, observed[other]))
            continue
        if overcurrent:
            checks.append(Check(Level.ERROR, "overcurrent", name, "tripped"))
        if config.waveform.periodic:
            checks.append(_amplitude(name, config, seen, overcurrent))
        else:
            checks.append(_level(name, config, seen))
        if config.role is Role.SOURCE_I:
            saturated = seen.rail_fraction > RAIL_FRACTION
            checks.append(Check(Level.ERROR if saturated else Level.OK, "pin_limit", name,
                                "saturated" if saturated else "clear", {"fraction_frac": seen.rail_fraction}))
        if config.circuit.kind is not CircuitKind.IVI_BOARD:
            checks.extend(_resistance(name, config, seen))
        checks.append(_return(name, config, seen))
        if not config.waveform.periodic:
            checks.append(_drift(name, config, seen))
    return checks


def halved_word_checks(name: str, halved: dict[str, tuple[int, float, float]] | None) -> list[Check]:
    """F-02 on one channel: a warning per quantity with halved words, or ok (spec 3.3.7)."""
    if halved is None:
        return []
    found = [halved_word_check(name, quantity, *values) for quantity, values in halved.items() if values[0]]
    return found or [Check(Level.OK, "halved_word", name, "ok")]


def halved_word_check(name: str, quantity: str, count: int, mean_shift: float, block_shift: float) -> Check:
    """The warning for `count` halved words of one quantity ("V" or "I") on a channel."""
    unit = "v" if quantity == "V" else "a"
    return Check(Level.WARNING, "halved_word", name, "found_one" if count == 1 else "found",
                 {"count_count": count, "quantity": "voltage" if quantity == "V" else "current",
                  f"mean_shift_{unit}": mean_shift, f"block_shift_{unit}": block_shift})


def _imposed(config: ChannelConfig, seen: Observed) -> tuple[float | None, float | None]:
    """(level, amplitude) of the quantity the channel imposes."""
    if config.role is Role.SOURCE_I:
        return seen.i_level, seen.i_amplitude
    return seen.v_level, seen.v_amplitude


def _amplitude(name: str, config: ChannelConfig, seen: Observed, overcurrent: bool) -> Check:
    unit = "a" if config.role is Role.SOURCE_I else "v"
    measured = _imposed(config, seen)[1]
    requested = seen.requested_amplitude
    error = abs(measured - requested) / abs(requested) if requested else 0.0
    numbers = {f"measured_{unit}": measured, f"requested_{unit}": requested, "error_frac": error}
    if error <= AMPLITUDE_TOLERANCE[config.role]:
        return Check(Level.OK, "amplitude", name, "ok", numbers)
    if overcurrent:
        cause = "the limiter acted"
    elif config.role is Role.SOURCE_I and seen.rail_fraction > RAIL_FRACTION:
        cause = "the pin reached its limit"
    elif config.waveform["freq"] * FEW_SAMPLES_PER_PERIOD > hw.SAMPLE_RATE:
        cause = "the frequency is high for the M1K"
    elif config.role is Role.SOURCE_I and requested < MICRO_AMPS_A:
        cause = "at microamperes the DAC step and the current zero weigh"
    else:
        cause = "check the circuit"
    return Check(Level.WARNING, "amplitude", name, "off", {**numbers, "cause": cause})


def _level(name: str, config: ChannelConfig, seen: Observed) -> Check:
    unit = "a" if config.role is Role.SOURCE_I else "v"
    measured = _imposed(config, seen)[0]
    delivered = seen.delivered_level
    absolute, relative = LEVEL_TOLERANCE[config.role]
    difference = measured - delivered
    off = abs(difference) > absolute + relative * abs(delivered)
    return Check(Level.WARNING if off else Level.OK, "level", name, "off" if off else "ok",
                 {f"measured_{unit}": measured, f"delivered_{unit}": delivered, f"difference_{unit}": difference})


def _resistance(name: str, config: ChannelConfig, seen: Observed) -> list[Check]:
    informed = config.circuit.ohms
    if config.waveform.periodic:
        if seen.i_amplitude is None or seen.i_amplitude < RESISTANCE_MIN_CURRENT_A:
            return []
        resistance = seen.v_amplitude / seen.i_amplitude
        absolute, relative = RESISTANCE_TOLERANCE
        off = abs(resistance - informed) > absolute + relative * informed
        return [Check(Level.WARNING if off else Level.OK, "resistance", name, "off" if off else "ok",
                      {"seen_ohm": resistance, "informed_ohm": informed})]
    if config.role is not Role.SOURCE_V:
        return []
    away_v, expected_a, fraction = OPEN_CIRCUIT
    expected = (seen.v_level - seen.return_voltage) / informed
    if abs(seen.v_level - seen.return_voltage) > away_v and abs(expected) > expected_a \
            and abs(seen.i_level) < fraction * abs(expected):
        return [Check(Level.WARNING, "resistance", name, "open", {"measured_a": seen.i_level, "expected_a": expected})]
    return []


def _return(name: str, config: ChannelConfig, seen: Observed) -> Check:
    circuit = config.circuit
    chosen = circuit.ret
    level_v, level_i = seen.v_level, seen.i_level
    if not config.waveform.periodic and config.role is Role.SOURCE_V \
            and circuit.ohms * LEVEL_TOLERANCE[Role.SOURCE_I][0] > INVISIBLE_RETURN_V:
        return Check(Level.NOTE, "return", name, "invisible", {"ohms_ohm": circuit.ohms})
    estimate = level_v - circuit.ohms * level_i
    numbers = {"estimate_v": estimate, "chosen": chosen}
    if _inside(estimate, _window(chosen, circuit.board_input)):
        return Check(Level.OK, "return", name, "ok", numbers)
    seen_pin = _seen_pin(estimate)
    if seen_pin is not None:
        return Check(Level.WARNING, "return", name, "other", {**numbers, "seen": seen_pin})
    return Check(Level.WARNING, "return", name, "far", numbers)


def _drift(name: str, config: ChannelConfig, seen: Observed) -> Check:
    if seen.other_drift is None or seen.other_u is None:
        return Check(Level.OK, "drift", name, "ok", {"drift": 0.0})
    unit = "v" if config.role is Role.SOURCE_I else "a"
    if config.role is Role.SOURCE_I:
        excursion = abs(seen.v_level - seen.return_voltage)
    else:
        excursion = abs(seen.i_level)
    times_u, fraction = DRIFT_LIMIT
    drifting = abs(seen.other_drift) > times_u * seen.other_u and abs(seen.other_drift) > fraction * excursion
    return Check(Level.WARNING if drifting else Level.OK, "drift", name, "drifting" if drifting else "ok",
                 {f"drift_{unit}": seen.other_drift, f"excursion_{unit}": excursion})


def _shunt_agreement(name: str, other: str, seen: Observed, source: Observed) -> Check:
    absolute, relative = SHUNT_AGREEMENT
    differences = [abs(seen.shunt_level - source.i_level)]
    if seen.shunt_amplitude is not None and source.i_amplitude is not None:
        differences.append(abs(seen.shunt_amplitude - source.i_amplitude))
    difference = max(differences)
    scale = max(abs(source.i_level), abs(source.i_amplitude or 0.0))
    off = difference > absolute + relative * scale
    return Check(Level.WARNING if off else Level.OK, "shunt_agreement", name, "off" if off else "ok",
                 {"other": other, "difference_a": difference})


# --- What the operator reads when the instrument cannot do something ------------------------------

INSTRUMENT_ERROR_TEXTS = {
    "no_pysmu": "pysmu was not found: activate the lab environment (micromamba activate phycovolt).",
    "no_board": "No M1K was found. Check the USB cable; close ALICE, Pixelpulse2 and any other Python "
                "session that uses the M1K; then run `smu -l` inside the lab environment.",
    "several_boards": "More than one M1K is connected: leave only one.",
    "firmware": "The M1K's firmware is too old or unreadable: 2.06 or newer is needed.",
    "lost_sample": "A sample was lost on the way from the M1K (USB): nothing of that read was kept.",
    "timeout": "The M1K stopped sending samples in time (USB): nothing of that read was kept.",
    "usb": "Communication with the M1K failed (USB): nothing of that read was kept.",
    "invalid_request": "The program asked the instrument for something it cannot do.",
    "vi_swap": "The M1K is in the V/I swap state (fault F-01): unplug its USB cable, wait a few "
               "seconds and plug it in again. No channel was driven.",
    "implausible_calibration": "A calibration record stored in the M1K is not plausible: keep a copy "
                               "with `smu -d`, reset it with `smu -r`, reconnect, and calibrate again.",
}


# --- Calibration: what can go wrong with a reading or a point (spec 3.7) ---------------------------

TEXTS.update({
    ("reading", "jump"): "{jumps} block(s) jumped in {quantity}: the reading is redone.",
    ("reading", "adc_limit"): "{fraction} of the {quantity} samples sit at the ADC's edge: the reading is redone.",
    ("reading", "overcurrent"): "The limiter acted: the reading is redone.",
    ("reading", "dmm_failed"): "The multimeter did not answer ({detail}): the reading is redone.",
    ("reading", "dmm_missing"): "No multimeter reading arrived after the settling: the reading is redone.",
    ("reading", "vi_swap"): "The V/I swap (fault F-01) appeared: unplug the M1K's USB cable, wait a few "
                            "seconds and plug it in again. The stage stops.",
    ("reading", "noisy"): "{quantity} drifted by {drift} with u = {u}, above the margin of {margin}.",
    ("reading", "few_decimals"): "At {point} the value was typed with {typed} decimals, and the display shows "
                                 "{shown} (in {unit}): the coarser resolution enters the uncertainty.",
    ("reading", "odd_typed"): "The typed value {value} is far from the other two ({median}): it is redone.",
    ("point", "dmm_overload"): "The multimeter shows an overload on its {range} range, where about {expected} "
                               "was expected: is it on the right channel and terminal?",
    ("point", "gnd_high"): "{channel} on GND reads {value}, more than {limit}: check its jumper to GND.",
    ("point", "disagree"): "The M1K reads {m1k} and the multimeter {dmm}: more than {limit} apart. Is the "
                           "multimeter on {channel}, on its {terminal} terminal?",
    ("point", "zero_request_high"): "0 V asked and the multimeter reads {dmm}, more than {limit}: is it on {channel}?",
    ("point", "zero_request_low"): "0 V asked and the multimeter reads only {dmm}: a negative offset would "
                                   "be cut at code 0 (H-05).",
    ("point", "open_high"): "{channel} with nothing connected reads {value}, more than {limit}: is "
                            "something connected?",
    ("point", "probe_load"): "The load seen is {ohms}, outside {low} to {high}: check the resistor and the "
                             "multimeter's {terminal} terminal.",
    ("point", "probe_pin"): "The loop returns to {pin}, outside {low} to {high}: is it on the 2.5V pin?",
    ("point", "probe_open"): "Only {current} flows: the loop is open. Is the multimeter on its {terminal} "
                             "terminal (B-01)?",
    ("point", "wrong_current"): "The multimeter reads {dmm} for about {expected} expected: wrong sign or "
                                "more than 40 % away.",
    ("point", "saturated"): "{channel} reached {value}: the current source saturated (it must stay between "
                            "{low} and {high}).",
    ("point", "request_off"): "The multimeter reads {dmm} for {request} asked: more than {limit} away.",
    ("stage", "attempts"): "The point {point} failed {attempts} times: the stage stops.",
    ("stage", "stopped"): "The stage was stopped by the operator.",
    ("stage", "interrupted"): "The calibration was interrupted (Ctrl+C or a signal); the channels are in "
                              "HI_Z, and what finished is kept.",
    ("stage", "aux_skipped"): "The zero's slope with the channel voltage was not measured (the auxiliary "
                              "zeros were skipped): the current pairs are not corrected for it (H-07).",
    ("stage", "dmm_pause"): "The multimeter failed {failures} times in a row: check its cable and settings.",
})

# What the operator mounts for each step of a stage (spec 3.7), with the profile's terminal names.
WIRING_TEXTS = {
    "gnd": "Jumper CH A and CH B to GND. Nothing else on the channels.",
    "pin": "Jumper CH A, CH B and the {pin} pin together, and put the multimeter's {voltage} terminal "
           "on them and its {common} terminal on GND (DC voltage).",
    "dmm_voltage": "Put the multimeter's {voltage} terminal on {channel} and its {common} terminal on GND "
                   "(DC voltage). Nothing else on {channel}.",
    "open": "Disconnect everything from CH A and CH B: the channels stay open.",
    "current_loop": "Mount {channel} → resistor (2.5 to 25 Ω, 1/2 W or more, metal film) → multimeter "
                    "{current} terminal → multimeter {common} terminal → 2.5V pin (DC current).",
    "dmm_current": "Set the multimeter to DC current on its {range} range (autorange off) before the load "
                   "probe: the probe must see the same loop as the current points.",
}
