"""The simulated M1K: the same contract as m1k.py, with this unit's real errors (spec C.8).

Question it answers: what would the M1K read, sample by sample, for a requested signal on a known
bench? It lets every flow be rehearsed and tested without the instrument (I-17).

Main choices: one plain chain of steps per read, each a short function:

    request -> DAC code (libsmu with the STORED calibration, truncated)
            -> true output (the unit's TRUE records)
            -> the bench: Ohm's law, the +-200 mA limiter, the 0-5 V rails, the real pin voltages,
               the HI_Z input's 1 MOhm to 0.34 V
            -> raw reading (TRUE records, the measured-current zero's slope with the voltage)
            -> noise: white, mains hum (phase drawn per read), slow noise (correlated 100 ms means)
            -> ADC: 16 bits, clipped at the edges; optional faults (V/I swap, a jump, crooked V)
            -> libsmu's correction with the STORED calibration, in float32 (libsmu_math)

No dynamics: the output follows the request in the same sample. The noise model is the one measured
at rest on 2026-11-21 (unit.ini). The virtual multimeter serves the calibration.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.signal import lfilter

from phycovolt import analysis, checks, libsmu_math, multimeter
from phycovolt import hardware as hw
from phycovolt.circuits import ChannelConfig, CircuitKind
from phycovolt.hardware import ChannelSetup, Chunk, InstrumentError, Kind, Record, Role, Unit

# --- The simulated bench --------------------------------------------------------------------------


@dataclass(frozen=True)
class Wiring:
    """What is connected to one channel's terminal.

    `ohms` is the resistance from the terminal to `pin` (math.inf: nothing connected). A channel can
    instead touch the OTHER channel's circuit at a node `tap_ohms` away from that circuit's pin (the
    node between a load and its shunt): it then sees the other circuit's pin voltage plus the other
    channel's current times `tap_ohms`.
    """

    ohms: float = math.inf
    pin: str = "GND"
    tap_ohms: float | None = None

    def __post_init__(self) -> None:
        if self.pin not in hw.PINS:
            raise ValueError(f"unknown pin {self.pin!r}")
        if not self.ohms > 0:
            raise ValueError("a resistance must be positive (use a small value for a jumper)")


@dataclass(frozen=True)
class Bench:
    """What the operator mounted: one Wiring per channel."""

    a: Wiring = field(default_factory=Wiring)
    b: Wiring = field(default_factory=Wiring)
    # What the multimeter measures: ("pin", "2.5V"), ("v", "A") a channel's terminal voltage,
    # ("i", "A") the current through a channel's circuit (the meter in series); None: nothing.
    dmm: tuple[str, str] | None = None

    def wiring(self, channel: str) -> Wiring:
        return self.a if channel == "A" else self.b


@dataclass
class Faults:
    """Test hooks (C.8). Times and samples count from the start of each read."""

    eeprom_default: bool = False                         # the EEPROM holds no calibration
    vi_swaps_per_second: dict[str, float] = field(default_factory=dict)   # F-01 on a channel
    crooked_v_at_5v: dict[str, float] = field(default_factory=dict)       # V of extra error at 5 V
    jump: tuple[str, str, int, float] | None = None      # (channel, "v" or "i", block, size), next read only
    halved_words_per_second: dict[str, float] = field(default_factory=dict)   # F-02 on a channel's current
    source_nonlinearity: bool = False                    # the voltage source bends near 5 V (H-06)
    usb_failure_at_chunk: int | None = None              # a stream loses a sample at that chunk
    usb_failure_in_rest_read: bool = False               # the next rest read loses a sample


def bench_for(a: ChannelConfig, b: ChannelConfig) -> Bench:
    """The bench as the operator described it: each informed circuit plus the wires' resistance
    (spec 9.2), on the unit's real pins. The i-v-i board's input is modelled by the 2.5V pin."""

    def wiring(config: ChannelConfig, other: ChannelConfig) -> Wiring:
        circuit = config.circuit
        if circuit.kind is CircuitKind.SHUNT_NODE:
            return Wiring(tap_ohms=other.circuit.shunt)
        if circuit.kind is CircuitKind.IVI_BOARD:
            return Wiring(checks.SIM_WIRE_OHMS, "2.5V")
        if circuit.kind is CircuitKind.FREE:
            return Wiring()
        return Wiring(circuit.ohms + checks.SIM_WIRE_OHMS, circuit.ret)

    return Bench(wiring(a, b), wiring(b, a))


def bend_near_5v(volts: np.ndarray, at_verification: float) -> np.ndarray:
    """H-06: zero up to mid-scale, growing as a square, `at_verification` volts at the 4.5 V point."""
    middle = hw.V_MAX / 2
    return at_verification * (np.clip(volts - middle, 0, None) / (checks.VERIFY_SOURCE_V - middle)) ** 2


def stored_records(unit: Unit) -> tuple[Record, ...]:
    """The simulator's EEPROM: the true records with an offset and gain error to correct (A.3)."""
    stored = []
    for (_, kind), true in zip(hw.RECORD_ORDER, unit.true_records):
        gain = 1 + unit.stored_gain_shift
        current = kind in (Kind.MEASURE_I, Kind.SOURCE_I)
        stored.append(Record(true.offset + unit.stored_offset_shift, true.gain_pos * gain,
                             true.gain_neg * gain if current else true.gain_neg))
    return tuple(stored)


# --- Noise ----------------------------------------------------------------------------------------


def slow_noise_coefficient(block_correlation: float, block: int = hw.CHUNK_SAMPLES) -> float:
    """The per-sample coefficient of an AR(1) process whose 100 ms means have this lag-1 correlation."""
    if block_correlation <= 0:
        return 0.0

    def correlation(phi: float) -> float:
        variance = block * (1 + phi) / (1 - phi) - 2 * phi * (1 - phi**block) / (1 - phi) ** 2
        covariance = phi * (1 - phi**block) ** 2 / (1 - phi) ** 2
        return covariance / variance

    low, high = 0.0, 1.0 - 1e-12             # correlation(phi) grows with phi
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if correlation(middle) < block_correlation else (low, middle)
    return (low + high) / 2


def slow_noise(rng: np.random.Generator, samples: int, block_std: float, phi: float) -> np.ndarray:
    """AR(1) noise whose 100 ms block means have standard deviation `block_std` (stationary start)."""
    if block_std == 0 or samples == 0:
        return np.zeros(samples)
    block = hw.CHUNK_SAMPLES
    if phi == 0:
        return np.repeat(rng.normal(0, block_std, math.ceil(samples / block)), block)[:samples]
    variance_factor = (block * (1 + phi) / (1 - phi) - 2 * phi * (1 - phi**block) / (1 - phi) ** 2) / block**2
    process_std = block_std / math.sqrt(variance_factor)
    innovations = rng.normal(0, process_std * math.sqrt(1 - phi**2), samples)
    start = rng.normal(0, process_std)
    noise, _ = lfilter([1.0], [1.0, -phi], innovations, zi=[phi * start])
    return noise


# --- The simulated instrument ---------------------------------------------------------------------


class _SimStream:
    def __init__(self, chunks: list[Chunk], failure_at: int | None, pace: bool,
                 advance: Callable[[float], None]) -> None:
        self._chunks = chunks
        self._failure_at = failure_at
        self._pace = pace
        self._advance = advance
        self._stopped = False
        self.chunks_read = 0
        self.overcurrent = False

    def stop(self) -> None:
        self._stopped = True

    def __iter__(self) -> Iterator[Chunk]:
        while self.chunks_read < len(self._chunks) and not self._stopped:
            if self._failure_at == self.chunks_read:
                raise InstrumentError("lost_sample", "simulated USB failure")
            if self._pace:
                time.sleep(hw.CHUNK_SECONDS)
            self._advance(hw.CHUNK_SECONDS)
            chunk = self._chunks[self.chunks_read]
            self.chunks_read += 1
            yield chunk


class SimulatedM1K:
    """A simulated M1K (implements hardware.Instrument).

    `bench` may be changed between reads (the operator remounting the circuit). `modes` is the state
    of each channel right now; after any read it is HI_Z on both, as on the real instrument.
    """

    def __init__(self, bench: Bench | None = None, unit: Unit | None = None, seed: int | None = 0,
                 faults: Faults | None = None, pace: bool = False) -> None:
        self.unit = unit or hw.load_unit()
        self.bench = bench or Bench()
        self.faults = faults or Faults()
        self.pace = pace
        self.rng = np.random.default_rng(seed)
        self.serial = "simulated"
        self.firmware = self.unit.firmware
        self.hardware = self.unit.hardware
        self.description = f"simulated (no board) · fw {self.firmware} · hw {self.hardware}"
        self.modes = {"A": Role.HI_Z, "B": Role.HI_Z}
        self.last_truth: dict[str, np.ndarray] = {}     # true V and I of the last read, for tests
        self.last_swaps: dict[str, np.ndarray] = {}     # samples given the V/I swap in the last read
        self.last_halved: dict[str, np.ndarray] = {}    # current words halved (F-02) in the last read
        self.time = 0.0                                  # the simulated clock (s)
        self.last_start = 0.0                            # the clock when the last read started
        self._closed = False
        self._phi = slow_noise_coefficient(self.unit.block_correlation)

    # --- the simulated clock ----------------------------------------------------------------------

    def clock(self) -> float:
        """The simulated time: it moves 100 ms with every chunk delivered and with rest reads."""
        return self.time

    def _advance(self, seconds: float) -> None:
        self.time += seconds

    def true_value(self, quantity: str, channel: str, at: float) -> float:
        """The true terminal voltage ("v") or current ("i") of a channel at a clock time, from the
        last read (what a multimeter wired there would see)."""
        values = self.last_truth.get(f"{quantity}_{channel.lower()}")
        if values is None or values.size == 0:
            return 0.0
        index = int(round((at - self.last_start) * hw.SAMPLE_RATE)) - 1
        return float(values[min(max(index, 0), values.size - 1)])

    # --- the contract ----------------------------------------------------------------------------

    def calibration(self) -> list[Record]:
        self._require_open()
        return list(hw.DEFAULT_RECORDS if self.faults.eeprom_default else stored_records(self.unit))

    def rest_read(self, seconds: float) -> Chunk:
        self._require_open()
        samples = round(seconds * hw.SAMPLE_RATE)
        if not 1 <= samples <= hw.QUEUE_SAMPLES:
            raise InstrumentError("invalid_request", f"a rest read of {seconds} s")
        self.modes = {"A": Role.HI_Z, "B": Role.HI_Z}
        if self.faults.usb_failure_in_rest_read:
            self.faults.usb_failure_in_rest_read = False
            raise InstrumentError("lost_sample", "simulated USB failure")
        self.last_start = self.time
        chunk, _ = self._read(ChannelSetup(), ChannelSetup(), samples)
        self._advance(samples / hw.SAMPLE_RATE)
        return replace(chunk, index=0)

    @contextmanager
    def stream(self, a: ChannelSetup, b: ChannelSetup, chunks: int) -> Iterator[_SimStream]:
        self._require_open()
        if not 1 <= chunks <= hw.MAX_STREAM_CHUNKS:
            raise InstrumentError("invalid_request", f"{chunks} chunks (1 to {hw.MAX_STREAM_CHUNKS})")
        self.last_start = self.time
        whole, overcurrent = self._read(a, b, chunks * hw.CHUNK_SAMPLES)
        pieces = [
            Chunk(k, *(array[k * hw.CHUNK_SAMPLES:(k + 1) * hw.CHUNK_SAMPLES]
                       for array in (whole.v_a, whole.i_a, whole.v_b, whole.i_b)))
            for k in range(chunks)
        ]
        stream = _SimStream(pieces, self.faults.usb_failure_at_chunk, self.pace, self._advance)
        self.modes = {"A": a.role, "B": b.role}
        try:
            yield stream
        finally:
            self.modes = {"A": Role.HI_Z, "B": Role.HI_Z}
            stream.stop()
            stream.overcurrent = overcurrent and stream.chunks_read > 0

    def close(self) -> None:
        self.modes = {"A": Role.HI_Z, "B": Role.HI_Z}
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise InstrumentError("invalid_request", "the M1K is closed")

    # --- one read, step by step ------------------------------------------------------------------

    def _read(self, a: ChannelSetup, b: ChannelSetup, samples: int) -> tuple[Chunk, bool]:
        stored = self.calibration()
        setups = {"A": a, "B": b}
        self.last_swaps = {}
        self.last_halved = {}
        outputs = {name: self._true_output(name, setups[name], stored, samples) for name in hw.CHANNELS}
        truth, overcurrent = self._bench(setups, outputs, samples)
        self.last_truth = {f"{q}_{name.lower()}": truth[name][k] for name in hw.CHANNELS
                           for k, q in enumerate("vi")}
        arrays = []
        for name in hw.CHANNELS:
            wiring = self.bench.wiring(name)
            is_open = not math.isfinite(wiring.ohms) and wiring.tap_ohms is None
            pickup = self.unit.open_input_pickup if setups[name].role is Role.HI_Z and is_open else 0.0
            v_word, i_word = self._adc(name, *truth[name], samples, pickup)
            arrays += self._libsmu(name, v_word, i_word, stored)
        return Chunk(0, *arrays), overcurrent

    def _true_output(self, channel: str, setup: ChannelSetup, stored: list[Record],
                     samples: int) -> np.ndarray | None:
        """What the channel really sources (V or A), sample by sample; None in HI_Z."""
        if setup.role is Role.HI_Z:
            return None
        kind = Kind.SOURCE_V if setup.role is Role.SOURCE_V else Kind.SOURCE_I
        index = hw.record_index(channel, kind)
        codes = libsmu_math.dac_code(setup.block, kind, stored[index])
        period = libsmu_math.source_output(codes, kind, self.unit.true_records[index])
        if kind is Kind.SOURCE_V and self.faults.source_nonlinearity:
            period = period + bend_near_5v(period, self.unit.source_nonlinearity[channel])
        return np.resize(period, samples)

    def _bench(self, setups: dict[str, ChannelSetup], outputs: dict[str, np.ndarray | None],
               samples: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], bool]:
        """True terminal voltage and current of each channel (C.8), and whether the limiter acted.
        A channel that touches the other's circuit is solved after it."""
        truth: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        overcurrent = False
        order = sorted(hw.CHANNELS, key=lambda name: self.bench.wiring(name).tap_ohms is not None)
        for name in order:
            wiring = self.bench.wiring(name)
            output = outputs[name]
            role = setups[name].role
            if role is Role.HI_Z:
                truth[name] = self._hi_z(name, wiring, truth, samples)
            elif role is Role.SOURCE_V:
                v, i, limited = self._svmi(wiring, output)
                truth[name] = (v, i)
                overcurrent |= limited
            else:
                truth[name] = self._simv(wiring, output)
        return truth, overcurrent

    def _pin(self, pin: str) -> float:
        return self.unit.pins[pin]

    def _hi_z(self, channel: str, wiring: Wiring, truth: dict, samples: int
              ) -> tuple[np.ndarray, np.ndarray]:
        """HI_Z: the 1 MOhm input towards 0.34 V, divided against what the terminal touches."""
        open_v = self.unit.open_input
        other = "B" if channel == "A" else "A"
        if wiring.tap_ohms is not None and other in truth:
            node = self._pin(self.bench.wiring(other).pin) + truth[other][1] * wiring.tap_ohms
            source_ohms = wiring.tap_ohms
        elif math.isfinite(wiring.ohms):
            node, source_ohms = self._pin(wiring.pin), wiring.ohms
        else:
            node, source_ohms = open_v, 0.0
        v = open_v + (node - open_v) * hw.HI_Z_INPUT_OHMS / (hw.HI_Z_INPUT_OHMS + source_ohms)
        return np.broadcast_to(np.asarray(v, dtype=float), (samples,)).copy(), np.zeros(samples)

    def _svmi(self, wiring: Wiring, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """SVMI: Ohm's law to the pin; above 200 mA the limiter holds the current and V follows."""
        if not math.isfinite(wiring.ohms):
            return output.copy(), np.zeros(output.size), False
        pin = self._pin(wiring.pin)
        current = (output - pin) / wiring.ohms
        limited = np.abs(current) > hw.CHANNEL_CURRENT_LIMIT
        current = np.clip(current, -hw.CHANNEL_CURRENT_LIMIT, hw.CHANNEL_CURRENT_LIMIT)
        voltage = np.where(limited, pin + current * wiring.ohms, output)
        return voltage, current, bool(np.any(limited))

    def _simv(self, wiring: Wiring, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SIMV: the terminal goes where the current needs, within the 0-5 V rails."""
        if not math.isfinite(wiring.ohms):
            rails = np.where(output > 0, hw.V_MAX, np.where(output < 0, hw.V_MIN, self.unit.open_input))
            return rails, np.zeros(output.size)
        pin = self._pin(wiring.pin)
        voltage = np.clip(pin + output * wiring.ohms, hw.V_MIN, hw.V_MAX)
        return voltage, (voltage - pin) / wiring.ohms

    def _adc(self, channel: str, v: np.ndarray, i: np.ndarray, samples: int, pickup: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray]:
        """Raw readings with the true errors and the noise, as 16-bit ADC words (plus faults)."""
        unit = self.unit
        true_v = unit.true_records[hw.record_index(channel, Kind.MEASURE_V)]
        true_i = unit.true_records[hw.record_index(channel, Kind.MEASURE_I)]
        v_raw = v / true_v.gain_pos + true_v.offset
        i_raw = i / np.where(i > 0, true_i.gain_pos, true_i.gain_neg) + true_i.offset
        i_raw = i_raw + unit.current_zero_slope * (v - hw.PIN_NOMINAL["2.5V"])
        crooked = self.faults.crooked_v_at_5v.get(channel, 0.0)
        if crooked:                       # zero up to mid-scale, growing as a square towards 5 V
            middle = hw.V_MAX / 2
            v_raw = v_raw + crooked * (np.clip(v - middle, 0, None) / middle) ** 2
        v_raw = v_raw + self._noise(samples, unit.voltage_sigma[channel], unit.voltage_hum + pickup,
                                    unit.voltage_slow)
        i_raw = i_raw + self._noise(samples, unit.current_sigma, unit.current_hum, unit.current_slow)
        if self.faults.jump is not None and self.faults.jump[0] == channel:
            _, quantity, block, size = self.faults.jump
            self.faults.jump = None                      # a jump happens once (the reading is redone)
            part = slice(block * hw.CHUNK_SAMPLES, (block + 1) * hw.CHUNK_SAMPLES)
            if quantity == "v":
                v_raw[part] += size
            else:
                i_raw[part] += size
        v_word = np.clip(np.rint(v_raw / hw.V_STEP), 0, hw.CODE_MAX)
        i_word = np.clip(np.rint((i_raw - hw.ADC_I_MIN) / hw.ADC_I_STEP), 0, hw.CODE_MAX)
        rate = self.faults.vi_swaps_per_second.get(channel, 0.0)
        if rate:
            count = self.rng.poisson(rate * samples / hw.SAMPLE_RATE)
            where = np.sort(self.rng.choice(samples, size=min(count, samples), replace=False))
            v_word[where], i_word[where] = i_word[where], v_word[where].copy()
            self.last_swaps[channel] = where
        rate = self.faults.halved_words_per_second.get(channel, 0.0)
        if rate:                                     # F-02: one bit shifted right in the current word
            count = self.rng.poisson(rate * samples / hw.SAMPLE_RATE)
            where = np.sort(self.rng.choice(samples, size=min(count, samples), replace=False))
            i_word[where] = np.floor(i_word[where] / 2)
            self.last_halved[channel] = where
        return v_word, i_word

    def _noise(self, samples: int, sigma: float, hum: float, slow: float) -> np.ndarray:
        t = np.arange(samples) / hw.SAMPLE_RATE
        phase = self.rng.uniform(0, 2 * np.pi)
        return (self.rng.normal(0, sigma, samples)
                + hum * np.sin(2 * np.pi * self.unit.mains * t + phase)
                + slow_noise(self.rng, samples, slow, self._phi))

    def _libsmu(self, channel: str, v_word: np.ndarray, i_word: np.ndarray,
                stored: Sequence[Record]) -> list[np.ndarray]:
        """What pysmu hands over: libsmu's correction with the stored calibration (float32)."""
        voltage = libsmu_math.correct_voltage(libsmu_math.adc_voltage(v_word),
                                              stored[hw.record_index(channel, Kind.MEASURE_V)])
        current = libsmu_math.correct_current(libsmu_math.adc_current(i_word),
                                              stored[hw.record_index(channel, Kind.MEASURE_I)])
        return [voltage.astype(np.float64), current.astype(np.float64)]


# --- The virtual multimeter (C.8) -----------------------------------------------------------------

VIRTUAL_DMM_NOISE = 2e-7                 # reading noise, as a fraction of the range


class VirtualMultimeter:
    """A multimeter on the simulated bench (implements multimeter.Multimeter).

    It reads what `bench.dmm` says it is wired to, at the simulated clock's time: the true value,
    plus a gain and a zero error drawn once per range within the profile's specification, plus a
    little reading noise, rounded to the displayed digits. `fail_reads` makes the next reads fail
    (a multimeter that stops answering).
    """

    def __init__(self, instrument: SimulatedM1K, profile: multimeter.Profile, seed: int | None = 0) -> None:
        self.profile = profile
        self._instrument = instrument
        self._rng = np.random.default_rng(seed)
        self._errors: dict[tuple[multimeter.Function, float], tuple[float, float]] = {}
        self._configured: tuple[multimeter.Function, float] | None = None
        self.fail_reads = 0

    def identify(self) -> str:
        return f"{self.profile.name} (virtual)"

    def configure(self, function: multimeter.Function, full_scale: float) -> None:
        if (function, full_scale) not in self._errors:
            accuracy = self.profile.ranges[function][full_scale]
            self._errors[(function, full_scale)] = (self._draw(accuracy.of_reading),
                                                    self._draw(accuracy.of_range * full_scale))
        self._configured = (function, full_scale)

    def _draw(self, limit: float) -> float:
        if self.profile.distribution == "rectangular":
            return float(self._rng.uniform(-limit, limit))
        return float(self._rng.normal(0, limit / 2))

    def true_errors(self, function: multimeter.Function, full_scale: float) -> tuple[float, float]:
        """(relative gain error, zero error) drawn for a range: for the tests and the Monte Carlo."""
        return self._errors[(function, full_scale)]

    def value_at(self, at: float) -> float:
        """What the display shows at a clock time."""
        if self._configured is None:
            raise multimeter.MultimeterError("not configured")
        function, full_scale = self._configured
        probe = self._instrument.bench.dmm
        true = 0.0
        if probe is not None:
            kind, where = probe
            if kind == "pin" and function is multimeter.Function.DC_VOLTAGE:
                true = self._instrument.unit.pins[where]
            elif kind == "v" and function is multimeter.Function.DC_VOLTAGE:
                true = self._instrument.true_value("v", where, at)
            elif kind == "i" and function is multimeter.Function.DC_CURRENT:
                true = self._instrument.true_value("i", where, at)
        if abs(true) > full_scale * self.profile.overrange:
            raise multimeter.MultimeterOverload("overload")
        gain, zero = self._errors[(function, full_scale)]
        shown = true * (1 + gain) + zero + self._rng.normal(0, VIRTUAL_DMM_NOISE * full_scale)
        step = multimeter.display_resolution(self.profile, full_scale)
        return round(shown / step) * step

    def read(self) -> multimeter.DmmReading:
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise multimeter.MultimeterError("no answer before the timeout (simulated)")
        at = self._instrument.clock()
        return multimeter.DmmReading(self.value_at(at), at)

    def close(self) -> None:
        return None


# --- The simulated operator (calibration) ---------------------------------------------------------


class SimulatedOperator:
    """The operator of a simulated calibration: mounts on the simulated bench what each wiring step
    asks (with `loop_ohms` in the current loop), keeps warned readings, tries again after a problem,
    and in typed mode types what the virtual multimeter shows in the window. It can make the
    mistakes the tests need: `miswire` (a step kind -> another multimeter probe), `swap_digits_once`
    (a typing error), `fewer_decimals` (fewer decimals than the display shows), `give_up` (stop at the first problem)."""

    def __init__(self, instrument: SimulatedM1K, dmm: VirtualMultimeter, loop_ohms: float = 10.0,
                 miswire: dict[str, tuple[str, str]] | None = None, swap_digits_once: bool = False,
                 give_up: bool = False, say: Callable[[str], None] | None = None,
                 fewer_decimals: int = 0) -> None:
        self.instrument = instrument
        self.dmm = dmm
        self.loop_ohms = loop_ohms
        self.miswire = miswire or {}
        self.swap_digits_once = swap_digits_once
        self.fewer_decimals = fewer_decimals         # types fewer decimals than the display shows
        self.give_up = give_up
        self.say = say
        self.steps: list[object] = []
        self.problems: list[checks.Check] = []
        self.warnings: list[checks.Check] = []
        self.typed: list[str] = []

    def wire(self, step: object) -> bool:
        self.steps.append(step)
        kind, channel, pin = step.kind, step.channel, step.pin
        if kind == "dmm_current":                    # a setting of the meter: the wiring stays
            return True
        jumper = 0.05
        if kind == "gnd":
            bench = Bench(Wiring(jumper, "GND"), Wiring(jumper, "GND"))
        elif kind == "pin":
            bench = Bench(Wiring(jumper, pin), Wiring(jumper, pin), dmm=("pin", pin))
        elif kind == "dmm_voltage":
            bench = Bench(dmm=("v", channel))
        elif kind == "current_loop":
            loop = Wiring(self.loop_ohms + checks.SIM_WIRE_OHMS, "2.5V")
            bench = Bench(a=loop if channel == "A" else Wiring(), b=loop if channel == "B" else Wiring(),
                          dmm=("i", channel))
        else:
            bench = Bench()
        if kind in self.miswire:
            bench = replace(bench, dmm=self.miswire[kind])
        self.instrument.bench = bench
        if self.say is not None:
            self.say(f"[simulated wiring: {kind}{' ' + channel if channel else ''}{' ' + pin if pin else ''}]")
        return True

    def confirm_warm_up(self) -> bool:
        return True

    def acquiring(self, stage: str, point: str, attempt: int, seconds: float) -> None:
        if self.say is not None:
            self.say(f"  {stage} · {point} · attempt {attempt} · {seconds:g} s")

    def problem(self, check: checks.Check) -> bool:
        self.problems.append(check)
        if self.say is not None:
            self.say(f"  {checks.SYMBOL[check.level]} {check.text}")
        return not self.give_up

    def accept_warning(self, check: checks.Check) -> bool:
        self.warnings.append(check)
        return True

    def typed_value(self, request: object) -> str:
        self.dmm.configure(request.function, request.full_scale)
        value = self.dmm.value_at((request.window[0] + request.window[1]) / 2)
        decimals = max(0, request.decimals - self.fewer_decimals)
        if request.unit == "mA":
            text = f"{value * 1e3:.{decimals}f} mA"
        else:
            text = f"{value:.{decimals}f}"
        if self.swap_digits_once:
            self.swap_digits_once = False
            text = _swap_two_digits(text)
        self.typed.append(text)
        return text


def _swap_two_digits(text: str) -> str:
    """The first two neighbouring digits that differ, swapped ("2.44570" -> "2.45470")."""
    chars = list(text)
    for k in range(len(chars) - 1):
        if chars[k].isdigit() and chars[k + 1].isdigit() and chars[k] != chars[k + 1] and k > text.find("."):
            chars[k], chars[k + 1] = chars[k + 1], chars[k]
            return "".join(chars)
    return text


# --- The Monte Carlo's readings (spec 8): simulated at the level of block means -------------------


def _block_series(rng: np.random.Generator, blocks: int, std: float, correlation: float) -> np.ndarray:
    series = np.empty(blocks)
    series[0] = rng.normal(0, std)
    innovation = std * math.sqrt(1 - correlation**2)
    for k in range(1, blocks):
        series[k] = correlation * series[k - 1] + rng.normal(0, innovation)
    return series


class BlockLevelBench:
    """The truth of one simulated calibration, and readings of it as `Observation`s.

    The M1K's value of a reading is the mean of its 100 ms block means, with the unit's noise model
    (white noise averaged in each block, plus slow noise with the measured block correlation); its u
    is method B on those blocks. The multimeter reads every 300 ms after the settling, with a gain
    and zero error drawn once per range within the profile's specification. Nothing else differs
    from a calibration on the simulated bench, and it is a thousand times faster.
    """

    def __init__(self, unit: Unit, profile: multimeter.Profile, rng: np.random.Generator,
                 loop_ohms: float = 10.0) -> None:
        self.unit = unit
        self.profile = profile
        self.rng = rng
        self.loop = loop_ohms + checks.SIM_WIRE_OHMS
        self.dmm_errors: dict[tuple[multimeter.Function, float], tuple[float, float]] = {}

    def true(self, channel: str, kind: Kind) -> Record:
        return self.unit.true_records[hw.record_index(channel, kind)]

    def m1k(self, true_raw: float, quantity: str, seconds: float, channel: str = "A") -> tuple[float, float]:
        sigma = self.unit.voltage_sigma[channel] if quantity == "V" else self.unit.current_sigma
        slow = self.unit.voltage_slow if quantity == "V" else self.unit.current_slow
        blocks = round((seconds - checks.CAL_SETTLING_S) / hw.CHUNK_SECONDS)
        std = math.hypot(slow, sigma / math.sqrt(hw.CHUNK_SAMPLES))
        series = true_raw + _block_series(self.rng, blocks, std, self.unit.block_correlation)
        return float(np.mean(series)), analysis.uncertainty_of_mean(series)

    def dmm(self, true: float, function: multimeter.Function, expected: float,
            seconds: float = checks.CAL_READING_S) -> tuple[tuple[float, float], float]:
        full_scale = multimeter.choose_range(self.profile, function, expected)
        if (function, full_scale) not in self.dmm_errors:
            accuracy = self.profile.ranges[function][full_scale]
            draw = (lambda limit: float(self.rng.uniform(-limit, limit))) if self.profile.distribution == "rectangular" \
                else (lambda limit: float(self.rng.normal(0, limit / 2)))
            self.dmm_errors[(function, full_scale)] = (draw(accuracy.of_reading), draw(accuracy.of_range * full_scale))
        gain, zero = self.dmm_errors[(function, full_scale)]
        count = int((seconds - checks.CAL_SETTLING_S) / (checks.DMM_EVERY_CHUNKS * hw.CHUNK_SECONDS))
        step = multimeter.display_resolution(self.profile, full_scale)
        shown = true * (1 + gain) + zero + self.rng.normal(0, VIRTUAL_DMM_NOISE * full_scale, count)
        shown = np.round(shown / step) * step
        return (float(np.mean(shown)), analysis.uncertainty_of_mean(shown)), full_scale

    def observations(self, stage: str, skip_optional: bool = False) -> dict[str, object]:
        """The observations of one simulated run of `stage`, named as the real stage names them."""
        from phycovolt import calibration as cal
        voltage, current = multimeter.Function.DC_VOLTAGE, multimeter.Function.DC_CURRENT
        found: dict[str, cal.Observation] = {}

        def add(name: str, role: str, m1k: dict, dmm: tuple | None = None, function=None, codes=None) -> None:
            reading, full_scale = dmm if dmm is not None else (None, None)
            found[name] = cal.Observation(name, role, m1k, reading, function, full_scale, codes or {})

        stored = list(stored_records(self.unit))
        if stage == "measure-v":
            pins = self.unit.pins
            for name, pin, role, seconds in (("zero", "GND", "fit", checks.CAL_ZERO_S),
                                             ("reference", "2.5V", "fit", checks.CAL_READING_S),
                                             ("verification", checks.VERIFY_V_PIN, "verification", checks.CAL_READING_S)):
                if role == "verification" and skip_optional:
                    continue
                m1k = {f"{c} V": self.m1k(pins[pin] / self.true(c, Kind.MEASURE_V).gain_pos
                                          + self.true(c, Kind.MEASURE_V).offset, "V", seconds, c) for c in hw.CHANNELS}
                dmm = self.dmm(pins[pin], voltage, hw.PIN_NOMINAL[pin]) if pin != "GND" else None
                add(name, role, m1k, dmm, voltage if dmm else None)
        elif stage == "source-v":
            for c in hw.CHANNELS:
                true = self.true(c, Kind.SOURCE_V)
                for volts in checks.SOURCE_V_FIT_V:
                    code = cal.code_for_raw(volts, Kind.SOURCE_V)
                    out = float(libsmu_math.source_output(code, Kind.SOURCE_V, true))
                    add(f"{c} {volts:g}V", "fit", {}, self.dmm(out, voltage, volts), voltage, {c: code})
                if not skip_optional:
                    new = cal.provisional_record("source-v", c, found, stored, self.profile)
                    code = int(libsmu_math.dac_code(checks.VERIFY_SOURCE_V, Kind.SOURCE_V, new))
                    out = float(libsmu_math.source_output(code, Kind.SOURCE_V, true))
                    add(f"{c} {checks.VERIFY_SOURCE_V:g}V", "verification", {}, self.dmm(out, voltage, checks.VERIFY_SOURCE_V),
                        voltage, {c: code})
        elif stage == "measure-i":
            slope = self.unit.current_zero_slope
            zeros = [(f"zero {checks.ZERO_V:g}V", checks.ZERO_V, "fit")]
            if not skip_optional:
                zeros += [(f"zero {v:g}V", v, "auxiliary") for v in checks.AUX_ZERO_V]
            def raw_v(c: str, volts: float) -> float:            # the channel voltage, as the ADC reads it
                true_v = self.true(c, Kind.MEASURE_V)
                return volts / true_v.gain_pos + true_v.offset

            for name, volts, role in zeros:
                m1k = {}
                for c in hw.CHANNELS:
                    true = self.true(c, Kind.MEASURE_I)
                    m1k[f"{c} I"] = self.m1k(true.offset + slope * (volts - checks.ZERO_V), "I", checks.CAL_ZERO_S, c)
                    m1k[f"{c} V"] = (raw_v(c, volts), 0.0)
                add(name, role, m1k)
            pin = self.unit.pins["2.5V"]
            for c in hw.CHANNELS:
                true = self.true(c, Kind.MEASURE_I)
                amps = cal.point_current(pin, self.loop)
                points = [("+fit", amps, "fit"), ("-fit", -amps, "fit")]
                if not skip_optional:
                    points += [("+verify", checks.VERIFY_A, "verification"), ("-verify", -checks.VERIFY_A, "verification")]
                for label, target, role in points:
                    volts = pin + target * self.loop
                    gain = true.gain_pos if target > 0 else true.gain_neg
                    raw = true.offset + target / gain + slope * (volts - checks.ZERO_V)
                    add(f"{c} {label}", role, {f"{c} I": self.m1k(raw, "I", checks.CAL_READING_S, c),
                                               f"{c} V": (raw_v(c, volts), 0.0)},
                        self.dmm(target, current, target), current)
        else:                                                   # source-i
            pin = self.unit.pins["2.5V"]
            for c in hw.CHANNELS:
                true = self.true(c, Kind.SOURCE_I)
                amps = cal.point_current(pin, self.loop)
                for label, target in (("zero", 0.0), ("+fit", amps), ("-fit", -amps)):
                    code = cal.code_for_raw(target, Kind.SOURCE_I)
                    out = float(libsmu_math.source_output(code, Kind.SOURCE_I, true))
                    add(f"{c} {label}", "fit", {}, self.dmm(out, current, target), current, {c: code})
                if not skip_optional:
                    new = cal.provisional_record("source-i", c, found, stored, self.profile)
                    for label, target in (("+verify", checks.VERIFY_A), ("-verify", -checks.VERIFY_A)):
                        code = int(libsmu_math.dac_code(target, Kind.SOURCE_I, new))
                        out = float(libsmu_math.source_output(code, Kind.SOURCE_I, true))
                        add(f"{c} {label}", "verification", {}, self.dmm(out, current, target), current, {c: code})
        return found
