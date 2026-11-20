"""The signals procedure: rest reading, before-checks, the burst, the analysis, the after-checks.

Question it answers: from what the operator asked of each channel, what was measured and is it
trustworthy? The command line only calls these functions and shows what they return (spec 3.3,
7.1: no math where it prints).

Main choices:

- `prepare` reads both channels at rest (explicitly HI_Z, I-03/I-14), then runs the before-checks;
  nothing is switched on. `run` is called only after the operator's confirmation, and only if no
  before-check gave an error.
- `run` streams the burst; Stop (`should_stop`), Ctrl+C or a signal end it at the current chunk with
  both channels in HI_Z (the instrument guarantees it); what was read stays and is analyzed, marked
  "stopped". A lost sample raises InstrumentError: nothing of that read is kept.
- `analyze` is a pure function of the recorded samples, the configuration, the stored calibration
  and the rest reading, so a recording can be analyzed again from its files alone (spec 3.4).
- A burst lasts as long as the longer of the two channels' waveforms; the shorter one repeats.
- With both channels in hi-z the burst only measures (nothing is parked or sourced); its duration
  is a parameter of its own (0.3 to 10 s, default 1 s; spec 3.3.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable

import numpy as np

from phycovolt import analysis, checks, circuits, units, waveforms
from phycovolt import hardware as hw
from phycovolt.analysis import CycleAverage, SineFit, Summary
from phycovolt.checks import Check, Observed
from phycovolt.circuits import ChannelConfig, CircuitKind, Prediction
from phycovolt.hardware import ChannelSetup, Chunk, Instrument, Record, Role
from phycovolt.waveforms import Shape


@dataclass(frozen=True)
class Prepared:
    """Everything known before switching on."""

    a: ChannelConfig
    b: ChannelConfig
    rest: dict[str, float]            # median rest voltage of each channel (V)
    records: list[Record]             # the stored calibration, read now
    checks: list[Check]
    predictions: dict[str, Prediction]
    chunks: int                       # length of the burst, in 100 ms chunks

    @property
    def blocked(self) -> bool:
        return checks.blocks_switch_on(self.checks)

    @property
    def measure_only(self) -> bool:
        return measure_only(self.a, self.b)

    def config(self, channel: str) -> ChannelConfig:
        return self.a if channel == "A" else self.b


DEFAULT_MEASURE_SECONDS = 1.0        # a measure-only burst, when no duration is given


def measure_only(a: ChannelConfig, b: ChannelConfig) -> bool:
    """Both channels hi-z: the burst only measures."""
    return a.role is Role.HI_Z and b.role is Role.HI_Z


def burst_chunks(a: ChannelConfig, b: ChannelConfig, measure_seconds: float | None = None) -> int:
    """The burst lasts as long as the longer waveform; a measure-only burst, `measure_seconds`
    (default 1 s). The before-checks judge the duration (0.3 to 10 s)."""
    if measure_only(a, b):
        seconds = DEFAULT_MEASURE_SECONDS if measure_seconds is None else measure_seconds
        if not seconds > 0:
            raise ValueError("the duration must be above 0 s")
        return max(round(seconds / hw.CHUNK_SECONDS), 1)
    if measure_seconds is not None:
        raise ValueError("a duration is only for a measure-only burst: a sourcing channel's waveform "
                         "sets the length")
    return max(waveforms.timing(c.waveform).chunks for c in (a, b) if c.waveform is not None)


def prepare(instrument: Instrument, a: ChannelConfig, b: ChannelConfig,
            open_input: float | None = None, measure_seconds: float | None = None) -> Prepared:
    """Read both channels at rest, then run the before-checks (nothing is switched on)."""
    circuits.check_pair(a, b)
    chunks = burst_chunks(a, b, measure_seconds)
    if measure_only(a, b) and measure_seconds is None:
        measure_seconds = DEFAULT_MEASURE_SECONDS
    if open_input is None:
        open_input = hw.load_unit().open_input
    rest = instrument.rest_read(checks.REST_SECONDS)
    records = instrument.calibration()
    found = checks.before_checks(a, b, rest, records, open_input, measure_seconds)
    resting = checks.rest_voltages(rest)
    predictions = {}
    for name, config in (("A", a), ("B", b)):
        if config.role is not Role.HI_Z and not any(c.code == "parameters" and c.channel == name for c in found):
            predictions[name] = circuits.predict(
                config, name, records, checks.return_voltage(config, resting[name], open_input))
    return Prepared(a, b, resting, records, found, predictions, chunks)


# --- The channel syntax ---------------------------------------------------------------------------


def parse_channel(text: str) -> ChannelConfig:
    """"source-i sine freq=100Hz pp=2mA circuit=resistor r=10 return=2.5V" -> a ChannelConfig.

    The circuit's return decides the waveform's defaults; typed values replace them."""
    tokens = text.split()
    if not tokens:
        raise ValueError("an empty channel description (use \"hi-z\" for a channel that only measures)")
    try:
        role = Role(tokens[0])
    except ValueError:
        raise ValueError(f"{tokens[0]!r} is not a role: use hi-z, source-v or source-i") from None
    rest = tokens[1:]
    shape = None
    if role is not Role.HI_Z:
        if not rest or "=" in rest[0]:
            raise ValueError(f"{role.value} needs a waveform: {', '.join(s.value for s in Shape)}")
        shape = Shape(rest[0])
        rest = rest[1:]
    pairs: dict[str, str] = {}
    for token in rest:
        if "=" not in token:
            raise ValueError(f"{token!r} is not key=value")
        key, value = token.split("=", 1)
        pairs[key] = value
    kind = CircuitKind(pairs.pop("circuit")) if "circuit" in pairs else circuits.default_circuit(role).kind
    if circuits.ROLE_OF[kind] is not role:
        raise ValueError(f"the {kind.value} circuit is not for {role.value}")
    typed = {key: pairs.pop(key) for key in list(pairs) if key in circuits.CIRCUIT_FIELDS[kind]}
    circuit = circuits.circuit_from_text(kind, typed)
    if role is Role.HI_Z:
        if pairs:
            raise ValueError(f"a hi-z channel has no fields {sorted(pairs)}")
        return ChannelConfig(role, circuit)
    waveform = waveforms.waveform_from_text(role, shape, pairs, circuit.ret)
    return ChannelConfig(role, circuit, waveform)


def channel_text(config: ChannelConfig) -> str:
    """A channel written out in full, every field and default included, in the syntax that
    `parse_channel` reads back to the same configuration (the "To repeat" command, spec 3.1)."""
    words = [config.role.value]
    if config.waveform is not None:
        words.append(config.waveform.shape.value)
        for item in waveforms.FIELDS[config.waveform.shape]:
            unit, prefix = waveforms.field_unit(item, config.role)
            words.append(f"{item.key}={units.exact(config.waveform[item.key], unit, prefix)}")
    circuit = config.circuit
    words.append(f"circuit={circuit.kind.value}")
    for key, (attribute, unit) in circuits.CIRCUIT_FIELDS[circuit.kind].items():
        value = getattr(circuit, attribute)
        words.append(f"{key}={value if attribute == 'ret' else units.exact(value, unit)}")
    return " ".join(words)


def board_check(instrument: Instrument) -> dict[str, int]:
    """2 s with both channels in HI_Z, looking for the V/I swap (F-01, I-13): swaps per channel."""
    rest = instrument.rest_read(checks.BOARD_CHECK_SECONDS)
    records = instrument.calibration()
    found = {}
    for name in hw.CHANNELS:
        voltage, current = rest.channel(name)
        found[name] = analysis.count_vi_swaps(voltage, current, records, name) or 0
    return found


# --- Results --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Samples:
    """Every sample of the burst, as libsmu returned them (V and A)."""

    v_a: np.ndarray
    i_a: np.ndarray
    v_b: np.ndarray
    i_b: np.ndarray

    @property
    def count(self) -> int:
        return int(self.v_a.size)

    def channel(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (self.v_a, self.i_a) if name == "A" else (self.v_b, self.i_b)


@dataclass(frozen=True)
class ChannelResult:
    """What the standard analysis found on one channel (spec 3.6)."""

    channel: str
    role: Role
    settling_samples: int
    period: int | None                # samples per period of the analysis (None: constant)
    voltage: Summary | None = None    # DC (and hi-z on a constant signal)
    current: Summary | None = None
    v_cycle: CycleAverage | None = None
    i_cycle: CycleAverage | None = None
    v_fit: SineFit | None = None
    i_fit: SineFit | None = None
    frequency: float | None = None    # the real frequency (Hz)
    vs_requested: tuple[float, float] | None = None   # cycle average - requested: (rms, max)
    vs_delivered: tuple[float, float] | None = None   # cycle average - DAC-delivered: (rms, max)
    impedance: float | None = None    # ohm, sine with >= 20 uA
    phase_vi: float | None = None     # rad, phase(V) - phase(I)
    shunt: Summary | CycleAverage | SineFit | None = None   # hi-z shunt / shunt-node: its current
    load_voltage: Summary | CycleAverage | SineFit | None = None   # shunt-node: V(other) - V(node)
    vi_swaps: int | None = None
    halved: dict[str, tuple[int, float, float]] | None = None   # F-02 by quantity
    rail_fraction: float = 0.0
    observed: Observed | None = None

    def numbers(self) -> dict[str, float | int | str]:
        """Every scalar result, with its unit in its name (as the checks' numbers)."""
        out: dict[str, float | int | str] = {"role": self.role.value, "settling_samples_count": self.settling_samples}
        if self.frequency is not None:
            out["frequency_hz"] = self.frequency
        for prefix, unit, summary in (("v", "v", self.voltage), ("i", "a", self.current)):
            if summary is not None:
                out.update(_summary_numbers(prefix, unit, summary))
        for prefix, unit, cycle in (("v", "v", self.v_cycle), ("i", "a", self.i_cycle)):
            if cycle is not None:
                out.update({f"{prefix}_cycle_level_{unit}": cycle.level, f"{prefix}_cycle_pp_{unit}": cycle.peak_to_peak,
                            f"{prefix}_cycle_rms_{unit}": cycle.rms, f"{prefix}_cycle_periods_count": cycle.periods})
        for prefix, unit, fit in (("v", "v", self.v_fit), ("i", "a", self.i_fit)):
            if fit is not None:
                out.update({f"{prefix}_fit_level_{unit}": fit.level, f"{prefix}_fit_amplitude_{unit}": fit.amplitude,
                            f"{prefix}_fit_phase_rad": fit.phase, f"{prefix}_fit_residual_{unit}": fit.residual})
                if self.frequency:
                    out[f"{prefix}_fit_delay_s"] = fit.delay(self.frequency)
        unit = "a" if self.role is Role.SOURCE_I else "v"
        if self.vs_requested is not None:
            out.update({f"vs_requested_rms_{unit}": self.vs_requested[0], f"vs_requested_max_{unit}": self.vs_requested[1]})
        if self.vs_delivered is not None:
            out.update({f"vs_delivered_rms_{unit}": self.vs_delivered[0], f"vs_delivered_max_{unit}": self.vs_delivered[1]})
        if self.impedance is not None:
            out.update({"impedance_ohm": self.impedance, "phase_vi_rad": self.phase_vi})
        if isinstance(self.shunt, Summary):
            out.update(_summary_numbers("shunt", "a", self.shunt))
        elif isinstance(self.shunt, SineFit):
            out.update({"shunt_fit_level_a": self.shunt.level, "shunt_fit_amplitude_a": self.shunt.amplitude})
        elif isinstance(self.shunt, CycleAverage):
            out.update({"shunt_cycle_level_a": self.shunt.level, "shunt_cycle_pp_a": self.shunt.peak_to_peak})
        if isinstance(self.load_voltage, Summary):
            out.update(_summary_numbers("load", "v", self.load_voltage))
        elif isinstance(self.load_voltage, SineFit):
            out.update({"load_fit_level_v": self.load_voltage.level, "load_fit_amplitude_v": self.load_voltage.amplitude})
        elif isinstance(self.load_voltage, CycleAverage):
            out.update({"load_cycle_level_v": self.load_voltage.level, "load_cycle_pp_v": self.load_voltage.peak_to_peak})
        if self.vi_swaps is not None:
            out["vi_swaps_count"] = self.vi_swaps
        for quantity, (count, mean_shift, block_shift) in (self.halved or {}).items():
            unit = "v" if quantity == "V" else "a"
            out[f"halved_{quantity.lower()}_count"] = count
            if count:
                out[f"halved_{quantity.lower()}_mean_shift_{unit}"] = mean_shift
                out[f"halved_{quantity.lower()}_block_shift_{unit}"] = block_shift
        out["rail_fraction_frac"] = self.rail_fraction
        return out


def _summary_numbers(prefix: str, unit: str, summary: Summary) -> dict[str, float | int | str]:
    numbers: dict[str, float | int | str] = {
        f"{prefix}_value_{unit}": summary.value, f"{prefix}_u_{unit}": summary.u,
        f"{prefix}_spread_{unit}": summary.spread, f"{prefix}_sigma_{unit}": summary.sigma,
        f"{prefix}_blocks_count": summary.blocks, f"{prefix}_jumps_count": summary.jumps,
        f"{prefix}_drift_{unit}": summary.drift, f"{prefix}_adc_limit_frac": summary.adc_limit_fraction,
        f"{prefix}_u_method": summary.method,
    }
    if summary.r1 is not None:
        numbers[f"{prefix}_r1"] = summary.r1
    return numbers


@dataclass(frozen=True)
class Measurement:
    """A finished (or stopped) burst, analyzed and checked."""

    prepared: Prepared
    samples: Samples
    ended: str                        # "complete" or "stopped"
    interrupted: bool                 # stopped by Ctrl+C or a signal (exit code 130)
    overcurrent: bool
    results: dict[str, ChannelResult]
    after: list[Check]
    started: datetime
    description: str
    serial: str
    firmware: str
    hardware: str
    extra: dict[str, object] = field(default_factory=dict)


def run(instrument: Instrument, prepared: Prepared,
        on_chunk: Callable[[Chunk, int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None) -> Measurement:
    """Switch on, stream the burst, switch off, analyze. Only after the operator's confirmation."""
    if prepared.blocked:
        raise ValueError("a before-check gave an error: the outputs stay off")
    setups = {}
    for name in hw.CHANNELS:
        config = prepared.config(name)
        setups[name] = ChannelSetup() if config.role is Role.HI_Z else \
            ChannelSetup(config.role, waveforms.period(config.waveform))
    started = datetime.now()
    collected: list[Chunk] = []
    interrupted = False
    stream = None
    try:
        with instrument.stream(setups["A"], setups["B"], prepared.chunks) as stream:
            for chunk in stream:
                collected.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk, len(collected), prepared.chunks)
                if should_stop is not None and should_stop():
                    stream.stop()
    except KeyboardInterrupt:
        interrupted = True               # both channels are in HI_Z: the `with` saw to it
    overcurrent = bool(stream.overcurrent) if stream is not None else False
    samples = Samples(*(np.concatenate([getattr(c, n) for c in collected]) if collected else np.empty(0)
                        for n in ("v_a", "i_a", "v_b", "i_b")))
    ended = "complete" if len(collected) == prepared.chunks else "stopped"
    results, after = analyze(prepared, samples, overcurrent)
    return Measurement(prepared, samples, ended, interrupted, overcurrent, results, after, started,
                       instrument.description, instrument.serial, instrument.firmware, instrument.hardware)


# --- The analysis ---------------------------------------------------------------------------------


def analyze(prepared: Prepared, samples: Samples, overcurrent: bool
            ) -> tuple[dict[str, ChannelResult], list[Check]]:
    """The standard analysis and the after-checks of a burst (pure: also used on a recording)."""
    period = _burst_period(prepared)
    results = {name: _channel(name, prepared, samples, period) for name in hw.CHANNELS}
    for name in hw.CHANNELS:                       # the shunt-node needs the other channel's current
        config = prepared.config(name)
        if config.circuit.kind is CircuitKind.SHUNT_NODE:
            other = "B" if name == "A" else "A"
            results[name] = _with_shunt_node(results[name], prepared, other, samples)
    observed = {name: results[name].observed for name in hw.CHANNELS}
    if samples.count == 0:
        return results, []
    after = checks.after_checks(prepared.a, prepared.b, observed, overcurrent)
    return results, after


def _burst_period(prepared: Prepared) -> int | None:
    """The period that sets the analysis of a channel without its own: A's first, then B's."""
    for config in (prepared.a, prepared.b):
        if config.waveform is not None and config.waveform.periodic:
            return waveforms.timing(config.waveform).samples_per_period
    return None


def _diffs(cycle: CycleAverage, reference: np.ndarray) -> tuple[float, float]:
    difference = cycle.mean - reference
    return float(np.sqrt(np.mean(np.square(difference)))), float(np.max(np.abs(difference)))


def _channel(name: str, prepared: Prepared, samples: Samples, burst_period: int | None) -> ChannelResult:
    config = prepared.config(name)
    v, i = samples.channel(name)
    waveform = config.waveform
    own_period = waveforms.timing(waveform).samples_per_period if waveform is not None and waveform.periodic else None
    period = own_period if config.role is not Role.HI_Z else burst_period
    settling = period if period is not None else hw.CHUNK_SAMPLES
    swaps = analysis.count_vi_swaps(v, i, prepared.records, name, period) if v.size else 0
    halved = analysis.count_halved(v, i, prepared.records, name, period) if v.size else None
    words_v, words_i = analysis.channel_words(v, i, prepared.records, name) if v.size else (v, i)
    return_v = _return_voltage(config, prepared, name)
    base = dict(channel=name, role=config.role, settling_samples=settling, period=period, vi_swaps=swaps, halved=halved,
                rail_fraction=analysis.rail_fraction(v[settling:]))
    if period is None:                                             # constant: reading summaries
        voltage = analysis.summarize(v[settling:], hw.V_STEP, words_v[settling:])
        current = analysis.summarize(i[settling:], hw.ADC_I_STEP, words_i[settling:])
        result = ChannelResult(**base, voltage=voltage, current=current)
        if config.circuit.kind is CircuitKind.SHUNT and return_v is not None and voltage is not None:
            r = config.circuit.r
            result = replace(result, shunt=replace(voltage, value=(voltage.value - return_v) / r, u=voltage.u / r,
                                                   spread=voltage.spread / r, sigma=voltage.sigma / r,
                                                   drift=voltage.drift / r))
        observed = _observed_dc(config, prepared, name, result, return_v)
        return replace(result, observed=observed)
    v_cycle, i_cycle = analysis.cycle_average(v, period), analysis.cycle_average(i, period)
    is_sine = waveform is not None and waveform.shape is Shape.SINE
    other_sine = config.role is Role.HI_Z and _other_is_sine(prepared, name)
    v_fit = analysis.sine_fit(v, period) if (is_sine or other_sine) else None
    i_fit = analysis.sine_fit(i, period) if is_sine else None
    frequency = hw.SAMPLE_RATE / period
    vs_requested = vs_delivered = None
    imposed = i_cycle if config.role is Role.SOURCE_I else v_cycle
    if config.role is not Role.HI_Z and imposed is not None:
        prediction = prepared.predictions[name]
        vs_requested = _diffs(imposed, prediction.requested)
        vs_delivered = _diffs(imposed, prediction.delivered)
    impedance = phase = None
    if v_fit is not None and i_fit is not None and i_fit.amplitude >= checks.RESISTANCE_MIN_CURRENT_A:
        impedance = v_fit.amplitude / i_fit.amplitude
        phase = math.remainder(v_fit.phase - i_fit.phase, 2 * math.pi)
    result = ChannelResult(**base, v_cycle=v_cycle, i_cycle=i_cycle, v_fit=v_fit, i_fit=i_fit,
                           frequency=frequency, vs_requested=vs_requested, vs_delivered=vs_delivered,
                           impedance=impedance, phase_vi=phase)
    if config.circuit.kind is CircuitKind.SHUNT and return_v is not None:
        shunt = analysis.sine_fit((v - return_v) / config.circuit.r, period) if other_sine else \
            analysis.cycle_average((v - return_v) / config.circuit.r, period)
        result = replace(result, shunt=shunt)
    return replace(result, observed=_observed_periodic(config, prepared, name, result, return_v))


def _other_is_sine(prepared: Prepared, name: str) -> bool:
    other = prepared.config("B" if name == "A" else "A")
    return other.waveform is not None and other.waveform.shape is Shape.SINE


def _return_voltage(config: ChannelConfig, prepared: Prepared, name: str) -> float | None:
    if name in prepared.predictions:
        return prepared.predictions[name].return_voltage
    if config.circuit.ret is None:
        return None
    return checks.return_voltage(config, prepared.rest[name], hw.load_unit().open_input)


def _levels(result: ChannelResult) -> tuple[float | None, float | None, float | None, float | None]:
    """(v level, i level, v amplitude, i amplitude): fit for a sine, cycle average otherwise."""
    def of(fit: SineFit | None, cycle: CycleAverage | None) -> tuple[float | None, float | None]:
        if fit is not None:
            return fit.level, fit.amplitude
        if cycle is not None:
            return cycle.level, cycle.peak_to_peak / 2
        return None, None
    v_level, v_amplitude = of(result.v_fit, result.v_cycle)
    i_level, i_amplitude = of(result.i_fit, result.i_cycle)
    return v_level, i_level, v_amplitude, i_amplitude


def _requested_amplitude(config: ChannelConfig, prediction: Prediction) -> float:
    if config.waveform.shape is Shape.SINE:
        return config.waveform["pp"] / 2
    return float(np.ptp(prediction.requested)) / 2


def _observed_periodic(config: ChannelConfig, prepared: Prepared, name: str, result: ChannelResult,
                       return_v: float | None) -> Observed:
    v_level, i_level, v_amplitude, i_amplitude = _levels(result)
    requested = None
    if config.role is not Role.HI_Z and name in prepared.predictions:
        requested = _requested_amplitude(config, prepared.predictions[name])
    if v_level is None:                                       # stopped before one whole period
        return Observed(result.vi_swaps, None, None, rail_fraction=result.rail_fraction, return_voltage=return_v,
                        halved=result.halved)
    return Observed(result.vi_swaps, v_level, i_level, v_amplitude, i_amplitude, requested_amplitude=requested,
                    rail_fraction=result.rail_fraction, return_voltage=return_v, halved=result.halved)


def _observed_dc(config: ChannelConfig, prepared: Prepared, name: str, result: ChannelResult,
                 return_v: float | None) -> Observed:
    voltage, current = result.voltage, result.current
    if voltage is None or current is None:
        return Observed(result.vi_swaps, None, None, rail_fraction=result.rail_fraction, return_voltage=return_v,
                        halved=result.halved)
    delivered = other_drift = other_u = None
    if name in prepared.predictions:
        delivered = float(prepared.predictions[name].delivered[0])
        other = voltage if config.role is Role.SOURCE_I else current
        other_drift, other_u = other.drift, other.u
    return Observed(result.vi_swaps, voltage.value, current.value, delivered_level=delivered,
                    rail_fraction=result.rail_fraction, other_drift=other_drift, other_u=other_u,
                    return_voltage=return_v, halved=result.halved)


def _with_shunt_node(result: ChannelResult, prepared: Prepared, other: str, samples: Samples) -> ChannelResult:
    """The shunt-node channel: the current through the other channel's shunt, (V_node - return) /
    shunt, and the load's voltage, V(other channel) - V(node)."""
    source = prepared.config(other)
    return_v = prepared.predictions[other].return_voltage if other in prepared.predictions else \
        float(source.circuit.nominal_return)
    observed = result.observed
    if observed is None or observed.v_level is None:
        return result
    shunt_level = (observed.v_level - return_v) / source.circuit.shunt
    shunt_amplitude = observed.v_amplitude / source.circuit.shunt if observed.v_amplitude is not None else None
    if result.v_fit is not None:
        shunt: Summary | CycleAverage | SineFit | None = SineFit(
            shunt_level, shunt_amplitude, result.v_fit.phase, result.v_fit.residual / source.circuit.shunt,
            result.v_fit.samples)
    elif result.v_cycle is not None:
        shunt = CycleAverage((result.v_cycle.mean - return_v) / source.circuit.shunt,
                             result.v_cycle.std / source.circuit.shunt, result.v_cycle.periods)
    else:
        voltage = result.voltage
        shunt = replace(voltage, value=shunt_level, u=voltage.u / source.circuit.shunt,
                        spread=voltage.spread / source.circuit.shunt, sigma=voltage.sigma / source.circuit.shunt,
                        drift=voltage.drift / source.circuit.shunt)
    load = samples.channel(other)[0] - samples.channel(result.channel)[0]
    if result.period is None:
        load_voltage = analysis.summarize(load[result.settling_samples:], hw.V_STEP)
    elif result.v_fit is not None:
        load_voltage = analysis.sine_fit(load, result.period)
    else:
        load_voltage = analysis.cycle_average(load, result.period)
    return replace(result, shunt=shunt, load_voltage=load_voltage,
                   observed=replace(observed, shunt_level=shunt_level, shunt_amplitude=shunt_amplitude))


def culture_values(config: ChannelConfig, result: ChannelResult) -> dict[str, float]:
    """The i-v-i board: the requested and measured current at the culture (scale x the M1K's)."""
    if config.circuit.kind is not CircuitKind.IVI_BOARD or result.observed is None:
        return {}
    observed = result.observed
    scale = config.circuit.scale
    values = {}
    if observed.i_level is not None:
        values["culture_level_a"] = scale * observed.i_level
    if observed.i_amplitude is not None:
        values["culture_amplitude_a"] = scale * observed.i_amplitude
    if observed.requested_amplitude is not None:
        values["culture_requested_amplitude_a"] = scale * observed.requested_amplitude
    if observed.delivered_level is not None:
        values["culture_delivered_level_a"] = scale * observed.delivered_level
    return values
