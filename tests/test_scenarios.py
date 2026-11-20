"""The signal acceptance scenarios S1 to S28 (spec 9.2), against the simulator.

"The right circuit is the one informed, simulated with 0.4 ohm of wires and the real pins": the
bench is `simulator.bench_for(...)` unless the scenario mounts something else on purpose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phycovolt import checks as ch
from phycovolt import circuits as ci
from phycovolt import hardware as hw
from phycovolt import recording as rec
from phycovolt import signals
from phycovolt import simulator as sim
from phycovolt import waveforms as wf
from phycovolt.checks import Check, Level
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import InstrumentError, Role
from phycovolt.waveforms import Shape

HI_Z = ChannelConfig()


def source(role: Role, shape: Shape, circuit: Circuit | None = None, **values: float) -> ChannelConfig:
    circuit = circuit or ci.default_circuit(role)
    waveform = wf.default_waveform(role, shape, circuit.ret)
    if values:
        waveform = wf.Waveform(shape, {**waveform.values, **values})
    return ChannelConfig(role, circuit, waveform)


def measure(a: ChannelConfig, b: ChannelConfig = HI_Z, bench: sim.Bench | None = None,
            faults: sim.Faults | None = None, seed: int = 1) -> signals.Measurement:
    m1k = sim.SimulatedM1K(bench or sim.bench_for(a, b), faults=faults, seed=seed)
    prepared = signals.prepare(m1k, a, b)
    assert not prepared.blocked, [c.text for c in prepared.checks if c.level is Level.ERROR]
    return signals.run(m1k, prepared)


def every_check(measurement: signals.Measurement) -> list[Check]:
    return measurement.prepared.checks + measurement.after


def no_warning(measurement: signals.Measurement) -> None:
    bad = [c.text for c in every_check(measurement) if c.level in (Level.WARNING, Level.ERROR)]
    assert bad == []


def find(found: list[Check], code: str, channel: str = "A") -> Check:
    matches = [c for c in found if c.code == code and c.channel == channel]
    assert len(matches) == 1, (code, channel, [c.text for c in found])
    return matches[0]


# --- Defaults and the right wiring ----------------------------------------------------------------


def test_s1_source_i_sine_defaults() -> None:
    measurement = measure(source(Role.SOURCE_I, Shape.SINE))
    no_warning(measurement)
    observed = measurement.results["A"].observed
    assert observed.i_amplitude == pytest.approx(0.1, rel=0.01)


def test_s2_source_i_sine_on_channel_b_into_5v() -> None:
    config = source(Role.SOURCE_I, Shape.SINE, Circuit(CircuitKind.RESISTOR, "5V", r=10.0))
    no_warning(measure(HI_Z, config))


def test_s3_source_i_dc_defaults() -> None:
    measurement = measure(source(Role.SOURCE_I, Shape.DC))
    no_warning(measurement)
    observed = measurement.results["A"].observed
    assert observed.i_level == pytest.approx(observed.delivered_level, abs=60e-6 + 0.01 * 10e-3)


def test_s4_source_i_dc_into_5v() -> None:
    no_warning(measure(source(Role.SOURCE_I, Shape.DC, Circuit(CircuitKind.RESISTOR, "5V", r=10.0))))


def test_s5_source_v_sine_defaults() -> None:
    measurement = measure(source(Role.SOURCE_V, Shape.SINE))
    no_warning(measurement)
    assert measurement.results["A"].observed.v_amplitude == pytest.approx(1.0, rel=0.005)


def test_s6_s7_charge_balance() -> None:
    circuit = Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0)
    first = measure(source(Role.SOURCE_V, Shape.SINE, circuit))
    note = find(first.prepared.checks, "charge_balance")
    assert note.level is Level.NOTE and note.numbers["center_v"] == pytest.approx(2.446, abs=1e-3)
    second = measure(source(Role.SOURCE_V, Shape.SINE, circuit, center=note.numbers["center_v"]))
    assert find(second.prepared.checks, "charge_balance").level is Level.OK
    assert abs(second.results["A"].observed.i_level) < 60e-6


@pytest.mark.parametrize("shape, ret", [(Shape.SINE, "5V"), (Shape.DC, "5V"), (Shape.DC, "2.5V")])
def test_s8_source_v_on_5v_and_2v5(shape: Shape, ret: str) -> None:
    measurement = measure(source(Role.SOURCE_V, shape, Circuit(CircuitKind.LOAD_SHUNT, ret, r=47.0, shunt=10.0)))
    no_warning(measurement)
    if ret == "5V":
        assert measurement.results["A"].observed.i_level < 0              # into the channel


# --- Wrong wiring and faults ----------------------------------------------------------------------


def test_s9_source_i_wired_to_gnd_instead_of_2v5() -> None:
    measurement = measure(source(Role.SOURCE_I, Shape.SINE), bench=sim.Bench(a=sim.Wiring(10.4, "GND")))
    assert find(measurement.prepared.checks, "wiring").level is Level.WARNING
    assert find(measurement.after, "pin_limit").level is Level.ERROR
    returned = find(measurement.after, "return")
    assert returned.level is Level.WARNING and returned.numbers.get("seen") == "GND"


def test_s10_and_s13_wiring_errors_keep_the_outputs_off() -> None:
    for config, bench in (
        (source(Role.SOURCE_V, Shape.SINE), sim.Bench(a=sim.Wiring(57.4, "2.5V"))),
        (source(Role.SOURCE_V, Shape.DC, Circuit(CircuitKind.LOAD_SHUNT, "5V", r=47.0, shunt=10.0)),
         sim.Bench(a=sim.Wiring(57.4, "GND"))),
    ):
        m1k = sim.SimulatedM1K(bench)
        prepared = signals.prepare(m1k, config, HI_Z)
        assert prepared.blocked and find(prepared.checks, "wiring").level is Level.ERROR
        with pytest.raises(ValueError):
            signals.run(m1k, prepared)


def test_s11_jumpers_add_25_ohm() -> None:
    config = source(Role.SOURCE_I, Shape.SINE, pp=0.1)
    measurement = measure(config, bench=sim.Bench(a=sim.Wiring(10.4 + 25.0, "2.5V")))
    check = find(measurement.after, "resistance")
    assert check.level is Level.WARNING and check.numbers["seen_ohm"] == pytest.approx(35.4, rel=0.01)


def test_s12_short_circuit_trips_the_limiter() -> None:
    measurement = measure(source(Role.SOURCE_V, Shape.SINE), bench=sim.Bench(a=sim.Wiring(5.0, "GND")))
    assert find(measurement.after, "overcurrent").level is Level.ERROR


def test_s14_circuit_on_the_other_channel() -> None:
    measurement = measure(HI_Z, source(Role.SOURCE_V, Shape.DC), bench=sim.Bench(a=sim.Wiring(57.4, "GND")))
    wiring = find(measurement.prepared.checks, "wiring", "B")
    assert wiring.level is Level.WARNING and "CH A" in wiring.text
    assert find(measurement.after, "resistance", "B").level is Level.WARNING


def test_s15_high_resistance() -> None:
    config = source(Role.SOURCE_I, Shape.DC, Circuit(CircuitKind.RESISTOR, "2.5V", r=100e3), level=10e-6)
    measurement = measure(config)
    assert find(measurement.prepared.checks, "wiring").level is Level.NOTE


def test_s16_vi_swap_on_channel_b() -> None:
    config = source(Role.SOURCE_I, Shape.SINE, Circuit(CircuitKind.RESISTOR, "5V", r=10.0))
    measurement = measure(HI_Z, config, faults=sim.Faults(vi_swaps_per_second={"B": 50}))
    assert find(measurement.after, "board", "B").level is Level.ERROR


def test_s17_and_s18_before_switching_on() -> None:
    m1k = sim.SimulatedM1K()
    big = source(Role.SOURCE_I, Shape.SINE, Circuit(CircuitKind.RESISTOR, "2.5V", r=100.0))
    m1k.bench = sim.bench_for(big, HI_Z)
    prepared = signals.prepare(m1k, big, HI_Z)
    assert find(prepared.checks, "pin_limit").level is Level.ERROR
    m1k.bench = sim.bench_for(source(Role.SOURCE_I, Shape.SINE), HI_Z)
    at_60 = signals.prepare(m1k, source(Role.SOURCE_I, Shape.SINE, freq=60, periods=60), HI_Z)
    at_10 = signals.prepare(m1k, source(Role.SOURCE_I, Shape.SINE, freq=10, periods=10), HI_Z)
    assert find(at_60.checks, "mains").level is Level.WARNING
    assert find(at_10.checks, "mains").level is Level.NOTE


@pytest.mark.parametrize("a_text, b_text", [
    ("source-v sine circuit=load-shunt load=47 shunt=10 return=2.5V", "hi-z circuit=shunt-node"),
    ("source-i sine circuit=resistor r=10 return=5V", "hi-z"),
])
def test_s20_a_new_waveform_keeps_the_channels_and_circuits(a_text: str, b_text: str) -> None:
    """S20 on the command line: in a repeat command the operator replaces only a channel's
    waveform (its word and its fields). The channel keeps its circuit, the other channel stays as it
    was, and the new waveform takes the defaults of that circuit's return (spec 3.3.3)."""
    a, b = signals.parse_channel(a_text), signals.parse_channel(b_text)
    kept = ("circuit", *ci.CIRCUIT_FIELDS[a.circuit.kind])
    circuit_words = [word for word in signals.channel_text(a).split() if word.split("=")[0] in kept]
    for shape in wf.offered_shapes(a.role, a.circuit.ret):
        changed = signals.parse_channel(" ".join([a.role.value, shape.value, *circuit_words]))
        assert (changed.role, changed.circuit) == (a.role, a.circuit)
        assert changed.waveform == wf.default_waveform(a.role, shape, a.circuit.ret)
        assert signals.parse_channel(signals.channel_text(b)) == b
        ci.check_pair(changed, b)                        # the pair is still one that can be measured


def test_s21_usb_failure_in_the_rest_reading_and_in_the_burst(tmp_path: Path) -> None:
    config = source(Role.SOURCE_I, Shape.DC)
    m1k = sim.SimulatedM1K(sim.bench_for(config, HI_Z), faults=sim.Faults(usb_failure_in_rest_read=True))
    with pytest.raises(InstrumentError):
        signals.prepare(m1k, config, HI_Z)
    prepared = signals.prepare(m1k, config, HI_Z)            # the session goes on
    m1k.faults.usb_failure_at_chunk = 3
    with pytest.raises(InstrumentError):
        signals.run(m1k, prepared)                           # nothing is returned, so nothing recorded
    assert m1k.modes == {"A": Role.HI_Z, "B": Role.HI_Z}
    m1k.faults.usb_failure_at_chunk = None
    assert signals.run(m1k, signals.prepare(m1k, config, HI_Z)).ended == "complete"
    assert not (tmp_path / "recordings").exists()


# --- Two channels, the other waveforms, the board -------------------------------------------------


def test_s22_shunt_node() -> None:
    a = source(Role.SOURCE_V, Shape.SINE, Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0))
    b = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE))
    measurement = measure(a, b)
    no_warning(measurement)
    assert find(measurement.after, "shunt_agreement", "B").level is Level.OK
    node = measurement.results["B"]
    assert node.observed.shunt_amplitude == pytest.approx(measurement.results["A"].observed.i_amplitude, rel=0.01)
    assert node.load_voltage.amplitude == pytest.approx(1.0 * 47.4 / 57.4, rel=0.01)


@pytest.mark.parametrize("shape", [Shape.SQUARE, Shape.TRIANGLE, Shape.PULSE, Shape.BIPHASIC])
def test_s23_other_waveforms(shape: Shape) -> None:
    measurement = measure(source(Role.SOURCE_I, shape))
    no_warning(measurement)
    rms, largest = measurement.results["A"].vs_delivered
    assert rms < 50e-6 and largest < 0.5e-3                  # the cycle average is the DAC's waveform


def test_s24_ivi_board() -> None:
    config = source(Role.SOURCE_I, Shape.SINE, Circuit(CircuitKind.IVI_BOARD, wf.BOARD, scale=0.01))
    measurement = measure(config)
    no_warning(measurement)
    culture = signals.culture_values(config, measurement.results["A"])
    observed = measurement.results["A"].observed
    assert culture["culture_amplitude_a"] == pytest.approx(0.01 * observed.i_amplitude)
    assert culture["culture_requested_amplitude_a"] == pytest.approx(20e-6)


def test_s25_stop_in_the_middle(tmp_path: Path) -> None:
    config = source(Role.SOURCE_I, Shape.SINE)
    m1k = sim.SimulatedM1K(sim.bench_for(config, HI_Z))
    prepared = signals.prepare(m1k, config, HI_Z)
    seen = []

    def on_chunk(chunk: hw.Chunk, done: int, total: int) -> None:
        seen.append(done)

    measurement = signals.run(m1k, prepared, on_chunk=on_chunk, should_stop=lambda: len(seen) >= 4)
    assert seen == [1, 2, 3, 4]                               # ended at the current chunk
    assert m1k.modes == {"A": Role.HI_Z, "B": Role.HI_Z}
    assert measurement.ended == "stopped" and measurement.samples.count == 4 * hw.CHUNK_SAMPLES
    folder = rec.write(measurement, tmp_path)
    assert "STOPPED · by the operator at 0.4 s of 1.0 s" in (folder / rec.SUMMARY_FILE).read_text(encoding="utf-8")


def test_stopped_before_one_period_is_still_recorded(tmp_path: Path) -> None:
    config = source(Role.SOURCE_I, Shape.PULSE)                # 10 Hz: one period is a chunk
    m1k = sim.SimulatedM1K(sim.bench_for(config, HI_Z))
    measurement = signals.run(m1k, signals.prepare(m1k, config, HI_Z), should_stop=lambda: True)
    assert measurement.ended == "stopped"
    text = rec.summary_text(measurement, None)
    assert "no complete period or block" in text
    rec.write(measurement, tmp_path)


# --- S26: measure-only ----------------------------------------------------------------------------

SHUNT_A = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT, "2.5V", r=10.0))


def test_s26_measure_only_sends_no_park_and_no_source() -> None:
    from phycovolt import m1k
    from tests import fake_pysmu

    fake = fake_pysmu.make()
    instrument = m1k.open_m1k(fake.module())
    opened = len(fake.log)
    prepared = signals.prepare(instrument, SHUNT_A, HI_Z, measure_seconds=2.0)
    measurement = signals.run(instrument, prepared)
    log = fake.log[opened:]
    rest = [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("flush",), ("get_samples", hw.CHUNK_SAMPLES)]
    burst = [("flush",), ("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("start", 20 * hw.CHUNK_SAMPLES)]
    assert log[:4] == rest                                   # the rest reading of prepare
    assert log[4:8] == burst                                 # no park, no write
    assert log[8:28] == [("read", hw.CHUNK_SAMPLES, hw.READ_TIMEOUT_MS)] * 20
    assert log[28:] == [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("end",), ("overcurrent",)]
    assert not any(entry[0] == "write" for entry in log)
    assert measurement.ended == "complete"


def test_s26_measure_only_recording(tmp_path: Path) -> None:
    m1k = sim.SimulatedM1K(sim.bench_for(SHUNT_A, HI_Z), seed=7)
    prepared = signals.prepare(m1k, SHUNT_A, HI_Z, measure_seconds=2.0)
    assert prepared.measure_only and prepared.chunks == 20 and not prepared.blocked
    measurement = signals.run(m1k, prepared)
    folder = rec.write(measurement, tmp_path)
    assert folder.name.endswith("_hi-z")
    results = rec.read(folder).measurement.results
    assert results["A"].voltage.value == pytest.approx(2.4457, abs=0.3e-3)
    assert results["B"].voltage.value == pytest.approx(hw.load_unit().open_input, abs=0.3e-3)   # open input
    assert results["A"].shunt.value == pytest.approx(0.0, abs=60e-6)           # no current flows
    text = (folder / rec.SUMMARY_FILE).read_text(encoding="utf-8")
    assert "COMPLETE · 2.0 s of 2.0 s; nothing sourced" in text and "  shunt current  " in text


def test_measure_only_default_and_limits() -> None:
    assert signals.burst_chunks(HI_Z, HI_Z) == 10                              # 1 s
    with pytest.raises(ValueError):
        signals.burst_chunks(source(Role.SOURCE_I, Shape.DC), HI_Z, 2.0)       # a waveform sets it


# --- S27, S28: the 2026-11-21 bench findings ------------------------------------------------------


def test_s27_halved_words_on_channel_b() -> None:
    measurement = measure(source(Role.SOURCE_I, Shape.SINE), faults=sim.Faults(halved_words_per_second={"B": 5}),
                          seed=27)
    halved = [c for c in measurement.after if c.code == "halved_word" and c.channel == "B"]
    assert [c.level for c in halved] == [Level.WARNING]
    assert halved[0].numbers["quantity"] == "current" and halved[0].numbers["count_count"] > 0
    assert find(measurement.after, "board", "B").level is Level.OK          # not a V/I swap
    assert find(measurement.after, "halved_word", "A").level is Level.OK


def test_s28_dac_ceiling() -> None:
    config = source(Role.SOURCE_V, Shape.DC, level=5.0)
    m1k = sim.SimulatedM1K(sim.bench_for(config, HI_Z))
    check = find(signals.prepare(m1k, config, HI_Z).checks, "dac_ceiling")
    assert check.level is Level.WARNING
    assert check.numbers["ceiling_v"] == pytest.approx(4.9936, abs=0.5e-3)    # H-16
