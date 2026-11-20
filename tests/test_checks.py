"""Do the before-checks catch what the instrument cannot do, and stay quiet otherwise? (spec 3.3.4)"""

from __future__ import annotations

import numpy as np
import pytest

from phycovolt import checks as ch
from phycovolt import circuits as ci
from phycovolt import hardware as hw
from phycovolt import simulator as sim
from phycovolt import waveforms as wf
from phycovolt.checks import Check, Level
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import Role
from phycovolt.waveforms import Shape

HI_Z = ChannelConfig()
UNIT = hw.load_unit()


def source(role: Role, shape: Shape, circuit: Circuit | None = None, **values: float) -> ChannelConfig:
    circuit = circuit or ci.default_circuit(role)
    waveform = wf.default_waveform(role, shape, circuit.ret)
    if values:
        waveform = wf.Waveform(shape, {**waveform.values, **values})
    return ChannelConfig(role, circuit, waveform)


def run(a: ChannelConfig, b: ChannelConfig = HI_Z, bench: sim.Bench | None = None,
        faults: sim.Faults | None = None) -> list[Check]:
    """The before-checks on a simulated rest reading of `bench` (default: the informed circuits)."""
    m1k = sim.SimulatedM1K(bench or sim.bench_for(a, b), faults=faults, seed=1)
    return ch.before_checks(a, b, m1k.rest_read(ch.REST_SECONDS), m1k.calibration())


def find(checks: list[Check], code: str, channel: str = "A") -> Check:
    found = [check for check in checks if check.code == code and check.channel == channel]
    assert len(found) == 1, (code, channel, found)
    return found[0]


def levels(checks: list[Check]) -> set[Level]:
    return {check.level for check in checks}


# --- S19: every default, every role and return, on both channels ---------------------------------

def default_configs() -> list[ChannelConfig]:
    configs = []
    for role in (Role.SOURCE_I, Role.SOURCE_V):
        for kind in ci.circuits_for(role):
            for ret in ci.RETURNS[kind]:
                circuit = ci.circuit_from_text(kind, {"scale": "0.01"} if kind is CircuitKind.IVI_BOARD
                                               else {"return": ret})
                for shape in wf.offered_shapes(role, ret):
                    configs.append(ChannelConfig(role, circuit, wf.default_waveform(role, shape, ret)))
    return configs


@pytest.mark.parametrize("channel", ["A", "B"])
def test_s19_defaults_are_never_refused(channel: str) -> None:
    configs = default_configs()
    assert len(configs) == 6 + 5 + 6 + 3 * 6                   # 2.5V, 5V, board; source-v x 3 returns
    for config in configs:
        a, b = (config, HI_Z) if channel == "A" else (HI_Z, config)
        checks = run(a, b)
        errors = [check.text for check in checks if check.level is Level.ERROR]
        assert errors == [], (config, errors)
        warnings = [check.text for check in checks if check.level is Level.WARNING]
        assert warnings == [], (config, warnings)          # not even a warning
        for check in checks:
            assert check.text                                  # every text formats


def test_s1_s3_s5_defaults_give_no_warning() -> None:
    for config in (source(Role.SOURCE_I, Shape.SINE), source(Role.SOURCE_I, Shape.DC),
                   source(Role.SOURCE_V, Shape.SINE)):
        assert Level.WARNING not in levels(run(config)) and Level.ERROR not in levels(run(config))


# --- Wiring (S9, S10, S13, S14, S15) -------------------------------------------------------------


def test_s10_svmi_with_the_return_on_another_pin_is_an_error() -> None:
    config = source(Role.SOURCE_V, Shape.SINE)                              # GND chosen
    check = find(run(config, bench=sim.Bench(a=sim.Wiring(57.4, "2.5V"))), "wiring")
    assert check.level is Level.ERROR and check.numbers["seen"] == "2.5V"
    assert "2.5V pin, not the GND return" in check.text


def test_s13_svmi_5v_chosen_wire_on_gnd() -> None:
    config = source(Role.SOURCE_V, Shape.DC, Circuit(CircuitKind.LOAD_SHUNT, "5V", r=47.0, shunt=10.0))
    check = find(run(config, bench=sim.Bench(a=sim.Wiring(57.4, "GND"))), "wiring")
    assert check.level is Level.ERROR and check.numbers["seen"] == "GND"


def test_s9_simv_with_the_return_on_another_pin_is_a_warning() -> None:
    config = source(Role.SOURCE_I, Shape.SINE)                              # 2.5V chosen
    check = find(run(config, bench=sim.Bench(a=sim.Wiring(10.4, "GND"))), "wiring")
    assert check.level is Level.WARNING and check.variant == "other_pin"


def test_s14_circuit_on_the_other_channel() -> None:
    config = source(Role.SOURCE_V, Shape.DC)                                # on CH B, GND
    checks = run(HI_Z, config, bench=sim.Bench(a=sim.Wiring(57.4, "GND")))
    check = find(checks, "wiring", "B")
    assert check.level is Level.WARNING and check.variant == "far_other"
    assert "CH A" in check.text


def test_s15_high_impedance_cannot_judge_the_return() -> None:
    config = source(Role.SOURCE_I, Shape.DC, Circuit(CircuitKind.RESISTOR, "2.5V", r=100e3), level=10e-6)
    check = find(run(config), "wiring")
    assert check.level is Level.NOTE and check.variant == "high_impedance"


def test_open_channel_is_far_from_every_pin() -> None:
    check = find(run(source(Role.SOURCE_I, Shape.SINE), bench=sim.Bench()), "wiring")
    assert check.level is Level.WARNING and check.variant == "far"


# --- The current source's compliance (S17) -------------------------------------------------------


def test_s17_pin_limit_gives_the_possible_range() -> None:
    config = source(Role.SOURCE_I, Shape.SINE, Circuit(CircuitKind.RESISTOR, "2.5V", r=100.0))
    check = find(run(config), "pin_limit")
    assert check.level is Level.ERROR and check.variant == "out_periodic"
    assert check.numbers["i_min_a"] == pytest.approx(-24e-3) and check.numbers["i_max_a"] == pytest.approx(24e-3)
    assert check.numbers["amplitude_a"] == pytest.approx(24e-3)
    assert ch.blocks_switch_on(run(config))


def test_pin_limit_into_5v() -> None:
    config = source(Role.SOURCE_I, Shape.DC, Circuit(CircuitKind.RESISTOR, "5V", r=10.0), level=10e-3)
    check = find(run(config), "pin_limit")                 # +10 mA cannot flow out towards 5V
    assert check.level is Level.ERROR and check.numbers["i_max_a"] == pytest.approx(-0.01e-3 * 1000)


# --- The waveform (S18 and the rest of the table) ------------------------------------------------


def test_s18_mains() -> None:
    at_60 = find(run(source(Role.SOURCE_I, Shape.SINE, freq=60, periods=60)), "mains")
    at_10 = find(run(source(Role.SOURCE_I, Shape.SINE, freq=10, periods=10)), "mains")
    at_15 = find(run(source(Role.SOURCE_I, Shape.SINE, freq=15, periods=15)), "mains")   # 6667 samples
    assert (at_60.level, at_10.level, at_15.level) == (Level.WARNING, Level.NOTE, Level.NOTE)
    assert at_10.numbers["ratio_count"] == 6
    assert [c for c in run(source(Role.SOURCE_I, Shape.SINE, freq=100)) if c.code == "mains"] == []


def test_frequency_rounding() -> None:
    check = find(run(source(Role.SOURCE_I, Shape.SINE, freq=3_000, periods=300)), "frequency")
    assert check.level is Level.WARNING and check.numbers["real_hz"] == pytest.approx(3030.3, abs=0.1)
    small = find(run(source(Role.SOURCE_I, Shape.SINE, freq=99.99, periods=100)), "frequency")
    assert small.level is Level.NOTE


def test_samples_per_period() -> None:
    few = find(run(source(Role.SOURCE_I, Shape.SINE, freq=2_000, periods=200)), "samples_per_period")
    assert few.level is Level.WARNING
    too_few = find(run(source(Role.SOURCE_I, Shape.SINE, freq=6_000, periods=600)), "samples_per_period")
    assert too_few.level is Level.ERROR


@pytest.mark.parametrize("config, variant", [
    (source(Role.SOURCE_I, Shape.SINE, periods=2_000), "too_long"),         # 20 s
    (source(Role.SOURCE_I, Shape.DC, seconds=0.2), "short_dc"),
    (source(Role.SOURCE_I, Shape.SINE, periods=1), "few_periods"),
])
def test_duration(config: ChannelConfig, variant: str) -> None:
    check = find(run(config), "duration")
    assert check.level is Level.ERROR and check.variant == variant


def test_micro_amps_staircase_and_dac_level() -> None:
    sine = run(source(Role.SOURCE_I, Shape.SINE, pp=20e-6, center=0.0))
    assert find(sine, "dac_staircase").level is Level.WARNING           # 20 uA p-p: 3 levels
    assert find(sine, "micro_amps").level is Level.NOTE
    dc = find(run(source(Role.SOURCE_I, Shape.DC, level=50e-6)), "dac_level")
    assert dc.level is Level.WARNING                                     # 47.5 uA for 50 uA
    square = run(source(Role.SOURCE_I, Shape.SQUARE))
    assert find(square, "dac_staircase").level is Level.OK               # 2 levels asked, 2 given


def test_voltage_range_and_dac_floor() -> None:
    out = find(run(source(Role.SOURCE_V, Shape.SINE, pp=6.0)), "range")
    assert out.level is Level.ERROR
    floor = find(run(source(Role.SOURCE_V, Shape.DC, level=0.0)), "dac_floor")
    assert floor.level is Level.WARNING and floor.numbers["floor_v"] == pytest.approx(2.56e-3, abs=0.05e-3)


def test_current_limit_and_power() -> None:
    small_load = Circuit(CircuitKind.LOAD_SHUNT, "GND", r=12.0, shunt=10.0)
    near = run(source(Role.SOURCE_V, Shape.DC, small_load, level=3.5))       # 159 mA
    assert find(near, "current_limit").level is Level.WARNING
    power = find(near, "power")                                             # the hottest resistor
    assert power.level is Level.WARNING and power.numbers["part"] == "load"
    assert power.numbers["power_w"] == pytest.approx(0.159**2 * 12, rel=0.01)
    assert power.numbers["rating_w"] == 1.0                                 # holds twice 0.30 W
    over = run(source(Role.SOURCE_V, Shape.DC, small_load, level=5.0))      # 227 mA
    assert find(over, "current_limit").level is Level.ERROR


def test_switch_on() -> None:
    simv = find(run(source(Role.SOURCE_I, Shape.SINE)), "switch_on")
    assert simv.level is Level.NOTE and simv.variant == "parked"
    assert simv.numbers["current_a"] == pytest.approx(-0.40e-3, abs=0.02e-3)   # CH A's source offset
    short = Circuit(CircuitKind.LOAD_SHUNT, "GND", r=5.0)
    svmi = find(run(source(Role.SOURCE_V, Shape.SINE, short, pp=0.2, center=0.5)), "switch_on")
    assert svmi.level is Level.WARNING and svmi.numbers["current_a"] == pytest.approx(0.4, abs=0.01)


# --- Charge balance (S6, S7) ---------------------------------------------------------------------


def test_s6_s7_charge_balance_with_the_2v5_return() -> None:
    circuit = Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0)
    note = find(run(source(Role.SOURCE_V, Shape.SINE, circuit)), "charge_balance")
    assert note.level is Level.NOTE and note.variant == "center"
    suggested = note.numbers["center_v"]
    assert suggested == pytest.approx(2.446, abs=1e-3)
    balanced = find(run(source(Role.SOURCE_V, Shape.SINE, circuit, center=suggested)), "charge_balance")
    assert balanced.level is Level.OK


def test_charge_balance_of_a_current_source() -> None:
    assert find(run(source(Role.SOURCE_I, Shape.BIPHASIC)), "charge_balance").level is Level.OK
    offset = find(run(source(Role.SOURCE_I, Shape.SINE, center=5e-3)), "charge_balance")
    assert offset.level is Level.NOTE and offset.numbers["mean_a"] == pytest.approx(5e-3, abs=0.02e-3)


# --- The calibration record ----------------------------------------------------------------------


def test_calibration_at_the_default() -> None:
    check = find(run(source(Role.SOURCE_I, Shape.SINE), faults=sim.Faults(eeprom_default=True)), "calibration")
    assert check.level is Level.WARNING and check.variant == "default"


def test_implausible_records() -> None:
    assert ch.record_problem(hw.Record(0.0, 1.0, 1.0), hw.Kind.MEASURE_V) == "default"
    assert ch.record_problem(hw.Record(0.0006, 1.02, 1.0), hw.Kind.MEASURE_V) is None
    assert ch.record_problem(hw.Record(0.0006, -1.001, 1.0), hw.Kind.MEASURE_I) == "implausible"  # a lost minus sign
    assert ch.record_problem(hw.Record(0.06, 1.02, 1.0), hw.Kind.SOURCE_V) == "implausible"
    statuses = ch.stored_calibration_checks([hw.Record(0.0006, 3e6, 1.0)] + [hw.Record()] * 7)
    assert statuses[0].level is Level.ERROR and statuses[1].level is Level.WARNING


# --- Two channels (S22, S24) ---------------------------------------------------------------------


def test_s22_shunt_node_before_checks() -> None:
    a = source(Role.SOURCE_V, Shape.SINE, Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0))
    b = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE))
    checks = run(a, b)
    assert find(checks, "hi_z_range", "B").level is Level.OK
    assert Level.ERROR not in levels(checks)


def test_s24_ivi_board_defaults() -> None:
    board = Circuit(CircuitKind.IVI_BOARD, wf.BOARD, scale=0.01)
    checks = run(source(Role.SOURCE_I, Shape.SINE, board))
    assert Level.WARNING not in levels(checks) and Level.ERROR not in levels(checks)
    assert find(checks, "wiring").level is Level.OK


def test_return_voltage_comes_from_the_rest_reading() -> None:
    config = source(Role.SOURCE_I, Shape.SINE)
    assert ch.return_voltage(config, 2.4457, UNIT.open_input) == pytest.approx(2.4457, abs=0.1e-3)
    assert ch.return_voltage(config, 0.0, UNIT.open_input) == 2.5          # wrong pin: the nominal


def test_unbuildable_waveform_is_an_error_not_a_crash() -> None:
    config = source(Role.SOURCE_I, Shape.SQUARE, duty=0.0001)
    check = find(run(config), "parameters")
    assert check.level is Level.ERROR and "duty" in check.text


def test_measure_only_duration() -> None:
    shunt = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT, "2.5V", r=10.0))
    m1k = sim.SimulatedM1K(sim.bench_for(shunt, HI_Z))
    rest = m1k.rest_read(ch.REST_SECONDS)
    ok = ch.before_checks(shunt, HI_Z, rest, m1k.calibration(), measure_seconds=2.0)
    assert Level.ERROR not in levels(ok) and find(ok, "wiring").level is Level.OK
    short = ch.before_checks(shunt, HI_Z, rest, m1k.calibration(), measure_seconds=0.2)
    assert find(short, "duration", None).variant == "short_measure"
    long = ch.before_checks(shunt, HI_Z, rest, m1k.calibration(), measure_seconds=11.0)
    assert find(long, "duration", None).variant == "too_long_measure"


def test_describe_writes_units() -> None:
    check = Check(Level.WARNING, "current_limit", "B", "near", {"peak_a": 0.1593})
    assert check.text == "CH B: The current would peak at 159.3 mA, close to the 200 mA limit."
    assert ch.worst([check, Check(Level.OK, "power", "B")]) is Level.WARNING
    assert np.isfinite(ch.MAX_BURST_SECONDS) and ch.MAX_BURST_SECONDS == pytest.approx(10.0)


def test_one_halved_word_reads_in_the_singular() -> None:
    one = ch.halved_word_check("B", "I", 1, -0.24e-6, 12e-6)
    two = ch.halved_word_check("B", "I", 2, -0.48e-6, 12e-6)
    assert one.text.startswith("CH B: 1 current sample carries a halved ADC word")
    assert two.text.startswith("CH B: 2 current samples carry a halved ADC word")
