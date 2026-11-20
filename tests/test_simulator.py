"""Does the simulated M1K follow the instrument contract and the physics of the bench (spec C.8)?"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from phycovolt import hardware as hw
from phycovolt import libsmu_math as lm
from phycovolt import simulator as sim
from phycovolt.hardware import ChannelSetup, InstrumentError, Kind, Role

UNIT = hw.load_unit()
PIN_2V5 = UNIT.pins["2.5V"]
HI_Z = ChannelSetup()


def dc(role: Role, value: float) -> ChannelSetup:
    return ChannelSetup(role, np.full(hw.CHUNK_SAMPLES, value))


def mean_of(m1k: sim.SimulatedM1K, a: ChannelSetup, b: ChannelSetup, chunks: int = 3
            ) -> tuple[float, float, float, float, bool]:
    with m1k.stream(a, b, chunks) as stream:
        data = list(stream)
    later = data[1:] or data            # without the first chunk (settling)
    means = [float(np.mean(np.concatenate([getattr(c, name) for c in later])))
             for name in ("v_a", "i_a", "v_b", "i_b")]
    return (*means, stream.overcurrent)


# --- The unit file --------------------------------------------------------------------------------


def test_unit_file_holds_the_project_unit() -> None:
    assert UNIT.firmware == "2.17" and UNIT.hardware == "D"
    assert UNIT.pins == {"GND": 0.0, "2.5V": 2.4457, "5V": 4.9300}
    assert UNIT.voltage_sigma == {"A": 64e-6, "B": 43e-6} and UNIT.block_correlation == 0.65
    assert UNIT.true_records[hw.record_index("A", Kind.MEASURE_V)] == hw.Record(0.00055, 1.02349, 1.0)
    assert len(UNIT.true_records) == 8


def test_stored_calibration_is_the_true_one_with_small_errors() -> None:
    stored = sim.stored_records(UNIT)
    a_mi = stored[hw.record_index("A", Kind.MEASURE_I)]
    true = UNIT.true_records[hw.record_index("A", Kind.MEASURE_I)]
    assert a_mi.offset == pytest.approx(true.offset - 10e-6)
    assert a_mi.gain_pos == pytest.approx(true.gain_pos * (1 + 40e-6))
    assert a_mi.gain_neg == pytest.approx(true.gain_neg * (1 + 40e-6))
    assert stored[hw.record_index("A", Kind.MEASURE_V)].gain_neg == 1.0


# --- The contract ---------------------------------------------------------------------------------


def test_identification() -> None:
    assert sim.SimulatedM1K().description == "simulated (no board) · fw 2.17 · hw D"


def test_stream_yields_chunks_and_ends_in_hi_z() -> None:
    m1k = sim.SimulatedM1K()
    with m1k.stream(dc(Role.SOURCE_V, 1.0), HI_Z, 4) as stream:
        assert m1k.modes == {"A": Role.SOURCE_V, "B": Role.HI_Z}
        chunks = list(stream)
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert all(c.samples == hw.CHUNK_SAMPLES for c in chunks)
    assert m1k.modes == {"A": Role.HI_Z, "B": Role.HI_Z}


def test_stop_ends_at_the_current_chunk_in_hi_z() -> None:
    m1k = sim.SimulatedM1K()
    seen = []
    with m1k.stream(dc(Role.SOURCE_I, 0.01), HI_Z, 50) as stream:
        for chunk in stream:
            seen.append(chunk.index)
            if chunk.index == 2:
                stream.stop()
    assert seen == [0, 1, 2]
    assert m1k.modes == {"A": Role.HI_Z, "B": Role.HI_Z}


def test_usb_failure_in_a_stream_and_in_a_rest_read() -> None:
    m1k = sim.SimulatedM1K(faults=sim.Faults(usb_failure_at_chunk=2, usb_failure_in_rest_read=True))
    with pytest.raises(InstrumentError) as error:
        m1k.rest_read(0.1)
    assert error.value.code == "lost_sample"
    m1k.rest_read(0.1)                                   # the session goes on
    with pytest.raises(InstrumentError):
        with m1k.stream(dc(Role.SOURCE_V, 1.0), HI_Z, 5) as stream:
            list(stream)
    assert m1k.modes == {"A": Role.HI_Z, "B": Role.HI_Z}


def test_same_seed_same_samples() -> None:
    first = sim.SimulatedM1K(seed=5).rest_read(0.2)
    second = sim.SimulatedM1K(seed=5).rest_read(0.2)
    assert np.array_equal(first.v_a, second.v_a) and np.array_equal(first.i_b, second.i_b)


def test_burst_length_is_limited() -> None:
    with pytest.raises(InstrumentError):
        with sim.SimulatedM1K().stream(HI_Z, HI_Z, hw.MAX_STREAM_CHUNKS + 1):
            pass


# --- The bench ------------------------------------------------------------------------------------


def test_open_input_reads_its_floating_voltage() -> None:
    chunk = sim.SimulatedM1K().rest_read(0.5)
    assert float(np.mean(chunk.v_a)) == pytest.approx(UNIT.open_input, abs=0.2e-3)


def test_rest_reading_shows_the_real_pin() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(10.4, "2.5V"), b=sim.Wiring(0.05, "5V")))
    chunk = m1k.rest_read(0.5)
    assert float(np.mean(chunk.v_a)) == pytest.approx(PIN_2V5, abs=0.2e-3)
    assert float(np.mean(chunk.v_b)) == pytest.approx(UNIT.pins["5V"], abs=0.3e-3)


def test_hi_z_divider_on_a_large_resistor() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(100e3, "2.5V")))
    expected = UNIT.open_input + (PIN_2V5 - UNIT.open_input) * 1e6 / 1.1e6
    assert float(np.mean(m1k.rest_read(0.5).v_a)) == pytest.approx(expected, abs=0.3e-3)


def test_uncalibrated_eeprom_reads_like_the_board_did() -> None:
    # H-02: with the EEPROM at the default, the 2.5V pin (2.4457 V) read 2.3901 V.
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "2.5V")), faults=sim.Faults(eeprom_default=True))
    assert m1k.calibration() == list(hw.DEFAULT_RECORDS)
    assert float(np.mean(m1k.rest_read(0.5).v_a)) == pytest.approx(2.3901, abs=0.3e-3)


def test_svmi_follows_ohms_law_to_the_pin() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(10.4, "2.5V")))
    v_a, i_a, _, _, overcurrent = mean_of(m1k, dc(Role.SOURCE_V, 3.0), HI_Z)
    assert v_a == pytest.approx(3.0, abs=1e-3)
    assert i_a == pytest.approx((3.0 - PIN_2V5) / 10.4, abs=0.1e-3)
    assert not overcurrent


def test_svmi_into_a_short_trips_the_limiter() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(5.0, "GND")))
    v_a, i_a, _, _, overcurrent = mean_of(m1k, dc(Role.SOURCE_V, 4.0), HI_Z)
    assert overcurrent
    assert i_a == pytest.approx(0.2, abs=2e-3)
    assert v_a == pytest.approx(1.0, abs=0.02)


def test_simv_pushes_both_ways_around_2_5_volts() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(10.4, "2.5V")))
    for current in (0.05, -0.05):
        _, _, v_b, i_b, _ = mean_of(m1k, HI_Z, dc(Role.SOURCE_I, current))
        assert i_b == pytest.approx(current, abs=0.3e-3)
        assert v_b == pytest.approx(PIN_2V5 + current * 10.4, abs=5e-3)


def test_simv_saturates_at_the_rail() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(100.0, "2.5V")))
    v_a, i_a, _, _, _ = mean_of(m1k, dc(Role.SOURCE_I, 0.1), HI_Z)
    assert v_a == pytest.approx(hw.V_MAX, abs=0.01)                      # stuck at the rail
    assert i_a == pytest.approx((5.0 - PIN_2V5) / 100.0, abs=1e-3)       # far short of 100 mA


def test_simv_with_nothing_connected_goes_to_a_rail() -> None:
    m1k = sim.SimulatedM1K()
    _, _, v_b, i_b, _ = mean_of(m1k, HI_Z, dc(Role.SOURCE_I, -0.01))
    assert v_b == pytest.approx(0.0, abs=2e-3)
    assert abs(i_b) < 0.1e-3


def test_true_output_comes_from_the_truncated_code() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(10.4, "2.5V")))
    with m1k.stream(dc(Role.SOURCE_I, 50e-6), HI_Z, 1) as stream:
        list(stream)
    index = hw.record_index("A", Kind.SOURCE_I)
    code = lm.dac_code(50e-6, Kind.SOURCE_I, m1k.calibration()[index])
    expected = lm.source_output(code, Kind.SOURCE_I, UNIT.true_records[index])
    assert m1k.last_truth["i_a"][0] == pytest.approx(float(expected), rel=1e-9)


def test_shunt_node_sees_the_other_channels_current() -> None:
    bench = sim.Bench(a=sim.Wiring(47.0 + 10.0 + 0.4, "GND"), b=sim.Wiring(tap_ohms=10.0))
    m1k = sim.SimulatedM1K(bench)
    _, i_a, v_b, _, _ = mean_of(m1k, dc(Role.SOURCE_V, 3.0), HI_Z)
    assert v_b == pytest.approx(i_a * 10.0, abs=0.3e-3)


def test_crooked_channel_at_5_volts() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(0.05, "5V")), faults=sim.Faults(crooked_v_at_5v={"B": 2e-3}))
    straight = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(0.05, "5V")))
    difference = np.mean(m1k.rest_read(0.5).v_b) - np.mean(straight.rest_read(0.5).v_b)
    expected = 2e-3 * ((UNIT.pins["5V"] - 2.5) / 2.5) ** 2 * 1.02382
    assert float(difference) == pytest.approx(expected, abs=0.1e-3)


def test_jump_in_one_block() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "GND")), faults=sim.Faults(jump=("A", "v", 3, 1e-3)))
    blocks = m1k.rest_read(0.8).v_a.reshape(-1, hw.CHUNK_SAMPLES).mean(axis=1)
    others = np.delete(blocks, 3)
    assert blocks[3] - np.mean(others) == pytest.approx(1.0235e-3, abs=0.1e-3)


# --- The noise ------------------------------------------------------------------------------------


def test_hum_cancels_in_100_ms_blocks() -> None:
    quiet = replace(UNIT, voltage_sigma={"A": 0.0, "B": 0.0}, voltage_slow=0.0, current_sigma=0.0,
                    current_slow=0.0, voltage_hum=150e-6)
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "2.5V")), unit=quiet)
    v = m1k.rest_read(0.5).v_a
    assert np.ptp(v) > 2 * 100e-6                                   # a 150 uV hum is there
    assert np.ptp(v.reshape(-1, hw.CHUNK_SAMPLES).mean(axis=1)) < 2 * hw.V_STEP


def test_noise_per_sample_matches_the_unit() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "GND")))
    chunk = m1k.rest_read(1.0)
    expected_v = math.hypot(UNIT.voltage_sigma["A"], UNIT.voltage_hum / math.sqrt(2)) * 1.0235
    assert float(np.std(chunk.v_a)) == pytest.approx(expected_v, rel=0.1)
    assert float(np.std(chunk.i_a)) == pytest.approx(math.hypot(UNIT.current_sigma, UNIT.current_hum / math.sqrt(2)),
                                                     rel=0.1)


def test_an_open_input_picks_up_the_room() -> None:
    open_channel = sim.SimulatedM1K().rest_read(0.5).v_a
    on_gnd = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "GND"))).rest_read(0.5).v_a
    assert np.std(open_channel) > 5 * np.std(on_gnd)                # ~1 mV of pickup (H-10)


def test_hooks_for_f02_and_h06() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(0.05, "GND")), faults=sim.Faults(halved_words_per_second={"B": 20}))
    chunk = m1k.rest_read(1.0)
    halved = m1k.last_halved["B"]
    assert len(halved) > 5
    assert np.all(np.abs(chunk.i_b[halved]) > 0.1)                  # half the zero word: about -0.12 A
    bent = sim.bend_near_5v(np.array([2.0, 2.5, 4.5, 5.0]), -1.37e-3)
    assert bent[:2].tolist() == [0.0, 0.0] and bent[2] == pytest.approx(-1.37e-3)


def test_slow_noise_block_means_are_correlated_as_asked() -> None:
    phi = sim.slow_noise_coefficient(0.75)
    rng = np.random.default_rng(9)
    noise = sim.slow_noise(rng, 400 * hw.CHUNK_SAMPLES, 6e-6, phi)
    blocks = noise.reshape(-1, hw.CHUNK_SAMPLES).mean(axis=1)
    assert np.corrcoef(blocks[:-1], blocks[1:])[0, 1] == pytest.approx(0.75, abs=0.1)
    assert float(np.std(blocks)) == pytest.approx(6e-6, rel=0.3)


def test_slow_noise_coefficient_inverts_the_correlation() -> None:
    for target in (0.3, 0.75, 0.94):
        phi = sim.slow_noise_coefficient(target)
        block = hw.CHUNK_SAMPLES
        variance = block * (1 + phi) / (1 - phi) - 2 * phi * (1 - phi**block) / (1 - phi) ** 2
        covariance = phi * (1 - phi**block) ** 2 / (1 - phi) ** 2
        assert covariance / variance == pytest.approx(target, abs=1e-6)
