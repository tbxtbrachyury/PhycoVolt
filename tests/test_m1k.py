"""Does m1k.py call pysmu in the order of the operating rules I-01 to I-08?

Every test runs against tests/fake_pysmu.py, which records calls and drives nothing. The expected
sequences are written out in full: they are the sequences the operator approves.
"""

from __future__ import annotations

import signal

import numpy as np
import pytest

from phycovolt import hardware as hw
from phycovolt import m1k
from phycovolt.hardware import ChannelSetup, InstrumentError, Role
from tests import fake_pysmu
from tests.fake_pysmu import Faults

N = hw.CHUNK_SAMPLES
T = hw.READ_TIMEOUT_MS

OPENING = [
    ("Session", {"add_all": True, "ignore_dataflow": False, "sample_rate": 0, "queue_size": hw.QUEUE_SAMPLES}),
    ("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("flush",), ("get_samples", N),
]
PARK = [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("flush",), ("get_samples", hw.PARK_SAMPLES)]
END_COMPLETE = [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("end",), ("overcurrent",)]
END_EARLY = [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("cancel",), ("end",), ("overcurrent",)]

HI_Z = ChannelSetup()


def sine(amplitude: float, center: float, n: int = 1_000) -> np.ndarray:
    k = np.arange(n)
    return center + amplitude * np.sin(2 * np.pi * k / n)


def opened(faults: Faults | None = None) -> tuple[m1k.M1K, fake_pysmu.Fake]:
    fake = fake_pysmu.make(faults)
    instrument = m1k.open_m1k(fake.module())
    return instrument, fake


def after_opening(fake: fake_pysmu.Fake) -> list[tuple]:
    assert fake.log[: len(OPENING)] == OPENING
    return fake.log[len(OPENING):]


def run(instrument: m1k.M1K, a: ChannelSetup, b: ChannelSetup, chunks: int) -> list[hw.Chunk]:
    with instrument.stream(a, b, chunks) as stream:
        return list(stream)


# --- Opening (I-04, I-05, I-06, I-07) -------------------------------------------------------------


def test_opening_sequence() -> None:
    instrument, fake = opened()
    assert fake.log == OPENING        # explicit HI_Z and a 100 ms read before any start/end (I-06)
    assert instrument.description == f"serial {'FAKE' + '0' * 27} · fw 2.17 · hw D"


def test_opening_restores_python_ctrl_c_handler() -> None:
    opened()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler      # I-04, P-02


@pytest.mark.parametrize("devices, code", [(0, "no_board"), (2, "several_boards")])
def test_exactly_one_board(devices: int, code: str) -> None:
    with pytest.raises(InstrumentError) as error:
        opened(Faults(devices=devices))
    assert error.value.code == code


def test_old_firmware_is_refused() -> None:
    with pytest.raises(InstrumentError) as error:
        opened(Faults(firmware="2.05"))
    assert error.value.code == "firmware"


def test_another_sample_rate_is_refused() -> None:          # I-07: below 100 kS/s libsmu hangs (L-12)
    with pytest.raises(InstrumentError) as error:
        opened(Faults(sample_rate=50_000))
    assert error.value.code == "invalid_request"


def test_without_pysmu() -> None:
    with pytest.raises(InstrumentError) as error:
        m1k.open_m1k(None, import_name="phycovolt_no_such_module")
    assert error.value.code == "no_pysmu"


def test_calibration_records() -> None:
    instrument, _ = opened()
    assert instrument.calibration() == list(hw.DEFAULT_RECORDS)


# --- Rest reads (I-03) ----------------------------------------------------------------------------


def test_rest_read_requests_hi_z_explicitly() -> None:
    instrument, fake = opened()
    chunk = instrument.rest_read(0.5)
    assert after_opening(fake) == [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"), ("flush",),
                                   ("get_samples", 50_000)]
    assert chunk.samples == 50_000
    assert chunk.v_b[0] == pytest.approx(0.0002)


def test_lost_sample_in_a_rest_read() -> None:
    instrument, fake = opened()
    fake.faults.drop_in_get_samples = True
    with pytest.raises(InstrumentError) as error:
        instrument.rest_read(0.1)
    assert error.value.code == "lost_sample"
    assert fake.log[-2:] == [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z")]


# --- Streams (I-01, I-02, I-05, I-07) -------------------------------------------------------------


def test_svmi_sequence() -> None:
    instrument, fake = opened()
    chunks = run(instrument, ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 3)
    assert after_opening(fake) == [
        *PARK,                            # the burst starts at code 26 214 (2.0 V), never elsewhere
        ("flush",), ("write", "A", 1_000, True),
        ("mode", "A", "SVMI"), ("mode", "B", "HI_Z"), ("start", 3 * N),
        ("read", N, T), ("read", N, T), ("read", N, T),
        *END_COMPLETE,
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.samples == N for chunk in chunks)


def test_simv_parks_the_dac_first() -> None:
    instrument, fake = opened()
    run(instrument, HI_Z, ChannelSetup(Role.SOURCE_I, sine(0.05, 0.0)), 2)
    assert after_opening(fake) == [
        *PARK,
        ("flush",), ("write", "B", 1_000, True),
        ("mode", "A", "HI_Z"), ("mode", "B", "SIMV"), ("start", 2 * N),
        ("read", N, T), ("read", N, T),
        *END_COMPLETE,
    ]


def test_both_channels_sourcing() -> None:
    instrument, fake = opened()
    run(instrument, ChannelSetup(Role.SOURCE_I, sine(0.05, 0.0)), ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), 1)
    assert after_opening(fake) == [
        *PARK,
        ("flush",), ("write", "A", 1_000, True), ("write", "B", 1_000, True),
        ("mode", "A", "SIMV"), ("mode", "B", "SVMI"), ("start", N),
        ("read", N, T),
        *END_COMPLETE,
    ]


def test_constant_value() -> None:
    instrument, fake = opened()
    run(instrument, ChannelSetup(Role.SOURCE_V, np.full(N, 1.0)), HI_Z, 3)
    assert after_opening(fake)[:9] == [*PARK, ("flush",), ("write", "A", N, True), ("mode", "A", "SVMI"),
                                       ("mode", "B", "HI_Z"), ("start", 3 * N)]


def test_back_to_back_svmi_bursts_are_each_parked() -> None:
    # Two calibration points in a row (+100 mA, then -100 mA): the second must not start at the
    # first one's voltage.
    instrument, fake = opened()
    run(instrument, ChannelSetup(Role.SOURCE_V, np.full(N, 3.5)), HI_Z, 1)
    run(instrument, ChannelSetup(Role.SOURCE_V, np.full(N, 1.5)), HI_Z, 1)
    starts = [k for k, entry in enumerate(fake.log) if entry[0] == "start"]
    for start in starts:
        assert fake.log[start - 8: start - 4] == PARK


def test_hi_z_only_stream() -> None:
    instrument, fake = opened()
    run(instrument, HI_Z, HI_Z, 1)
    assert after_opening(fake) == [("flush",), ("mode", "A", "HI_Z"), ("mode", "B", "HI_Z"),
                                   ("start", N), ("read", N, T), *END_COMPLETE]


def test_error_in_the_middle_leaves_hi_z() -> None:
    instrument, fake = opened(Faults(drop_at_read=2))
    with pytest.raises(InstrumentError) as error:
        run(instrument, ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 5)
    assert error.value.code == "lost_sample"
    assert fake.log[-7:] == [("read", N, T), ("read", N, T), *END_EARLY]


def test_short_read_is_an_error() -> None:
    instrument, fake = opened(Faults(short_at_read=1))
    with pytest.raises(InstrumentError) as error:
        run(instrument, ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 3)
    assert error.value.code == "timeout"
    assert fake.log[-5:] == END_EARLY


def test_stop_in_the_middle() -> None:
    instrument, fake = opened()
    seen = []
    with instrument.stream(ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 10) as stream:
        for chunk in stream:
            seen.append(chunk.index)
            if chunk.index == 1:
                stream.stop()
    assert seen == [0, 1]
    assert fake.log[-7:] == [("read", N, T), ("read", N, T), *END_EARLY]


def test_break_out_of_the_loop() -> None:
    instrument, fake = opened()
    with instrument.stream(ChannelSetup(Role.SOURCE_I, sine(0.05, 0.0)), HI_Z, 10) as stream:
        for _ in stream:
            break
    assert fake.log[-6:] == [("read", N, T), *END_EARLY]


def test_ctrl_c_during_a_read() -> None:
    instrument, fake = opened(Faults(interrupt_at_read=3))
    with pytest.raises(KeyboardInterrupt):
        run(instrument, ChannelSetup(Role.SOURCE_I, sine(0.05, 0.0)), HI_Z, 10)
    assert fake.log[-5:] == END_EARLY


def test_exception_in_the_callers_code() -> None:
    instrument, fake = opened()
    with pytest.raises(ZeroDivisionError):
        with instrument.stream(ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 3) as stream:
            for _ in stream:
                1 / 0
    assert fake.log[-5:] == END_EARLY


def test_every_end_step_is_tried_even_when_usb_fails() -> None:
    instrument, fake = opened()
    with pytest.raises(InstrumentError) as error:
        with instrument.stream(ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 3) as stream:
            for _ in stream:
                # The cable is pulled: the next read fails, and so does every mode change.
                fake.faults.usb_error_on_mode = True
                fake.faults.drop_at_read = fake.reads + 1
    assert error.value.code == "lost_sample"          # the first error is the one reported
    assert fake.log[-5:] == END_EARLY


def test_a_stream_cannot_read_after_its_with_block() -> None:
    instrument, fake = opened()
    with instrument.stream(ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 3) as stream:
        pass
    count = len(fake.log)
    assert list(stream) == []
    assert len(fake.log) == count


def test_overcurrent_is_read_after_the_end() -> None:
    instrument, fake = opened()
    fake.sessions[0].devices[0]._overcurrent = 1
    with instrument.stream(ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 1) as stream:
        list(stream)
    assert stream.overcurrent is True


@pytest.mark.parametrize("chunks", [0, hw.MAX_STREAM_CHUNKS + 1])
def test_burst_length_is_limited(chunks: int) -> None:
    instrument, fake = opened()
    with pytest.raises(InstrumentError):
        with instrument.stream(HI_Z, HI_Z, chunks):
            pass
    assert fake.log == OPENING          # nothing was sent


def test_close_leaves_hi_z() -> None:
    instrument, fake = opened()
    instrument.close()
    assert after_opening(fake) == [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z")]
    with pytest.raises(InstrumentError):
        instrument.rest_read(0.1)


# --- Rules checked on every sequence --------------------------------------------------------------


BOTH_HI_Z = [("mode", "A", "HI_Z"), ("mode", "B", "HI_Z")]


def check_rules(log: list[tuple]) -> None:
    """The rules on any log.

    I-01: the two modes are set right before start, after every write of that stream.
    I-02: a stream with any sourcing channel is preceded by a park since the previous stream ended.
    I-03: every get_samples (rest read, park) is preceded by HI_Z on both channels.
    I-04: every end is preceded by HI_Z on both channels.
    """
    previous_end = 0
    for position, (name, *args) in enumerate(log):
        if name == "get_samples":
            assert log[position - 3: position - 1] == BOTH_HI_Z and log[position - 1] == ("flush",)
        elif name == "start":
            mode_a, mode_b = log[position - 2], log[position - 1]
            assert mode_a[:2] == ("mode", "A") and mode_b[:2] == ("mode", "B")
            preparation = log[previous_end:position]
            if (mode_a[2], mode_b[2]) != ("HI_Z", "HI_Z"):
                assert ("get_samples", hw.PARK_SAMPLES) in preparation, "a sourcing burst without a park"
        elif name == "end":
            modes = [entry for entry in log[previous_end:position] if entry[0] == "mode"]
            assert modes[-2:] == BOTH_HI_Z
            previous_end = position


def test_rules_hold_over_a_session_with_everything() -> None:
    instrument, fake = opened()
    run(instrument, ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), HI_Z, 2)
    instrument.rest_read(0.1)
    run(instrument, ChannelSetup(Role.SOURCE_I, sine(0.05, 0.0)), ChannelSetup(Role.SOURCE_V, sine(1.0, 2.5)), 1)
    with instrument.stream(HI_Z, ChannelSetup(Role.SOURCE_I, np.full(N, 0.01)), 4) as stream:
        for chunk in stream:
            stream.stop()
    run(instrument, ChannelSetup(Role.SOURCE_V, np.full(N, 3.0)), HI_Z, 1)
    instrument.close()
    check_rules(fake.log)


def test_the_program_never_writes_the_eeprom() -> None:
    from pathlib import Path

    for path in Path(m1k.__file__).parent.glob("*.py"):
        assert "write_calibration(" not in path.read_text(encoding="utf-8"), path.name   # I-08
