"""Does each waveform give the right period on the 100 kS/s grid? (spec 3.3.3, I-15, C.7)"""

from __future__ import annotations

import numpy as np
import pytest

from phycovolt import hardware as hw
from phycovolt import waveforms as wf
from phycovolt.hardware import Role
from phycovolt.waveforms import Shape, Waveform


def make(shape: Shape, **values: float) -> Waveform:
    return Waveform(shape, values)


def test_sine_starts_at_its_center_with_a_whole_period() -> None:
    sine = make(Shape.SINE, freq=100, pp=2.0, center=2.5, periods=10)
    block = wf.period(sine)
    assert block.size == 1_000
    assert block[0] == 2.5                                  # I-15: no step from rest
    assert block.max() == pytest.approx(3.5) and block.min() == pytest.approx(1.5)
    assert block[250] == pytest.approx(3.5)                 # the crest a quarter later


def test_real_frequency_of_a_rounded_period() -> None:
    timing = wf.timing(make(Shape.SINE, freq=3_000, pp=1.0, center=2.5, periods=10))
    assert timing.samples_per_period == 33
    assert timing.frequency == pytest.approx(3030.30, abs=0.01)       # P-01's 3030 Hz, now shown


def test_square_edges_on_samples_and_real_duty() -> None:
    square = make(Shape.SQUARE, freq=300, low=-0.05, high=0.05, duty=0.5, periods=3)
    block = wf.period(square)
    assert block.size == 333
    assert np.count_nonzero(block == 0.05) == 166            # round(0.5 x 333)
    assert wf.timing(square).real["duty"] == pytest.approx(166 / 333)
    assert set(np.unique(block)) == {-0.05, 0.05}


def test_pulse_width_in_whole_samples() -> None:
    pulse = make(Shape.PULSE, freq=10, baseline=0.0, pulse=0.05, width=1.234e-3, periods=3)
    block = wf.period(pulse)
    assert np.count_nonzero(block == 0.05) == 123
    assert wf.timing(pulse).real["width"] == pytest.approx(1.23e-3)
    assert block[0] == 0.05 and block[-1] == 0.0


def test_biphasic_is_charge_balanced() -> None:
    biphasic = make(Shape.BIPHASIC, freq=10, center=0.0, amplitude=0.05, width=1e-3, gap=0.1e-3, periods=3)
    block = wf.period(biphasic)
    assert np.sum(block) == pytest.approx(0.0, abs=1e-12)
    assert block[:100].tolist() == [0.05] * 100
    assert block[100:110].tolist() == [0.0] * 10
    assert block[110:210].tolist() == [-0.05] * 100


def test_triangle_starts_at_its_center_rising() -> None:
    block = wf.period(make(Shape.TRIANGLE, freq=100, pp=2.0, center=2.5, periods=3))
    assert block[0] == 2.5 and block[1] > 2.5
    assert block[250] == pytest.approx(3.5) and block[750] == pytest.approx(1.5)


def test_dc_is_one_block_and_its_burst_whole_chunks() -> None:
    dc = make(Shape.DC, level=0.01, seconds=1.0)
    assert wf.period(dc).size == hw.CHUNK_SAMPLES
    timing = wf.timing(dc)
    assert timing.chunks == 10 and timing.seconds == pytest.approx(1.0)


def test_periodic_burst_covers_whole_chunks() -> None:
    sine = make(Shape.SINE, freq=30, pp=1.0, center=2.5, periods=4)      # 4 x 3333 = 13 332 samples
    timing = wf.timing(sine)
    assert timing.chunks == 2
    burst = wf.burst(sine)
    assert burst.size == 20_000
    assert np.array_equal(burst[3_333:6_666], wf.period(sine))


@pytest.mark.parametrize("waveform", [
    make(Shape.SQUARE, freq=100, low=0, high=1, duty=0.0001, periods=3),
    make(Shape.PULSE, freq=100, baseline=0, pulse=1, width=0.02, periods=3),
    make(Shape.BIPHASIC, freq=1000, center=0, amplitude=0.01, width=0.6e-3, gap=0, periods=3),
    make(Shape.SINE, freq=0, pp=1, center=2.5, periods=3),
    make(Shape.SINE, freq=100, pp=1, center=2.5, periods=2.5),
])
def test_what_cannot_be_built_is_refused(waveform: Waveform) -> None:
    with pytest.raises(ValueError):
        wf.timing(waveform)


def test_fields_must_match_the_shape() -> None:
    with pytest.raises(ValueError):
        Waveform(Shape.SINE, {"freq": 100, "pp": 1})


def test_defaults_depend_on_the_return() -> None:
    around = wf.default_waveform(Role.SOURCE_I, Shape.SINE, "2.5V")
    into_5v = wf.default_waveform(Role.SOURCE_I, Shape.SINE, "5V")
    assert (around["pp"], around["center"]) == (0.2, 0.0)
    assert (into_5v["pp"], into_5v["center"]) == (0.08, -0.06)
    assert Shape.BIPHASIC not in wf.offered_shapes(Role.SOURCE_I, "5V")
    assert wf.offered_shapes(Role.SOURCE_V, "5V") == list(Shape)


def test_typed_fields_override_defaults_and_keep_the_base() -> None:
    sine = wf.waveform_from_text(Role.SOURCE_I, Shape.SINE, {"freq": "1 kHz", "pp": "20"}, "2.5V")
    assert sine["freq"] == 1000 and sine["pp"] == pytest.approx(20e-3) and sine["center"] == 0.0
    kept = wf.waveform_from_text(Role.SOURCE_I, Shape.SINE, {"center": "5 mA"}, "2.5V", base=sine)
    assert kept["pp"] == pytest.approx(20e-3) and kept["center"] == pytest.approx(5e-3)
    with pytest.raises(ValueError):
        wf.waveform_from_text(Role.SOURCE_I, Shape.SINE, {"duty": "50%"}, "2.5V")
