"""Are the V/I swap (fault F-01) and the halved word (F-02) found, and only they? (spec C.5, 3.3.7)"""

from __future__ import annotations

import numpy as np
import pytest

from phycovolt import analysis
from phycovolt import hardware as hw
from phycovolt import simulator as sim
from phycovolt.hardware import ChannelSetup, Role

ZERO_CURRENT_WORD = 31_950     # about 0 A on the current scale
GND_WORD = 7                   # about 0.5 mV on the voltage scale


def rest_words(rng: np.random.Generator, samples: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """A channel on GND at rest: V near word 7, I near word 31 950, a few codes of noise."""
    return rng.normal(GND_WORD, 1.5, samples), rng.normal(ZERO_CURRENT_WORD, 8.0, samples)


def swap(v: np.ndarray, i: np.ndarray, where: np.ndarray) -> None:
    v[where], i[where] = i[where], v[where].copy()


def test_twenty_injected_swaps_are_counted() -> None:
    rng = np.random.default_rng(3)
    v, i = rest_words(rng)
    where = rng.choice(v.size, 20, replace=False)
    swap(v, i, where)
    found = analysis.vi_swaps(v, i)
    assert np.flatnonzero(found).tolist() == sorted(where.tolist())


def test_noise_is_not_a_swap() -> None:
    v, i = rest_words(np.random.default_rng(4), 1_000_000)
    assert np.count_nonzero(analysis.vi_swaps(v, i)) == 0


def test_a_jump_is_not_a_swap() -> None:
    rng = np.random.default_rng(5)
    v, i = rest_words(rng)
    v[100_000:] += 300            # 23 mV: the contact moved
    i[150_000:] -= 2_000          # 15 mA: something changed in the circuit
    assert np.count_nonzero(analysis.vi_swaps(v, i)) == 0


def test_close_levels_cannot_be_told_apart() -> None:
    # A channel at 2.437 V: its voltage word sits on the current-zero word. A swap changes almost
    # nothing there, and it is not reported (C.5).
    rng = np.random.default_rng(6)
    v = rng.normal(ZERO_CURRENT_WORD + 50, 1.5, 100_000)
    i = rng.normal(ZERO_CURRENT_WORD, 8.0, 100_000)
    swap(v, i, np.arange(10, 100_000, 5_000))
    assert np.count_nonzero(analysis.vi_swaps(v, i)) == 0


def test_periodic_signal_uses_each_phase_level() -> None:
    rng = np.random.default_rng(7)
    period, periods = 1_000, 20
    phase = 2 * np.pi * np.arange(period * periods) / period
    i = ZERO_CURRENT_WORD + 6_500 * np.sin(phase) + rng.normal(0, 8, phase.size)   # +-50 mA
    v = 32_000 + 13_000 * np.sin(phase) + rng.normal(0, 1.5, phase.size)          # 2.44 +- 1 V
    where = rng.choice(phase.size, 15, replace=False)
    swap(v, i, where)
    found = np.flatnonzero(analysis.vi_swaps(v, i, period=period))
    # Near the crossings the two words come close, and those swaps cannot be seen (C.5).
    assert set(found) <= set(where)
    assert len(found) >= 10
    assert analysis.vi_swaps(v, i) is not None          # (as a constant signal it still runs)


def test_fewer_than_three_periods_cannot_be_checked() -> None:
    v = np.zeros(2_500)
    assert analysis.vi_swaps(v, v, period=1_000) is None


def test_robust_spread_estimates_sigma_and_ignores_outliers() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(0, 2.0, 100_000)
    values[:500] = 1e6
    assert analysis.robust_spread(values) == pytest.approx(2.0, rel=0.03)


def test_swaps_in_simulated_samples_after_libsmu() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(0.05, "GND")),
                           faults=sim.Faults(vi_swaps_per_second={"B": 10}), seed=11)
    chunk = m1k.rest_read(hw.CHUNK_SECONDS * 20)
    records = m1k.calibration()
    swapped = analysis.vi_swaps(*analysis.channel_words(chunk.v_b, chunk.i_b, records, "B"))
    assert np.flatnonzero(swapped).tolist() == m1k.last_swaps["B"].tolist()
    assert len(m1k.last_swaps["B"]) > 5
    assert analysis.count_vi_swaps(chunk.v_a, chunk.i_a, records, "A") == 0


def test_swap_while_sourcing_a_current_sine() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(10.4, "5V")),
                           faults=sim.Faults(vi_swaps_per_second={"B": 50}), seed=12)
    k = np.arange(1_000)
    block = -0.06 + 0.04 * np.sin(2 * np.pi * k / 1_000)
    with m1k.stream(ChannelSetup(), ChannelSetup(Role.SOURCE_I, block), 10) as stream:
        chunks = list(stream)
    v = np.concatenate([c.v_b for c in chunks])
    i = np.concatenate([c.i_b for c in chunks])
    found = analysis.count_vi_swaps(v, i, m1k.calibration(), "B", period=1_000)
    assert found >= 0.8 * len(m1k.last_swaps["B"]) > 0


# --- F-02: the halved word -----------------------------------------------------------------------


def test_halved_current_words_are_counted() -> None:
    rng = np.random.default_rng(13)
    v, i = rest_words(rng)
    where = rng.choice(i.size, 7, replace=False)
    i[where] = np.floor(i[where] / 2)                       # one bit shifted right
    assert np.flatnonzero(analysis.halved_words(i)).tolist() == sorted(where.tolist())
    assert np.count_nonzero(analysis.vi_swaps(v, i)) == 0   # not a V/I swap (C.5 rightly ignores it)


def test_noise_jumps_and_swaps_are_not_halved_words() -> None:
    rng = np.random.default_rng(14)
    v, i = rest_words(rng, 1_000_000)
    assert np.count_nonzero(analysis.halved_words(i)) == 0
    i[300_000:] -= 2_000
    swap(v, i, rng.choice(v.size, 10, replace=False))
    assert np.count_nonzero(analysis.halved_words(i)) == 0


def test_a_word_near_zero_cannot_be_told() -> None:
    v = np.random.default_rng(15).normal(GND_WORD, 1.5, 100_000)
    v[::10_000] = np.floor(v[::10_000] / 2)                  # 7 -> 3: half of almost nothing
    assert np.count_nonzero(analysis.halved_words(v)) == 0


def test_halved_words_and_their_effect_after_libsmu() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(b=sim.Wiring(0.05, "GND")),
                           faults=sim.Faults(halved_words_per_second={"B": 4}), seed=16)
    chunk = m1k.rest_read(5.0)
    found = analysis.count_halved(chunk.v_b, chunk.i_b, m1k.calibration(), "B")
    count, mean_shift, block_shift = found["I"]
    assert count == len(m1k.last_halved["B"]) > 5
    assert found["V"][0] == 0
    # Each moves a 100 ms mean by about half the zero word (~0.12 A) / 10 000 samples, about -12 uA;
    # two in the same block, twice that.
    assert 9e-6 < block_shift < 3 * 13e-6
    assert mean_shift == pytest.approx(-count * 12e-6 / 50, rel=0.3)
