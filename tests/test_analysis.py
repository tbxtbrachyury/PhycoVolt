"""Do the reading summary, the cycle average and the sine fit tell the truth? (spec 3.6, 8, 9.1)"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phycovolt import analysis
from phycovolt import hardware as hw
from phycovolt import simulator as sim

BLOCK = hw.CHUNK_SAMPLES


def test_white_noise_block_means_agree_with_sigma_over_root_n() -> None:
    rng = np.random.default_rng(1)
    sigma, samples = 1e-4, 50 * BLOCK
    x = rng.normal(0, sigma, samples)
    means = analysis.block_means(x)
    # The standard error of the mean from the blocks, s_b / sqrt(n), equals sigma / sqrt(N).
    assert np.std(means, ddof=1) / math.sqrt(means.size) == pytest.approx(sigma / math.sqrt(samples), rel=0.25)


def ar1_block_means(rng: np.random.Generator, n: int, correlation: float, std: float) -> np.ndarray:
    values = np.empty(n)
    values[0] = rng.normal(0, std)
    innovation = std * math.sqrt(1 - correlation**2)
    for k in range(1, n):
        values[k] = correlation * values[k - 1] + rng.normal(0, innovation)
    return values


def test_correlated_blocks_sigma_over_root_n_under_covers_and_method_b_does_not() -> None:
    # H-14: neighbouring 100 ms means correlated at 0.75. The true mean is 0; count how often
    # "value +- 2u" contains it.
    rng = np.random.default_rng(2)
    trials, blocks = 1_000, 30
    covered_b = covered_naive = 0
    for _ in range(trials):
        means = ar1_block_means(rng, blocks, 0.75, 1.0)
        value = np.mean(means)
        u_b = analysis.uncertainty_of_mean(means)
        u_naive = np.std(means, ddof=1) / math.sqrt(blocks)      # what sigma / sqrt(N) amounts to
        covered_b += abs(value) <= 2 * u_b
        covered_naive += abs(value) <= 2 * u_naive
    assert covered_b / trials >= 0.95
    assert covered_naive / trials < 0.75


def test_method_b_is_the_spread_of_the_block_means() -> None:
    means = np.array([1.0, 2.0, 3.0, 4.0])
    assert analysis.uncertainty_of_mean(means) == pytest.approx(np.std(means, ddof=1))
    assert analysis.UNCERTAINTY_METHOD == "B"


def test_summary_of_a_quiet_reading_adds_the_step() -> None:
    x = np.full(5 * BLOCK, 1.0)                                 # no noise at all
    summary = analysis.summarize(x, hw.V_STEP)
    assert summary.value == 1.0
    assert summary.u == pytest.approx(hw.V_STEP / math.sqrt(12))


def test_a_jump_is_caught_and_noise_is_not() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1e-4, 40 * BLOCK)
    assert analysis.summarize(x, hw.V_STEP).jumps == 0
    x[7 * BLOCK:8 * BLOCK] += 1e-3                               # one block moved by 1 mV
    assert analysis.summarize(x, hw.V_STEP).jumps == 1


def test_a_clipped_adc_is_caught() -> None:
    words = np.full(3 * BLOCK, 1_000.0)
    words[: BLOCK // 2] = hw.CODE_MAX                            # 1/6 of the samples at the edge
    summary = analysis.summarize(words * hw.V_STEP, hw.V_STEP, words)
    assert summary.adc_limit_fraction == pytest.approx(1 / 6)


def test_drift_is_the_slope_times_the_length() -> None:
    t = np.arange(20 * BLOCK) / hw.SAMPLE_RATE
    summary = analysis.summarize(1.0 + 1e-3 * t, hw.V_STEP)         # 1 mV/s for 2 s
    assert summary.drift == pytest.approx(1e-3 * 1.9, rel=1e-6)    # between the first and last block centers
    assert summary.r1 is not None and summary.r1 > 0.8


def test_noisy_sine_is_recovered_within_sigma_root_two_over_n() -> None:
    rng = np.random.default_rng(4)
    n, periods, sigma = 1_000, 101, 58e-6
    k = np.arange(n * periods)
    amplitude, level, phase = 0.1, 2e-3, 0.3
    y = level + amplitude * np.sin(2 * np.pi * k / n + phase) + rng.normal(0, sigma, k.size)
    fit = analysis.sine_fit(y, n)
    used = n * (periods - 1)
    bound = 4 * sigma * math.sqrt(2 / used)                         # 4 standard errors
    assert fit.amplitude == pytest.approx(amplitude, abs=bound)
    assert fit.level == pytest.approx(level, abs=4 * sigma / math.sqrt(used))
    assert fit.phase == pytest.approx(phase, abs=bound / amplitude)
    assert fit.residual == pytest.approx(sigma, rel=0.05)
    assert fit.delay(100.0) == pytest.approx(-phase / (2 * math.pi * 100.0))


def test_cycle_average_recovers_a_square_waves_edges() -> None:
    rng = np.random.default_rng(5)
    n, periods = 1_000, 30
    period = np.where(np.arange(n) < 300, 0.05, -0.05)
    y = np.tile(period, periods) + rng.normal(0, 58e-6, n * periods)
    cycle = analysis.cycle_average(y, n)
    assert cycle.periods == periods - 1                             # the first one is left out
    assert np.max(np.abs(cycle.mean - period)) < 60e-6
    edges = np.flatnonzero(np.abs(np.diff(cycle.mean)) > 0.05)
    assert edges.tolist() == [299]
    assert cycle.peak_to_peak == pytest.approx(0.1, abs=1e-4)


def test_too_short_for_analysis() -> None:
    assert analysis.summarize(np.zeros(BLOCK - 1), hw.V_STEP) is None
    assert analysis.cycle_average(np.zeros(1_500), 1_000) is None
    assert analysis.sine_fit(np.zeros(1_500), 1_000) is None


def test_rail_fraction() -> None:
    v = np.concatenate([np.full(90, 2.5), np.full(10, 4.99)])
    assert analysis.rail_fraction(v) == pytest.approx(0.1)


def test_simulated_reading_summary_reports_correlated_blocks() -> None:
    m1k = sim.SimulatedM1K(sim.Bench(a=sim.Wiring(0.05, "2.5V")), seed=6)
    chunk = m1k.rest_read(5.0)
    summary = analysis.summarize(chunk.v_a, hw.V_STEP)
    assert summary.blocks == 50
    assert summary.value == pytest.approx(2.4457, abs=0.2e-3)
    assert summary.r1 is not None and summary.r1 > 0.3
