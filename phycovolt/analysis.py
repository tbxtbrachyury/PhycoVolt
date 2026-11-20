"""What a recording says: reading summaries, cycle averages, the sine regression, and the V/I swap.

Questions it answers: what are a quantity's level, spread and uncertainty; what shape did a periodic
signal really have, phase by phase; what are a sine's level, amplitude and phase; and is the
channel's data trustworthy (no V/I swap, fault F-01)?

Main choices (spec 3.6 and 8, C.4, C.5, C.7):

- Nothing is decided from one sample: a reading is summarized by its 100 ms block means (6 cycles of
  60 Hz, so the hum cancels in each). Value = mean of the block means; spread = their standard
  deviation. The uncertainty of the value is method B (spec 8): the spread
  itself, without dividing by the square root of the number of blocks, because
  neighbouring block means are correlated (H-14). It lives in ONE function, `uncertainty_of_mean`.
- A periodic signal is laid period over period (the cycle average), without the first period; a
  sine is fitted with level + sine + cosine at the known frequency (synchronous detection).
- The robust spread for jumps is one function too (open item 3; provisional 1.4826 x MAD).
- The V/I swap (F-01) and the halved word (F-02) are looked for in ADC words, with the stored
  calibration undone (libsmu_math).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phycovolt import checks
from phycovolt import hardware as hw
from phycovolt import libsmu_math
from phycovolt.hardware import Kind, Record

# --- The two open statistical choices (spec 8): one function each ---------------------------------

UNCERTAINTY_METHOD = "B"              # recorded with every result


def uncertainty_of_mean(block_means: np.ndarray) -> float:
    """The standard uncertainty of a reading's value, from its 100 ms block means.

    Method B (spec 8, decided 2026-11-21): the standard deviation of the block
    means, NOT divided by the square root of their number, because neighbouring blocks are
    correlated (H-14) and the division would promise an uncertainty several times too small. On the
    2026-11-21 rest recordings the real spread of repeated bursts was 0.67 times B's value (an exact
    method gives about 0.75), against 6.2 for the division and 2.4 for method C. Another method
    would be a change in this function only (and in UNCERTAINTY_METHOD).
    """
    return float(np.std(block_means, ddof=1)) if len(block_means) > 1 else math.nan


def robust_spread(values: np.ndarray) -> float:
    """A standard deviation that ignores outliers: 1.4826 x the median absolute deviation.

    Provisional (spec 8, open item 3): the operator may choose one based on the interquartile range;
    switching is a change in this function only.
    """
    values = np.asarray(values, dtype=np.float64)
    return float(checks.MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


# --- The reading summary (C.4) --------------------------------------------------------------------


def block_means(samples: np.ndarray, block: int = hw.CHUNK_SAMPLES) -> np.ndarray:
    """The mean of each whole block of `block` samples (a partial last block is left out)."""
    samples = np.asarray(samples, dtype=np.float64)
    count = samples.size // block
    return samples[: count * block].reshape(count, block).mean(axis=1)


@dataclass(frozen=True)
class Summary:
    """One quantity of one reading, without its settling (C.4). V or A."""

    value: float                      # mean of the block means
    u: float                          # standard uncertainty (method in `method`)
    spread: float                     # standard deviation of the block means
    sigma: float                      # standard deviation per sample (noise)
    blocks: int
    r1: float | None                  # lag-1 correlation of the block means (reported)
    jumps: int                        # blocks far from the median (robust)
    drift: float                      # slope of the block means x the reading's length
    adc_limit_fraction: float         # samples within half a step of the ADC's edges
    method: str = UNCERTAINTY_METHOD


def summarize(samples: np.ndarray, step: float, words: np.ndarray | None = None,
              block: int = hw.CHUNK_SAMPLES) -> Summary | None:
    """The summary of one quantity's samples (already without the settling); None if there is less
    than one block. `step` is the ADC step of the quantity; `words` its ADC words, if known."""
    samples = np.asarray(samples, dtype=np.float64)
    means = block_means(samples, block)
    if means.size == 0:
        return None
    sigma = float(np.std(samples))
    u = uncertainty_of_mean(means)
    if sigma < step / 2 and math.isfinite(u):      # a quiet reading: the ADC step weighs (C.4)
        u = math.hypot(u, step / math.sqrt(12))
    r1 = None
    if means.size >= 3 and np.std(means[:-1]) > 0 and np.std(means[1:]) > 0:
        r1 = float(np.corrcoef(means[:-1], means[1:])[0, 1])
    floor = step * checks.JUMP_FLOOR_STEPS
    limit = checks.JUMP_ROBUST_DEVIATIONS * max(robust_spread(means), floor)
    jumps = int(np.count_nonzero(np.abs(means - np.median(means)) > limit))
    drift = 0.0
    if means.size >= 2:
        times = (np.arange(means.size) + 0.5) * block / hw.SAMPLE_RATE
        slope = np.polyfit(times, means, 1)[0]
        drift = float(slope * (times[-1] - times[0]))
    at_limit = 0.0
    if words is not None:
        words = np.asarray(words)
        edge = (words <= 0.5) | (words >= hw.CODE_MAX - 0.5)
        at_limit = float(np.count_nonzero(edge) / words.size)
    return Summary(float(np.mean(means)), u, float(np.std(means, ddof=1)) if means.size > 1 else math.nan,
                   sigma, int(means.size), r1, jumps, drift, at_limit)


# --- Periodic signals (C.7) -----------------------------------------------------------------------


@dataclass(frozen=True)
class CycleAverage:
    """The periods laid over each other: mean and standard deviation, phase by phase."""

    mean: np.ndarray
    std: np.ndarray
    periods: int

    @property
    def level(self) -> float:
        return float(np.mean(self.mean))

    @property
    def peak_to_peak(self) -> float:
        return float(np.ptp(self.mean))

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.mean))))


def whole_periods(samples: np.ndarray, n: int, skip: int = checks.SETTLING_PERIODS) -> np.ndarray:
    """The whole periods after the first `skip`, one per row (no rows if there are none)."""
    samples = np.asarray(samples, dtype=np.float64)
    count = samples.size // n - skip
    if count <= 0:
        return np.empty((0, n))
    return samples[skip * n:(skip + count) * n].reshape(count, n)


def cycle_average(samples: np.ndarray, n: int, skip: int = checks.SETTLING_PERIODS) -> CycleAverage | None:
    rows = whole_periods(samples, n, skip)
    if rows.shape[0] == 0:
        return None
    std = rows.std(axis=0, ddof=1) if rows.shape[0] > 1 else np.zeros(n)
    return CycleAverage(rows.mean(axis=0), std, int(rows.shape[0]))


@dataclass(frozen=True)
class SineFit:
    """level + a sin + b cos at the requested frequency, with the request's phase (C.7)."""

    level: float
    amplitude: float
    phase: float                      # rad, relative to the requested sine
    residual: float                   # rms of what the fit leaves
    samples: int

    def delay(self, frequency: float) -> float:
        """s; positive when the measured sine comes after the request."""
        return -self.phase / (2 * math.pi * frequency)


def sine_fit(samples: np.ndarray, n: int, skip: int = checks.SETTLING_PERIODS) -> SineFit | None:
    """Synchronous detection over the whole periods after the first; sample 0 of the burst is phase 0."""
    rows = whole_periods(samples, n, skip)
    if rows.shape[0] == 0:
        return None
    y = rows.reshape(-1)
    k = np.arange(y.size) % n
    sine, cosine = np.sin(2 * np.pi * k / n), np.cos(2 * np.pi * k / n)
    level = float(np.mean(y))
    a = 2 * float(np.mean((y - level) * sine))
    b = 2 * float(np.mean((y - level) * cosine))
    residual = float(np.sqrt(np.mean(np.square(y - level - a * sine - b * cosine))))
    return SineFit(level, math.hypot(a, b), math.atan2(b, a), residual, int(y.size))


def rail_fraction(voltage: np.ndarray) -> float:
    """The fraction of samples within 20 mV of 0 or 5 V (a current source touching a rail)."""
    voltage = np.asarray(voltage, dtype=np.float64)
    if voltage.size == 0:
        return 0.0
    near = (voltage <= hw.V_MIN + checks.RAIL_MARGIN_V) | (voltage >= hw.V_MAX - checks.RAIL_MARGIN_V)
    return float(np.count_nonzero(near) / voltage.size)


# --- The V/I swap (C.5) ---------------------------------------------------------------------------


def levels(words: np.ndarray, period: int | None) -> np.ndarray | None:
    """The level each sample should sit on: the median (constant signal) or the median of each phase
    across periods (periodic signal of `period` samples). None if there are too few periods."""
    words = np.asarray(words, dtype=np.float64)
    if period is None:
        return np.full(words.size, np.median(words[:: checks.VI_SWAP_LEVEL_DECIMATION]))
    periods = words.size // period
    if periods < checks.VI_SWAP_MIN_PERIODS:
        return None
    per_phase = np.median(words[: periods * period].reshape(periods, period), axis=0)
    return per_phase[np.arange(words.size) % period]


def vi_swaps(v_words: np.ndarray, i_words: np.ndarray, period: int | None = None) -> np.ndarray | None:
    """Which samples of one channel carry the V/I swap (C.5); None when it cannot be told
    (a periodic signal with fewer than 3 periods)."""
    v_words = np.asarray(v_words, dtype=np.float64)
    i_words = np.asarray(i_words, dtype=np.float64)
    level_v = levels(v_words, period)
    level_i = levels(i_words, period)
    if level_v is None or level_i is None:
        return None
    near = checks.VI_SWAP_NEAR_CODES
    far = checks.VI_SWAP_ROBUST_DEVIATIONS
    v_off = np.abs(v_words - level_v)
    i_off = np.abs(i_words - level_i)
    return (
        (np.abs(v_words - level_i) <= near)
        & (np.abs(i_words - level_v) <= near)
        & (v_off > near)
        & (i_off > near)
        & (v_off > far * robust_spread(v_words - level_v))
        & (i_off > far * robust_spread(i_words - level_i))
    )


def halved_words(words: np.ndarray, period: int | None = None) -> np.ndarray | None:
    """Which samples carry a halved ADC word (fault F-02): the word sits near half its level (one
    bit shifted right), far from the level itself. None when it cannot be told (fewer than 3 periods)."""
    words = np.asarray(words, dtype=np.float64)
    level = levels(words, period)
    if level is None:
        return None
    off = np.abs(words - level)
    return ((np.abs(words - level / 2) <= checks.HALVED_WORD_NEAR_CODES)
            & (level / 2 > checks.HALVED_WORD_MIN_CODES)
            & (off > checks.VI_SWAP_ROBUST_DEVIATIONS * robust_spread(words - level)))


def count_halved(voltage: np.ndarray, current: np.ndarray, records: list[Record], channel: str,
                 period: int | None = None) -> dict[str, tuple[int, float, float]] | None:
    """F-02 on one channel, by quantity: (count, how far the halved words move the level, how far
    they move one 100 ms mean at most), in V or A. None when it cannot be told."""
    found = {}
    for quantity, words, step in zip(("V", "I"), channel_words(voltage, current, records, channel),
                                     (hw.V_STEP, hw.ADC_I_STEP)):
        mask = halved_words(words, period)
        if mask is None:
            return None
        level = levels(words, period)
        shift = np.where(mask, (words - level) * step, 0.0)
        blocks = block_means(shift)
        block_shift = float(np.max(np.abs(blocks))) if blocks.size else float(np.sum(shift))
        found[quantity] = (int(np.count_nonzero(mask)), float(np.mean(shift)), block_shift)
    return found


def channel_words(voltage: np.ndarray, current: np.ndarray, records: list[Record], channel: str,
                  transfer: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The ADC words behind one channel's samples (from the start of a read), with the stored
    calibration undone (I-09)."""
    v_raw = libsmu_math.raw_voltage(voltage, records[hw.record_index(channel, Kind.MEASURE_V)])
    i_raw = libsmu_math.raw_current(current, records[hw.record_index(channel, Kind.MEASURE_I)], transfer)
    return libsmu_math.adc_words(v_raw, i_raw)


def count_vi_swaps(voltage: np.ndarray, current: np.ndarray, records: list[Record], channel: str,
                   period: int | None = None) -> int | None:
    """How many samples of a channel carry the V/I swap; None when it cannot be told."""
    swapped = vi_swaps(*channel_words(voltage, current, records, channel), period=period)
    return None if swapped is None else int(np.count_nonzero(swapped))
