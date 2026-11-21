"""The Monte Carlo validation of the calibration's uncertainty (spec 8, A.2).

The ± the program prints is a promise: the true error is inside "predicted error ± U" about 95 % of
the time. In the simulator the true error is known. This repeats each stage 1000 times (fixed seed)
with new noise and new multimeter range errors, runs the very `calibration.solve` of the program on
the readings, and counts, for every value of the effect table, how often the true error falls
outside. An honest k = 2 misses about 45 of 1000 (standard deviation about 7): above 65 the test
fails; below 25 U is exaggerated (expected where the multimeter's rectangular specification or
method B dominates) and is reported. No verification may fail.

Run as a script to print the table:
    python -m tests.test_monte_carlo         (from the project root)
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pytest

from phycovolt import calibration as cal
from phycovolt import checks
from phycovolt import hardware as hw
from phycovolt import libsmu_math, multimeter
from phycovolt import simulator as sim
from phycovolt.hardware import Kind

PROFILE = multimeter.load_profile(multimeter.PROFILES / "agilent_34401a.ini")
UNIT = hw.load_unit()
SEED = 20261120


def monte_carlo(stage: str, runs: int = checks.MONTE_CARLO_RUNS, seed: int = SEED) -> tuple[Counter, int, int]:
    """(misses per (channel, value), verifications that failed, verifications made)."""
    rng = np.random.default_rng(seed)
    stored = list(sim.stored_records(UNIT))
    misses: Counter = Counter()
    failed = made = 0
    for _ in range(runs):
        bench = sim.BlockLevelBench(UNIT, PROFILE, rng)
        result = cal.solve(cal.StageData(stage, hw.CHANNELS, bench.observations(stage), stored, PROFILE))
        for channel, item in result.channels.items():
            true = UNIT.true_records[hw.record_index(channel, item.kind)]
            source = item.kind in (Kind.SOURCE_V, Kind.SOURCE_I)
            for index, row in enumerate(item.effect):
                code = int(libsmu_math.dac_code(row.x, item.kind, item.record)) if source else None
                true_error = cal.predicted_error(item.kind, row.x, true, item.record, code=code)
                label = "reference" if cal.EFFECT_VALUES[stage][index] == "reference" else f"{row.x:g}"
                misses[(channel, label)] += abs(true_error - row.after) > checks.COVERAGE * row.u
            for verification in item.verifications:
                made += 1
                failed += not verification.passed
    return misses, failed, made


@pytest.mark.parametrize("stage", cal.STAGES)
def test_coverage_of_predicted_error_plus_minus_u(stage: str) -> None:
    misses, failed, made = monte_carlo(stage)
    too_many = {key: count for key, count in misses.items() if count > checks.MONTE_CARLO_FAIL}
    assert too_many == {}, f"{stage}: coverage too low at {too_many}"
    assert failed == 0, f"{stage}: {failed} of {made} verifications failed"
    generous = {key: count for key, count in misses.items() if count < checks.MONTE_CARLO_WARN}
    if generous:
        warnings.warn(f"{stage}: U exaggerated (fewer than {checks.MONTE_CARLO_WARN} misses of "
                      f"{checks.MONTE_CARLO_RUNS}) at {sorted(generous)}", stacklevel=1)


if __name__ == "__main__":
    for stage in cal.STAGES:
        misses, failed, made = monte_carlo(stage)
        print(f"{stage}: verifications failed {failed} of {made}")
        for (channel, label), count in sorted(misses.items()):
            mark = "FAIL" if count > checks.MONTE_CARLO_FAIL else "low" if count < checks.MONTE_CARLO_WARN else "ok"
            print(f"  CH {channel} {label:>10}: {count:3d} misses of {checks.MONTE_CARLO_RUNS}  {mark}")
