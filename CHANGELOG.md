# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

Released after its acceptance on the bench.

### Added

- The `phycovolt` command line: `status`, `signals`, `calibrate`, `analyze` and `plot`, a start page
  with tested examples, `--simulate` against a simulated M1K and multimeter, and styled output whose
  badge color is the verdict.
- Signals on each channel (`hi-z`, `source-v`, `source-i`) with DC, sine, square, triangle, pulse and
  biphasic waveforms, six circuit descriptions, 18 checks before switching on and 10 after the
  measurement, including the V/I swap and the halved ADC word.
- Bursts that park the DAC first, run in 100 ms chunks and leave both channels in HI_Z on every exit.
- The standard analysis: 100 ms block means, the uncertainty of a mean as the spread of the block
  means, the cycle average and the sine regression.
- Recordings that can be analyzed again from their files and that store the command that repeats them.
- Calibration of the four functions on both channels against the Agilent 34401A (RS-232 or typed), with
  the uncertainty of every value, verifications outside the fit, a report and a plot. The EEPROM is
  never written by the program.
- A test suite that runs without the instrument on Python 3.10 and on a newer Python: call-order tests
  for every operating rule, the acceptance scenarios and a Monte Carlo validation of the calibration's
  uncertainty.
