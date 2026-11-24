# PhycoVolt

PhycoVolt drives the **ADALM1000 (M1K)**, a two-channel source-measure unit from Analog Devices, to
apply and measure weak electrical stimuli (µA to mA) on cultures of microalgae and bacteria. From one
command line it:

1. **calibrates** the M1K against a bench multimeter, with the uncertainty of every value and a
   verification at points outside the fit;
2. **generates and measures signals** on each channel (a voltage source, a current source, or voltage
   measurement only), checking before switching on what the instrument and the circuit cannot do, and
   checking the result against the circuit's physics;
3. **records** everything, analyzes a recording again at any time and plots it on request.

Version 0.1.0, not released yet: it is released after its acceptance on the bench, and no culture is
connected before that.

## Safety

- **Nothing is switched on without `--confirm-output`.** Without it, `phycovolt signals` reads both
  channels at rest, prints its checks and stops (`NOT SWITCHED ON`).
- **A check error (✗) blocks switching on.** Warnings (⚠) do not: read them and decide.
- **Every exit leaves both channels in HI_Z** (nothing sourced): the normal end, an error, Ctrl+C, the
  terminal closed, SIGTERM. Ctrl+C acts within 100 ms and keeps what was measured.
- **A program that dies** (`kill -9`, a crash in the vendor library, the USB cable pulled, the laptop
  suspended) runs no code, and the M1K probably keeps its last output: unplug its USB cable.
- **Rewire only when nothing is sourced:** at a calibration prompt (both channels wait in HI_Z) or with
  the program closed.
- **The program never writes the instrument's memory (EEPROM).** A calibration ends with the `smu -w`
  command; running it is your decision.
- **One program at a time on the M1K:** close ALICE, Pixelpulse2 and any other session that uses it.
- **An oscilloscope's ground clip goes on the M1K's GND only.** It is tied to mains earth; anywhere
  else it shorts that point to GND.
- **The V/I swap** (voltage and current samples exchanged, a known M1K fault): unplug the USB cable,
  wait a few seconds and plug it in again. Nothing is driven in that state.

## Installation

PhycoVolt needs micromamba (or conda) and conda-forge. libsmu 1.0.3, the vendor library, exists for
Python 3.10 at most, so there are two environments:

```bash
micromamba create -f environment.yml        # lab: Python 3.10 with libsmu/pysmu, drives the M1K
```

```bash
micromamba create -f environment-dev.yml    # development: a newer Python, no pysmu, no instrument
```

On Linux the M1K needs its udev rule, shipped with libsmu but not activated. Once, then reconnect the
M1K:

```bash
sudo cp "$CONDA_PREFIX/lib/udev/rules.d/53-adi-m1k-usb.rules" /etc/udev/rules.d/
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Run it from the project root with `python -m phycovolt`, or install the `phycovolt` command into the
active environment:

```bash
pip install -e . --no-deps --no-build-isolation
```

## Quick start

- `phycovolt` with no subcommand opens no instrument: it lists the commands with examples ready to copy.
- `phycovolt <command> --help` ends with that command's examples.
- **Rehearse with `--simulate`:** everything runs against a simulated M1K and multimeter, and the
  results go to `scratch/` unless `--output DIR` is given.

  ```bash
  phycovolt --simulate signals --a "source-i sine freq=100Hz pp=2mA circuit=resistor r=10 return=2.5V"
  ```

- `phycovolt status`, with nothing mounted: the identification, the 8 stored calibration records, the
  rest voltages and the board check (2 s at rest, looking for the V/I swap).

Exit codes: **0** done (also when nothing was switched on for want of `--confirm-output`); **1** could
not start, a check error, a lost read or a calibration stage that stopped; **130** stopped by you.

## Signals

```bash
phycovolt signals --a "<channel A>" --b "<channel B>" [--confirm-output] [--duration "2 s"]
```

A channel is described as `<role> [<waveform> <field>=<value> …] [circuit=<circuit> <field>=<value> …]`.
Fields left out take their defaults, which depend on the return pin and are never refused by the checks.
`--duration` (0.3 to 10 s) applies only when both channels are `hi-z`.

| Role | Mode | The channel |
|---|---|---|
| `hi-z` | HI_Z | sources nothing and measures voltage (0–5 V, 1 MΩ): the safe state |
| `source-v` | SVMI | sources a voltage and measures the current |
| `source-i` | SIMV | sources a current and measures the voltage |

| Circuit | For | Fields | Mounted |
|---|---|---|---|
| `resistor` | source-i | `r`, `return` (`2.5V`, `5V`) | channel → R → return pin |
| `ivi-board` | source-i | `scale` (required), `input` (2.5 V) | channel → the conditioning board's input |
| `load-shunt` | source-v | `load`, `shunt` (0 = none), `return` (`GND`, `2.5V`, `5V`) | channel → load → shunt → return pin |
| `free` | hi-z | — | any node; no physics check |
| `shunt-node` | hi-z | — | the node between the other channel's load and shunt |
| `shunt` | hi-z | `r`, `return` | the node of a known resistor whose other end is on a pin |

Waveforms: `dc` (`level`, `seconds`), `sine` and `triangle` (`freq`, `pp`, `center`), `square` (`freq`,
`low`, `high`, `duty`), `pulse` (`freq`, `baseline`, `pulse`, `width`), `biphasic` (`freq`, `center`,
`amplitude`, `width`, `gap`); periodic ones last `periods`. Values take SI prefixes and a point or a comma
(`200mA`, `4,7k`, `20uA`, `1kHz`, `1ms`). A period is a whole number of samples at 100 kS/s (5 kHz down to
0.1 Hz), and a burst lasts at most 10 s.

Positive current **leaves** the channel's terminal. The terminal only spans 0 to 5 V, so the return
decides which way current can flow: through a resistor to 2.5V both ways, to 5V only into the channel,
to GND only out of it.

```bash
phycovolt signals --a "source-v dc level=1 circuit=load-shunt load=47 shunt=10 return=GND" --b "hi-z circuit=shunt-node"
```

```bash
phycovolt signals --a "source-i biphasic amplitude=2mA circuit=ivi-board scale=0.01"
```

A burst goes: the rest reading (100 ms, both channels in HI_Z, which shows where the return wire really
is) → the checks "Before switching on" → `SWITCHING ON` → the burst in 100 ms chunks with a progress line
→ the result and the checks "After the measurement".

## Reading the output

Every command ends with its verdict and the next actions (`Folder`, `Plot`, `Repeat`), so they stay in
view above the prompt; the details come above them. The badge's color is the verdict:

| Color | Means | Examples |
|---|---|---|
| green | nothing to look at | `COMPLETE` with no warning; `PASS` |
| yellow | look above before going on | warnings; `STOPPED` by you; `NOT SWITCHED ON` without `--confirm-output`; a failed calibration verification |
| red | it failed | `FAILED`; `NOT SWITCHED ON` for a check error; `FAIL` |
| blue | what is happening | `SWITCHING ON`, `MEASURING` |

Symbols: ✓ passed, ◆ note, ⚠ warning, ✗ error. They carry the meaning without color; styles turn off in
pipes and with `NO_COLOR`, and the files are plain text.

- A DC level is shown as value ± u, with u the standard deviation of the 100 ms block means (spec §8).
  The ± is statistical only.
- "DAC delivers" is what the DAC's code gives with the stored calibration: up to one step (76.3 µV, or
  9.5 µA) from the request.
- `Repeat` writes both channels in full and never includes `--confirm-output`, so a repeated command
  first shows the checks.

## Calibration

```bash
phycovolt calibrate [--stage all|measure-v|source-v|measure-i|source-i] [--typed] [--dmm PROFILE] [--base CALIB]
```

It measures the M1K's errors against the multimeter (the Agilent 34401A over RS-232, or `--typed` readings)
and writes a new `calib.txt`. It works with the EEPROM already calibrated, so `smu -r` is never needed
first. `--base` (default: the project's `calib.txt`, only read) completes a partial calibration. Before
it starts, the M1K must have been on for 15 min and the multimeter for 1 h.

At each prompt the program says what to mount, and waits with both channels in HI_Z: **Enter**
continues, **q** stops. A reading is redone after a jump, the ADC at its limits, the limiter, a failed
physics check (for example, the M1K and the multimeter disagreeing by more than 5 %) or a multimeter
error. In typed mode, type every decimal the display shows; the program says how many.

| Stage | Points |
|---|---|
| measure-v | zero (channels on GND), reference (the 2.5V pin, multimeter on DC V), verification (the 5V pin) |
| source-v | 0 V and 2.5 V, verification at 4.5 V (the multimeter on the channel) |
| measure-i | zeros (channels open), a load probe, ±100 mA, verifications at ±10 mA (channel → 2.5–25 Ω, ½ W or more → multimeter I → 2.5V pin) |
| source-i | the same loop: a load probe, 0 and ±100 mA, verifications at ±10 mA |

The report gives each record ± U (U = 2u), the error at each value before and after, the verifications
(`PASS`/`FAIL`), the uncertainty budget, and the `smu -w` and `smu -d` commands. On this M1K two
verifications are expected to fail: source-v at 4.5 V (the source bends near 5 V, H-06) and measure-i at
±10 mA (the zero is measured open and cold, H-08). After a calibration: keep a copy with
`smu -d > old_calibration.txt`, write with the `Write` command if you accept it, check with `smu -d`,
reconnect the M1K and run `phycovolt status`.

## Analyze and plot

```bash
phycovolt analyze recordings/2026-11-22_101500_source-i-sine
```

```bash
phycovolt plot recordings/2026-11-22_101500_source-i-sine
```

`analyze` runs the standard analysis again from the files and never rewrites the folder. `plot` opens
a window in its own process and gives the terminal back at once: signals in time with the request and
what the DAC delivered, the cycle average, or for DC the block means and a histogram; for a calibration
folder, the errors before and after, ±U, the verifications and the budget.

## Files

- `recordings/<date>_<time>_<role>-<waveform>/`: `samples.csv.gz` (every sample of both channels),
  `metadata.json` (configuration, stored calibration, commands, checks and results) and `summary.txt`.
- `calibrations/<date>_<time>_<stage>/`: `calib.txt` (for `smu -w`, written only with all 8 blocks),
  `blocks.txt`, `readings.csv`, `dmm.csv`, `report.txt` and `metadata.json`.

A stopped calibration still writes what finished, in a folder ending in `_interrupted`. These folders
and `lab/` stay out of git: back them up yourself.

## Checks

| Check | Means | What to do |
|---|---|---|
| `wiring`, `return` | the rest reading or the result puts the circuit's far end elsewhere than the chosen return (an error for a voltage source) | move the return wire, or choose the pin it is on |
| `pin_limit` | a current source would leave, or left, 0–5 V | lower the current or the resistance, or change the return |
| `range`, `dac_floor`, `dac_ceiling` | the request is outside what the DAC gives (about +1 to +3 mV at code 0, about 4.993 V at the top) | keep the waveform inside |
| `current_limit`, `overcurrent` | above 150 mA a warning, above 200 mA the limiter acts | lower the current or raise the load |
| `power` | a resistor would dissipate more than 0.25 W | use the rating given, or lower the current |
| `duration`, `samples_per_period`, `frequency`, `mains` | too long, too fast, rounded, or 60 Hz | change the waveform |
| `dac_staircase`, `dac_level`, `micro_amps` | at µA the DAC step and the current zero weigh | expected; measure µA as a voltage over a `shunt` |
| `switch_on`, `charge_balance` | what the channel does before the first sample; a mean current that is not zero | notes; use the center suggested, or a biphasic waveform |
| `board` | the V/I swap: the measurement is invalid | re-plug the USB cable and repeat |
| `halved_word` | single current samples halved on CH B (a known fault, about 0.1 µA on a 10 s level) | note it; re-plug if it keeps appearing |
| `amplitude`, `level`, `shunt_agreement` | the result differs from the request or from the other channel | follow the cause given |
| `resistance`, `drift` | the resistance seen differs from the one given (bad jumpers add 15–30 Ω); a level drifts (a heating resistor) | check the jumpers; measure the resistors; use a higher wattage |

## Tests

The tests never touch the instrument: they run against the simulated M1K and a fake pysmu that records
every call, in about 2 minutes.

```bash
micromamba run -n phycovolt python -m pytest
```

```bash
micromamba run -n phycovolt-dev python -m pytest
```

`docs/specification.md` holds the design, the methods with their numbers, and what is known about the
instrument.

## License

Apache License 2.0: see `LICENSE` and `NOTICE`.
