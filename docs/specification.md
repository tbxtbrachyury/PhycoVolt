# PhycoVolt — Project specification

---

## 0. Summary

**What to build.** PhycoVolt: one Python program that drives the **ADALM1000 (M1K)**, a low-cost
two-channel source-measure unit from Analog Devices connected over USB. From one command line it:

1. **calibrates** the M1K against a bench multimeter, produces the calibration file the operator
   writes into the instrument's memory, states the uncertainty of every value and verifies the
   result at a point that did not enter the computation;
2. **generates and/or measures signals**, each channel configured on its own (source voltage, source
   current, or measure voltage only), with parametric waveforms (DC, sine, square, triangle, pulses,
   biphasic pulses); it checks before switching on what the instrument cannot do, measures, checks
   the result against the physics of the circuit, and records everything;
3. **analyzes recordings** again at any time, and opens plots on request.

**Why.** The research applies weak currents (µA to mA) to cultures of microalgae and bacteria. At µA, errors of 1–2 % or of tens of µA invalidate an experiment. The program must
**calibrate rigorously**, **check everything** and **never apply to a circuit (or a culture) anything
that was not requested**.

**Non-negotiable constraints.**

- Nobody but the operator drives the instrument (§2.3). Development runs against a simulator.
- The operating rules of §6.1 are followed by every call sequence.
- v0.1 runs on Python 3.10 with libsmu 1.0.3; the way to an own driver and to newer Python stays
  open and paved (§4.3).
- **English everywhere** in the software: code, user interface, messages, comments, documents, file
  and column names.
- **Simple and readable**: a flat package of single-purpose files, plain functions and dataclasses
  (§7).

**Most important section:** §6, what is known about the instrument and its library. It was learned
on the bench and by reading libsmu's source code, and most of it is not in the vendor documentation.
A programmer without it would break several rules without noticing, and some of those rules protect
the circuit and the culture from stimuli nobody asked for.

**Normative words.** MUST / MUST NOT: mandatory; not complying is a defect. SHOULD / SHOULD NOT:
recommended; deviating needs a written reason in the decision log (§2.3). MAY: optional.

---

## 1. Context

### 1.1 The research

Weak electrical stimulation (µA to mA; DC, AC and pulses) of microalgae and bacteria cultures.
PhycoVolt is the stimulation system's software: it calibrates the M1K, applies the stimuli and
measures voltage and current; the decision on filtering the measured signals, and impedance, come in
later versions (§4.2). The research protocol is the research lead's (§11 item 4).

### 1.2 The signal chain

```
M1K (USB, laptop)  ──►  i-v-i conditioning board  ──►  electrodes in the culture
```

The **i-v-i board** is an external conditioning circuit with its own supply. The M1K sources a current
into the board's input, which sits at 2.5 V; the board delivers to the electrodes a current
proportional to the M1K's and much smaller, by a scale factor still to be measured (§11 item 6).

Consequences:

- the **current source** is the M1K's most important function for the application;
- the M1K works in mA, where it is relatively more accurate; at the culture, the board's own errors
  (gain from resistor tolerances, amplifier offsets, output impedance, drift) dominate;
- the M1K's HI_Z "safe state" means zero input current to the board, so the culture receives only
  the board's own offset current;
- the board is not characterized yet (roadmap 0.3).

On a culture, the solution's impedance can be low, so the voltage across it may be near the M1K's
resolution. Synchronous detection and cycle averaging (§3.6) address that.

### 1.3 Electronics concepts used in this document

- **Voltage source** imposes a voltage; the current is whatever the circuit draws (I = V ÷ R).
  **Current source** imposes a current; the voltage is whatever is needed (V = R × I).
- **Compliance:** a current source holds its current only while the needed voltage fits in its range.
  Outside it, it saturates: the voltage sticks to the limit and the current falls short.
- **SMU** (source-measure unit): on the same terminal and at the same time, it sources one quantity
  and measures the other. The M1K has two SMU channels.
- **Channel modes:** **HI_Z** (high impedance): sources nothing, measures only the voltage; the safe
  resting state. **SVMI** (source voltage, measure current). **SIMV** (source current, measure
  voltage).
- **Current sign:** positive current **leaves** the channel's terminal (the channel sources);
  negative **enters** it (the channel sinks).
- **Return:** a circuit on a channel closes its current path at a fixed-voltage pin. The M1K offers
  **GND** (0 V), **2.5V** and **5V**. Since the channel's terminal only spans 0 to 5 V, the return
  decides which way current can flow: through a resistor to 2.5V a current source can push both ways
  (the terminal goes above or below 2.5 V); to 5V, only into the channel; to GND, only out of it.
- **Shunt:** a small, known resistor in series; its voltage (I × R_shunt) shows the current on a
  voltage instrument.
- **DAC and ADC:** the digital-to-analog converter turns an integer, the **code**, into a voltage or
  current; the analog-to-digital converter does the opposite. With 16 bits, codes go from 0 to
  65 535, and the smallest change is the **step** (on the M1K: 76.3 µV in voltage and about 9.5 µA in
  sourced current). At µA, the current step is large: the output is a staircase.
- **Calibration:** each function of the instrument errs in its own way, well described by a line:
  `raw value = true value ÷ gain + offset`. Calibrating means measuring the **offset** (error at zero)
  and the **gain** (error in slope) against a better reference (a **digital multimeter**), and writing
  them into the instrument's memory (the EEPROM), so that the library applies them to every sample.
- **Noise and hum:** every sample has random noise plus 60 Hz interference from the mains (the
  "hum"). Nothing is decided from one sample: averages are used.
- **Oscilloscope ground:** a bench oscilloscope's ground clip is tied to mains earth, and the M1K's
  GND to the computer's ground through USB. Clipping the ground anywhere but the M1K's GND shorts
  that point to GND through earth.

### 1.4 The instrument: ADALM1000 (M1K)

| Item | Value |
|---|---|
| Channels | 2 (CH A and CH B), each an SMU |
| Converters | one 16-bit DAC and one 16-bit ADC per channel; 100 000 samples/s per channel |
| Ranges | voltage 0 to 5 V; current −200 to +200 mA (source and measure) |
| Modes | HI_Z, SVMI, SIMV, per channel |
| Fixed pins | GND, 2.5V and 5V (each up to 200 mA; 5V is the USB supply) |
| Input in HI_Z | 1 MΩ |
| Protection | current limiter at ±200 mA; an ADM1177 that records overcurrent |
| The project's unit | hardware rev D, firmware 2.17; came **without** factory calibration; calibrated 2026-11-18 |

### 1.5 The instrument's software stack

```
 PhycoVolt (Python 3.10)    ALICE Desktop    Pixelpulse2    smu (command line)
          │                       │               │               │
          └──────────── pysmu (the Python binding of libsmu) ─────┘
                                  │
                          libsmu 1.0.3 (C++)
                                  │
                         libusb  →  M1K firmware
```

**libsmu** is the vendor library, and it is not a neutral driver: it **applies the calibration**
stored in the EEPROM to every measured sample and every requested value, and decides the code sent to
the DAC. **pysmu** is its Python binding. The **smu** utility comes in the same package: `smu -l`
lists devices, `smu -d` shows the stored calibration, `smu -w file` writes a calibration, `smu -r`
resets it to the default. libsmu has been frozen since 1.0.4 (2022); the only packaged version
(conda-forge) is 1.0.3, for Python ≤ 3.10. Many rules of §6 exist to work around this version's
behavior, found by reading its source code and by measuring on the instrument.

---

## 2. People, equipment and working rules

### 2.1 Roles

- **Operator:** runs the bench and the research; switches the instrument on, mounts circuits, runs
  everything that touches hardware, approves what affects use.
- **Developer:** whoever implements (a Claude Code session or a person).
- **Reviewer:** reviews each milestone (§10).
- **Research lead:** defines the research protocol.

### 2.2 Equipment

- **M1K**, rev D, firmware 2.17, calibrated on 2026-11-18 (`calib.txt`).
- **Agilent 34401A** 6½-digit multimeter: the **calibration reference**. It has RS-232 and GPIB (one
  active at a time) and **no USB**. The RS-232 link needs a null-modem cable (available since
  2026-11-24) and a USB-to-RS-232 adapter. Until the link is validated on the bench (roadmap 0.2),
  calibrations use the typed mode. Another bench multimeter with a computer link needs its own
  profile (§3.8).
- **Keysight DSOX2022A oscilloscope** (InfiniiVision 2000 X: 200 MHz, 2 channels, 8 bits, a USB device
  port on the rear).
  It is **not** a calibration reference: its DC accuracy is about 2–3 % of full scale (8-bit), roughly
  the error of an uncalibrated M1K, while the 34401A is hundreds of times better; averaging reduces
  random error, never a systematic one. It is the right instrument for **dynamics**: switch-on
  transients, gaps between bursts, pulse shapes, the board's frequency response (roadmap 0.3 and 0.4).
- Resistors (2.5–25 Ω for calibration; 10 Ω, 47 Ω and others for signals), jumpers, the i-v-i board
  and its supply.
- **The development laptop** (Ubuntu, micromamba): used for development and for the operator's bench
  sessions; it will **not** be used by the researchers. The M1K's udev rule is installed and the user
  is in `plugdev`; the Linux kernel's `usbtmc` driver is available (oscilloscope over USB).
- **The research computer**, where the researchers will run experiments (including long stimulation),
  is not defined yet: operating system, administration and network are open (§11); it may be a
  Windows machine.

### 2.3 Working rules for developers

1. **You MUST NOT drive the instrument.** No `smu`, no `pysmu.Session()`, no running PhycoVolt
   without `--simulate`, even with the M1K plugged into your computer. It may be wired to a circuit
   or a live culture, and a mode change applies voltage or current (L-04, L-05). When you need
   something from the hardware, write a procedure and ask the operator. Reading libsmu's source code
   and the system state (udev, permissions, sysfs) is free.
2. **The data are the operator's.** `calib.txt`, `recordings/`, `calibrations/`, `lab/` and `drafts/`
   are never deleted or rewritten. Your test runs write to a scratch folder (`--output`).
3. **The EEPROM is written only by the operator** (`smu -w`), never by the program (I-08).
4. **No new dependency without the operator's OK.** Approved for v0.1: numpy, scipy, matplotlib with
   the Qt backend (PySide6), rich, pyserial, and libsmu/pysmu 1.0.3. To try something else, use a
   separate venv.
5. **Tests first where there is hardware:** code that talks to an instrument starts with a call-order
   test against the fake library.
6. **Decisions:** a deviation from a SHOULD or from §7 → a record in the decision log
   (`docs/decisions.md`, kept locally, outside version control: context, decision, alternatives,
   consequences). A method error or a new instrument behavior → record it and
   tell the operator; do not change the method on your own. **Statistical choices are the operator's**
   (§8).
7. **Git:** small, descriptive commits; one branch per work package, merged after review.
8. **Appearance:** check the terminal output at 80 columns, on dark and light terminals.
9. **The repository is public.** Versioned files carry no personal data, no names of people or
   institutions, and no content from unpublished research documents. Local material stays out of
   git: the operator's folders (`lab/`, `drafts/`, results) through `.gitignore`, anything else
   through `.git/info/exclude`.

---

## 3. What v0.1 does

### 3.1 One program, one command line

`python -m phycovolt` from the project root, or `phycovolt` after `pip install -e . --no-deps`. Every
operation is a subcommand. SHOULD look like:

```
phycovolt status
phycovolt signals --a "source-i sine freq=100Hz pp=2mA center=0 circuit=resistor r=10 return=2.5V" \
                  --b "hi-z" --confirm-output
phycovolt calibrate --stage measure-v --dmm phycovolt/profiles/agilent_34401a.ini --base calib.txt
phycovolt analyze recordings/2026-11-22_101500_source-i-sine
phycovolt plot recordings/2026-11-22_101500_source-i-sine
```

**The start page.** Without a subcommand, `phycovolt` opens no instrument and prints: the name and
version; each subcommand with one line; examples grouped by task, ready to copy (measure both
channels at rest; a current sine into a resistor; a DC voltage into a load and a shunt; a typed
calibration of one stage; analyze and plot a folder), with how to try any of them with `--simulate`;
how to get help on one subcommand; and that Ctrl+C stops at once with both channels in HI_Z. Each
subcommand's `--help` ends with its own examples. Every example on the start page and in the help MUST
be accepted by the parser (a test), so that they never go stale.

**Styles by relevance.** The terminal output is styled by meaning, with a few semantic styles defined
in one place (the texts say what a part is, never which color it has):

| Style | For | Look (SHOULD) |
|---|---|---|
| action / danger | errors (✗); "Nothing was switched on"; `--confirm-output`; an interruption | bold red or bold yellow |
| attention | warnings (⚠) | yellow |
| result | the measured values and their ± | bold; the asked or expected value next to it dimmed |
| context | identification, start time, folder, secondary details | dimmed |
| safe | passed checks (✓); "both channels are in HI_Z" | green |
| command | commands and options in examples and hints | cyan |

If everything is highlighted, nothing is: a line gets a style only when it falls in one of these
rows; notes (◆) keep the terminal's plain look. Style never carries information alone (the symbols
✓ ◆ ⚠ ✗ and the words stay). The files written (`summary.txt`, `report.txt`) stay plain text. Styles
turn off when the output is not a terminal or when `NO_COLOR` is set. Readable at 80 columns on dark
and light terminals. Library: `rich`.

**Hierarchy.** Structure comes before color:

1. **Order for the screen.** A terminal scrolls, and the last lines stay in view above the prompt. So
   every command ends with its verdict and the operator's next actions (folder, plot, repeat; in a
   calibration, the `smu -w` command). The details come above them.
2. **A verdict line** at each status moment, as a badge (reverse video, a word in capitals): `SWITCHING
   ON` (with what is applied), `NOT SWITCHED ON` (with why: an error, or no `--confirm-output`),
   `COMPLETE` / `STOPPED` / `FAILED` at the end of a burst (with the duration, "both channels are in
   HI_Z" and the count of errors, warnings and notes), `PASS` / `FAIL` for each verification of a
   calibration. Badges are only for these moments, one or two per run. In plain text they stay as the
   words. **The badge's color is the verdict, the same rule in every command:** green when there is nothing to look at; yellow when the operator must look above (warnings, a
   failed calibration verification, `NOT SWITCHED ON` for want of `--confirm-output`, a stop by the
   operator); red when it failed (an error, the limiter, a lost read). `SWITCHING ON` and `MEASURING`
   say what is happening, not a verdict, and keep a color of their own. Notes do not change it. A
   `COMPLETE` with warnings is yellow, never green. When a calibration verification failed, the
   `smu -w` line says so ("a verification failed (above): write it only if you accept the record
   there").
3. **Sections** with a rule and a title ("Before switching on", "Result", "After the measurement"),
   and a blank line between groups. Channel headers ("CH A source-i sine 100.0 Hz → R 10.00 Ω → 2.5V")
   in bold: they are where the eye lands.
4. **Dim the normal, show the exception.** Passed checks (✓) keep their green symbol and a dimmed
   text; notes plain; warnings and errors stand out. After a burst, on the terminal, the passed
   checks are counted on one line ("✓ 8 passed"), and only notes, warnings and errors are written out.
   The before-checks are written out in full before switching on (the operator decides on them) and
   not repeated at the end ("Before switching on: ✓ 8 passed, ◆ 1 note, shown above").
5. **Aligned labels.** In a result, the quantity names form a column (dimmed), the measured values are
   bold, the asked or expected values dimmed next to them. The same for "Folder", "Plot" and "Repeat".
6. **Long explanations stay in the files.** On the terminal, the paragraph on how the numbers were
   obtained is one dimmed line pointing to `summary.txt`; the full text and every check stay there.
7. **No boxes around anything to copy** (their border characters would be copied with it).

The start page follows the same rules: section titles in bold, each example's description dimmed and
its command in the command style. In a calibration, each stage opens with a section rule, and what
the operator must do at a prompt (what to mount, what to type, which key) is its strongest line.
`phycovolt analyze` shows a recording as the end of a burst does.

**Repeatable.** Every recording and calibration folder stores the command line that produced it (a
recording in its `metadata.json`), with each channel's description written out in full (every field, defaults
included), so that a later version with other defaults repeats the same burst. The summary ends with
it ("To repeat: …") **without** `--confirm-output`, so that a repeated command first shows the
before-checks; the operator adds the flag. A summary also ends with the command that opens its plot.

Global options: `--simulate` (everything against the simulated instrument), `--output DIR` (where
recordings and calibrations go; default the project root's `recordings/` and `calibrations/`, and,
for a simulated run, the project's `scratch/`, so that simulated and real results never mix).

**Switching an output on always needs the operator's confirmation:** the explicit `--confirm-output`
flag (the operator's responsibility). Without it, the command stops after the before-checks, which is
how the operator reads them before switching on. The calibration's steps (mounting, warm-up, typed
values) are prompts in the terminal: Enter continues, `q` stops.

**Exit codes:** 0 done; 1 could not start (no pysmu, no board, more than one board, firmware,
implausible stored calibration, V/I swap); 130 interrupted by the operator.

**Simulation.** With `--simulate`, every flow runs against the simulated instrument (and simulated
multimeter), with no difference for the operator except the identification "simulated (no board)".

### 3.2 Opening the instrument, and `status`

- Exactly one M1K MUST be connected. None: ask to check the cable, close ALICE, Pixelpulse2 and any
  other Python session using the M1K, and run `smu -l` inside the environment. More than one: ask to
  leave only one.
- Right after opening, a 100 ms read with both channels **explicitly** in HI_Z (I-06).
- **Board check:** 2 s with both channels in HI_Z, looking for the V/I swap (F-01, C.5). If found: the
  message "Unplug the M1K's USB cable, wait a few seconds and plug it in again", no channel driven,
  exit code 1.
- **`phycovolt status`:** serial, firmware, hardware revision; the 8 stored calibration records
  (warning if a record is at the default, error if implausible); the rest voltage of both channels;
  the board check result.

### 3.3 Signals

#### 3.3.1 Channel roles

Each channel is configured on its own to one role:

| Role | Mode | What it does |
|---|---|---|
| `hi-z` | HI_Z | sources nothing; measures voltage only (0–5 V, 1 MΩ input); the safe state |
| `source-v` | SVMI | sources a voltage waveform; measures the current |
| `source-i` | SIMV | sources a current waveform; measures the voltage |

Both channels MAY source at the same time, each with its own circuit and checks. A `hi-z` channel
measures only voltage; a current is obtained from a voltage across a known resistor.

Both channels MAY also be `hi-z`: a **measure-only** recording, in which nothing is sourced. Its
duration is then a parameter of its own (0.3 to 10 s, default 1 s; no waveform sets it), no DAC is
parked, and each channel is analyzed with the reading summary (§3.6), plus the current for a `shunt`
circuit.

Every M1K voltage is measured against the M1K's GND; a voltage between two other points is the
difference of two measurements.

#### 3.3.2 Circuits

The checks need to know what is connected. v0.1 circuits:

| Circuit | For role | Description | Returns offered (default first) |
|---|---|---|---|
| `resistor` | source-i | channel → R → return pin | 2.5V, 5V |
| `load-shunt` | source-v | channel → load → shunt → return pin (shunt on the return side; shunt 0 = no shunt) | GND, 2.5V, 5V |
| `ivi-board` | source-i | channel → i-v-i board input (held at about 2.5 V by the board) | the board |
| `free` | hi-z | measures a node; no physics check | — |
| `shunt-node` | hi-z | the node between the load and the shunt of the **other** channel's `load-shunt` circuit: reports the shunt current and the load voltage, and checks them against the other channel | — |
| `shunt` | hi-z | the node of a known resistor R whose other end is on a pin (GND, 2.5V, 5V): reports the current (V − V_pin) ÷ R | GND, 2.5V, 5V |

GND is **not** offered as a return for `resistor` with `source-i`: at small currents the pin would sit
at 0 V, where the "pin at the limit" check cannot tell saturation from a good measurement.

`ivi-board` in v0.1 is a black box with two operator-entered numbers: the **scale factor** k
(culture current = k × M1K current, signed) and the input voltage (nominal 2.5 V; the rest reading
gives the real one). The program shows the requested and measured values at the M1K and at the
culture. The confirmation before switching on MUST name the circuit and, for `ivi-board`, remind the
operator that the board's output may be connected to a culture.

#### 3.3.3 Waveforms

Every waveform is one function that returns **one period** of samples at 100 kS/s (for DC, one block
of 100 ms with the same value); the channel repeats it cyclically (L-08). Only that block is written to
the channel.

| Waveform | Parameters |
|---|---|
| `dc` | level |
| `sine` | frequency, peak-to-peak, center |
| `square` | frequency, low, high, duty cycle |
| `triangle` | frequency, peak-to-peak, center |
| `pulse` | frequency, baseline, pulse level, pulse width |
| `biphasic` | frequency, baseline (center), amplitude, phase width, inter-phase gap (charge-balanced by construction) |

Plus, for every waveform: the duration (number of periods, or seconds for DC). Values are V for
`source-v` and A for `source-i`.

Rules:

- A period has `N = round(100 000 ÷ f)` samples, with 20 ≤ N ≤ 1 000 000 (5 kHz to 0.1 Hz). The real
  frequency `100 000 ÷ N` is shown (I-15).
- A sine starts at its center: `center + amplitude × sin(2πk/N)` (I-15). Square and pulse edges fall on
  sample boundaries; widths are rounded to whole samples (10 µs) and the real width is shown.
- A measurement lasts at most 10 s (I-07). DC lasts at least 0.3 s (3 blocks: one for settling and two
  for the statistics).
- Values accept SI prefixes and a decimal point or comma: "200 mA", "4,7 k", "20 µA" (also "20 uA"),
  "1 kHz". A number without a unit is in the field's display unit.
- **Defaults depend on the return.** When the operator changes the return, the waveform parameters go
  back to the defaults that fit it; when the return is kept, typed values stay. **Accepting every
  default, for every waveform, role and offered return, MUST never produce a waveform that the
  before-checks refuse** (acceptance S19). Proposed defaults (SHOULD), with `resistor` 10 Ω or
  `load-shunt` 47 Ω + 10 Ω:

| Role | Waveform | Return 2.5V (or GND for source-v) | Return 5V |
|---|---|---|---|
| source-i | sine | 100 Hz, 200 mA p-p, center 0 | 100 Hz, 80 mA p-p, center −60 mA |
| source-i | dc | +10 mA, 1 s | −20 mA, 1 s |
| source-i | square | 100 Hz, −50 / +50 mA, 50 % | 100 Hz, −80 / −20 mA, 50 % |
| source-i | triangle | 100 Hz, 100 mA p-p, center 0 | 100 Hz, 60 mA p-p, center −50 mA |
| source-i | pulse | 10 Hz, baseline 0, pulse +50 mA, 1 ms | 10 Hz, baseline −20 mA, pulse −80 mA, 1 ms |
| source-i | biphasic | 10 Hz, center 0, ±50 mA, 1 ms, gap 0.1 ms | not offered (one direction only) |
| source-v | sine | 100 Hz, 2 V p-p, center 2.5 V | same |
| source-v | dc | 1 V, 1 s | same |
| source-v | square | 100 Hz, 1.5 / 3.5 V, 50 % | same |
| source-v | triangle | 100 Hz, 2 V p-p, center 2.5 V | same |
| source-v | pulse | 10 Hz, baseline 0.5 V, pulse 3 V, 1 ms | 10 Hz, baseline 4.5 V, pulse 2.5 V, 1 ms (a 0.5 V baseline would dissipate 0.28 W in the load) |
| source-v | biphasic | 10 Hz, center 2.5 V, ±1 V, 1 ms, gap 0.1 ms | same |

For `ivi-board`, defaults are expressed at the M1K (source-i, return = the board) and SHOULD stay at a
few mA (for example sine 100 Hz, 4 mA p-p, center 0), which gives tens of µA at the culture with a
scale near 0.01.

When the operator changes the waveform, the chosen channels and circuits are kept.

#### 3.3.4 Before switching on

1. **Rest reading:** 100 ms with both channels **explicitly** in HI_Z (I-03, I-14). The median voltage
   of each channel shows where the other end of its circuit is (with no current, resistors drop
   nothing). It feeds the "wiring" check and every expected value that depends on a pin's real voltage
   (H-04), and it leaves the DAC at code 26 214 (L-04).
2. **Checks**, computed on the actual sample block, the circuit and the rest reading, so they work for
   any waveform. Levels: **error** (✗, blocks switching on), **warning** (⚠, informs, the operator
   decides), **note** (◆, explains), ok. Thresholds in appendix A.2.

| Check (code) | Where | Rule | Level |
|---|---|---|---|
| `frequency` | periodic | relative error of the real frequency > 0.1 % → warning; smaller and not zero → note | warning / note / ok |
| `mains` | periodic | f = 60 Hz → warning; 60 Hz ÷ f an integer > 1 → note | warning / note |
| `samples_per_period` | periodic | fewer than 100 samples per period | warning |
| `duration` | all | the measurement would exceed 10 s | error |
| `dac_staircase` | periodic | fewer than 30 distinct DAC levels per period; and the shift of the mean | warning / ok |
| `dac_level` | dc | what the DAC delivers differs from the request by more than 2 % | warning / ok |
| `calibration` | sourcing channel | the channel's source record is at the default, or implausible | warning / ok |
| `wiring` | all | the rest voltage falls in the window of the chosen return (GND ±0.15 V; 2.5V 2.2–2.7 V; 5V 4.4–5.3 V). On another pin: **error in SVMI**, warning in SIMV. Far from every pin: warning (open channel; circuit on the other channel, saying whether the other channel is at the return; electrodes). Circuit above 50 kΩ: note (the 1 MΩ input pulls the pin) | error / warning / note / ok |
| `pin_limit` | source-i | V_return + R·I leaves the compliance range of the return; the message gives the possible current range for this R (and, for periodic waves, the center and amplitude that fit) | error / ok |
| `range` | source-v | the waveform leaves 0–5 V | error / ok |
| `dac_floor` | source-v | the waveform asks for less than what code 0 delivers (H-05) | warning |
| `dac_ceiling` | source-v | the waveform asks for more than what the top code delivers (H-16; about 4.993 V on this unit) | warning |
| `current_limit` | all | peak above 200 mA → error; above 150 mA → warning | error / warning / ok |
| `power` | all | dissipation (rms for periodic waves) of a resistor above 0.25 W → warning, suggesting the commercial rating (⅛, ¼, ½, 1, 2, 5 W) that holds twice that | warning / ok |
| `switch_on` | sourcing channel | SIMV: the current of the parked code on that channel, the source's uncalibrated zero (note). SVMI: the current during the milliseconds at ≈ 2.0 V before the first sample, (2.0 − V_return) ÷ R, above 200 mA → warning | note / warning |
| `micro_amps` | source-i | peak below 0.5 mA: the DAC step and the measured-current zero weigh | note |
| `charge_balance` | all | the mean current, when it is not zero. In SVMI it uses the real return voltage; with the 2.5V return it says the center that zeroes the mean, and is ok when the center is within 1 mV of it | note / ok |
| `hi_z_range` | hi-z | the expected node voltage (from the other channel's circuit, when known) leaves 0–5 V | error / ok |

#### 3.3.5 The sequence on the instrument

rest reading → operator confirmation → (if any channel sources, SVMI or SIMV: **park the DAC**, 10 ms
with both channels explicitly in HI_Z) → discard stale samples → write each sourcing channel's block (cyclic) →
set the modes, right before the read starts → read in 100 ms chunks → end: both channels in HI_Z; read
the overcurrent flag. This follows I-01 to I-07 and MUST be covered by call-order tests.

#### 3.3.6 During the measurement

A progress line, no ASCII plots: a bar and the seconds done of the total, under the channels' roles
and circuits. The burst loop itself redraws it at each 100 ms chunk (no extra thread), and drawing
never delays the acquisition. **Ctrl+C is the Stop** (§3.9): both channels go to HI_Z within one chunk
and the recording is kept, marked as stopped. The progress line says so.

Per-channel live numbers (the latest 100 ms means of V and I, the min and max of the latest block; for
SIMV, the pin status, "away from 0 and 5 V" or "touching a rail"; for SVMI, the current against
200 mA) are for later (§4.2).

#### 3.3.7 After the measurement

The standard analysis (§3.6), then the after-checks:

| Check (code) | Where | Rule | Level |
|---|---|---|---|
| `board` | all | V/I swap in the channel's samples (constant: against the median; periodic: against each phase's median, with ≥ 3 periods): the measurement is invalid | error |
| `halved_word` | all | samples whose ADC word is half the expected level (F-02), on any channel and quantity: their count, and how far they move the levels; suggest re-plugging the USB cable if they keep appearing | warning / ok |
| `amplitude` | periodic | the measured amplitude (or peak-to-peak) differs from the requested by more than 5 % (current) or 2 % (voltage); the message gives the probable cause (limiter, pin at the limit, high frequency, µA) | warning / ok |
| `level` | dc | the measured level differs from what the DAC delivers by more than 60 µA + 1 % (current) or 2 mV + 0.2 % (voltage) | warning / ok |
| `pin_limit` | source-i | more than 1 % of the samples within 20 mV of 0 or 5 V | error / ok |
| `overcurrent` | sourcing | the limiter acted | error |
| `resistance` | periodic | the resistance seen (ratio of V and I amplitudes, if the I amplitude ≥ 20 µA) differs from the informed one by more than 10 % + 1 Ω | warning / ok |
| `resistance` | dc, SVMI | almost no current (< 10 % of the expected) where the circuit would give more than 1 mA and the channel is more than 0.5 V from the return | warning |
| `return` | all | the other end is outside the chosen return's window; say which pin it looks like. In DC it is estimated with the informed R (R and return are confounded: say both causes). In SVMI DC with R × 60 µA > 75 mV: note "the return cannot be seen" | warning / note / ok |
| `drift` | dc | the drift of the measured quantity exceeds 10 × its uncertainty and 0.5 % of the excursion | warning / ok |
| `shunt_agreement` | hi-z `shunt-node` | the shunt current and the other channel's measured current differ by more than 60 µA + 1 % | warning / ok |

**Result** (on the terminal, styled and ordered for the screen as in §3.1, Hierarchy; and in
`summary.txt`, plain and complete): a header with the waveform, main parameters and circuits
("CH A source-i sine 100 Hz → R 10 Ω → 2.5V · CH B hi-z"); duration and folder; requested × measured
(periodic: amplitude and level of the imposed quantity, amplitude and level of the other against the
expected, real frequency, delay, phase between V and I, impedance seen, residual; DC: level ± u of the
imposed quantity, what the DAC delivers, the other quantity against the expected, σ, drift); one
paragraph on how each number was obtained; the before- and after-checks; the files; the commands to
plot it and to repeat it (§3.1).

**Plot** (on request, in a separate window): the imposed and the other quantity in time (opening on
the first 5 periods; the rest by zooming), the requested waveform dashed and the DAC-delivered one
dotted, the DAC staircase when there are fewer than 200 levels, the relevant limit in red when the
curve gets close; the cycle average with a ±1σ band. DC: block means over time with the target, and a
histogram of the samples (bins aligned to the ADC step) with mean and ±1σ. Every axis shows full
values, never matplotlib's offset notation (a "+1" in a corner would make 0.9999868 V look like a value
below zero). The window is matplotlib on the Qt backend (PySide6), run as a separate process (`python -m phycovolt.plot <folder>`), so the terminal is
free at once; the operator saves the figure from the window if wanted. Nothing is saved automatically.

### 3.4 Recordings

Every measurement that ends normally, or that the operator stops, creates
`recordings/<YYYY-MM-DD_HHMMSS>_<role>-<waveform>/` (with `_2`, `_3` … in the same second). A read
that fails (lost sample, USB) creates nothing. Contents:

- `samples.csv.gz`: every sample of both channels, gzip-compressed CSV (readable with any tool, and
  with Python's `gzip` and `csv` modules). Columns:
  `sample, time_s, a_requested, a_dac, a_v, a_i, b_requested, b_dac, b_v, b_i`
  (V and A; `requested` and `dac` empty for a `hi-z` channel; the requested and DAC-predicted values
  repeated cyclically).
- `metadata.json`: program version (and git commit, if available); date and time; simulated, or serial,
  firmware and hardware revision; the 8 calibration records read from the EEPROM at that moment; both
  channels' configuration (role, waveform and parameters, circuit, return, scale factor); the rest
  reading; the command line that produced it, written out in full (§3.1); how the measurement ended
  (complete, stopped by the operator); every check (code, level,
  numbers with units); every analysis result (numbers with units and uncertainties); the file list.
- `summary.txt`: the result as plain text, no styles, at most 100 columns (a command to copy is never
  cut, so it may be longer).

A recording MUST be analyzable again from `samples.csv.gz` and `metadata.json` alone.

### 3.5 Analyze a recording

Choose a recording folder → show its summary → run the standard analysis again → open the plot. This is
where later methods (a filter, if roadmap 0.2 adopts one) are applied and compared with the standard
analysis.

### 3.6 The standard analysis

- **Reading summary** of one quantity (C.4): discard the settling (0.5 s for calibration readings; the
  first period or block for signals); 100 ms block means; value = mean of the block means; spread =
  standard deviation of the block means; **uncertainty of the value: method B** (decided, §8); σ per sample; jumps; drift; correlation between neighbouring blocks (reported); fraction of
  samples at the ADC limits.
- **Periodic signals** (C.7): the **cycle average** (mean and standard deviation, phase by phase, over
  all periods except the first); level, peak-to-peak and rms of the cycle average; its rms and maximum
  difference from the requested and from the DAC-delivered waveform. For a sine, also the **regression**
  `c + a·sin + b·cos` (amplitude, phase, delay, residual rms).
- **Two quantities:** the impedance seen at the stimulus frequency (sine): ratio of the V and I
  amplitudes (if the I amplitude ≥ 20 µA) and the phase between them; for `shunt` and `shunt-node`,
  the current through the shunt.
- The summary MUST say that the ± shown is statistical only; whether to add the calibration's
  uncertainty is open (§8).

### 3.7 Calibration

**Purpose:** measure the M1K's errors against the reference multimeter, produce the `calib.txt` the
operator writes into the EEPROM, state the uncertainty of every value, and verify at a point outside
the computation.

**Before starting:** firmware ≥ 2.06; no implausible stored record (gain outside 0.9–1.1, or offset
above 50 mV for voltage or 5 mA for current; in that case guide the operator: keep a copy with `smu -d`,
reset with `smu -r`, reconnect); the board check; the multimeter connected and identified (`*IDN?`), or
typed mode chosen; the operator confirms that the M1K has been on for 15 min and the multimeter for 1 h
(H-11).

**Raw values (I-09).** Calibration MUST work with the EEPROM already calibrated, without asking for
`smu -r`: every measured sample has the stored correction undone (C.2), and every source request is
pre-distorted so that libsmu sends the DAC exactly the desired raw value (C.3).

**Four stages**, each calibrating one function on both channels:

| Stage | Point | Wiring | Mode and drive | Read | Role | Duration |
|---|---|---|---|---|---|---|
| measure-v | zero | CH A and CH B to GND (jumpers) | HI_Z | M1K (V) | fit | 5 s |
| | reference | CH A, CH B and the 2.5V pin on the multimeter's voltage terminal | HI_Z | M1K and multimeter (DC V) | fit | 10 s |
| | verification | the same wires on the 5V pin | HI_Z | both | verification, optional | 10 s |
| source-v (per channel) | 0 V | multimeter DC V on the channel's output | SVMI 0 V | multimeter | fit | 10 s |
| | 2.5 V | same | SVMI 2.5 V | multimeter | fit | 10 s |
| | 4.5 V | same | SVMI 4.5 V **through the new record** | multimeter | verification, optional | 10 s |
| measure-i | zero at 2.5 V | channels open | SVMI 2.5 V | M1K (I) | fit | 5 s |
| | zeros at 0 V and 4.9 V | channels open | SVMI 0 V / 4.9 V | M1K (I) | auxiliary | 5 s each |
| (per channel) | load probe | channel → resistor (2.5–25 Ω) → multimeter current terminal → 2.5V pin | SVMI 3 V and 2 V | M1K (I) | probe | 2 s |
| | ±100 mA | same | SVMI at the voltage giving the current (from the probe) | M1K and multimeter | fit | 10 s |
| | ±10 mA | same | same | both | verification, optional | 10 s |
| source-i (per channel) | load probe | same as measure-i | SVMI 3 V and 2 V | M1K (I) | probe | 2 s |
| | 0, +100, −100 mA | same | SIMV | multimeter (the M1K reads V) | fit | 10 s |
| | ±10 mA | same | SIMV **through the new record** | multimeter | verification, optional | 10 s |

Optional points MAY be skipped (and the following optional points of the same phase). The current of
the "±100 mA" points is `min(100 mA, (4.9 − pin)/R, (pin − 0.1)/R)`, rounded to 0.1 mA, with R and the
pin's real voltage from the probe.

**Reading the multimeter (automatic, RS-232).** The multimeter is read continuously during the
point's acquisition, on a fixed range chosen by the profile's rule (autorange off), every reading
time-stamped. The point's reference value is the mean of the multimeter readings taken after the
settling; the M1K value is the mean of the block means over the same interval. Both instruments
therefore see the same drift (a resistor that heats changes the current within seconds). One reading per point; it is redone if invalid.

**Typed fallback.** If no link is available: the terminal points out the 40–60 % window of the
acquisition; the operator types the value ("2.44570", "2,4457 V", "2445.70 mV", "63.930" in mA for
current, "-0,393 mA"); the number of typed digits sets the resolution (u = last digit ÷ √12). The
terminal says how many decimals the multimeter shows on that range at full resolution (6½ digits on the
34401A: for example 5 decimals on the 10 V range) and asks to type them all, and it warns when fewer
were typed (a coarser resolution weighs in the budget). Three
readings per point; one that deviates from the median of the three by more than
max(3·√2·u, margin) is redone (catches typing errors).

**What invalidates a reading** (it is redone whole; pieces are never discarded): a jump; more than 1 %
of samples at the ADC limits; overcurrent; a failed physics check of the point; a multimeter
communication error. Drift and noise above the margin (30 µV + 20 ppm for voltage; 5 µA + 200 ppm for
current) give a **warning**, and the operator decides. A V/I swap stops the stage (the board must be
re-plugged).

**Physics checks of each point:** a channel on GND reads less than 20 mV; the M1K and the multimeter on
the same node (or the same current) agree within 5 % + 20 mV (V) or 5 % + 2 mA (I), which catches the multimeter on the wrong channel or terminal; the current
has the right sign and is within 40 % of the expected; the current source does not saturate (channel
between 0.1 and 4.9 V) and the multimeter reads within 5 % + 2 mA of the request; a 0 V request shows no
more than 20 mV on the multimeter, and less than 0.3 mV gives a warning (a negative offset would be cut
by code 0); the probe refuses a load outside 2–40 Ω, a pin outside 2.2–2.7 V and a current below 1 mA
(open circuit).

**Before the load probe**, the multimeter is set to DC current on the range of the stage's fit points
(automatic mode; in typed mode the program tells the operator to set it). The probe then measures the
same loop, multimeter shunt included, that the fit points will use, and the loop is never open
because the multimeter was left in another function.

**Zero-current slope (measure-i).** `k = (z₄.₉ − z₀) ÷ 4.9`; the M1K readings at the ±100 mA points are
corrected by `−k·(V − 2.5)` before becoming pairs, with the uncertainties of z₀ and z₄.₉ propagated
(H-07). If the auxiliary zeros are skipped, the report warns.

**Pairs and the record.** Measure: `<reference, M1K>` (the multimeter, or 0 at the zero; the M1K's raw
mean). Source: `<request, multimeter>`, with the request equal to the **value of the DAC code**
(truncated, I-11). Only "fit" pairs go into the file (I-10); the record is computed exactly as libsmu
does (C.1), from the formatted text of the pairs, in float32.

**Uncertainty** (C.6): numerical propagation through the same code that computes the record; every
multimeter range is two named error sources (range gain and range zero), so that correlations are
handled by themselves; plus the multimeter's reading spread, the M1K's block statistics (method open,
§8) and, in typed mode, the typed resolution. Expanded uncertainty U = 2·u.

**Verification:** the residual (`corrected M1K − multimeter` for measure; `multimeter − request` for
source, minus the predicted truncation) passes if it fits in its expanded uncertainty plus one M1K step.
A failed source-V verification at 4.5 V is reported with its expected cause, the known non-linearity
near 5 V (H-06), and compared with the sizes recorded there; it stays a failure (the linear record does
not hold up there).

**Report** (on the terminal, styled as in §3.1; and in `report.txt`, plain), per stage: the `calib.txt` lines (no borders, left-aligned, ready
to copy); the predicted coefficients (float32); the effect table (error without calibration or with the
stored one, against the error with the new one, as "predicted error ± U", value by value — measure-v:
0, 0.1, 1 V, the reference, 5 V; source-v: 0, 0.1, 1, 2.5, 4.5, 5 V; measure-i: ±200, ±100, ±10,
±0.1 mA and 0; source-i: ±150, ±100, ±10, ±1 mA and 0); the verifications (residual × limit); the weight
of each uncertainty source; the stage's notes (noise per sample, step, DAC floor, DAC ceiling (H-16),
zero slope). Plot on
request: error before and after; what remains with ±U; verification; uncertainty budget.

**Files:** `calibrations/<YYYY-MM-DD_HHMMSS>_<stage or "all">/` with `calib.txt` (the 8 blocks: complete
when all four stages ran, or the new blocks merged into `--base`, default `./calib.txt`; the base is
never overwritten), `blocks.txt` (only this run's blocks, with a header saying it is not for `smu -w`,
which needs all 8), `readings.csv` (B.2), `dmm.csv` (every multimeter reading with its time stamp),
`report.txt`, `metadata.json`. Every generated `calib.txt` MUST pass the reader that imitates libsmu
(B.1) before it is written to disk. A calibration that stops early or is interrupted (Ctrl+C, SIGTERM,
SIGHUP) still writes what finished: the completed stages' blocks and results and every reading and
multimeter reading taken so far, in a folder whose name and `metadata.json` say it was interrupted.

**The program never writes the EEPROM** (I-08). At the end it shows the exact `smu -w <path>` command for
the operator, and how to check the result with `smu -d`.

**Start of a stage:** the material needed and the wiring, using the multimeter's terminal names
from its profile.

### 3.8 The multimeter

- One small interface: identify, configure (function and fixed range), read (value and time stamp).
  Implementations in v0.1: **SCPI over RS-232** (pyserial) for the 34401A; **typed entry**; a **virtual
  multimeter** in the simulator, whose errors are drawn within the profile's specification.
- The SCPI transport is isolated, so that SCPI over USBTMC (another multimeter, or the oscilloscope) is
  a small addition.
- **Profile** (INI file, B.3): name, specification source, accuracy table per function and range,
  distribution, terminal names, connection settings, the SCPI commands used. A template with `FILL_IN`
  fields is refused until filled.
- 34401A notes (to confirm against its user's guide at the link's first bench use, roadmap 0.2): RS-232 needs remote
  mode (`SYST:REM`) before measurement commands; the instrument is a DTE, so a null-modem cable is
  needed; DTR/DSR handshake; baud rate and parity as set in its I/O menu.

### 3.9 Safety behavior

- **Ctrl+C (the Stop), terminal closed (SIGHUP), SIGTERM:** both channels go to HI_Z, at the latest at the
  end of the current 100 ms chunk; what was already recorded stays; the program exits with code 130 and
  says where the results are. Python's default Ctrl+C handler is restored after the session opens
  (pysmu replaces it, P-02).
- **Communication failure during a read** (lost sample, USB): it does not crash the session; nothing of
  that read is recorded; calibration redoes the reading; signals report it and leave both channels
  in HI_Z (the operator runs the command again).
- **Program death** (kill -9, a crash inside libsmu): no code of ours runs. The target state is every
  channel in HI_Z at once. Not covered in v0.1; roadmap 0.4 adds a watchdog process and a bench
  measurement of what the M1K does on its own (§6.3).

---

## 4. Scope and roadmap

### 4.1 v0.1 (this document)

Calibration (multimeter over RS-232, typed fallback); per-channel signals with the parametric
waveforms and the circuits of §3.3.2 (including `ivi-board` with a scale factor); recordings and the
standard analysis; plots on request; the rich command line (§3.1); the simulator; the tests and the
Monte Carlo validation; the documentation.

v0.1 is for the bench and for the building blocks of the application. **No culture is connected
before the operator has validated the program on the bench.**

### 4.2 After v0.1 (sketch)

The order: calibration, then filtering of the measured signals (if the bench shows a need), then
impedance from the load's voltage and the current.

| Version | Content | Depends on |
|---|---|---|
| 0.2 | **Calibration over RS-232 on the bench**: the automatic reading of the 34401A (coded in v0.1, tested only against a fake) checked against the typed mode, with the fixes the bench shows. **Filtering, if needed**: the standard analysis already filters (a 100 ms block mean cancels 60 Hz and its harmonics; the sine regression is a synchronous detection at the stimulus frequency; the cycle average keeps what repeats every period and averages down the rest). Bench evidence decides whether a filter (Butterworth or another) adds anything: the spectrum of the regression's residual, repeated identical bursts, and the oscilloscope for content above 50 kHz, which folds into the band when sampled and only an analog filter before the ADC removes. A filter that is adopted goes in "Analyze a recording" and is compared with the standard analysis on the same recordings. **The open statistical decisions of §8** are settled here | v0.1 recordings and bench acceptance (§9.3); the RS-232 cable and a USB adapter; the statistical decisions of §8 |
| 0.3 | **Impedance**: the i-v-i board's scale measured (gain, offset, sign), the current at the culture measured by a `hi-z` channel over a shunt, impedance (modulus and phase) at several frequencies; the board's frequency response with the oscilloscope over USB | the board; measurements on it |
| 0.4 | **Long stimulation** by chained bursts (each ≤ 10 s, DAC parked between them), or by longer bursts if the burst-length test (§6.3 item 13) shows them stable; logging by per-block summaries (raw samples only in snapshots: 100 kS/s × 4 values ≈ 1.6 MB/s ≈ 6 GB per hour), a watchdog process, an on-screen monitor. Bench measurements with the oscilloscope: switch-on transient, gap between bursts, program death | the burst-length test; those bench measurements; the research protocol; the research computer |
| later | **own USB driver** (pyusb; about 16 control requests and two bulk endpoints, 0x81 in and 0x02 out) and newer Python; arbitrary waveform from a CSV file; stimulation protocols (schedules); live numbers during a burst; documentation in Portuguese | — |

If short pauses of zero current are not acceptable for the research protocol, the own driver comes
before long stimulation (0.4).

### 4.3 How v0.1 paves the way (MUST)

1. **Acquisition is a stream of 100 ms chunks** (§7.2). A burst takes N chunks; a long run (0.4) takes
   chunks until stopped. The same loop serves both.
2. **Two recording levels** are designed from the start: raw samples, and per-block summaries (mean,
   min, max per 100 ms) that long runs will use.
3. **One small `Instrument` interface**, implemented by the libsmu driver and by the simulator; an own
   driver is a third implementation.
4. **A waveform is one function** returning one period; **a circuit is one small description** used by
   the checks and the simulator. A new waveform or circuit = one function + its checks + a test.
5. The code runs on Python 3.10 **and** on a newer Python; the test suite runs on both (the second
   without pysmu).
6. Instrument transports (serial now, USBTMC later) are isolated.

### 4.4 Out of scope for v0.1

Long stimulation; Butterworth filtering; the board's model and frequency response; impedance over
time; a calibrated SIMV current zero with the board connected (H-08); arbitrary CSV waveforms;
oscilloscope automation; an own driver.

---

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| NF-01 | **Experiment safety:** every call sequence to the instrument follows §6.1; a new sequence does not go to the instrument without the operator's approval. |
| NF-02 | **Testable without the instrument:** all code, including the command line, is exercised by tests against the simulator and a fake pysmu. |
| NF-03 | **Suite time:** under ~2 min on an ordinary machine, Monte Carlo included. Anything slower is marked and run on demand. |
| NF-04 | **Environments:** the lab environment (Python 3.10, libsmu 1.0.3 from conda-forge, numpy, scipy, matplotlib with Qt (PySide6), rich, pyserial; exact versions pinned in `environment.yml`); a development environment with a newer Python and without pysmu, where the whole suite also runs. |
| NF-05 | **Python 3.10 compatibility:** no 3.11+ syntax or library (no `tomllib`, `typing.Self`, `StrEnum`, exception groups). Configuration files use INI (`configparser`) or JSON. |
| NF-06 | **Installation:** runs from the project root without installing (`python -m phycovolt`) and as a package (`pip install -e . --no-deps`, command `phycovolt`). Version 0.1.0. |
| NF-07 | **Language:** English everywhere in the software. |
| NF-08 | **Readability:** every module starts with a docstring saying the question it answers and its main choices; type hints on every public function; no operator text in domain code; no math in the code that prints; no threshold outside `checks.py`; no instrument number outside `hardware.py` and `unit.ini`. |
| NF-09 | **Run-time performance:** the progress line is redrawn at each 100 ms chunk without delaying acquisition; analyzing 1 000 000 samples takes a few seconds; the command line starts fast (matplotlib is imported only by the plot process). |
| NF-10 | **Record keeping:** no reading or sample is discarded silently; every automatic decision (a reading redone, a point discarded) is in the files with its reason. |
| NF-11 | **No network access.** |
| NF-12 | **Documentation:** one README for the user (installation, safety rules, use, how to read the output, the checks), `CHANGELOG.md`, this specification (the methods, with the reasoning and the numbers, and the instrument) and `CLAUDE.md`, all describing the delivered program. Few files, kept short. |
| NF-13 | **Portability:** v0.1 is developed and accepted on Linux (the operator's laptop). Outside the instrument transports and the installation notes, the code MUST NOT assume Linux: paths with `pathlib`, no shell commands, signals that do not exist on Windows (SIGHUP) handled only where available, so that installing it on the research computer stays possible if that computer runs Windows (§11). |

---

## 6. What is known about the instrument

Learned between 2026-11-10 and 2026-11-20: manual calibrations with pysmu and the 34401A, a complete
calibration of the 8 functions on 2026-11-18, sine and DC measurements on resistors, the diagnosis of a
fault, and libsmu 1.0.3's source code read wherever the instrument did not behave as documented.

Evidence tags: **[source]** read in libsmu 1.0.3's source (`src/device_m1000.cpp`, `src/session.cpp`)
or pysmu's; **[board]** measured on the project's unit (with the date); **[ADI]** Analog Devices
documentation; **[issue]** reported in libsmu's repository; **[data]** computed from recorded data;
**[inferred]** deduction not yet confirmed.

### 6.1 Operating rules (non-negotiable)

| # | The program MUST | Because |
|---|---|---|
| I-01 | write the constant value or the waveform to the channel **before** setting the mode, and set the mode right before the read starts | changing the mode switches the channel on at once, at the last code the DAC received (L-05); this way the channel sits on the old code for only a few milliseconds (L-07) |
| I-02 | before every read in which a channel sources (SVMI or SIMV), do a short read (10 ms) with **both** channels **explicitly** in HI_Z ("park the DAC") | a HI_Z read makes libsmu put the DAC at code 26 214, which is 2.0 V in SVMI and, in SIMV, the source's uncalibrated zero: a fraction of a mA (about −0.40 mA on CH A and −0.66 mA on CH B of this unit; L-04); otherwise the channel would start at the previous measurement's code: after a voltage sine around 2.5 V, a SIMV would start at **+62.5 mA** for a few milliseconds, and back-to-back SVMI points (calibration, chained bursts) would start at the previous point's voltage. Parking makes the switch-on state always the one the `switch_on` check predicts (R1 review, 2026-11-20) |
| I-03 | request HI_Z explicitly in every read meant to be at rest (board check, rest reading, parking) | at the end of a read libsmu sets HI_Z but keeps the last requested mode, and restores it at the start of the next read (L-06) |
| I-04 | end every path (normal end, error, stop, Ctrl+C, signals) with both channels in HI_Z, and restore Python's default Ctrl+C handler after opening the session | pysmu installs a handler that kills the process at once, without switching the channels off (P-02) |
| I-05 | create the session with `ignore_dataflow=False` and read in 100 ms chunks with a timeout (2 s); a lost sample is an error | otherwise libsmu drops samples silently (L-15); and `get_samples()` blocks the program until the end, with no progress and no Ctrl+C (P-03) |
| I-06 | do a 100 ms read right after opening the session, before any manual `start`/`end` | libsmu 1.0.3 only sets `end()`'s timeout inside `run()` (L-10) |
| I-07 | use one M1K at a time, 100 000 samples/s, and finite bursts (≤ 10 s, this program's cap: §6.3 item 13) | with two instruments the start is scheduled and the I-01 window grows (L-07); below 100 kS/s libsmu hangs (L-12); 1.0.3's continuous mode hangs or drops samples (L-11) |
| I-08 | **never** write the EEPROM; that is the operator's job, with `smu -w` | safety; and the calibration utility that comes with pysmu writes without asking, and is wrong (P-04) |
| I-09 | calibrate with raw values: undo the stored correction in every measured sample and pre-distort every source request | libsmu applies the stored calibration to everything (L-02, L-03); without this the good calibration would have to be erased (`smu -r`) before knowing whether the new one is better |
| I-10 | put in `calib.txt` only the necessary pairs; extra points become verifications | `smu -w` averages ratios, it does not fit a line: a third point hides the non-linearity instead of correcting it (L-01) |
| I-11 | use, in the source model, the value each DAC code represents (truncated), not the requested value | the DAC truncates: 0 A becomes −3.8 µA (L-03); in a simulation validation of the calibration method, without this the predicted source-current uncertainty failed 1 time in 3 |
| I-12 | never compute the uncertainty of a reading as σ/√N over samples; work on 100 ms block means (method in §8) | neighbouring samples are correlated; σ/√N promises an uncertainty tens of times smaller than the real one (H-10); in a simulation with strong slow noise, ±2σ/√N covered the true value only 44 % of the time. Block means are correlated too on this board (H-14) |
| I-13 | check for the V/I swap at opening (2 s at rest) and in every read | F-01 appears without warning and contaminates measurements; only re-plugging the USB cable fixes it |
| I-14 | before switching a signal on, read the channels at rest and check where the return is; in SVMI, the return on the wrong pin blocks switching on | in SVMI the return decides the current: with 5V chosen and the wire on GND, 4.5 V on a 57 Ω load gives +79 mA instead of −9 mA, and 0.36 W in a resistor sized for 4 mW (H-04) |
| I-15 | generate periodic waveforms with an integer number of samples per period; a sine starts at its center | pysmu's ready-made sine truncates the period (3 kHz becomes 3030 Hz) and starts at the crest, with a step (P-01) |
| I-16 | stop at a prompt, with the channels in HI_Z, while the operator mounts or unmounts the circuit; switch an output on only after the operator confirms | mounting with the output on applies spikes to the circuit and the culture |
| I-17 | be entirely testable without the instrument (simulator with the real errors; fake library checking the call order) | nobody runs tests on the instrument without the operator |

### 6.2 What was observed

#### libsmu 1.0.3

- **L-01 — `smu -w` calibration is an average of ratios.** [source] For each block of the file,
  `offset = val₀ − ref₀` (the first pair is the zero) and each gain is the **mean** of
  `ref ÷ (val − offset)` over the pairs with positive reference (gain+) or reference ≤ 0 (gain−), in
  float32. The file has rigid reading rules (B.1): a line out of format is misread without warning.
  `smu -w` does not check that `<` is in the first column; every line
  inside a block goes to `sscanf("<%f, %f>")`, whose result is not checked, and still counts as a point,
  so a blank or comment line inside a block repeats the previous pair's values; and a line containing
  `</>` anywhere, even in a comment, opens a block. libsmu itself does not check the 2.06 firmware
  minimum for the calibration (it comes from ADI's documentation).
  → Reproduce the computation exactly (C.1); validate every `calib.txt` with the reader's rules before
  writing it; I-10.
- **L-02 — How each measured sample is corrected.** [source] Voltage: `(raw − offset) × gain+` (gain− is
  never used for voltage). Current: `(raw − offset) × gain±`, with the gain chosen by the sign of the
  **previous** sample, already corrected (0 at the start of each USB **transfer**: 1024 samples at
  100 kS/s on Linux, 2560 on Windows). The correction applies in
  every mode: calibrating measure-V in HI_Z corrects the voltage read in SVMI and SIMV. → The inversion
  (I-09) and the simulator imitate this rule (C.2).
- **L-03 — The DAC code is truncated.** [source] `code = trunc((request − offset) × gain ÷ 76.29 µV)` for
  voltage; `trunc(65536 × (0.4 + 1.6 × (request − offset) × gain±))` for current (gain by the sign of the
  request); the corrected request is first clamped to 0–5 V or ±0.2 A. Consequences: 0 A becomes code
  26 214 = **−3.8 µA**; every output is up to one step below the request; at µA the waveform is a
  staircase (20 µA p-p comes out in 3 levels). With the calibration stored today, requesting 50 µA on
  CH A delivers **47.52 µA**, and 100 µA delivers 94.81 µA. → Predict the output (C.3) and show it; I-11.
- **L-04 — A HI_Z read puts the DAC at code 26 214**, which is **2.0 V** in SVMI and, in SIMV, −3.8 µA
  as a raw value; what comes out is the source's own uncalibrated zero, about **−0.40 mA (CH A) and
  −0.66 mA (CH B)** on this unit (H-02). [source] It is the cause of libsmu issue **#188** ("the DAC jumps to ~2 V going from HI_Z to SVMI").
  After any rest read, the next SVMI starts at 2.0 V for a few milliseconds (L-05); on a culture, an
  unrequested stimulus. → Use the effect on purpose to park the DAC before every burst that sources (I-02); in SVMI,
  compute and show the current of that instant with the chosen return, and warn above 200 mA
  (`switch_on`).
- **L-05 — The DAC keeps its last code, and changing the mode switches the channel on at once.** [source]
  `channel.mode = …` drives the channel immediately, outside any read, with the DAC at the last code
  received; the DAC only receives a new code while samples flow. The same code is a voltage in SVMI and a
  current in SIMV (`I = (code ÷ 65536 − 0.4) ÷ 1.6`): after a voltage sine centered at 2.5 V (code
  32 768), a SIMV would start at **+62.5 mA**. There is no way to put a code in the DAC without samples
  flowing, and only HI_Z flows without driving the terminal. → I-01, I-02.
- **L-06 — `off()` keeps the last mode, and `on()` restores it.** [source] At the end of each
  non-continuous read libsmu puts the channels in HI_Z without forgetting the requested mode; at the next
  `start()` it restores it. A "rest read" made without requesting HI_Z drives the channel in the last
  mode. → I-03.
- **L-07 — With one M1K, sampling starts without scheduling.** [source] `Session::start` calls `on()`, then
  `sync()` only if there is more than one device, then `run()`. The L-05 window stays in milliseconds
  (not measured: §6.3). → I-07.
- **L-08 — The cyclic waveform.** [source] `write(data, cyclic=True)` stores a buffer that libsmu repeats
  while the read lasts; every write restarts from sample 0. The maximum size is the session queue (10 s
  with `queue_size` = 1 000 000). → Write one block (one period, or 100 ms of a level) per measurement.
- **L-09 — Overcurrent is read at the end, with no time stamp.** [source] The board's ADM1177 records it;
  libsmu reads it at the end of the read, and only firmware ≥ 2.07 reports it. A millisecond pulse at
  switch-on (L-04) can mark the whole measurement. → Check after (`overcurrent`) and warn before
  (`switch_on`).
- **L-10 — `end()`'s timeout only exists after a `run()`.** [source] Reading in chunks
  (`start`/`read`/`end`) in a new session, without a complete read first, can wait wrongly. → I-06.
- **L-11 — Continuous mode is not reliable.** [issue] #181 (`session.end()` hangs after continuous mode),
  #201 (lost samples in continuous write and read from Python); [source] `encode_out` busy-waits inside
  the libusb callback when the program falls behind. Every non-continuous read ends in HI_Z, so chaining
  short reads interrupts the stimulus between them. 1.0.4 fixes one continuous-mode hang but has no
  package. → Bursts only (I-07). Long stimulation needs chained bursts (with pauses), a watched
  continuous mode, compiling 1.0.4, or an own driver (§4.2).
- **L-12 — Below 100 kS/s, `run()` hangs** with more than 256–512 samples. [issue] #193. → Always
  100 kS/s.
- **L-13 — Without the udev rule, segfault.** [board, 11/11] and [source]: `libusb_open` fails with
  `EACCES`, `Session::probe_device` ignores the error and uses a null pointer, and `smu -l` and
  `pysmu.Session()` crash with SIGSEGV (issue #209). → Document the rule's installation (S-02).
- **L-14 — libsmu is frozen.** No change since 1.0.4 (February 2022); conda-forge only has 1.0.3
  (November 2021), for Python ≤ 3.10. → Isolate the dependency in `m1k.py` and `libsmu_math.py`.
- **L-15 — `ignore_dataflow=True` hides lost samples.** [source] → I-05.

#### pysmu and the utilities

- **P-01 — `Channel.sine` truncates the period and starts at the crest.** [source] It writes
  `int(rate ÷ f)` samples: at 3 kHz, 33, i.e. **3030 Hz** (+1 %); a "requested" curve computed with the
  fractional period ends in phase opposition with the real one after ~50 periods; and the first sample
  is the crest (a cosine): a step from rest to maximum. → I-15; do not use it.
- **P-02 — pysmu replaces the Ctrl+C handler** with one that ends the process at once (in the
  interactive terminal, Ctrl+C closed the whole session). [board, 11/11] → I-04.
- **P-03 — `get_samples()` blocks Python** until the end, in chunks of the queue size (default
  100 000 = 1 s), and ends with `off()`. [source] → Only for short rest reads; everything else with
  `start` and chunked reads.
- **P-04 — The package's `pysmu` command is unusable.** [source] It is Python 2 code (dies at
  `from urllib import urlretrieve`), and its calibration (`pysmu cal -R`) has serious errors: the 2.5 V
  pair of measure-V inverted, the measure-I zero reading voltage, CH B's voltage search driving CH A,
  source pairs with the measured value in place of the request, rounding to 4 decimals, continuous mode,
  and writing the EEPROM **without asking**. → Do not use it; use `smu`.
- **P-05 — `smu -d` shows 4 decimals; the EEPROM stores float32.** [board, 13/11] A gain of 1.02385 shows
  as 1.0238. `smu -w` and pysmu's `write_calibration()` call the same function. → Put the predicted
  float32 coefficients in each block's comment in `calib.txt`, for the operator to compare with
  `smu -d`.
- **P-06 — Pixelpulse2 does not calibrate; ALICE applies its own gains outside the EEPROM.** [source]
  Pixelpulse2 only sets mode and value and shows the buffer's mean with 4 decimals. ALICE has an
  "auto-calibration" (with an AD584 reference) that applies gains in software, invisible to pysmu.
  → PhycoVolt's calibration is the EEPROM's, which applies to every program.

#### The project's unit (rev D, firmware 2.17)

- **H-01 — It came without factory calibration.** [board, 11/11] The EEPROM was at the default (offset 0,
  gain 1). → Do not assume a calibrated board; warn when the EEPROM is at the default.
- **H-02 — Errors without calibration.** [board, 11 and 13/11, against the 34401A] Measure-V: gain ~2.3 %
  low and offset +0.55 mV (the 2.5V pin, 2.4457 V on the multimeter, read 2.3901 V by the M1K);
  measure-I: zero of 0.53 mA with the channel open, gain ~2 % low; source-V: +2.6 mV (CH A) and +1.1 mV
  (CH B) when asked for 0 V, gain ~0.2 %; source-I: −0.39 mA (CH A) when asked for 0, gain ~0.9 %. At µA
  none of this is acceptable uncalibrated.
- **H-03 — The calibration stored today** (2026-11-18, `calib.txt`; `smu -d` confirmed the 24
  coefficients): appendix A.3. The measure-V gains were within ~30 ppm of those of 13/11: the board is
  stable from day to day in this function. [board, 21/11] Measure-V and source-V were calibrated again
  with PhycoVolt (typed mode, both instruments warm): the new records were within 20–30 ppm (gain) and
  20 µV (offset) of the stored ones, and the stored records' errors stayed inside the new U at every
  point. The stored calibration still holds; the EEPROM was not rewritten.
- **H-04 — The fixed pins are not at their nominal values.** [board] The 2.5V pin measures **2.4457 V**
  (11 and 13/11) and 2.4451 V (18/11): stable, but 54 mV low. The 5V pin is the USB supply reaching the
  board: **~4.93 V** here; by the USB specification, 4.75 to 5.25 V at the port, minus the drops in the
  cable and the board, and noisier than 2.5V. Computations with nominal values err: 54 mV over ~15 Ω is
  3.6 mA at a calibration current point; in SVMI with the 2.5V return, a sine centered at 2.5 V leaves
  ~0.95 mA of mean current in a 57 Ω circuit. → Measure the pin when it matters (the calibration load
  probe; the signals' rest reading) and use the measured value; wide windows in the return checks; a
  compliance margin on 5V.
- **H-05 — The voltage source does not reach 0 V.** [board] Asking for 0 V puts the DAC at code 0, and
  what comes out is the output's positive offset: **+2.752 mV (CH A) and +1.327 mV (CH B)** on 18/11.
  Calibration cannot correct it (it would need a negative code). If 0 V is the "off" of a circuit (the
  i-v-i board's input, for example), count on these mV. → "DAC floor" note in calibration; `dac_floor`
  warning in signals.
- **H-06 — The voltage source leaves the line near 5 V.** [board, 13 and 18/11] At 4.5 V, already
  calibrated, the output was **0.36 mV (CH A) and 1.37 mV (CH B)** below the 0–2.5 V line (limits
  ±0.2 mV); on 13/11 CH B's 5 V point was 1.2 mV below. A hardware non-linearity (~0.03 %) that two
  calibration points cannot correct. [board, 21/11] Reproduced: **−0.29 mV (CH A) and −1.37 mV (CH B)**
  at 4.5 V, CH B exactly as on 18/11. → The 4.5 V verification flags it and shows its size, and the
  report names H-06 as the expected cause when it fails.
- **H-16 — The voltage source does not reach 5 V.** [board and source, 21/11] The corrected request is
  clamped at the top code, so asking for 5.0 V delivers about **4.9936 V (CH A) and 4.9928 V (CH B)**
  with the stored calibration: a "DAC ceiling", symmetric to the floor of H-05. → "DAC ceiling" note in
  calibration; `dac_ceiling` warning in signals.
- **H-07 — The measured-current zero changes with the channel's voltage, and the slope is not stable.**
  [board] With the channel open in SVMI, the current read (which should be zero) depends on the voltage:
  **−18 µA/V** on 11/11 (CH A: 0.529 mA at 0 V; 0.467 mA at 3.4 V); **+6.7 µA/V (CH A) and +22.5 µA/V
  (CH B)** on 18/11 (zeros at 0 V and 4.9 V). The sign changed between the two days: a fixed value
  cannot serve as a correction. In a current sine with a large R, the voltage follows the current, and
  the slope alters the measured amplitude (≈ k × R: about 10 % with 10 kΩ). → Measure the slope in every
  current calibration and remove it from the gain points; never hard-code a value (the simulator uses
  one only to rehearse).
- **H-08 — The measured-current zero changes with the mode and the condition.** [board] On 13/11, SIMV
  asking for 0 A, with the channel at 2.443 V through 10 Ω and the 34401A: the multimeter showed
  0.005 ± 0.005 mA and the M1K read **0.0282 mA** (+23 µA), against 1.5 µA with the channel open in SVMI
  at 0 V; the output was right, the measurement was wrong. On 18/11, the ±10 mA verifications, done after
  the ±100 mA points, read **−54 and −63 µA (CH A) and −38 and −61 µA (CH B)** below the multimeter, in
  both polarities (limits ±22 to ±34 µA). Probable explanation: the zero measured with the channel open
  and cold, the verifications with it loaded and warm [inferred]. → At µA, the current the M1K measures
  is not the truth: 60 µA + 1 % tolerance on current levels in signals, and the advice to measure the
  voltage across a precision resistor. The SIMV zero with the i-v-i board connected (the application's
  condition) is for later.
- **H-09 — The current measured in SIMV seemed to use another gain.** [board, 13/11, one measurement]
  SIMV asking for 50 mA through 10 Ω and the 34401A to 2.5V: the multimeter measured 49.987 mA (the
  source was right) and the M1K read **50.10 mA** (+0.23 %); the implied gain (1.0172) matches the
  negative pairs of measure-I in SVMI, not the positive one (1.0195) [inferred]. → Record it; do not use
  it without confirmation.
- **H-10 — Noise, hum and resolution.** [board, 18/11] Noise per sample: voltage **σ ≈ 64 µV (CH A) and
  41 µV (CH B)**; current **σ ≈ 57 µA**; plus the 60 Hz hum. Steps: ADC 76.3 µV and 7.63 µA; DAC 76.3 µV
  and 9.54 µA. A single sample cannot see µA. → 100 ms block means; synchronous detection for sines (the
  amplitude's deviation is σ·√(2/N): ~0.26 µA in 1 s); I-12. [board, 21/11, both channels at rest in
  HI_Z, cold and warm] Confirmed on GND: **64 µV (CH A), 42–43 µV (CH B)**, current 57–58 µA; on the
  2.5V pin ~165 µV (the pin's own noise adds). **No measurable 60 Hz hum** on GND or the pins (≤ 2 µV),
  with the laptop on its charger; open inputs pick up ~1 mV.
- **H-11 — Warm-up.** [bench practice] The M1K drifts in its first minutes on USB, and a calibration
  holds at the temperature it was made. [board, 21/11] Cold, the M1K read the 2.5V pin 1.4–1.7 mV below
  the 34401A; warm, read at the same moment, they agreed within 0.05 mV. The pin itself moves by a few
  tenths of a mV over tens of minutes, so only simultaneous readings compare. → Ask for 15 min of the
  instrument on (and the multimeter warm: 1 h for the 34401A) before calibrating.
- **H-12 — The HI_Z input.** [ADI] 1 MΩ. Open, it does not read 0 V. [board, 21/11] It floats between
  **0.25 and 0.35 V** (0.25 V soon after power-up, 0.35 V an hour later), drifting up by tenths of a mV
  per second, with ~1 mV of noise picked up from the room. With the circuit at rest, the terminal reads the return's voltage divided by
  the input: 57 Ω gives a 0.01 % error; 50 kΩ, 5 %. → The `wiring` check only judges the return up to
  50 kΩ and undoes the divider when using the value read.
- **H-13 — Limits.** [ADI] ±200 mA per channel (the limiter acts and the ADM1177 records it); 200 mA per
  fixed pin. → Check before and after.
- **H-14 — Block means are strongly correlated on this board.** [data, 2026-11-20, from the 151 accepted
  real readings of the 18/11 calibration] The correlation between neighbouring 100 ms block means had a
  median of **0.75** (quartiles 0.59–0.94); with independent blocks it would be near 0. Dividing the
  spread of the block means by √n would understate the uncertainty of the mean by about 2.5–3 times
  (about 6 times in source-I, where the resistor heated). [board, 21/11, rest at GND] The correlation
  was 0.54–0.81, and still 0.46–0.75 after removing each burst's linear trend: slow noise, not only
  drift (the spread of 100 ms block means is ~2 µV). Over four groups of three 10 s bursts, the real
  spread of the bursts' means against each option of §8: option A promised ~6 times less uncertainty
  than there was, C ~2 times less, and B was conservative. → §8.
- **H-15 — The multimeter dominates the calibration's uncertainty.** [data, 18/11 report] In three of the
  four stages, the multimeter's specification and reading spread were ~99 % of the uncertainty's
  variance; in measure-I, drift (resistor heating) dominated. → Simple statistics on the M1K side cost
  little; reading the multimeter at the same time as the M1K cancels drift.

#### Faults

- **F-01 — The V/I swap.** [board, 18/11] CH B entered a state in which, **4 to 5 times per second**, at
  irregular intervals, a single sample carried the voltage in place of the current and vice versa. With
  the inputs on GND, at rest: CH B's voltage read **+2.444 V** (the current-zero code, on the voltage
  scale) and its current **−0.2438 A** (the 0 V code, on the current scale). In the same state, CH B's
  current zero was 0.238 mA instead of the usual 0.459 mA, and its voltage noise ~14 mV against 62 µV on
  CH A. Unaware of this, the analysis blamed "a block jump, contact moved?", the wrong cause. Swapping
  the jumpers between A and B changed nothing; changing USB port fixed it, and going back to the old port
  did not bring it back: what fixed it was re-powering the board. The cause is unknown (a digital state of
  the ADC or the firmware) and the fault may return. The signature, in ADC codes: in the same sample, V
  and I each leave their own level and fall near the other's (C.5); when the two levels are close, the
  swap cannot be told apart (and changes almost nothing). The 18/11 runs of 13:46 and 13:56 were hit by
  it. → I-13; the message to the operator is "unplug the M1K's USB cable, wait a few seconds and plug it
  in again".
- **F-02 — A halved current word on CH B.** [board, 21/11] In 18 bursts of 10 s at rest, five single
  samples on CH B had their current ADC word exactly halved (~16 000 instead of ~32 000: one bit
  shifted right), with the voltage of the same sample normal; three while the instruments were cold,
  two warm; none on CH A; about one every 36 s of acquisition. Not the V/I swap (C.5 rightly does not
  count it). Each one moves a 100 ms current block mean by ~12 µA (about 0.1 µA over a 10 s mean). It is
  the same channel as F-01; the cause is unknown [inferred: a digital fault of the ADC readout].
  → Detect it on every channel and quantity, reported like the V/I swap (`halved_word`), and record
  whether it grows or appears on CH A (§6.3).

#### System and environment

- **S-01 — Python 3.10.** libsmu is only on conda-forge for Python ≤ 3.10 (L-14).
- **S-02 — udev rules.** The package ships them in `$CONDA_PREFIX/lib/udev/rules.d/` but does not activate
  them; without them, segfault (L-13). Copy `53-adi-m1k-usb.rules` to `/etc/udev/rules.d/`, run
  `udevadm control --reload-rules && udevadm trigger`, reconnect the M1K (access through the `plugdev`
  group; VID:PID `064b:784c`, and `03eb:6124` in firmware-update mode). Already done on the operator's laptop
  (2026-11-20).
- **S-03 — `smu` only exists with the environment activated** ("command not found" outside it).
- **S-04 — One program at a time on the M1K.** ALICE, Pixelpulse2, an interactive pysmu session and
  PhycoVolt compete for the instrument. → A message asking to close the others.
- **S-05 — No display** (SSH, tests): no plot window. → Plots only on request; tests use the Agg backend.
- **S-06 — The 34401A has no USB;** RS-232 or GPIB, one active at a time (I/O menu).
- **S-07 — The operator's laptop kernel has the `usbtmc` driver** (oscilloscope over USB). Talking to the
  device will probably need a permission rule, like the M1K's.

### 6.3 Open questions about the instrument

Not part of v0.1, but the program must not close the door on them, and anything observed about them is
recorded:

1. The measured-current zero in SIMV with the i-v-i board connected (H-08): the application's condition.
2. The stability of the zero's slope with the channel voltage (H-07).
3. The gain of the current measured in SIMV (H-09).
4. ~~The open-input voltage in HI_Z~~ — answered on 2026-11-21: 0.25–0.35 V, drifting (H-12).
5. How much the 5V pin drops at 200 mA.
6. How long the channel stays on the old code at switch-on (L-05, L-07), with the oscilloscope.
7. The frequency response (the calibration is static; AC and impedance need amplitude and phase
   characterized).
8. Whether the V/I swap (F-01) returns, on which channel and after what (not seen in 18 bursts and two
   calibration stages on 2026-11-21); and whether the halved word (F-02) grows, or appears on CH A.
9. **What the M1K does when the program dies with an output on** (kill -9, libsmu crash, USB unplugged,
   laptop suspended): does it keep the last mode and code? [inferred: probably yes].
10. **The gap between chained bursts** (duration, and the channel's state during it).
11. **The voltage range of the i-v-i board's nodes** (the board has its own supply) against the M1K's 0–5 V input, and
    the M1K's absolute maximum input voltage.
12. ~~The noise structure at rest~~ — answered on 2026-11-21: H-10 and H-14 (no hum; slow noise with a
    block correlation ~0.65 that survives detrending).
13. **How long one finite burst can last.** The 10 s cap (I-07, the `duration` check) is this
    program's: it equals libsmu's queue as the program opens it (1 000 000 samples), so that a whole
    burst fits in it. [source] libsmu 1.0.3 takes the sample count as a 64-bit integer in
    `Session::start` and `Session::run`, and pysmu passes it as `uint64_t`; reading in 100 ms chunks,
    the queue only absorbs reading delays. No burst longer than 10 s has been run. Before 0.4 is
    designed: bursts of 1 min, 10 min and 1 h, first in the simulator, then on the bench with a
    resistor and no culture, recording whether each completes, lost samples, the timing of the
    chunks, drift, the overcurrent flag and F-01/F-02. The test needs first a way to lift the cap,
    recording by per-block summaries (§4.3 item 2), the swap and lost-sample checks per chunk, and an
    answer for overcurrent, which libsmu reports only at the end of a read (L-09). If long bursts are
    stable, long stimulation needs only a few pauses per hour, and the own driver may not be needed.

---

## 7. Architecture

### 7.1 A flat package

```
phycovolt/
  __init__.py       the version
  __main__.py       entry point: the command line (subcommands, the start page and its examples, the
                    prompts, the progress line); the only file that prints, with the styles of §3.1
  hardware.py       the M1K's fixed numbers (appendix A.1): the ONLY place for them
  unit.ini          the project unit's measured values (appendix A.3); another unit = another file
  m1k.py            the real instrument through pysmu: the ONLY file that imports pysmu; rules I-01 to I-08
  libsmu_math.py    the arithmetic libsmu does with the calibration and the DAC (C.1–C.3), and
                    calib.txt (read, write, validate, merge)
  simulator.py      the simulated M1K, the simulated circuits and the virtual multimeter (C.8)
  waveforms.py      waveform parameters → one period of samples; the time base
  circuits.py       circuit descriptions and their physics: expected V and I, compliance, power
  checks.py         the Check type, every threshold (appendix A.2) with its reason, the before- and
                    after-checks, and the English text of every check code
  analysis.py       reading summary, cycle average, sine regression, V/I swap detection
  signals.py        the signals procedure: rest reading and before-checks, the burst, the analysis
                    and the after-checks
  units.py          reading and writing values with units and SI prefixes ("200 mA", "4,7 k")
  calibration.py    the calibration procedure (stages as data), pairs, record, uncertainty, verification
  multimeter.py     the multimeter interface: SCPI over RS-232, typed entry, profiles
  recording.py      recording and calibration folders and files; the plain-text summaries; reading back
  plot.py           the matplotlib figures, in a Qt window, run as a separate process
  profiles/         multimeter profiles (agilent_34401a.ini, template.ini)
tests/
docs/
```

Two boundaries are non-negotiable:

- **The instrument lives in one place** (`m1k.py`, with `libsmu_math.py`): for safety, and so that an own
  driver later replaces exactly these files.
- **No math where it prints**: `__main__.py` only asks, calls and shows; everything is testable
  without a terminal.

A new file needs a reason in the decision log.

### 7.2 The contracts

Names are suggestions; what each contract guarantees is not.

```python
class Role(Enum):
    HI_Z = "hi-z"
    SOURCE_V = "source-v"
    SOURCE_I = "source-i"

@dataclass(frozen=True)
class ChannelSetup:
    role: Role = Role.HI_Z
    block: np.ndarray | None = None   # one period, in V (SOURCE_V) or A (SOURCE_I), at 100 kS/s;
                                      # repeated cyclically; None for HI_Z

@dataclass(frozen=True)
class Chunk:                          # 100 ms = 10 000 samples
    index: int                        # chunk number since the start of the stream
    v_a: np.ndarray                   # V, as libsmu returns them (stored calibration applied)
    i_a: np.ndarray                   # A
    v_b: np.ndarray
    i_b: np.ndarray

class Stream(Protocol):
    def __iter__(self) -> Iterator[Chunk]: ...
    def stop(self) -> None: ...       # ends at the current chunk
    overcurrent: bool                 # valid after the stream ends

class Instrument(Protocol):            # implemented by m1k.py and simulator.py
    description: str                  # "serial … · fw 2.17 · hw D" or "simulated (no board) · fw 2.17 · hw D"
    firmware: str
    def calibration(self) -> list[Record]: ...                 # the 8 EEPROM records, in calib.txt order
    def rest_read(self, seconds: float) -> Chunk: ...          # both channels explicitly in HI_Z
    def stream(self, a: ChannelSetup, b: ChannelSetup,
               chunks: int) -> ContextManager[Stream]: ...     # v0.1: chunks ≤ 100 (10 s)
    def close(self) -> None: ...                               # both channels in HI_Z; device released
```

- `stream` parks the DAC when a channel sources (SVMI or SIMV), writes the blocks, sets the modes right before the
  read, and yields exactly `chunks` chunks. **Leaving the `with` block by any path** (end, `break`,
  `stop()`, exception, Ctrl+C) leaves both channels in HI_Z. A lost sample raises an error; it is never a
  silent gap.
- `Record = (offset, gain+, gain−)`; `Kind ∈ {measure V, measure I, source V, source I}`.
- `Check(level, code, numbers)`; the text comes from `checks.py`.
- Procedures receive the instrument and the multimeter as parameters; tests pass fakes.

---

## 8. Statistics: what is decided and what is open

**Decided:**

- every reading is summarized by **100 ms block means** (6 cycles of 60 Hz: the hum cancels in each mean);
- value = **mean** of the block means; noise = **standard deviation** per sample; spread = standard
  deviation of the block means;
- jumps are found against the **median** with a robust spread; drift is the slope of a **line** fitted to
  the block means against time;
- the **correlation** between neighbouring block means is computed and reported;
- offset and gain come from **lines** (libsmu's own computation, C.1); a sine's level, amplitude and phase
  from a **regression** on sine and cosine; any periodic waveform from its **cycle average**;
- independent error sources combine in **quadrature**, √(u₁² + u₂² + …);
- a multimeter specification limit L becomes u = L ÷ √3 (uniform) or L ÷ 2 (normal at 95 %), as the
  profile says;
- uncertainties reach the offset and the gain by **perturbation** (move one input by ±u, see how far the
  result moves; C.6);
- expanded uncertainty **U = 2·u** (about 95 % for a normal distribution);
- **the uncertainty of a reading's mean is method B** (decided 2026-11-21; it was open
  item 1): u = the standard deviation of the block means, **not** divided by √n. The code keeps it in
  one function (`analysis.uncertainty_of_mean`). The evidence, from the 2026-11-21 rest recordings
  (four groups of three identical 10 s bursts at GND, CH A and CH B, cold and warm):
  - the ratio of the real spread of the three means to what each method promised for one burst,
    as a geometric mean over the four groups: **A** (s ÷ √n) 6.2; **C** (below) 2.4; **B** 0.67;
  - with only three bursts per group, a method that is exact gives about 0.75, and 0.42–1.2 over
    four groups (by simulation). So B is right, while A and C promise 3 to 8 times too little;
  - the block correlation survives detrending (r₁ 0.46–0.75): the noise wanders, which the AR(1)
    model behind C does not capture. A simulation of AR(1) noise plus a slow wander reproduces the
    pattern (A ≈ 10, B ≈ 1, C ≈ 3);
  - the cost: in the 2026-11-21 measure-V calibration the M1K's part was 9–15 % of the variance at
    the main point, so B makes U 5–8 % larger than A would;
  - the limits: voltage at rest only. The first bench session with current repeats the test (three
    identical bursts) for current.

**Open — the operator decides:**

1. *(Decided on 2026-11-21: method B, above. The candidate it was weighed against, C, is a
   correlation-corrected method: an effective number of blocks from the lag-1
   correlation, with a bias correction; its formula stays in C.4 for reference.)*
2. Whether the ± of signal results includes the calibration's uncertainty.
3. The robust spread used for jumps: 1.4826 × MAD (provisional) or one based on the interquartile range.
4. The uncertainty of periodic results (amplitude, level, phase, impedance seen): shown without ± for
   now, with the residual rms.
5. Whether a filter (Butterworth or another) is applied before the standard analysis, which already
   averages in blocks, by cycles and by synchronous detection (§4.2, 0.2). v0.1 applies none.

**Monte Carlo validation (why and how).** The ± the program prints is a promise: the true value is inside
it about 95 % of the time. On the bench nobody knows the true value; in the simulator we choose it. The
validation repeats the simulated calibration many times, with new noise and new multimeter errors each
time, and counts how often the true error falls outside "predicted error ± U". (A validation like this
catches method errors such as a DAC truncation missing from the source model, the correlation bias
in short series, or a jump limit that is too tight.)

- 1000 simulated calibrations per stage, with a **fixed seed** (the test always gives the same result);
- simulated at the level of block means, with the noise model taken from real data (H-14 and the
  2026-11-21 recording), so that it fits the suite's time budget;
- for every value of the effect table, count the misses; with k = 2 an honest method misses about 45 of
  1000 (standard deviation ≈ 7, the normal approximation to the binomial);
- **fail above 65 misses** (the coverage is clearly too low; false-alarm risk about 0.1 %); **warn below
  25** (U exaggerated; expected for some M1K-dominated values under method B, and reported as such);
- no verification failed and no reading redone without a reason.

---

## 9. Tests and acceptance

### 9.1 Unit and integration tests (no instrument)

- **Fake pysmu** that only records calls; call-order tests for I-01 to I-07: SVMI, SIMV, both channels
  sourcing, waveform, constant value, rest read, an error in the middle, a stop in the middle.
- **libsmu_math** against hand-computed values: float32 coefficients; exact voltage inversion; current
  gain chosen by the previous sample; 0 A → code 26 214 → −3.8 µA; 50 µA → 47.52 µA on CH A with the
  stored calibration (L-03).
- **calib.txt**: round trip; refusal of whatever libsmu would misread; the project's `calib.txt` parses and
  reproduces its 24 coefficients.
- **V/I swap**: 20 injected swaps are counted; noise and jumps are not swaps.
- **No literals**: no instrument number outside `hardware.py`/`unit.ini`, no threshold outside `checks.py`
  (a test searches for a curated list of values such as 2.4457, 26214, 76.29e-6).
- **Statistics**: with white noise, the block method agrees with σ/√N; with slow (correlated) noise,
  σ/√N under-covers and the chosen method does not; jumps and a clipped ADC are caught; a noisy sine has
  amplitude, level and phase recovered within σ·√(2/N); the cycle average recovers a square wave's edges;
  the perturbation propagation equals the analytic formula on a linear model; the same error in two places
  cancels.
- **Multimeter**: the SCPI dialogue against a fake serial port; profile parsing; `FILL_IN` refused.
- **Command line**: every subcommand end to end with `--simulate`; every example of the start page and
  of the help is accepted by the parser; a recording's repeat command, parsed again, gives the same
  configuration; output that is not a terminal, and every file written, carries no style codes; Ctrl+C
  and SIGTERM during a burst leave both channels in HI_Z (code 130).
- The whole suite runs in the lab environment (Python 3.10) and in the development environment (newer
  Python, no pysmu).

### 9.2 Acceptance scenarios (simulator)

The "right" circuit is the one informed, simulated with 0.4 Ω of wires and the real pins (A.3).

**Signals:**

| # | Scenario | Expected |
|---|---|---|
| S1 | source-i sine, defaults, CH A, 2.5V | no warning; amplitude within 1 % |
| S2 | source-i sine, CH B, 5V (its defaults) | no warning |
| S3 | source-i dc, defaults | no warning; level within 60 µA + 1 % |
| S4 | source-i dc, 5V (−20 mA) | no warning |
| S5 | source-v sine, defaults (GND) | no warning; amplitude within 0.5 % |
| S6 | source-v sine, 2.5V, center 2.5 V | `charge_balance` = note, with the center that zeroes the mean (≈ 2.446 V) |
| S7 | source-v sine, 2.5V, with the suggested center | `charge_balance` = ok; mean current < 60 µA |
| S8 | source-v sine on 5V; source-v dc on 5V; source-v dc on 2.5V | no warning; negative current on 5V |
| S9 | source-i sine with the circuit wired to GND instead of 2.5V | `pin_limit` = error (after); `return` = warning; `wiring` = warning |
| S10 | source-v sine with the return wire on 2.5V and GND chosen | `wiring` = error (before) |
| S11 | source-i sine, jumpers adding 25 Ω, 100 mA p-p | `resistance` = warning |
| S12 | source-v sine into a short (5 Ω) | `overcurrent` = error |
| S13 | source-v dc, 5V chosen, wire on GND | `wiring` = error |
| S14 | source-v dc configured on CH B with the circuit on CH A | `wiring` = warning naming CH A; `resistance` (DC) = warning |
| S15 | source-i dc, R = 100 kΩ, 10 µA | `wiring` = note |
| S16 | source-i sine, 5V, CH B, V/I swap 50 times per second on CH B | `board` = error |
| S17 | source-i sine, R = 100 Ω, 200 mA p-p | `pin_limit` = error before switching on, with the possible range |
| S18 | source-i sine at 60 Hz; at 10 Hz | `mains` = warning; `mains` = note |
| S19 | defaults for every waveform × role × offered return | no error before switching on |
| S20 | change of waveform after a measurement | channels and circuits kept |
| S21 | USB failure during the rest reading and during the measurement | back to the checks; nothing recorded; the session continues |
| S22 | CH A source-v sine on load + shunt to 2.5V; CH B hi-z `shunt-node` | shunt current agrees with CH A's current; load voltage reported; no warning |
| S23 | square, triangle, pulse and biphasic defaults on CH A, 2.5V | no warning; the cycle average matches the DAC-delivered waveform |
| S24 | `ivi-board` with scale 0.01, sine 4 mA p-p | culture-side values = 0.01 × the M1K's; no warning |
| S25 | Stop pressed in the middle of a measurement | both channels in HI_Z within one chunk; recording marked "stopped by the operator" |
| S26 | measure-only: both channels hi-z, 2 s, CH A on a `shunt` circuit | nothing parked or sourced (call order); a recording with both channels' summaries and CH A's shunt current |
| S27 | halved words injected on CH B's current (F-02) | `halved_word` = warning with their count; `board` = ok (not a V/I swap) |
| S28 | source-v dc at 5.0 V on load + shunt to GND | `dac_ceiling` = warning with what the top code delivers |

**Calibration** (simulated bench, virtual multimeter over a fake serial port unless stated):

| # | Scenario | Expected |
|---|---|---|
| C1–C4 | each stage, right wiring | the simulator's true records within ±U; verifications pass |
| C5 | measure-v with CH B "reading crooked" at 5 V | CH B's verification fails |
| C6 | typed mode, one value with two digits swapped | the reading is redone |
| C7 | a jump in one block | the reading is redone |
| C8 | V/I swap on CH B | the stage stops (the board must be re-plugged) |
| C9 | one stage with a base `calib.txt` | the output has the new blocks and the base's others |
| C10 | full command line (all stages, optional points skipped) | complete files; exit code 0 |
| C11 | the multimeter stops answering in the middle of a reading | the reading is redone; after 3 failures the stage pauses and asks the operator |
| C12 | the EEPROM at the default | the calibration runs; the report compares against "no calibration" |
| C13 | source-v with the simulator's non-linearity near 5 V (H-06) | the 4.5 V verification fails, and the report names H-06 with the sizes recorded there |
| C14 | typed mode, values typed with fewer decimals than the display shows | a warning; the coarser resolution enters the budget |

**Monte Carlo:** §8.

### 9.3 Bench acceptance (by the operator, after v0.1)

The developer prepares the procedure (kept with the operator's lab notes, outside the repository) and
analyzes the results; the operator runs it:

1. Open the program: the board check passes (2 s at rest, no V/I swap).
2. CH A source-v sine, 100 Hz, 2 V p-p around 2.5 V, load 47 Ω + shunt 10 Ω to GND, CH B hi-z on the
   middle node: amplitude within 0.5 % of the request, mean current ≈ 2.5 V ÷ 57 Ω, shunt current agrees,
   no warning.
3. CH A source-i dc, +10 mA through 10 Ω to 2.5V, with the multimeter in DC V across the resistor
   (should read ~0.1 V). The same on CH B.
4. CH A source-i square and pulse defaults on 10 Ω to 2.5V, observed on the oscilloscope (ground clip on
   GND only): shape, frequency, edges.
5. Calibration, measure-v stage only, with `--output` to a test folder (EEPROM untouched; typed mode, as
   the RS-232 link is validated on the bench in 0.2, §4.2): the new record within the uncertainty of the
   stored one (H-03).
6. Method B with current (§8): step 3's burst repeated three times, identical, 10 s each, within a few
   minutes, on CH A and on CH B. The developer compares the spread of the three current means with the u
   that method B gave for one burst, as on 2026-11-21 for voltage at rest, and reports the ratio.

Accepted when the results are as predicted and no check flags a problem that the mounted circuit does not
have.

---

## 10. Work plan for v0.1

| WP | Content | Review |
|---|---|---|
| WP0 | `environment.yml` (pinned) and the development environment; `pyproject.toml` (0.1.0, command `phycovolt`, pytest); git repository (with the operator's OK) and `.gitignore` (recordings, calibrations, scratch output, Python caches); the decision log | — |
| WP1 | `hardware.py`, `unit.ini`, `libsmu_math.py`, `m1k.py` (call-order tests first), `simulator.py` (noise model provisional until the 2026-11-21 recording), V/I swap detection | R1 |
| WP2 | `waveforms.py`, `circuits.py`, `checks.py` | — |
| WP3 | `analysis.py`, `recording.py`, `plot.py`; signals end to end on the command line | R2 |
| WP4 | `multimeter.py`, `calibration.py`; calibration end to end on the command line; Monte Carlo | R3 |
| WP5 | the rich command line (§3.1): the start page and examples, the styles, the progress line, the repeat command; the plots' axes without offset notation | — |
| WP6 | documentation (README, `CHANGELOG.md`); the bench procedure (§9.3) | R4 → bench session |

At every review: what was done, the tests, the decisions (and deviations), the questions. Do not move on
past a rejected review.

**Definition of done:** §3 and §5 met, or deviated with an approved record; every rule of §6.1 with a
test; the suite green without the instrument in under ~2 min, on both environments; every scenario of
§9.2 run by a test named after it, and the Monte Carlo result reported by its test; reviews R1–R4
approved; documentation delivered; the bench acceptance (§9.3) run by the operator, with the results in
the lab notes.

---

## 11. Open decisions (as of 2026-11-24)

1. *(Decided on 2026-11-21: the uncertainty of a reading's mean is method B, §8.)*
2. Whether signal results include the calibration's uncertainty (§8).
3. The robust spread for jumps (§8).
4. The research protocol (shapes, currents at the culture, duration, duty cycle,
   tolerance to short pauses), which sets long stimulation (roadmap 0.4).
5. ~~The RS-232 link for the 34401A~~ — the cable is available (2026-11-24); the link is validated on
   the bench in 0.2 (§4.2). Until then, the typed mode.
6. The i-v-i board's scale factor and sign, and its nodes' voltage range.
7. ~~The oscilloscope's model~~ — a DSOX2022A (automation in 0.3).
8. The exact command-line syntax (§3.1 gives the recommended form).
9. The research computer: operating system (possibly Windows), who administers it, network access. It
   is not the development laptop. It decides how PhycoVolt is installed for the
   researchers (libsmu's availability on that system included) and where long stimulation runs (0.4).
10. The uncertainty of periodic results (§8, item 4).
11. Whether a filter is applied before the standard analysis (§8, item 5; roadmap 0.2).

---

# Appendices

## A. Reference values

### A.1 The instrument (goes to `hardware.py`)

| Quantity | Value |
|---|---|
| Sample rate | 100 000 samples/s per channel |
| Channels | A, B; each sample carries (V_A, I_A, V_B, I_B) |
| Voltage range (source and measure) | 0 to 5 V |
| Current range (source) | −0.2 to +0.2 A (the corrected request is clamped to this before the DAC) |
| ADC range, raw | voltage: 0 to 4.99992 V; current: −0.24375 to +0.25624 A |
| ADC and DAC voltage step | 5 ÷ 65 536 = 76.29 µV |
| ADC current step | 0.4 ÷ 65 536 × 1.25 = 7.629 µA |
| DAC current step | 1 ÷ (65 536 × 1.6) = 9.537 µA |
| ADC code | voltage: V ÷ 76.29 µV (0 V at code 0); current: (I + 0.24375) ÷ 7.629 µA (zero near code 31 950) |
| DAC code at rest (HI_Z) | 26 214 = 2.0 V in SVMI; −3.8 µA in SIMV as a raw value (the real output is the source's offset, see L-04) |
| Fixed pins (nominal) | GND 0 V; 2.5V 2.5 V; 5V 5.0 V; 200 mA each |
| Channel current limit | ±200 mA (limiter; overcurrent recorded by the ADM1177) |
| Input in HI_Z | 1 MΩ |
| Minimum firmware | 2.06 for the EEPROM calibration; 2.07 to report overcurrent |
| Reading | chunks of 10 000 samples (100 ms); 2 s timeout per chunk; queue of 1 000 000 samples (10 s) |
| Parking the DAC | 1 000 samples (10 ms) with both channels in HI_Z |
| USB identification | VID:PID `064b:784c` (`03eb:6124` in firmware-update mode) |

### A.2 Thresholds and decision constants (go to `checks.py`, each with its reason)

**Reading and calibration**

| Name | Value | Why |
|---|---|---|
| settling | 0.5 s | the start of every calibration reading: acquisition starts and the output settles |
| block | 100 ms | 6 cycles of 60 Hz (5 of 50 Hz): the hum cancels in every mean |
| reading duration | 5 s (zeros); 10 s with the multimeter; 2 s for the probe | time for the multimeter; the M1K's share of the uncertainty becomes negligible |
| typed readings per point | 3 | the median points at the odd one |
| typed reading window | 40–60 % of the acquisition | both instruments at the same moment if the value drifts |
| jump | a block more than 8 × 1.4826 × MAD from the median (floor: step ÷ 10) | with 5, ~0.4 % false alarms on current readings |
| ADC limit | more than 1 % of the samples within half a step of the edges | the mean would be biased |
| practical margin | voltage: 30 µV + 20 ppm; current: 5 µA + 200 ppm | below it a difference changes no result |
| odd typed reading | deviation from the median > max(3·√2·u, margin) | unlikely and relevant at the same time |
| coverage | U = 2·u | about 95 % for a normal distribution |
| auxiliary zeros (measure-i) | 0 V and 4.9 V | they measure the zero's slope |
| load probe | channel at 3 V and 2 V; load accepted 2–40 Ω; pin accepted 2.2–2.7 V; minimum current 1 mA | R and the pin's real voltage; open circuit |
| point current | min(100 mA, (4.9 − pin)/R, (pin − 0.1)/R), rounded to 0.1 mA | the channel stays between 0.1 and 4.9 V |
| verification points | 4.5 V (source-v); ±10 mA (measure-i and source-i); 5 V (measure-v) | outside the computation; in a smaller multimeter range, close to use |
| plausibility | M1K × multimeter: 5 % + 20 mV (V) or + 2 mA (I); GND < 20 mV; open < 5 mA; current within 40 % of the expected; source-i with the channel between 0.1 and 4.9 V; 0 V request: > 20 mV error, < 0.3 mV warning | wiring mistakes |
| implausible record | gain outside 0.9–1.1; offset > 50 mV (V) or > 5 mA (I) | not from an M1K |
| verification pass | residual ≤ U(residual) + 1 M1K step | an error below the resolution cannot be corrected |
| multimeter failures | 3 consecutive communication failures pause the stage | a cable or setting problem, not noise |
| Monte Carlo | 1000 calibrations per stage, fixed seed; fail above 65 misses per value; warn below 25 | §8 |

**Signals**

| Name | Value | Why |
|---|---|---|
| maximum duration | 10 s | bursts (L-11) |
| samples per period | 20 to 1 000 000 | 5 kHz to 0.1 Hz (the period fits the queue) |
| settling | 1 period (periodic); 1 block (DC) | the channel leaving HI_Z |
| minimum DC duration | 0.3 s | 3 blocks: settling and 2 for the statistics |
| frequency | warning if the real frequency's error exceeds 0.1 % | integer period |
| mains | 60 Hz | the hum |
| samples per period, warning | below 100 | staircase; frequency response not calibrated |
| DAC staircase | warning below 30 levels per period; the plot draws the staircase below 200 | µA |
| DAC level (DC) | warning if the DAC delivers more than 2 % away from the request | µA |
| pin at the limit | within 20 mV of 0 or 5 V; error above 1 % of the samples (SIMV) | saturation |
| compliance (SIMV) | R·I from −2.4 to +2.4 V (2.5V); from −4.65 to −0.1 V (5V) | the pin 0.1 V from the edges; margin for a USB at 4.75 V |
| current limit | error above 200 mA; warning above 150 mA | the limiter; the calibration's correction near the limit |
| power | warning above 0.25 W; commercial ratings ⅛, ¼, ½, 1, 2, 5 W; suggest the one holding twice | heating |
| µA | note when the SIMV peak is below 0.5 mA | step and zero weigh |
| amplitude | tolerance 5 % (current) and 2 % (voltage) | — |
| level (DC) | current: 60 µA + 1 %; voltage: 2 mV + 0.2 % | the measured-current zero's variation (H-08) |
| resistance seen | warning above 10 % + 1 Ω; only with a current amplitude ≥ 20 µA | jumpers |
| open circuit (DC, SVMI) | channel more than 0.5 V from the return, expected current > 1 mA, measured < 10 % of it | load off the channel |
| drift (DC) | warning above 10·u and 0.5 % of the excursion | resistor heating |
| pin windows | GND −0.15…0.15 V; 2.5V 2.2…2.7 V; 5V 4.4…5.3 V | the pin yields under load; USB varies |
| rest reading | 100 ms; the return is not judged above 50 kΩ; the 1 MΩ divider is undone toward the open-input voltage of A.3 (0.25–0.35 V measured) | the HI_Z input |
| invisible return (DC, SVMI) | R × 60 µA > 75 mV | the current zero hides the return |
| charge balance (2.5V) | ok with the center within 1 mV of the pin read | zero mean |
| switch-on in SVMI | the channel at ≈ 2.0 V for a few ms; warning if (2.0 − return) ÷ R exceeds 200 mA | L-04 |
| V/I swap | 500 codes from the other quantity's level; 8 robust deviations from its own level; ≥ 3 periods for periodic waves; 2 s at opening | F-01 |
| shunt agreement | 60 µA + 1 % | H-08 |
| wires in the simulator | 0.4 Ω added to the informed circuit | realism of the rehearsals |

### A.3 The project's unit (goes to `unit.ini`)

Calibration stored in the EEPROM on 2026-11-18 (`calib.txt`):

| Record | CH A (offset; gain+; gain−) | CH B |
|---|---|---|
| measure V | 0.566 mV; 1.023517; 1 | 0.538 mV; 1.023770; 1 |
| measure I | 0.5275 mA; 1.019914; 1.017460 | 0.5136 mA; 1.020277; 1.018271 |
| source V | 2.752 mV; 1.001784; 1 | 1.327 mV; 1.001694; 1 |
| source I | −0.3932 mA; 1.008366; 1.008412 | −0.6588 mA; 1.008778; 1.009436 |

Values for the simulator (measurements of 11 and 13/11, and 2026-11-20's data analysis):

| Constant | Value |
|---|---|
| "true" records, CH A | measure V (0.00055; 1.02349; 1.0) · measure I (0.000528; 1.0195; 1.0173) · source V (0.00257; 1.0018; 1.0) · source I (−0.000389; 1.0084; 1.0083) |
| "true" records, CH B | measure V (0.00063; 1.02382; 1.0) · measure I (0.000459; 1.0177; 1.0178) · source V (0.00114; 1.0017; 1.0) · source I (−0.000646; 1.0089; 1.0096) |
| the simulator's "stored" calibration | the true one with offset −10 µ(V or A) and gain +40 ppm |
| current-zero slope | −18 µA/V around 2.5 V (unstable on the board: H-07) |
| real pins | 2.5V 2.4457 V; 5V 4.9300 V (this laptop's USB, warm, 2026-11-21) |
| open input in HI_Z | 0.30 V (it floats between 0.25 and 0.35 V: H-12) |
| noise per sample (σ) | voltage 64 µV (CH A) and 43 µV (CH B) on GND, ~165 µV on the 2.5V pin; current 58 µA (2026-11-21) |
| 60 Hz hum (amplitude) | voltage ≤ 1 µV on GND and the pins, ~1 mV on an open input; current ≤ 0.2 µA (2026-11-21) |
| slow noise | spread of 100 ms block means ~2 µV, lag-1 correlation ≈ 0.65 (rest on GND, 2026-11-21) |
| source non-linearity near 5 V | at 4.5 V, −0.3 mV (CH A) and −1.4 mV (CH B) below the 0–2.5 V line (H-06) |
| halved current word | CH B, about one sample every 36 s (F-02); off by default, a test hook |
| identification | "simulated (no board) · fw 2.17 · hw D" |

## B. File formats

### B.1 `calib.txt` (as libsmu reads it)

```
# comments outside blocks: free, any length
# Channel A, measure V   ->  offset 0.000566, gain+ 1.023517, gain- 1.000000
</>
<0.0000, 0.000566>
<2.445100, 2.389485>
<\>
```

1. A block opens on a line containing `</>` and closes on one containing `<\>` (anywhere on the line:
   comments must never contain these marks).
2. Inside a block, every line is a pair `<a, b>` with the `<` in the **first column** and at most **126
   bytes** (libsmu uses a 127-byte `fgets` and `sscanf`). No blank or comment line inside a block:
   libsmu would read it as a repeat of the previous pair.
3. Exactly **8 blocks**, in this order: CH A measure V, measure I, source V, source I; CH B in the same
   order. The position decides; the comment is for people.
4. 2 to 16 pairs per block (current: at least 3, one with a positive and one with a negative reference).
   The first pair is the zero (reference 0); no other pair has reference 0.
5. Measure: `<reference, M1K>`. Source: `<request, multimeter>`. In V and A.
6. Formatting: the zero as `0.0000`; source-V requests with 4 decimals (their fit codes are exact);
   source-I requests with 7 decimals, the value of the DAC code sent (I-11; 4 decimals would add a 15–61
   ppm gain error); measured values with 6 (V) or 7 (A) decimals. Each block's comment carries
   the predicted coefficients (float32).

### B.2 Folders and tables

- `recordings/<YYYY-MM-DD_HHMMSS>_<role>-<waveform>[_N]/`: `samples.csv.gz`, `metadata.json`,
  `summary.txt` (§3.4).
- `calibrations/<YYYY-MM-DD_HHMMSS>_<stage or "all">/`: `calib.txt`, `blocks.txt`, `readings.csv`,
  `dmm.csv`, `report.txt`, `metadata.json` (§3.7).
- `readings.csv`: one line per channel per reading, accepted or not, with the columns
  `time, stage, point, attempt, status, channel, quantity, mean, u, sigma, drift, r1, blocks, jumps,
  adc_limit_fraction, dmm_mean, dmm_u, dmm_readings, vi_swaps` (`status` carries the reason when a
  reading is redone).
- `dmm.csv`: `time, stage, point, attempt, function, range, value`.
- Text summaries: the terminal's text, no styles, at most 100 columns.

### B.3 Multimeter profile (`phycovolt/profiles/*.ini`)

```ini
[multimeter]
name = Agilent 34401A
source = datasheet 5968-0162: 1-year accuracy, 23 ± 5 °C, after 1 h warm-up
overrange = 1.2               ; a range covers values up to 120 % of its full scale
distribution = rectangular    ; or normal-95 (limit ÷ 2) if the datasheet states the confidence
terminal_voltage = HI
terminal_common = LO
terminal_current = I

[connection]
transport = serial            ; serial | typed (usbtmc later)
port = /dev/ttyUSB0
baud = 9600                   ; as set in the instrument's I/O menu
parity = none
data_bits = 8
handshake = dtr_dsr

[scpi]                        ; to confirm against the instrument's user's guide
remote = SYST:REM
identify = *IDN?
configure_dc_voltage = CONF:VOLT:DC {range}
configure_dc_current = CONF:CURR:DC {range}
read = READ?

[dc_voltage]                  ; full scale = % of reading + % of range
0.1 = 0.0050 + 0.0035
1 = 0.0040 + 0.0007
10 = 0.0035 + 0.0005
100 = 0.0045 + 0.0006
1000 = 0.0045 + 0.0010

[dc_current]
0.01 = 0.050 + 0.020
0.1 = 0.050 + 0.005
1 = 0.100 + 0.010
3 = 0.120 + 0.020
```

The range used is the smallest whose full scale × `overrange` covers the value. Every range becomes two
named error sources (the range's gain and zero), rectangular (limit ÷ √3) or normal (limit ÷ 2). The
template (`template.ini`) has `FILL_IN` fields, and a profile with any of them is refused.

## C. Formulas

### C.1 `smu -w` (per block, in float32)

```
offset = val₀ − ref₀
gain+  = mean{ ref_k ÷ (val_k − offset) : ref_k > 0 }        (1 if empty)
gain−  = mean{ ref_k ÷ (val_k − offset) : ref_k ≤ 0, k ≥ 1 }  (1 if empty)
```

### C.2 Measurement (libsmu) and its inversion (PhycoVolt)

```
voltage:  corrected = (raw − offset) × gain+                raw = corrected ÷ gain+ + offset
current:  corrected = (raw − offset) × g                    raw = corrected ÷ g + offset
          g = gain+ if the PREVIOUS corrected sample > 0, else gain−
          (previous = 0 at a USB transfer's start: every 1024 samples on Linux, 2560 on Windows)
the "true" M1K:  raw = true ÷ gain± + offset
```

### C.3 The source

```
corrected request:  c = clamp((request − offset) × g, limits)    g: gain+ (V); gain± by the request's sign (I)
code:               V: trunc(c ÷ 76.29 µV)        I: trunc(65536 × (0.4 + 1.6 × c))      clamped to 0…65535
value of a code:    V: code × 76.29 µV            I: (code ÷ 65536 − 0.4) ÷ 1.6
pre-distortion:     request = raw ÷ gain+ + offset;  for current, if that request is not > 0: raw ÷ gain− + offset
true output:        output = true_offset + code_value ÷ true_gain±
"what the DAC delivers" = the true output assuming the stored calibration is exact
note: libsmu writes the current factor 1.6 as 0.8*0.2*20.*0.5, which in double precision is
      1.6000000000000003; the model uses that exact value
```

### C.4 The summary of a reading

```
x = raw samples without the settling;  n blocks of 100 ms;  b_j = mean of block j
value = mean(b)
s_b   = standard deviation of b
u     = s_b                                                 (method B, decided 2026-11-21: §8)
        [method C, not chosen, for reference: ρ₁ = Σ(b_j − b̄)(b_{j+1} − b̄) ÷ Σ(b_j − b̄)²;
         ρ = clamp(ρ₁ + (1 + 4·max(ρ₁, 0)) ÷ n, 0, 0.95);  n_ef = n·(1 − ρ) ÷ (1 + ρ);  u = s_b ÷ √n_ef]
σ = standard deviation of x;  if σ < step/2: u = √(u² + (step/√12)²)
r₁ = correlation between (b_1 … b_{n−1}) and (b_2 … b_n)    (reported)
jumps = #{ j : abs(b_j − median(b)) > 8 × max(1.4826 × MAD(b), step/10) }
drift = slope of the line through (t_j, b_j) × (t_n − t_1)
at limit = fraction of x within step/2 of the ADC edges
```

Typed mode, three readings: value = mean; u = max(√(Σu_k²) ÷ 3, s ÷ √3).

### C.5 The V/I swap

```
v, i = ADC codes of a channel's raw samples
levels: constant → median (of 1 in 16 samples); periodic with N per period (≥ 3 periods) →
        the median of each phase across periods, repeated
swapped(k) ⇔ abs(v_k − level_i) ≤ 500  and  abs(i_k − level_v) ≤ 500
             and abs(v_k − level_v) > 500  and  abs(i_k − level_i) > 500
             and abs(v_k − level_v) > 8 × 1.4826 × MAD(v − level_v)
             and abs(i_k − level_i) > 8 × 1.4826 × MAD(i − level_i)
```

### C.6 Uncertainty propagation

```
each error source: (name, u)
each measured value:  measured = true + Σ c_k × error_k
for an output y = f(…) computed by the SAME code as the calibration:
    (c·u)_k = [ f(error_k = +u_k) − f(error_k = −u_k) ] ÷ 2
    u(y)    = √Σ (c·u)_k²
    U       = 2 · u(y)
predicted error at x: with the new record, from the "true" behavior (the pairs minus the errors)
verification: residual ± U(residual); passes if abs(residual) ≤ U + 1 step
```

### C.7 Periodic signals and DC levels

```
one period:      s_k = shape(k ÷ N), k = 0 … N−1,  N = round(100 000 ÷ f)
                 sine: s_k = center + amplitude × sin(2πk/N)
cycle average:   over whole periods without the first: mean and standard deviation, phase by phase
sine regression  (y, with sin and cos at the request's phase):
                 c = mean(y);  a = 2·mean((y − c)·sin);  b = 2·mean((y − c)·cos)
                 amplitude = √(a² + b²);  phase = atan2(b, a);  delay = −phase ÷ (2πf)
                 residual = rms(y − c − a·sin − b·cos)
impedance seen:  amplitude(V) ÷ amplitude(I)  (if amplitude(I) ≥ 20 µA);  phase(V) − phase(I)
return:          level(V) − R × level(I)
DC level:        the reading summary (C.4) on the samples without the first block
```

### C.8 The simulated instrument

```
true source output = C.3 (corrected by the STORED calibration, truncated at the DAC, answering
                     with the TRUE one)
circuit → true V and I at the terminal:
    HI_Z:  V = the return's voltage through the 1 MΩ divider (the open-input voltage of A.3 with
           nothing connected, plus ~1 mV of room pickup), I = 0
    SVMI:  I = (V − V_return) ÷ R, limited to ±200 mA (overcurrent; V follows)
    SIMV:  V = clamp(V_return + I·R, 0, 5);  I = (V − V_return) ÷ R   (no circuit: V goes to 0 or 5 V)
raw:   V ÷ gain + offset;   I ÷ gain± + offset + slope × (V − 2.5)
       + 60 Hz hum (phase drawn per read) + white noise + slow noise (block correlation as in A.3)
ADC:   quantize (16 bits), clip at the edges;  [optional V/I swap]
libsmu: the correction of C.2, in float32
no dynamics: the output follows the request in the same sample
virtual multimeter: true value + a range error drawn within the profile's specification
                    (once per simulated session) + reading noise, rounded to the displayed digits
test hooks: a channel reading crooked at 5 V, a jump in a block, the V/I swap, the halved word (F-02),
            the source's non-linearity near 5 V (H-06), overcurrent on a short,
            EEPROM at the default, a USB failure, a multimeter that stops answering
```

## D. Glossary

| Term | Meaning |
|---|---|
| SMU | source and meter on the same terminal at the same time |
| HI_Z, SVMI, SIMV | rest (measures voltage only); source voltage and measure current; source current and measure voltage |
| role | what a channel does in PhycoVolt: hi-z, source-v, source-i |
| code | the integer (0 to 65 535) the DAC receives or the ADC returns |
| step | the smallest change of the ADC or the DAC |
| raw | a value without any calibration correction |
| record | (offset, gain+, gain−) of one function of one channel; there are 8 |
| pair | one `<a, b>` line of `calib.txt` |
| park the DAC | a short HI_Z read that puts the DAC at code 26 214 (in SIMV the source's uncalibrated zero, a fraction of a mA; 2.0 V in SVMI), done before every burst that sources |
| rest reading | the 100 ms in HI_Z before switching a signal on, which show where the return is |
| return | the pin where the other end of a circuit goes: GND, 2.5V or 5V |
| compliance | the voltage range in which a current source keeps the requested current |
| shunt | a small known resistor whose voltage shows the current |
| charge balance | the mean current of a stimulus; when not zero, it is a net charge on an electrode |
| cycle average | the periods of a periodic signal laid over each other and averaged, phase by phase |
| synchronous detection | the fit of level + sine + cosine at the known frequency |
| block | 100 ms of samples |
| chunk | the 100 ms unit in which the instrument delivers samples |
| V/I swap | fault F-01 |
| u, U | standard uncertainty; expanded uncertainty (U = 2u, about 95 %) |
| MAD | median absolute deviation |
| verification | a point measured outside the calibration's computation |
| burst | a finite acquisition (≤ 10 s) with the outputs on only while it lasts |
| operator | who runs the instrument and answers for the research |

## E. References

- libsmu: <https://github.com/analogdevicesinc/libsmu> (version 1.0.3; `src/device_m1000.cpp`:
  `write_calibration`, `handle_in_transfer`, `encode_out`, `set_mode`; `src/session.cpp`; issues #181,
  #188, #193, #201, #209).
- ADALM1000: the product page at Analog Devices and the ADI wiki
  (<https://wiki.analog.com/university/tools/m1k/analog-inputs>: the 1 MΩ input in HI_Z; the M1K
  calibration pages).
- Agilent 34401A: datasheet 5968-0162 (the 1-year accuracy used in B.3) and the user's guide (RS-232 and
  SCPI).
- GUM: JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement*.
- N. F. Zhang, "Calculation of the uncertainty of the mean of autocorrelated measurements", *Metrologia*
  43 (2006) — background for method C.
