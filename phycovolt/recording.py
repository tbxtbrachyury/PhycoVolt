"""Recording folders and their files; the plain-text summaries; reading a recording back.

Questions it answers: where does a measurement go, what is in its files, how does it read as text
(on the terminal and in `summary.txt`), and how is it analyzed again from its files alone?

Main choices (spec 3.4, B.2):

- `recordings/<YYYY-MM-DD_HHMMSS>_<role>-<waveform>/`, with `_2`, `_3` … in the same second; a
  folder is never reused, so nothing is ever overwritten.
- `samples.csv.gz`: every sample, gzip-compressed CSV, columns
  `sample, time_s, a_requested, a_dac, a_v, a_i, b_requested, b_dac, b_v, b_i` (V and A; requested
  and dac empty for a hi-z channel); 9 significant digits keep libsmu's float32 values exactly.
- `metadata.json`: everything needed to analyze the samples again (configuration, stored
  calibration, rest reading), plus every check and result with its unit in its name.
- `summary.txt`: the result as plain text, at most 100 columns (a command to repeat may be longer).
- The texts are lines of `rich.text.Text` whose parts are marked by meaning (`DANGER`, `RESULT`,
  `CONTEXT`…, spec 3.1); `__main__.THEME` gives each meaning its look on the terminal, and `plain`
  writes the files with no styles. Built from parts, never parsed from markup, so a "[" in a path or
  a description stays as it is.
- All text is UTF-8 with "\\n" line ends; paths are pathlib (NF-13).
"""

from __future__ import annotations

import gzip
import json
import math
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.text import Text

from phycovolt import __version__, analysis, checks, circuits, libsmu_math, signals, units, waveforms
from phycovolt import hardware as hw
from phycovolt.checks import Check, Level
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind, Prediction
from phycovolt.hardware import Record, Role
from phycovolt.analysis import CycleAverage, SineFit
from phycovolt.signals import ChannelResult, Measurement, Prepared, Samples
from phycovolt.waveforms import Shape, Waveform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_FILE = "samples.csv.gz"
METADATA_FILE = "metadata.json"
SUMMARY_FILE = "summary.txt"
COLUMNS = ("sample", "time_s", "a_requested", "a_dac", "a_v", "a_i", "b_requested", "b_dac", "b_v", "b_i")
WIDTH = 100                            # summary.txt columns
VALUE_FORMAT = "%.9g"                  # float32 needs 9 significant digits to come back exactly


def recordings_folder(output: Path | None = None) -> Path:
    """Where recordings go: `<output>/recordings`, or the project root's `recordings/` (spec 3.1)."""
    return (output if output is not None else PROJECT_ROOT) / "recordings"


def git_commit() -> str | None:
    """The commit of the program's folder, read from .git without running git (None if absent)."""
    git = PROJECT_ROOT / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        loose = git / ref
        if loose.exists():
            return loose.read_text(encoding="utf-8").strip()
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        return None
    return None


# --- Folders --------------------------------------------------------------------------------------


def label(a: ChannelConfig, b: ChannelConfig) -> str:
    """"source-i-sine": the first sourcing channel's role and waveform."""
    for config in (a, b):
        if config.waveform is not None:
            return f"{config.role.value}-{config.waveform.shape.value}"
    return "hi-z"


INCOMPLETE_PREFIX = ".incomplete_"     # a folder being written; renamed when every file is there


def free_folder(base: Path, started: datetime, name: str) -> Path:
    """A folder name `<date>_<time>_<name>` not taken yet (also not by one being written),
    with _2, _3 … if that second is taken. Nothing is created."""
    stem = f"{started:%Y-%m-%d_%H%M%S}_{name}"
    for count in range(1, 1000):
        folder = base / (stem if count == 1 else f"{stem}_{count}")
        if not folder.exists() and not (base / (INCOMPLETE_PREFIX + folder.name)).exists():
            return folder
    raise RuntimeError(f"no free folder name for {stem}")


# --- To and from JSON -----------------------------------------------------------------------------


def config_to_dict(config: ChannelConfig) -> dict:
    circuit = config.circuit
    data: dict = {"role": config.role.value,
                  "circuit": {"kind": circuit.kind.value, "return": circuit.ret, "r_ohm": circuit.r,
                              "shunt_ohm": circuit.shunt, "scale": circuit.scale,
                              "board_input_v": circuit.board_input}}
    if config.waveform is not None:
        timing = waveforms.timing(config.waveform)
        data["waveform"] = {"shape": config.waveform.shape.value, "values": dict(config.waveform.values),
                            "samples_per_period": timing.samples_per_period,
                            "real_frequency_hz": timing.frequency, "real": timing.real,
                            "chunks": timing.chunks}
    return data


def config_from_dict(data: dict) -> ChannelConfig:
    circuit = data["circuit"]
    waveform = None
    if "waveform" in data:
        waveform = Waveform(Shape(data["waveform"]["shape"]), data["waveform"]["values"])
    return ChannelConfig(Role(data["role"]),
                         Circuit(CircuitKind(circuit["kind"]), circuit["return"], circuit["r_ohm"],
                                 circuit["shunt_ohm"], circuit["scale"], circuit["board_input_v"]),
                         waveform)


def check_to_dict(check: Check) -> dict:
    return {"level": check.level.value, "code": check.code, "channel": check.channel,
            "variant": check.variant, "numbers": check.numbers, "text": check.text}


def check_from_dict(data: dict) -> Check:
    return Check(Level(data["level"]), data["code"], data["channel"], data["variant"], data["numbers"])


def _clean(value: object) -> object:
    """JSON has no NaN: write null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def metadata(measurement: Measurement, files: list[str], command: dict | None = None) -> dict:
    prepared = measurement.prepared
    return _clean({
        "program": {"name": "PhycoVolt", "version": __version__, "git_commit": git_commit()},
        "started": measurement.started.isoformat(timespec="seconds"),
        "command": command,
        "instrument": {"simulated": measurement.serial == "simulated", "description": measurement.description,
                       "serial": measurement.serial, "firmware": measurement.firmware,
                       "hardware": measurement.hardware},
        "calibration": [[r.offset, r.gain_pos, r.gain_neg] for r in prepared.records],
        "calibration_order": [f"{channel} {kind.value}" for channel, kind in hw.RECORD_ORDER],
        "channels": {"A": config_to_dict(prepared.a), "B": config_to_dict(prepared.b)},
        "rest": {"seconds": checks.REST_SECONDS, "median_v": prepared.rest},
        "burst": {"chunks": prepared.chunks, "samples": measurement.samples.count,
                  "sample_rate_hz": hw.SAMPLE_RATE, "measure_only": prepared.measure_only},
        "ended": measurement.ended,
        "interrupted": measurement.interrupted,
        "overcurrent": measurement.overcurrent,
        "checks": {"before": [check_to_dict(c) for c in prepared.checks],
                   "after": [check_to_dict(c) for c in measurement.after]},
        "results": {"A": measurement.results["A"].numbers(), "B": measurement.results["B"].numbers(),
                    "uncertainty_method": analysis.UNCERTAINTY_METHOD,
                    "statistical_only": True},
        "files": files,
    })


# --- Writing --------------------------------------------------------------------------------------


def _columns(measurement: Measurement) -> tuple[np.ndarray, str]:
    """The table of samples and its row format (requested and dac left empty for hi-z)."""
    prepared = measurement.prepared
    count = measurement.samples.count
    sample = np.arange(count)
    table = [sample, sample / hw.SAMPLE_RATE]
    formats = ["%d", "%.5f"]
    for name in hw.CHANNELS:
        prediction = prepared.predictions.get(name)
        if prediction is not None:
            table += [np.resize(prediction.requested, count), np.resize(prediction.delivered, count)]
            formats += [VALUE_FORMAT, VALUE_FORMAT]
        else:
            formats += ["", ""]
        voltage, current = measurement.samples.channel(name)
        table += [voltage, current]
        formats += [VALUE_FORMAT, VALUE_FORMAT]
    return np.column_stack(table) if count else np.empty((0, len(table))), ",".join(formats)


def repeat_words(command: dict | None) -> list[str] | None:
    """The words of the command that repeats a recording, from its metadata (None if it has none)."""
    if not command or not command.get("repeat"):
        return None
    return shlex.split(command["repeat"])


def write(measurement: Measurement, base: Path, command: dict | None = None) -> Path:
    """Write a recording; returns its folder. `command` is the command line that made it (spec
    3.1): {"typed": what was typed, "repeat": every channel written out, without --confirm-output}.

    The files are written into `.incomplete_<name>` and the folder is renamed only when all three
    are there, so an interrupted write (Ctrl+C during the seconds a long burst takes) never leaves a
    folder that looks complete.
    """
    prepared = measurement.prepared
    base.mkdir(parents=True, exist_ok=True)
    folder = free_folder(base, measurement.started, label(prepared.a, prepared.b))
    partial = base / (INCOMPLETE_PREFIX + folder.name)
    partial.mkdir()
    table, row = _columns(measurement)
    with gzip.open(partial / SAMPLES_FILE, "wt", encoding="utf-8", newline="\n", compresslevel=6) as file:
        file.write(",".join(COLUMNS) + "\n")
        if table.size:
            np.savetxt(file, table, fmt=row)
    files = [SAMPLES_FILE, METADATA_FILE, SUMMARY_FILE]
    (partial / METADATA_FILE).write_text(json.dumps(metadata(measurement, files, command), indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    (partial / SUMMARY_FILE).write_text(summary_text(measurement, folder, repeat=repeat_words(command)),
                                        encoding="utf-8", newline="\n")
    partial.rename(folder)
    return folder


# --- Reading back (spec 3.5) ----------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    folder: Path
    metadata: dict
    measurement: Measurement           # with the analysis done again from the files


def read_samples(folder: Path) -> Samples:
    """The measured columns of samples.csv.gz (the requested ones follow from the metadata)."""
    columns = [COLUMNS.index(name) for name in ("a_v", "a_i", "b_v", "b_i")]
    with gzip.open(folder / SAMPLES_FILE, "rt", encoding="utf-8") as file:
        data = np.loadtxt(file, delimiter=",", skiprows=1, usecols=columns, ndmin=2)
    # libsmu's samples are float32: 9 digits name each one exactly once read back as float32.
    data = data.astype(np.float32).astype(np.float64)
    return Samples(*(data[:, k].copy() for k in range(4)))


def read(folder: Path) -> Recording:
    """Read a recording and analyze it again from samples.csv.gz and metadata.json alone."""
    folder = Path(folder)
    meta = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    a = config_from_dict(meta["channels"]["A"])
    b = config_from_dict(meta["channels"]["B"])
    records = [Record(*values) for values in meta["calibration"]]
    rest = {name: float(value) for name, value in meta["rest"]["median_v"].items()}
    open_input = hw.load_unit().open_input
    predictions = {}
    for name, config in (("A", a), ("B", b)):
        if config.role is not Role.HI_Z:
            predictions[name] = circuits.predict(config, name, records,
                                                 checks.return_voltage(config, rest[name], open_input))
    before = [check_from_dict(c) for c in meta["checks"]["before"]]
    prepared = Prepared(a, b, rest, records, before, predictions, int(meta["burst"]["chunks"]))
    samples = read_samples(folder)
    results, after = signals.analyze(prepared, samples, bool(meta["overcurrent"]))
    instrument = meta["instrument"]
    measurement = Measurement(prepared, samples, meta["ended"], bool(meta["interrupted"]),
                              bool(meta["overcurrent"]), results, after,
                              datetime.fromisoformat(meta["started"]), instrument["description"],
                              instrument["serial"], instrument["firmware"], instrument["hardware"])
    return Recording(folder, meta, measurement)


# --- Text marked by meaning, in two views (spec 3.1) ----------------------------------------------

# What a part of a text is; __main__.THEME says how each looks. "If everything is highlighted,
# nothing is": a part is marked only when it is one of these.
DANGER = "danger"          # errors (✗), "nothing was switched on", --confirm-output, an interruption
ATTENTION = "attention"    # warnings (⚠)
RESULT = "result"          # a measured value and its ±
ASKED = "asked"            # the asked or expected value next to a result
CONTEXT = "context"        # identification, start time, secondary details, a passed check's text
SAFE = "safe"              # a passed check's ✓, "both channels are in HI_Z"
COMMAND = "command"        # a command or an option to type
HEADING = "heading"        # a section's title, a channel's header: where the eye lands
LABEL = "label"            # the column of names in front of values
RULE = "rule"              # the line of a section's rule
ACTION = "action"          # what the operator must do at a prompt
GOOD = "verdict.good"      # badges: COMPLETE, PASS
LIVE = "verdict.live"      # SWITCHING ON, MEASURING
STOP = "verdict.stop"      # STOPPED, NOT SWITCHED ON (no --confirm-output), COMPLETE with a FAIL
BAD = "verdict.bad"        # FAILED, FAIL, NOT SWITCHED ON (an error)
STYLES = (DANGER, ATTENTION, RESULT, ASKED, CONTEXT, SAFE, COMMAND, HEADING, LABEL, RULE, ACTION,
          GOOD, LIVE, STOP, BAD)
LEVEL_STYLE = {Level.OK: SAFE, Level.NOTE: "", Level.WARNING: ATTENTION, Level.ERROR: DANGER}
BODY_STYLE = {Level.OK: CONTEXT, Level.NOTE: "", Level.WARNING: ATTENTION, Level.ERROR: DANGER}
LABEL_WIDTH = 14           # "at the culture"
NEXT_WIDTH = 7             # "Folder", "Repeat"


@dataclass(frozen=True)
class View:
    """Where a text goes (spec 3.1, Hierarchy). The terminal: ordered for the screen, the normal
    condensed (passed checks counted, the method in the file). A file: complete and plain. The
    same builders make both, so a file never loses what the terminal condenses."""

    terminal: bool = False
    width: int = WIDTH


FILE = View()


def plain(lines: list[Text]) -> str:
    """The lines as plain text, one per line (the files: no styles, no escape codes)."""
    return "".join(line.plain + "\n" for line in lines)


# A word, kept with the unit after it ("2.446 V", "10.40 Ω;", "CH A"): a line never breaks between them.
_WORD = re.compile(r"\S+(?: (?=[pnuµμmkM]?(?:V|A|Hz|s|Ω|W|%)(?:[\s),;:.]|$))\S+)?")


def wrapped(text: Text | str, indent: str = "", width: int = WIDTH, hang: str | None = None,
            lead: Text | None = None) -> list[Text]:
    """Word-wrapped at `width` with its marks kept. The first line starts with `lead` (as it is: a
    badge, a label) or `indent`; the next ones with `hang` (two more than `indent` when not given).
    A word longer than the width (a path) is never cut."""
    text = Text(text) if isinstance(text, str) else text
    hang = indent + "  " if hang is None else hang
    prefix = lead if lead is not None else Text(indent)
    words = [match.span() for match in _WORD.finditer(text.plain)]
    if not words:
        return [prefix]
    lines = []
    start, end = words[0]
    for word_start, word_end in words[1:]:
        if len(prefix.plain) + word_end - start <= width:
            end = word_end
            continue
        lines.append(prefix + text[start:end])
        prefix, start, end = Text(hang), word_start, word_end
    lines.append(prefix + text[start:end])
    return lines


def labeled(label: str, value: Text | str, width: int = WIDTH, indent: str = "  ",
            label_width: int = LABEL_WIDTH) -> list[Text]:
    """"  current        amplitude 1.000 mA …": the label in a column, the value wrapped beside it."""
    lead = Text.assemble(indent, (f"{label:<{label_width}}", LABEL), " ")
    return wrapped(value, width=width, hang=" " * len(lead.plain), lead=lead)


def section(title: str, view: View) -> list[Text]:
    """A blank line, then the section's rule with its title ("== title ===" in a file)."""
    rest = max(view.width - len(title) - 4, 2)
    if view.terminal:
        return [Text(), Text.assemble(("── ", RULE), (title, HEADING), (" " + "─" * rest, RULE))]
    return [Text(), Text(f"== {title} " + "=" * rest)]


def verdict(word: str, style: str, parts: list[str | tuple[str, str]], view: View,
            then: list[list[str | tuple[str, str]]] = ()) -> list[Text]:
    """A status line: a badge (a word in capitals, filled) and what it says; each of `then` on a
    line of its own, aligned under it. In a file, the word."""
    # The badge is a span of its own: a Text's base style would color the whole line.
    lead = Text.assemble((f" {word} ", style), " ") if view.terminal else Text(f"{word} · ")
    hang = " " * len(lead.plain)
    lines = wrapped(Text.assemble(*parts), width=view.width, hang=hang, lead=lead)
    for more in then:
        lines += wrapped(Text.assemble(*more), width=view.width, hang=hang, lead=Text(hang))
    return lines


def check_lines(check: Check, indent: str = "  ", width: int = WIDTH) -> list[Text]:
    """"✓ CH A: …": a passed check's symbol green and its text dimmed, a note plain, a warning's or
    an error's whole text marked."""
    lead = Text.assemble(indent, (checks.SYMBOL[check.level], LEVEL_STYLE[check.level]), " ")
    return wrapped(Text(check.text, style=BODY_STYLE[check.level]), width=width, hang=indent + "  ", lead=lead)


_TALLY = ((Level.OK, "passed", "passed"), (Level.NOTE, "note", "notes"), (Level.WARNING, "warning", "warnings"),
          (Level.ERROR, "error", "errors"))


def tally(found: list[Check]) -> list[str | tuple[str, str]]:
    """"✓ 8 passed, ◆ 1 note, ⚠ 2 warnings" (the levels there are), marked."""
    parts: list[str | tuple[str, str]] = []
    for level, one, many in _TALLY:
        count = sum(check.level is level for check in found)
        if count:
            parts += [", " if parts else "", (checks.SYMBOL[level], LEVEL_STYLE[level]),
                      (f" {count} {one if count == 1 else many}", BODY_STYLE[level])]
    return parts or [("no check", CONTEXT)]


def problems(found: list[Check]) -> list[str | tuple[str, str]]:
    """"0 errors, 1 warning, 2 notes" for a verdict; a count above zero is marked."""
    parts: list[str | tuple[str, str]] = []
    for level, one, many in reversed(_TALLY[1:]):
        count = sum(check.level is level for check in found)
        parts += [", " if parts else "", (f"{count} {one if count == 1 else many}", BODY_STYLE[level] if count else "")]
    return parts


def checks_block(found: list[Check], view: View) -> list[Text]:
    """Every check in a file; on the terminal, the passed ones counted on one line."""
    lines: list[Text] = []
    shown = found
    if view.terminal:
        passed = sum(check.level is Level.OK for check in found)
        if passed:
            lines.append(Text.assemble("  ", (checks.SYMBOL[Level.OK], SAFE), (f" {passed} passed", CONTEXT)))
        shown = [check for check in found if check.level is not Level.OK]
    for check in shown:
        lines.extend(check_lines(check, "  ", view.width))
    return lines


_SAFE_WORD = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")


def quoted(word: str) -> str:
    """A word of a command, quoted for pasting: as it is, in double quotes (bash, zsh, PowerShell
    and cmd all read them), or in POSIX single quotes when it holds " $ ` \\ or !."""
    if _SAFE_WORD.match(word):
        return word
    if not any(char in word for char in '"$`\\!'):
        return f'"{word}"'
    return shlex.quote(word)


def command_text(words: list[str]) -> str:
    """A command on one line, ready to paste."""
    return " ".join(quoted(word) for word in words)


def command_lines(words: list[str], indent: str = "  ", width: int = WIDTH, label: str | None = None,
                  label_width: int = NEXT_WIDTH) -> list[Text]:
    """A command to copy, after its label when it has one; on one line when it fits, else broken
    before options with " \\" (bash). A quoted channel description is never cut, and nothing is
    drawn around it (a box's border would be copied with it)."""
    pieces: list[str] = []                 # a break comes only before an option
    for word in words:
        complete_option = pieces and pieces[-1].startswith("--") and " " in pieces[-1]
        if not pieces or word.startswith("--") or complete_option:
            pieces.append(quoted(word))
        else:
            pieces[-1] += " " + quoted(word)
    lead = Text(indent) + (Text.assemble((f"{label:<{label_width}}", LABEL), " ") if label is not None else Text())
    room = width - len(lead.plain)
    rows, current = [], pieces[0]
    for piece in pieces[1:]:
        if len(current) + 1 + len(piece) <= room - 2:
            current += " " + piece
        else:
            rows.append(current)
            current = "  " + piece
    rows.append(current)
    hang = Text(" " * len(lead.plain))
    return [(lead if k == 0 else hang) + Text(row + (" \\" if k < len(rows) - 1 else ""), style=COMMAND)
            for k, row in enumerate(rows)]


def plot_command(folder: Path) -> list[str]:
    return ["phycovolt", "plot", str(folder)]


def _signal_unit(role: Role) -> str:
    return "A" if role is Role.SOURCE_I else "V"


def _show(value: float | None, unit: str) -> str:
    return "—" if value is None or not math.isfinite(value) else units.show(value, unit)


def describe_channel(name: str, config: ChannelConfig) -> str:
    """"CH A source-i sine 100 Hz → R 10 Ω → 2.5V"."""
    circuit = config.circuit
    text = f"CH {name} {config.role.value}"
    if config.waveform is not None:
        text += f" {config.waveform.shape.value}"
        if config.waveform.periodic:
            text += f" {units.show(waveforms.timing(config.waveform).frequency, 'Hz', 4)}"
    kind = circuit.kind
    if kind is CircuitKind.RESISTOR:
        text += f" → R {units.show(circuit.r, 'Ω')} → {circuit.ret}"
    elif kind is CircuitKind.LOAD_SHUNT:
        shunt = f" + shunt {units.show(circuit.shunt, 'Ω')}" if circuit.shunt else ""
        text += f" → load {units.show(circuit.r, 'Ω')}{shunt} → {circuit.ret}"
    elif kind is CircuitKind.IVI_BOARD:
        text += f" → i-v-i board (scale {circuit.scale:g})"
    elif kind is CircuitKind.SHUNT_NODE:
        text += " on the other channel's shunt node"
    elif kind is CircuitKind.SHUNT:
        text += f" on R {units.show(circuit.r, 'Ω')} → {circuit.ret}"
    return text


def channel_header(name: str, config: ChannelConfig, width: int = WIDTH) -> list[Text]:
    return wrapped(Text(describe_channel(name, config), style=HEADING), "", width)


def describe_parameters(config: ChannelConfig) -> str:
    """"100 Hz, pp 200.0 mA, center 0 A, 100 periods"."""
    if config.waveform is None:
        return ""
    parts = []
    for item in waveforms.FIELDS[config.waveform.shape]:
        value = config.waveform[item.key]
        unit, _ = waveforms.field_unit(item, config.role)
        parts.append(f"{item.key} {units.show(value, unit) if unit else f'{value:g}'}")
    return ", ".join(parts)


def _expected(prediction: Prediction, role: Role) -> tuple[float, float]:
    """(level, amplitude) of the quantity the channel does not impose, from the circuit."""
    other = prediction.voltage if role is Role.SOURCE_I else prediction.current
    return float(np.mean(other)), float(np.ptp(other)) / 2


def _result(value: float | None, unit: str) -> tuple[str, str]:
    return (_show(value, unit), RESULT)


def _channel_rows(name: str, config: ChannelConfig, result: ChannelResult,
                  prepared: Prepared) -> list[tuple[str, Text]]:
    """A channel's result, as (label, value) rows: the measured values marked, the asked and
    expected ones beside them."""
    rows: list[tuple[str, Text]] = []
    parameters = describe_parameters(config)
    if parameters:
        rows.append(("asked", Text(parameters, style=ASKED)))
    observed = result.observed
    if observed is None or observed.v_level is None:
        rows.append(("measured", Text("no complete period or block was measured", style=ATTENTION)))
        return rows
    imposed_unit = _signal_unit(config.role)
    if config.role is not Role.HI_Z:
        prediction = prepared.predictions[name]
        imposed_name, other_name = ("current", "voltage") if config.role is Role.SOURCE_I else ("voltage", "current")
        other_unit = "V" if config.role is Role.SOURCE_I else "A"
        imposed_level, other_level = (observed.i_level, observed.v_level) if config.role is Role.SOURCE_I \
            else (observed.v_level, observed.i_level)
        expected_level, expected_amplitude = _expected(prediction, config.role)
        if config.waveform.periodic:
            imposed_amplitude, other_amplitude = (observed.i_amplitude, observed.v_amplitude) \
                if config.role is Role.SOURCE_I else (observed.v_amplitude, observed.i_amplitude)
            rows.append((imposed_name, Text.assemble(
                "amplitude ", _result(imposed_amplitude, imposed_unit),
                " ", (f"(asked {_show(observed.requested_amplitude, imposed_unit)})", ASKED),
                ", level ", _result(imposed_level, imposed_unit),
                " ", (f"(asked {_show(float(np.mean(prediction.requested)), imposed_unit)})", ASKED))))
            rows.append((other_name, Text.assemble(
                "amplitude ", _result(other_amplitude, other_unit),
                " ", (f"(expected {_show(expected_amplitude, other_unit)})", ASKED),
                ", level ", _result(other_level, other_unit),
                " ", (f"(expected {_show(expected_level, other_unit)})", ASKED))))
            fit = result.i_fit if config.role is Role.SOURCE_I else result.v_fit
            extra = [f"real frequency {_show(result.frequency, 'Hz')}"]
            if fit is not None:
                extra.append(f"delay {_show(fit.delay(result.frequency), 's')}")
                extra.append(f"residual {_show(fit.residual, imposed_unit)} rms")
            if result.impedance is not None:
                extra.append(f"impedance seen {_show(result.impedance, 'Ω')}")
                extra.append(f"phase V-I {math.degrees(result.phase_vi) + 0.0:.2f}°")
            rows.append(("details", Text("; ".join(extra), style=CONTEXT)))
            if result.vs_requested is not None:
                rows.append(("cycle average", Text(
                    f"- asked: {_show(result.vs_requested[0], imposed_unit)} rms, "
                    f"{_show(result.vs_requested[1], imposed_unit)} max; - DAC-delivered: "
                    f"{_show(result.vs_delivered[0], imposed_unit)} rms, "
                    f"{_show(result.vs_delivered[1], imposed_unit)} max", style=CONTEXT)))
        else:
            imposed = result.current if config.role is Role.SOURCE_I else result.voltage
            other = result.voltage if config.role is Role.SOURCE_I else result.current
            rows.append((imposed_name, Text.assemble(
                (f"{_show(imposed.value, imposed_unit)} ± {_show(imposed.u, imposed_unit)}", RESULT),
                " ", (f"(DAC delivers {_show(observed.delivered_level, imposed_unit)}; "
                      f"asked {_show(float(prediction.requested[0]), imposed_unit)})", ASKED))))
            rows.append((other_name, Text.assemble(
                (f"{_show(other.value, other_unit)} ± {_show(other.u, other_unit)}", RESULT),
                " ", (f"(expected {_show(expected_level, other_unit)})", ASKED))))
            rows.append(("spread", Text(f"σ per sample {_show(other.sigma, other_unit)}; drift "
                                        f"{_show(other.drift, other_unit)}", style=CONTEXT)))
        culture = signals.culture_values(config, result)
        if culture:
            parts: list[object] = []
            for k, (key, value) in enumerate(culture.items()):
                label = key.replace("culture_", "").replace("_a", "").replace("_", " ")
                parts += [", " if k else "", f"{label} ", _result(value, "A")]
            rows.append(("at the culture", Text.assemble(*parts)))
    else:
        if result.voltage is not None:
            rows.append(("voltage", Text(f"{_show(result.voltage.value, 'V')} ± {_show(result.voltage.u, 'V')}",
                                         style=RESULT)))
            rows.append(("spread", Text(f"σ per sample {_show(result.voltage.sigma, 'V')}", style=CONTEXT)))
        else:
            rows.append(("voltage", Text.assemble("amplitude ", _result(observed.v_amplitude, "V"),
                                                  ", level ", _result(observed.v_level, "V"))))
        if observed.shunt_level is not None:
            parts = []
            if observed.shunt_amplitude:
                parts += ["amplitude ", _result(observed.shunt_amplitude, "A"), ", "]
            rows.append(("shunt current", Text.assemble(*parts, "level ", _result(observed.shunt_level, "A"))))
            load = result.load_voltage
            if isinstance(load, SineFit):
                rows.append(("load voltage", Text.assemble("amplitude ", _result(load.amplitude, "V"),
                                                           ", level ", _result(load.level, "V"))))
            elif isinstance(load, CycleAverage):
                rows.append(("load voltage", Text.assemble("amplitude ", _result(load.peak_to_peak / 2, "V"),
                                                           ", level ", _result(load.level, "V"))))
            elif load is not None:
                rows.append(("load voltage", Text(f"{_show(load.value, 'V')} ± {_show(load.u, 'V')}", style=RESULT)))
        elif isinstance(result.shunt, analysis.Summary):
            rows.append(("shunt current", Text(f"{_show(result.shunt.value, 'A')} ± {_show(result.shunt.u, 'A')}",
                                               style=RESULT)))
        elif result.shunt is not None:
            amplitude = result.shunt.amplitude if isinstance(result.shunt, SineFit) else result.shunt.peak_to_peak / 2
            rows.append(("shunt current", Text.assemble("amplitude ", _result(amplitude, "A"),
                                                        ", level ", _result(result.shunt.level, "A"))))
    return rows


HOW = ("The first period (or the first 100 ms block for DC) is left out while the channel settles. A "
       "sine's level, amplitude and phase come from a fit of level + sine + cosine at the asked frequency; "
       "other waveforms from the cycle average (the periods laid over each other, phase by phase; "
       "amplitude = half the peak-to-peak). A DC level is the mean of the 100 ms block means, and its ± is "
       "their standard deviation (method {method}, spec §8). The ± shown is statistical only: the "
       "calibration's uncertainty is not included (an open decision). Expected values use Ohm's law with "
       "the informed resistance and the return read at rest.")


# --- Signals: before switching on -----------------------------------------------------------------


def before_lines(prepared: Prepared, view: View = FILE) -> list[Text]:
    """What the operator reads before switching on: the channels and every before-check, in full
    (the operator decides on them)."""
    lines = section("Before switching on", view)
    for name in hw.CHANNELS:
        lines += channel_header(name, prepared.config(name), view.width)
    for check in prepared.checks:
        lines.extend(check_lines(check, "  ", view.width))
    return lines


def switch_lines(prepared: Prepared, confirmed: bool, view: View = FILE) -> list[Text]:
    """The verdict before the burst: SWITCHING ON (what is applied), MEASURING, or NOT SWITCHED ON
    (why: an error, or no --confirm-output)."""
    lines = [Text()]
    if prepared.blocked:
        errors = sum(check.level is Level.ERROR for check in prepared.checks)
        return lines + verdict("NOT SWITCHED ON", BAD, [
            (f"{errors} before-check error{'s' if errors > 1 else ''} (✗ above): the outputs stay off.", DANGER),
            " Fix the circuit or the description, then run the command again."], view)
    if prepared.measure_only:
        return lines + verdict("MEASURING", LIVE, [
            f"nothing is sourced: both channels stay in HI_Z for {prepared.chunks * hw.CHUNK_SECONDS:.1f} s"], view)
    if not confirmed:
        return lines + verdict("NOT SWITCHED ON", STOP, [
            "without ", ("--confirm-output", DANGER), ". Read the checks above; to switch on, run the same "
            "command with ", ("--confirm-output", DANGER), " (your responsibility)."], view)
    applied = " · ".join(describe_channel(name, prepared.config(name)) for name in hw.CHANNELS)
    lines += verdict("SWITCHING ON", LIVE, [applied], view)
    for name in hw.CHANNELS:
        if prepared.config(name).circuit.kind is CircuitKind.IVI_BOARD:
            lines += wrapped(Text(f"CH {name}: the i-v-i board's output may be connected to a culture.",
                                  style=ATTENTION), "  ", view.width)
    return lines


def before_text(prepared: Prepared, confirmed: bool = False) -> str:
    return plain(before_lines(prepared) + switch_lines(prepared, confirmed))


# --- Signals: the result --------------------------------------------------------------------------


def verdict_style(found: list[Check], failed: bool = False, look: bool = False) -> str:
    """The badge's color is the verdict, the same rule in every command (spec 3.1): red when it
    failed (an error, or `failed`); yellow when the operator must look above (a warning, or
    `look`); green when there is nothing to look at. Notes do not change it."""
    if failed or any(check.level is Level.ERROR for check in found):
        return BAD
    if look or any(check.level is Level.WARNING for check in found):
        return STOP
    return GOOD


def _warnings(found: list[Check]) -> list[str | tuple[str, str]]:
    """"; 2 warnings (⚠ above)" next to a yellow COMPLETE: why it is yellow."""
    count = sum(check.level is Level.WARNING for check in found)
    if not count:
        return []
    return ["; ", (f"{count} warning{'s' if count > 1 else ''} (⚠ above)", ATTENTION)]


def burst_verdict(measurement: Measurement, view: View = FILE) -> list[Text]:
    """COMPLETE, STOPPED or FAILED, with the duration, "both channels are in HI_Z" and the count of
    errors, warnings and notes of every check (before and after). A COMPLETE with warnings is
    yellow, never green."""
    prepared = measurement.prepared
    found = prepared.checks + measurement.after
    done = f"{measurement.samples.count / hw.SAMPLE_RATE:.1f} s of {prepared.chunks * hw.CHUNK_SECONDS:.1f} s"
    failed = measurement.overcurrent or any(check.level is Level.ERROR for check in measurement.after)
    if measurement.ended != "complete":
        word, style, what = "STOPPED", verdict_style(found, failed, look=True), [(f"by the operator at {done}", DANGER)]
    elif failed:
        word, style, what = "FAILED", BAD, [(f"{done}; " + ("the limiter acted" if measurement.overcurrent else
                                                          "an after-check gave an error (✗ above)"), DANGER)]
    else:
        word, style = "COMPLETE", verdict_style(found)
        what = [done + ("; nothing sourced" if prepared.measure_only else ""), *_warnings(found)]
    return verdict(word, style, what + [" · ", ("both channels are in HI_Z", SAFE)], view,
                   [[("checks: ", CONTEXT), *problems(found)]])


def next_lines(folder: Path | None, view: View, commands: list[tuple[str, list[str]]],
               note: list[str | tuple[str, str]] | None = None,
               cautions: dict[str, list[str | tuple[str, str]]] | None = None) -> list[Text]:
    """What the operator does next, in a column of labels: the folder (its full path on the
    terminal; its name in a file, which is in it), then commands to copy, each with its caution
    under it when it has one, then a note."""
    column = " " * (NEXT_WIDTH + 3)
    lines = []
    if folder is not None:
        lines += labeled("Folder", Text(str(folder) if view.terminal else Path(folder).name), view.width, "  ",
                         NEXT_WIDTH)
    for label, words in commands:
        lines += command_lines(words, "  ", view.width, label, NEXT_WIDTH)
        if cautions and label in cautions:
            lines += wrapped(Text.assemble(*cautions[label]), width=view.width, lead=Text(column), hang=column)
    if note:
        lines += wrapped(Text.assemble(*note), width=view.width, lead=Text(column), hang=column)
    return lines


def summary_lines(measurement: Measurement, folder: Path | None, view: View = FILE,
                  repeat: list[str] | None = None, before_shown: bool = False) -> list[Text]:
    """The result of a burst (spec 3.3.7), ordered for the screen: the result, the checks, then the
    verdict and the next actions (folder, plot, repeat) last. `before_shown`: the before-checks
    are on the screen above (a burst just run), so they are only counted."""
    prepared = measurement.prepared
    lines: list[Text] = []
    if not (view.terminal and before_shown):
        lines += wrapped(Text(" · ".join(describe_channel(name, prepared.config(name)) for name in hw.CHANNELS),
                              style=HEADING), "", view.width)
        lines += wrapped(Text(f"{measurement.description} · started {measurement.started:%Y-%m-%d %H:%M:%S}",
                              style=CONTEXT), "", view.width)
    lines += section("Result", view)
    for k, name in enumerate(hw.CHANNELS):
        lines += ([Text()] if k else []) + channel_header(name, prepared.config(name), view.width)
        for label, value in _channel_rows(name, prepared.config(name), measurement.results[name], prepared):
            lines += labeled(label, value, view.width)
    how = HOW.format(method=analysis.UNCERTAINTY_METHOD)
    if view.terminal:
        lines += [Text()] + wrapped(Text(f"How the numbers were obtained, and every check: {SUMMARY_FILE} in the "
                                         f"folder.", style=CONTEXT), "", view.width)
    else:
        lines += section("How the numbers were obtained", view) + wrapped(how, "", view.width, "")
    if not (view.terminal and before_shown):
        lines += section("Before switching on", view) + checks_block(prepared.checks, view)
    lines += section("After the measurement", view) + checks_block(measurement.after, view)
    if view.terminal and before_shown:
        lines += wrapped(Text.assemble(("Before switching on: ", CONTEXT), *tally(prepared.checks),
                                       (", shown above", CONTEXT)), "  ", view.width)
    lines += [Text()] + burst_verdict(measurement, view)
    if not view.terminal:
        lines += labeled("Files", Text(f"{SAMPLES_FILE}, {METADATA_FILE}, {SUMMARY_FILE}"), view.width, "  ",
                         NEXT_WIDTH)
    commands = [("Plot", plot_command(Path(folder)))] if folder is not None else []
    note = None
    if repeat is not None:
        commands.append(("Repeat", repeat))
        note = [("it shows the before-checks first; add ", CONTEXT), ("--confirm-output", DANGER),
                (" to switch on", CONTEXT)]
    return lines + next_lines(folder, view, commands, note)


def summary_text(measurement: Measurement, folder: Path | None, repeat: list[str] | None = None) -> str:
    """The result as plain text and complete (summary.txt)."""
    return plain(summary_lines(measurement, folder, FILE, repeat))


# --- Instrument status ----------------------------------------------------------------------------


def status_lines(description: str, records: list[Record], rest: dict[str, float], board: dict[str, int],
                 view: View = FILE) -> list[Text]:
    """The instrument status (spec 3.2): identification, the 8 stored records, then the rest voltages
    and the board check."""
    lines = wrapped(Text(f"PhycoVolt · instrument status · {description}", style=CONTEXT), "", view.width)
    lines += section("Stored calibration (offset; gain+; gain-)", view)
    for (channel, kind), record, check in zip(hw.RECORD_ORDER, records, checks.stored_calibration_checks(records)):
        lines.append(Text.assemble("  ", (checks.SYMBOL[check.level], LEVEL_STYLE[check.level]),
                                   (f" CH {channel} {kind.value:9s} {record.offset:+.7f}; {record.gain_pos:.6f}; "
                                    f"{record.gain_neg:.6f}", BODY_STYLE[check.level])))
    lines.append(Text())
    lines += labeled("Rest", Text.assemble("CH A ", (f"{rest['A']:.4f} V", RESULT), " · CH B ",
                                           (f"{rest['B']:.4f} V", RESULT)), view.width, "", 12)
    swaps = sum(board.values())
    lines += labeled("Board check", Text("no V/I swap (2 s at rest)", style=SAFE) if not swaps else Text(
        f"{swaps} V/I swaps: unplug the M1K's USB cable, wait a few seconds and plug it in again", style=DANGER),
        view.width, "", 12)
    return lines


def status_text(description: str, records: list[Record], rest: dict[str, float], board: dict[str, int]) -> str:
    return plain(status_lines(description, records, rest, board))


# --- Calibration folders and the report (spec 3.7, B.2) -------------------------------------------

CALIB_FILE = "calib.txt"
BLOCKS_FILE = "blocks.txt"
READINGS_FILE = "readings.csv"
DMM_FILE = "dmm.csv"
REPORT_FILE = "report.txt"
READING_COLUMNS = ("time", "stage", "point", "attempt", "status", "channel", "quantity", "mean", "u", "sigma",
                   "drift", "r1", "blocks", "jumps", "adc_limit_fraction", "dmm_mean", "dmm_u", "dmm_readings",
                   "vi_swaps", "halved_words")
DMM_COLUMNS = ("time", "stage", "point", "attempt", "function", "range", "value")


def calibrations_folder(output: Path | None = None) -> Path:
    """Where calibrations go: `<output>/calibrations`, or the project root's `calibrations/`."""
    return (output if output is not None else PROJECT_ROOT) / "calibrations"


def is_calibration(folder: Path) -> bool:
    """A calibration folder has its report next to its metadata; a recording has samples instead."""
    return (Path(folder) / REPORT_FILE).is_file() and (Path(folder) / METADATA_FILE).is_file()


def read_calibration(folder: Path) -> dict:
    """A calibration folder's results as its metadata.json holds them (for the plot, spec 3.7):
    per stage and channel, the effect table, the verifications and the budget."""
    meta = json.loads((Path(folder) / METADATA_FILE).read_text(encoding="utf-8"))
    if not isinstance(meta.get("results"), dict) or "stored_calibration" not in meta:
        raise ValueError(f"{folder} is not a calibration folder")
    return meta


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _csv(rows: list, columns: tuple[str, ...]) -> str:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_cell(getattr(row, name)) for name in columns))
    return "\n".join(lines) + "\n"


def _block_lines(channel: str, result: "calibration.ChannelCalibration") -> list[str]:
    record = result.record
    return [f"# Channel {channel}, {result.kind.value}   ->  offset {record.offset:.7g}, "
            f"gain+ {record.gain_pos:.7f}, gain- {record.gain_neg:.7f}",
            "</>", *(f"<{a}, {b}>" for a, b in result.block.pairs), "<\\>"]


def _shown(value: float, unit: str) -> str:
    return units.show(value, unit, 4)


def wiring_text(step: "calibration.WiringStep", profile: "multimeter.Profile") -> str:
    """What the operator mounts for a calibration step, with the profile's terminal names."""
    terminals = profile.terminals
    full_scale = units.show(step.full_scale, "A") if step.full_scale is not None else ""
    return f"[{step.stage}] " + checks.WIRING_TEXTS[step.kind].format(
        pin=step.pin, channel=f"CH {step.channel}", voltage=terminals["voltage"], common=terminals["common"],
        current=terminals["current"], range=full_scale)


WARM_UP_QUESTION = "Has the M1K been on for 15 min and the multimeter for 1 h (H-11)?"


def acquiring_text(stage: str, point: str, attempt: int, seconds: float, typed: bool) -> str:
    """One acquisition starts; in typed mode, when to read the display."""
    text = f"{stage} · {point} · attempt {attempt} · {seconds:g} s"
    if typed:
        low, high = (fraction * seconds for fraction in checks.TYPED_WINDOW)
        text += f" (typed mode: read the display between {low:g} and {high:g} s)"
    return text


def typed_hint(request: "calibration.TypedRequest") -> str:
    """Typed mode: how many decimals to type (all the display shows on this range)."""
    return (f"{request.stage} · {request.point}: the display shows {request.decimals} decimals in "
            f"{request.unit} on this range: type them all.")


def h06_note() -> str:
    """Why a source-V verification at 4.5 V fails, and the sizes recorded (H-06)."""
    sizes = hw.load_unit().source_nonlinearity
    return (f"Expected cause: the voltage source leaves its 0-2.5 V line near 5 V (H-06), which two "
            f"calibration points cannot correct. On 2026-11-21 it was {_shown(sizes['A'], 'V')} on CH A and "
            f"{_shown(sizes['B'], 'V')} on CH B at {checks.VERIFY_SOURCE_V:g} V. It stays a failure: the "
            f"linear record does not hold there.")


def calibration_report_lines(run: "calibration.CalibrationRun", started: datetime, description: str,
                             calib_path: Path | None, base: Path | None, view: View = FILE,
                             folder: Path | None = None) -> list[Text]:
    """The report (terminal and report.txt): per stage, opened by a rule, the calib.txt lines, the
    coefficients, the effect table, each verification's PASS or FAIL, the budget and the notes;
    then the verdict and what to do next (the smu -w command) last."""
    from phycovolt import calibration as cal
    width = view.width
    lines = wrapped(Text(f"PhycoVolt calibration · {description} · started {started:%Y-%m-%d %H:%M:%S}",
                         style=CONTEXT), "", width)
    lines.extend(wrapped(Text(f"Multimeter: {run.multimeter_id}", style=CONTEXT), "", width))
    verifications = []
    for stage, result in run.stages.items():
        lines += section(stage, view)
        for channel, item in result.channels.items():
            unit = cal.UNIT[item.kind]
            lines += [Text()] + wrapped(Text(f"CH {channel} {item.kind.value}", style=HEADING), "", width)
            lines.append(Text("The calib.txt lines:", style=CONTEXT))
            lines += [Text(line) for line in _block_lines(channel, item)]
            u_offset, u_pos, u_neg = item.record_u
            lines.extend(wrapped(Text.assemble(
                "Estimated record ± U: ", (f"offset {_shown(item.estimate.offset, unit)} ± {_shown(2 * u_offset, unit)}, "
                f"gain+ {item.estimate.gain_pos:.6f} ± {2 * u_pos:.6f}"
                + (f", gain- {item.estimate.gain_neg:.6f} ± {2 * u_neg:.6f}" if u_neg else ""), RESULT)), "", width))
            lines.extend(wrapped(Text(f"Error at each value ({item.before_label} → with the new record ± U):",
                                      style=CONTEXT), "", width))
            for row in item.effect:
                lines.append(Text.assemble(f"  {_shown(row.x, unit):>12}   {_shown(row.before, unit):>12}  →  ",
                                           (f"{_shown(row.after, unit):>11} ± {_shown(2 * row.u, unit)}", RESULT)))
            for check in item.verifications:
                verifications.append(check)
                lines += verdict("PASS" if check.passed else "FAIL", GOOD if check.passed else BAD, [
                    f"verification at {check.name}: residual {_shown(check.residual, unit)}, limit "
                    f"{_shown(check.limit, unit)} (U + one step)"], view)
                if item.kind is hw.Kind.SOURCE_V and not check.passed:
                    lines.extend(wrapped(h06_note(), "  ", width))
            if item.budget:
                lines.append(Text("Uncertainty budget at the main point:", style=CONTEXT))
                for name, share in item.budget[:5]:
                    lead = Text(f"  {share * 100:5.1f} %  ")
                    lines.extend(wrapped(name, width=width, hang=" " * len(lead.plain), lead=lead))
            if item.slope is not None:
                lines.extend(wrapped(f"The zero's slope with the channel voltage: {_shown(item.slope, 'A')} per V "
                                     f"(H-07)", "", width))
            if item.kind is hw.Kind.SOURCE_V:
                lines.extend(wrapped(f"DAC floor: asking 0 V gives {_shown(item.effect[0].after, 'V')} with the "
                                     f"new record (H-05)", "", width))
                ceiling = float(libsmu_math.source_output(hw.CODE_MAX, hw.Kind.SOURCE_V, item.estimate))
                lines.extend(wrapped(f"DAC ceiling: the top code gives {_shown(ceiling, 'V')}; nothing above it "
                                     f"(H-16)", "", width))
        if result.notes:
            lines.append(Text())
        for note in result.notes:
            lines.extend(check_lines(note, "", width))
    if run.warnings:
        lines += section("Warnings kept by the operator", view)
        for warning in run.warnings:
            lines.extend(check_lines(warning, "  ", width))
    lines += [Text()] + wrapped(Text(f"The ± are expanded uncertainties (U = 2u). The M1K's part uses method "
                                     f"{analysis.UNCERTAINTY_METHOD}: the standard deviation of the 100 ms block "
                                     f"means.", style=CONTEXT), "", width)
    if base is not None and calib_path is not None and len(run.stages) < len(cal.STAGES):
        lines.extend(wrapped(Text(f"The blocks of the stages not run come from {base}.", style=CONTEXT), "", width))
    failed = sum(not check.passed for check in verifications)
    counted = [f"{len(verifications) - failed} pass"] + ([", ", (f"{failed} FAIL (above)", DANGER)] if failed else []) \
        if verifications else ["none"]
    stages = ", ".join(run.stages) or "no stage"
    kept = list(run.warnings) + [note for result in run.stages.values() for note in result.notes]
    if run.interrupted:
        word, style, what = "STOPPED", STOP, [("by the operator", DANGER), f" · finished: {stages}"]
    elif run.stopped is not None:
        word, style, what = "STOPPED", BAD, [(run.stopped.text, DANGER), f" · finished: {stages}"]
    else:
        word, style, what = "COMPLETE", verdict_style(kept, look=bool(failed)), [stages, *_warnings(kept)]
    lines += [Text()] + verdict(word, style, what + [" · ", ("both channels are in HI_Z", SAFE)], view,
                                [[("verifications: ", CONTEXT), *counted]])
    commands = []
    note = None
    cautions = {}
    if calib_path is not None:
        commands = [("Write", ["smu", "-w", str(calib_path)]), ("Check", ["smu", "-d"])]
        note = [("The program never writes the EEPROM: smu -w writes all 8 blocks. Then compare each record "
                 "that smu -d shows with the float32 coefficients in the file's comments.", CONTEXT)]
        if failed:
            cautions["Write"] = [(f"a verification failed (above): write it only if you accept the record "
                                  f"there", ATTENTION)]
    lines += next_lines(folder, view, commands, note, cautions)
    if calib_path is None:
        lines += wrapped(Text("No complete calib.txt: not every stage ran and there was no base file. blocks.txt "
                              "has this run's blocks (it is not for smu -w, which needs all 8).", style=ATTENTION),
                         "  ", width)
    return lines


def calibration_report(run: "calibration.CalibrationRun", started: datetime, description: str,
                       calib_path: Path | None, base: Path | None, folder: Path | None = None) -> str:
    """The report as plain text (report.txt)."""
    return plain(calibration_report_lines(run, started, description, calib_path, base, FILE, folder))


def write_calibration_folder(run: "calibration.CalibrationRun", base: Path | None, output: Path, started: datetime,
                             description: str, profile_name: str, command: dict | None = None) -> Path:
    """Write a calibration folder; returns it. The base calib.txt is only read, never written.
    `command` is the command line that ran it (spec 3.1): {"typed": …, "repeat": …}."""
    from phycovolt import calibration as cal
    base_blocks = libsmu_math.parse_calib(base.read_text(encoding="utf-8")) if base is not None and base.exists() else None
    complete = cal.complete_calib(run, base_blocks)
    name = "all" if len(run.stages) == len(cal.STAGES) else "-".join(run.stages) or "none"
    if run.stopped is not None:                          # it says so in its name
        name += "_interrupted"
    output.mkdir(parents=True, exist_ok=True)
    folder = free_folder(output, started, name)
    partial = output / (INCOMPLETE_PREFIX + folder.name)
    partial.mkdir()
    header = [f"Calibration of the ADALM1000 · PhycoVolt {__version__} · {started:%Y-%m-%d %H:%M}",
              f"{description} · multimeter {run.multimeter_id} ({profile_name})",
              "Pairs: measure = <reference, M1K>; source = <request, multimeter>; V and A; the zero first.",
              f"Stages measured now: {', '.join(run.stages) or 'none'}"
              + (f"; the other blocks from {base.name}" if base is not None and len(run.stages) < len(cal.STAGES) else "")]
    files = []
    if complete is not None:
        text = libsmu_math.format_calib(complete, header)          # refuses what libsmu would misread
        (partial / CALIB_FILE).write_text(text, encoding="utf-8", newline="\n")
        files.append(CALIB_FILE)
    blocks = ["# This run's blocks only. NOT for smu -w, which needs all 8 blocks: use calib.txt.", "#"]
    for stage, result in run.stages.items():
        for channel, item in result.channels.items():
            blocks += [""] + _block_lines(channel, item)
    (partial / BLOCKS_FILE).write_text("\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
    (partial / READINGS_FILE).write_text(_csv(run.readings, READING_COLUMNS), encoding="utf-8", newline="\n")
    (partial / DMM_FILE).write_text(_csv(run.dmm_rows, DMM_COLUMNS), encoding="utf-8", newline="\n")
    files += [BLOCKS_FILE, READINGS_FILE, DMM_FILE, REPORT_FILE, METADATA_FILE]
    calib_path = folder / CALIB_FILE if complete is not None else None
    (partial / REPORT_FILE).write_text(calibration_report(run, started, description, calib_path, base, folder),
                                       encoding="utf-8", newline="\n")
    meta = {
        "program": {"name": "PhycoVolt", "version": __version__, "git_commit": git_commit()},
        "started": started.isoformat(timespec="seconds"),
        "instrument": description,
        "multimeter": {"profile": profile_name, "identification": run.multimeter_id},
        "stored_calibration": [[r.offset, r.gain_pos, r.gain_neg] for r in run.stored],
        "stages": list(run.stages),
        "ended": "interrupted" if run.interrupted else "stopped" if run.stopped is not None else "complete",
        "interrupted": run.interrupted or run.stopped is not None,
        "stopped": check_to_dict(run.stopped) if run.stopped is not None else None,
        "base": str(base) if base is not None else None,
        "command": command,
        "uncertainty_method": analysis.UNCERTAINTY_METHOD,
        "results": {stage: {channel: {
            "kind": item.kind.value, "pairs": [list(pair) for pair in item.block.pairs],
            "record": [item.record.offset, item.record.gain_pos, item.record.gain_neg],
            "estimate": [item.estimate.offset, item.estimate.gain_pos, item.estimate.gain_neg],
            "record_u": list(item.record_u),
            "effect": [{"x": r.x, "before": r.before, "after": r.after, "u": r.u} for r in item.effect],
            "verifications": [{"name": v.name, "residual": v.residual, "u": v.u, "limit": v.limit,
                               "passed": v.passed} for v in item.verifications],
            "budget": [[name, share] for name, share in item.budget],
            "slope_a_per_v": item.slope,
        } for channel, item in result.channels.items()} for stage, result in run.stages.items()},
        "warnings": [check_to_dict(w) for w in run.warnings],
        "files": files,
    }
    (partial / METADATA_FILE).write_text(json.dumps(_clean(meta), indent=1) + "\n", encoding="utf-8", newline="\n")
    partial.rename(folder)
    return folder
