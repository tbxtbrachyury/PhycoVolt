"""The entry point: the command line (spec 3.1), the only file that prints.

Question it answers: what did the operator ask for on the command line, what does the terminal show
of it, and which exit code says how it went (0 done; 1 could not start; 130 interrupted)?

Main choices:

- This file only asks, calls and shows; the work is in signals.py, calibration.py and recording.py,
  whose texts come as lines marked by meaning (recording.DANGER, RESULT, CONTEXT…). `THEME` gives
  each meaning its look, with rich; the files stay plain. Styles are off when the output is not a
  terminal or when NO_COLOR is set (no escape codes at all, spec 3.1).
- Without a subcommand: the start page, which opens no instrument. Its examples and those at the
  end of each `--help` come from `EXAMPLES`, which the tests run through the parser.
- Nothing is switched on without --confirm-output (the operator's responsibility): without it the
  command stops after the before-checks. Every recording stores its command written out in full,
  and its summary ends with the command to repeat it, without --confirm-output (spec 3.1).
- The burst's progress line is redrawn by the burst loop at each chunk (rich, no refresh thread);
  without a terminal, one plain line per second. Ctrl+C is the Stop: SIGTERM and SIGHUP (where it
  exists) become the same interrupt, so both channels go to HI_Z before the program ends (3.9).
- The real instrument is imported only when --simulate is absent, so a simulated run never touches
  pysmu; a simulated run without --output writes to the project's scratch/.
- `plot` starts the window as its own process and does not wait for it, so the terminal is free at
  once (spec 3.3.7): the program's own Python, a list of arguments, no shell.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text
from rich.theme import Theme

from phycovolt import __version__, calibration, checks, circuits, libsmu_math, multimeter, recording, signals, \
    simulator, units
from phycovolt import hardware as hw
from phycovolt.checks import Level
from phycovolt.hardware import Instrument, InstrumentError
from phycovolt.recording import (ACTION, ASKED, ATTENTION, BAD, COMMAND, CONTEXT, DANGER, GOOD, HEADING, LABEL, LIVE,
                                 RESULT, RULE, SAFE, STOP)
from phycovolt.units import UnitError

EXIT_DONE = 0
EXIT_NOT_STARTED = 1
EXIT_INTERRUPTED = 130
PROGRESS_EVERY_CHUNKS = 10            # without a terminal: one progress line per second

# How each meaning looks on a terminal (spec 3.1): the 16 standard colors, which follow the
# terminal's own dark or light palette. A badge is a word on a filled background, with its text
# color given (not reverse video, whose white-on-yellow would not read on a light palette).
THEME = Theme({
    DANGER: "bold red",
    ATTENTION: "yellow",
    RESULT: "bold",
    ASKED: "dim",
    CONTEXT: "dim",
    SAFE: "green",
    COMMAND: "cyan",
    HEADING: "bold",
    LABEL: "dim",
    RULE: "dim",
    ACTION: "bold",
    GOOD: "bold black on green",
    LIVE: "bold white on blue",
    STOP: "bold black on yellow",
    BAD: "bold white on red",
})


class NotStarted(Exception):
    """Something the operator must fix before anything can run; the message says what."""


# --- The terminal ---------------------------------------------------------------------------------


class Terminal:
    """Standard output and error through rich, styled by THEME only on a terminal without NO_COLOR."""

    def __init__(self) -> None:
        plain = bool(os.environ.get("NO_COLOR"))            # no-color.org: set and not empty
        options = dict(theme=THEME, highlight=False, emoji=False, markup=False,
                       color_system=None if plain else "auto")
        self.out = Console(**options)
        self.err = Console(stderr=True, **options)

    @property
    def width(self) -> int:
        """The width the texts are wrapped at: the terminal's, at most 100 columns."""
        return min(self.out.width, recording.WIDTH)

    @property
    def view(self) -> recording.View:
        """What is printed is the terminal's view (spec 3.1, Hierarchy), piped or not."""
        return recording.View(terminal=True, width=self.width)

    def lines(self, lines: list[Text], err: bool = False) -> None:
        console = self.err if err else self.out
        for line in lines:
            console.print(line, soft_wrap=True)              # wrapped already; a long command stays whole

    def say(self, *parts: str | tuple[str, str], indent: str = "", err: bool = False) -> None:
        """One message, wrapped at the width with a hanging indent."""
        self.lines(recording.wrapped(Text.assemble(*parts), indent, self.width), err)


_terminal: Terminal | None = None


def terminal() -> Terminal:
    global _terminal
    if _terminal is None:
        _terminal = Terminal()
    return _terminal


# --- Signals: Ctrl+C, SIGTERM, SIGHUP -------------------------------------------------------------


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    """SIGTERM and SIGHUP behave like Ctrl+C: the running burst ends with both channels in HI_Z."""
    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGHUP"):                  # not on Windows (NF-13)
        signal.signal(signal.SIGHUP, _interrupt)


# --- The commands, their examples, and the start page ---------------------------------------------

parse_channel = signals.parse_channel
board_check = signals.board_check

COMMANDS = {
    "status": "the M1K, its stored calibration and its rest state",
    "signals": "generate and/or measure signals",
    "calibrate": "calibrate the M1K against the multimeter",
    "analyze": "the summary of a recording, again",
    "plot": "the plot window of a recording",
}
A_FOLDER = "recordings/2026-11-22_101500_source-i-sine"


@dataclass(frozen=True)
class Example:
    """What an example does, its commands (each a list of words) and a line after them."""

    title: str
    commands: tuple[tuple[str, ...], ...]
    after: tuple[str | tuple[str, str], ...] = ()
    start_page: bool = True


EXAMPLES = [
    Example("Both channels measured at rest for 2 s (nothing is sourced):",
            (("phycovolt", "signals", "--a", "hi-z", "--b", "hi-z", "--duration", "2 s"),)),
    Example("A 100 Hz current sine on CH A, through 10 Ω to the 2.5V pin; the checks first:",
            (("phycovolt", "signals", "--a", "source-i sine freq=100Hz pp=2mA circuit=resistor r=10 return=2.5V"),),
            ("then the same command with ", ("--confirm-output", DANGER), " switches it on.")),
    Example("1 V on CH A into 47 Ω and a 10 Ω shunt to GND; CH B measures the node between:",
            (("phycovolt", "signals", "--a", "source-v dc level=1 circuit=load-shunt load=47 shunt=10 return=GND",
              "--b", "hi-z circuit=shunt-node"),)),
    Example("A 10 Hz charge-balanced pulse into the i-v-i board (its scale factor is yours):",
            (("phycovolt", "signals", "--a", "source-i biphasic amplitude=2mA circuit=ivi-board scale=0.01"),),
            start_page=False),
    Example("Calibrate the voltage measurement, typing the multimeter's readings:",
            (("phycovolt", "calibrate", "--stage", "measure-v", "--typed"),)),
    Example("All four stages with the multimeter on RS-232, the optional points skipped:",
            (("phycovolt", "calibrate", "--skip-optional"),), start_page=False),
    Example("The instrument, its stored calibration and its rest state:", (("phycovolt", "status"),),
            start_page=False),
    Example("A recording's summary again, and its plot window:",
            (("phycovolt", "analyze", A_FOLDER), ("phycovolt", "plot", A_FOLDER))),
]


def examples_for(command: str | None) -> list[Example]:
    """The examples of one subcommand (its --help), or of the start page (None)."""
    if command is None:
        return [example for example in EXAMPLES if example.start_page]
    return [example for example in EXAMPLES if any(words[1] == command for words in example.commands)]


def example_lines(examples: list[Example], width: int) -> list[Text]:
    """Each example: its description dimmed, its commands to copy, a blank line between them."""
    lines: list[Text] = []
    for k, example in enumerate(examples):
        lines += ([Text()] if k else []) + recording.wrapped(Text(example.title, style=CONTEXT), "  ", width)
        for words in example.commands:
            lines += recording.command_lines(list(words), "    ", width)
        if example.after:
            lines += recording.wrapped(Text.assemble(*example.after), "    ", width)
    return lines


def epilog(command: str | None) -> str:
    """The examples at the end of a --help, plain (argparse prints them as they are)."""
    return "examples:\n" + recording.plain(example_lines(examples_for(command), 80))


def start_page(width: int) -> list[Text]:
    """Without a subcommand: what the program does, its commands and examples (no instrument)."""
    view = recording.View(terminal=True, width=width)
    lines = recording.wrapped(Text.assemble((f"PhycoVolt {__version__}", HEADING), (
        " · calibrate the ADALM1000; generate and measure signals with it", CONTEXT)), "", width)
    lines += recording.section("Commands", view)
    for name, text in COMMANDS.items():
        lines += recording.wrapped(Text.assemble((f"{name:10s}", COMMAND), "  ", text), "  ", width, " " * 14)
    lines += recording.section("Examples", view)
    lines += recording.wrapped(Text.assemble(
        "Add ", ("--simulate", COMMAND), " after phycovolt to try one without the instrument (its results "
        "go to scratch/)."), "", width, "") + [Text()]
    lines += example_lines(examples_for(None), width)
    lines += [Text()] + recording.wrapped(Text.assemble(
        "Help and more examples for one command: ", ("phycovolt signals --help", COMMAND)), "", width)
    lines += recording.wrapped(Text("Ctrl+C stops at once: both channels go to HI_Z.", style=SAFE), "", width)
    return lines


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--simulate", action="store_true", default=argparse.SUPPRESS,
                        help="run against the simulated instrument (no M1K is touched)")
    common.add_argument("--output", type=Path, default=argparse.SUPPRESS,
                        help="folder for recordings and calibrations (default: the project root; "
                             "with --simulate, the project's scratch/)")
    raw = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="phycovolt", parents=[common], formatter_class=raw,
                                     description="Calibrate the ADALM1000 and generate and measure signals.",
                                     epilog=epilog(None))
    parser.add_argument("--version", action="version", version=f"PhycoVolt {__version__}")
    commands = parser.add_subparsers(dest="command")

    def command(name: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, parents=[common], help=COMMANDS[name], description=COMMANDS[name],
                                   epilog=epilog(name), formatter_class=raw)

    command("status")
    run = command("signals")
    run.add_argument("--a", default="hi-z", help='channel A, e.g. "source-i sine freq=100Hz pp=2mA"')
    run.add_argument("--b", default="hi-z", help='channel B, e.g. "hi-z"')
    run.add_argument("--confirm-output", action="store_true",
                     help="switch the outputs on (the operator's responsibility)")
    run.add_argument("--duration", default=None,
                     help='measure-only bursts (both channels hi-z): "2 s", 0.3 to 10 s (default 1 s)')
    calibrate = command("calibrate")
    calibrate.add_argument("--stage", default="all", choices=["all", *calibration.STAGES])
    calibrate.add_argument("--dmm", type=Path, default=multimeter.PROFILES / "agilent_34401a.ini",
                           help="the multimeter's profile (profiles/*.ini)")
    calibrate.add_argument("--typed", action="store_true", help="type the multimeter's readings (no link)")
    calibrate.add_argument("--base", type=Path, default=recording.PROJECT_ROOT / "calib.txt",
                           help="the calib.txt whose other blocks complete a partial calibration (only read)")
    calibrate.add_argument("--skip-optional", action="store_true", help="skip the optional and auxiliary points")
    for name in ("analyze", "plot"):
        command(name).add_argument("folder", type=Path)
    return parser


# --- The commands written out in full (spec 3.1) --------------------------------------------------


def global_words(simulate: bool, given_output: Path | None) -> list[str]:
    """--simulate stays (a pasted repeat never drives the real M1K); --output stays when given."""
    words = ["phycovolt"]
    if simulate:
        words.append("--simulate")
    if given_output is not None:
        words += ["--output", str(given_output.resolve())]
    return words


def repeat_signals(simulate: bool, given_output: Path | None, a: circuits.ChannelConfig,
                   b: circuits.ChannelConfig, seconds: float | None) -> list[str]:
    """The command that repeats a burst: every field written out, never --confirm-output."""
    words = global_words(simulate, given_output)
    words += ["signals", "--a", signals.channel_text(a), "--b", signals.channel_text(b)]
    if signals.measure_only(a, b):
        words += ["--duration", units.exact(signals.DEFAULT_MEASURE_SECONDS if seconds is None else seconds, "s")]
    return words


def repeat_calibration(simulate: bool, given_output: Path | None, stage: str, profile: Path, typed: bool,
                       base: Path, skip_optional: bool) -> list[str]:
    words = global_words(simulate, given_output)
    words += ["calibrate", "--stage", stage, "--dmm", str(profile.resolve())]
    words += ["--typed"] if typed else []
    words += ["--base", str(base.resolve())]
    words += ["--skip-optional"] if skip_optional else []
    return words


def command_record(typed: Sequence[str], repeat: list[str]) -> dict:
    return {"typed": recording.command_text(["phycovolt", *typed]), "repeat": recording.command_text(repeat)}


# --- Opening the instrument (spec 3.2) ------------------------------------------------------------


def open_instrument(simulate: bool, bench: simulator.Bench | None = None, pace: bool = True,
                    implausible_ok: bool = False) -> Instrument:
    """The simulated or the real M1K, after the board check; raises NotStarted with the reason.

    An implausible stored record refuses every command but `status`, which opens with
    `implausible_ok` to show the records (spec 3.2); the calls are the same either way."""
    if simulate:
        instrument: Instrument = simulator.SimulatedM1K(bench, pace=pace)
    else:
        from phycovolt import m1k                   # imported only for the real instrument
        try:
            instrument = m1k.open_m1k()
        except InstrumentError as error:
            raise NotStarted(checks.INSTRUMENT_ERROR_TEXTS.get(error.code, str(error))) from error
    try:
        implausible = any(c.level is Level.ERROR for c in checks.stored_calibration_checks(instrument.calibration()))
        if implausible and not implausible_ok:
            raise NotStarted(checks.INSTRUMENT_ERROR_TEXTS["implausible_calibration"])
        board = board_check(instrument)
        if any(board.values()):
            raise NotStarted(checks.INSTRUMENT_ERROR_TEXTS["vi_swap"])
    except BaseException:
        instrument.close()
        raise
    return instrument


# --- The progress line (spec 3.3.6) ---------------------------------------------------------------


class ProgressLine:
    """A bar and the seconds done of the total, redrawn by the burst loop at each chunk: rich's
    Progress with auto_refresh off starts no thread. Without a terminal, one plain line per second.
    Leaving it (also on Ctrl+C) gives the cursor back."""

    def __init__(self, console: Console, total: int) -> None:
        self.console = console
        self.total = total
        self.progress: Progress | None = None
        self.task = 0

    def _seconds(self, done: int) -> str:
        return f"{done * hw.CHUNK_SECONDS:.1f} s of {self.total * hw.CHUNK_SECONDS:.1f} s"

    def _running(self, done: int) -> str:          # markup made of our own words and numbers only
        if done >= self.total:
            return self._seconds(done)
        return f"{self._seconds(done)} · [{COMMAND}]Ctrl+C[/{COMMAND}] stops"

    def _progress(self) -> Progress:
        progress = Progress(TextColumn("  "), BarColumn(bar_width=30), TextColumn("{task.fields[text]}"),
                            console=self.console, auto_refresh=False, transient=False,
                            redirect_stdout=False, redirect_stderr=False)
        self.task = progress.add_task("", total=self.total, text=self._running(0))
        return progress

    def __enter__(self) -> "ProgressLine":
        if self.console.is_terminal:
            self.progress = self._progress()
            self.progress.start()
        return self

    def chunk(self, done: int) -> None:
        if self.progress is not None:
            self.progress.update(self.task, completed=done, text=self._running(done), refresh=True)
        elif done % PROGRESS_EVERY_CHUNKS == 0 or done == self.total:
            self.console.print(Text(f"  {self._seconds(done)}"), soft_wrap=True)

    def stopped(self, done: int) -> None:
        """Ctrl+C: the line says so."""
        if self.progress is not None:
            self.progress.update(self.task, completed=done, refresh=True,
                                 text=f"[{DANGER}]stopped at {self._seconds(done)}[/{DANGER}]")
        else:
            self.console.print(Text.assemble(f"  {self._seconds(done)} · ", ("stopped", DANGER)), soft_wrap=True)

    def __exit__(self, *exc: object) -> None:
        if self.progress is not None:
            self.progress.stop()


# --- Subcommands ----------------------------------------------------------------------------------


def command_start() -> int:
    term = terminal()
    term.lines(start_page(term.width))
    return EXIT_DONE


def command_status(simulate: bool) -> int:
    """The 8 stored records (an implausible one as ✗), the rest voltages and the board check; rest
    reads only. Exit 1 when a record is implausible, with what to do (spec 3.2)."""
    term = terminal()
    instrument = open_instrument(simulate, simulator.Bench(), implausible_ok=True)
    try:
        rest = checks.rest_voltages(instrument.rest_read(checks.REST_SECONDS))
        board = board_check(instrument)
        records = instrument.calibration()
        term.lines(recording.status_lines(instrument.description, records, rest, board, term.view))
    finally:
        instrument.close()
    if any(c.level is Level.ERROR for c in checks.stored_calibration_checks(records)):
        term.lines([Text()])
        term.say((checks.INSTRUMENT_ERROR_TEXTS["implausible_calibration"], DANGER))
        return EXIT_NOT_STARTED
    return EXIT_DONE


def command_signals(simulate: bool, output: Path | None, a_text: str, b_text: str, confirm: bool,
                    duration: str | None = None, given_output: Path | None = None,
                    typed: Sequence[str] = ()) -> int:
    term = terminal()
    try:
        a, b = parse_channel(a_text), parse_channel(b_text)
        circuits.check_pair(a, b)
        seconds = units.value(duration, "s") if duration is not None else None
        signals.burst_chunks(a, b, seconds)
    except (ValueError, UnitError) as error:
        raise NotStarted(f"The channel description cannot be used: {error}") from error
    instrument = open_instrument(simulate, simulator.bench_for(a, b))
    try:
        term.say((f"PhycoVolt · signals · {instrument.description}", CONTEXT))
        try:
            prepared = signals.prepare(instrument, a, b, measure_seconds=seconds)
        except InstrumentError as error:
            raise NotStarted(checks.INSTRUMENT_ERROR_TEXTS.get(error.code, str(error))) from error
        term.lines(recording.before_lines(prepared, term.view) + recording.switch_lines(prepared, confirm, term.view))
        if prepared.blocked:
            return EXIT_NOT_STARTED
        if not confirm and not prepared.measure_only:
            return EXIT_DONE
        with ProgressLine(term.out, prepared.chunks) as line:
            try:
                measurement = signals.run(instrument, prepared, on_chunk=lambda chunk, done, total: line.chunk(done))
            except InstrumentError as error:
                failure = error
            else:
                failure = None
                if measurement.ended != "complete":
                    line.stopped(measurement.samples.count // hw.CHUNK_SAMPLES)
        if failure is not None:                    # spec 3.9: report it; both channels are in HI_Z
            term.lines([Text()] + recording.verdict("FAILED", BAD, [
                (checks.INSTRUMENT_ERROR_TEXTS.get(failure.code, str(failure)), DANGER), " · ",
                ("both channels are in HI_Z", SAFE), " · no recording: run the command again."], term.view))
            return EXIT_NOT_STARTED
        repeat = repeat_signals(simulate, given_output, a, b, seconds)
        folder = recording.write(measurement, recording.recordings_folder(output), command_record(typed, repeat))
        term.lines(recording.summary_lines(measurement, folder, term.view, repeat, before_shown=True))
        return EXIT_INTERRUPTED if measurement.interrupted else EXIT_DONE
    finally:
        instrument.close()


class ConsoleOperator:
    """The operator at the keyboard during a calibration: Enter continues, q stops (spec 3.1). What
    the operator must do (mount, type, which key) is the strongest line."""

    def __init__(self, profile: multimeter.Profile, typed: bool) -> None:
        self.profile = profile
        self.typed = typed
        self.term = terminal()

    def _ask(self, *parts: str | tuple[str, str]) -> str:
        return self.term.out.input(Text.assemble(*parts))

    def wire(self, step: calibration.WiringStep) -> bool:
        self.term.out.print()
        self.term.say((recording.wiring_text(step, self.profile), ACTION))
        return self._ask(("Press ", ACTION), ("Enter", COMMAND), (" when it is mounted", ACTION), " (",
                         ("q", COMMAND), " stops): ").strip().lower() != "q"

    def confirm_warm_up(self) -> bool:
        answer = self._ask((recording.WARM_UP_QUESTION, ACTION), " [y/N] ")
        return answer.strip().lower() in ("y", "yes", "s", "sim")

    def acquiring(self, stage: str, point: str, attempt: int, seconds: float) -> None:
        text = recording.acquiring_text(stage, point, attempt, seconds, self.typed)
        self.term.say((text, ACTION if self.typed else CONTEXT), indent="  ")

    def problem(self, check: checks.Check) -> bool:
        self.term.lines(recording.check_lines(check, "", self.term.width))
        return self._ask(("Fix it and press ", ACTION), ("Enter", COMMAND), (" to try again", ACTION), " (",
                         ("q", COMMAND), " stops the stage): ").strip().lower() != "q"

    def accept_warning(self, check: checks.Check) -> bool:
        self.term.lines(recording.check_lines(check, "", self.term.width))
        return self._ask(("Keep this reading?", ACTION), " [Y/n] ").strip().lower() not in ("n", "no", "nao", "não")

    def typed_value(self, request: calibration.TypedRequest) -> str:
        self.term.say((recording.typed_hint(request), ACTION), indent="  ")
        return self._ask("  ", (f"The multimeter's value ({request.unit}):", ACTION), " ")


class Staged:
    """An operator whose calibration stages each open with a section rule (spec 3.1, Hierarchy)."""

    def __init__(self, operator: object, term: Terminal) -> None:
        self.operator = operator
        self.term = term
        self.stage: str | None = None

    def _open(self, stage: str) -> None:
        if stage != self.stage:
            self.stage = stage
            self.term.lines(recording.section(stage, self.term.view))

    def wire(self, step: calibration.WiringStep) -> bool:
        self._open(step.stage)
        return self.operator.wire(step)

    def acquiring(self, stage: str, point: str, attempt: int, seconds: float) -> None:
        self._open(stage)
        self.operator.acquiring(stage, point, attempt, seconds)

    def __getattr__(self, name: str) -> object:
        return getattr(self.operator, name)


def command_calibrate(simulate: bool, output: Path | None, stage: str, profile_path: Path, typed: bool,
                      base: Path, skip_optional: bool, given_output: Path | None = None,
                      typed_words: Sequence[str] = ()) -> int:
    term = terminal()
    try:
        profile = multimeter.load_profile(profile_path)
    except (OSError, multimeter.ProfileError) as error:
        raise NotStarted(f"The multimeter profile cannot be used: {error}") from error
    stages = list(calibration.STAGES) if stage == "all" else [stage]
    if base.exists():
        try:
            libsmu_math.parse_calib(base.read_text(encoding="utf-8"))
        except ValueError as error:
            raise NotStarted(f"The base {base} cannot be used: {error}") from error
    instrument = open_instrument(simulate, simulator.Bench(), pace=False)
    description = instrument.description
    dmm = None
    try:
        typed = typed or profile.typed
        if simulate:
            virtual = simulator.VirtualMultimeter(instrument, profile)
            dmm = None if typed else virtual
            operator = simulator.SimulatedOperator(instrument, virtual,
                                                   say=lambda text: term.say((text, CONTEXT)))
            clock = instrument.clock
        else:
            if not typed:
                try:
                    dmm = multimeter.open_multimeter(profile)
                except (multimeter.MultimeterError, multimeter.ProfileError) as error:
                    raise NotStarted(f"The multimeter cannot be opened: {error} (or use --typed)") from error
            operator = ConsoleOperator(profile, typed)
            clock = time.monotonic
        term.say((f"PhycoVolt · calibration · {description} · stages: {', '.join(stages)}", CONTEXT))
        context = calibration.Context(instrument, dmm, profile, Staged(operator, term), clock,
                                      instrument.calibration(), skip_optional)
        started = datetime.now()
        try:
            run = calibration.calibrate(context, stages)
        except InstrumentError as error:
            raise NotStarted(checks.INSTRUMENT_ERROR_TEXTS.get(error.code, str(error))) from error
        except multimeter.MultimeterError as error:
            raise NotStarted(f"The multimeter did not answer: {error}") from error
    finally:
        instrument.close()
        if dmm is not None:
            dmm.close()
    used_base = base if base.exists() else None
    repeat = repeat_calibration(simulate, given_output, stage, profile_path, typed, base, skip_optional)
    folder = recording.write_calibration_folder(run, used_base, recording.calibrations_folder(output), started,
                                                description, profile.name, command_record(typed_words, repeat))
    calib_path = folder / recording.CALIB_FILE if (folder / recording.CALIB_FILE).exists() else None
    term.lines(recording.section("Report", term.view))
    term.lines(recording.calibration_report_lines(run, started, description, calib_path, used_base, term.view,
                                                  folder))
    if run.interrupted:
        return EXIT_INTERRUPTED
    return EXIT_NOT_STARTED if run.stopped is not None else EXIT_DONE


def command_analyze(folder: Path) -> int:
    term = terminal()
    try:
        found = recording.read(folder)
    except (OSError, ValueError, KeyError) as error:
        raise NotStarted(f"{folder} cannot be read as a recording: {error}") from error
    repeat = recording.repeat_words(found.metadata.get("command"))
    term.lines(recording.summary_lines(found.measurement, found.folder, term.view, repeat))   # as after a burst
    return EXIT_DONE


def start_plot_window(folder: Path) -> None:
    """`python -m phycovolt.plot <folder>` in its own process, not waited for."""
    subprocess.Popen([sys.executable, "-m", "phycovolt.plot", str(folder.resolve())],
                     cwd=recording.PROJECT_ROOT, stdin=subprocess.DEVNULL)


def command_plot(folder: Path) -> int:
    if not (folder / recording.METADATA_FILE).is_file():
        raise NotStarted(f"{folder} is not a recording or a calibration folder (it has no "
                         f"{recording.METADATA_FILE}).")
    start_plot_window(folder)
    terminal().say(("The plot window opens in its own process; this terminal is free.", CONTEXT))
    return EXIT_DONE


def output_folder(simulate: bool, output: Path | None) -> Path | None:
    """Where results go. A simulated run without --output goes to the project's scratch/, so that
    rehearsals never mix with the operator's real recordings and calibrations."""
    if output is None and simulate:
        return recording.PROJECT_ROOT / "scratch"
    return output


def main(argv: Sequence[str] | None = None) -> int:
    global _terminal
    _terminal = Terminal()                         # the environment of this run (NO_COLOR, a terminal)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    simulate = getattr(args, "simulate", False)
    given_output = getattr(args, "output", None)
    output = output_folder(simulate, given_output)
    install_signal_handlers()
    try:
        if args.command == "status":
            return command_status(simulate)
        if args.command == "signals":
            return command_signals(simulate, output, args.a, args.b, args.confirm_output, args.duration,
                                   given_output, argv)
        if args.command == "calibrate":
            return command_calibrate(simulate, output, args.stage, args.dmm, args.typed, args.base,
                                     args.skip_optional, given_output, argv)
        if args.command == "analyze":
            return command_analyze(args.folder)
        if args.command == "plot":
            return command_plot(args.folder)
        return command_start()
    except NotStarted as reason:
        terminal().say((str(reason), DANGER), err=True)
        return EXIT_NOT_STARTED
    except KeyboardInterrupt:
        term = terminal()
        term.lines([Text()] + recording.verdict("STOPPED", STOP, [("Interrupted by the operator: ", DANGER),
                                                                   ("both channels are in HI_Z.", SAFE)], term.view),
                   err=True)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
