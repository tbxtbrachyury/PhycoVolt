"""The rich command line (spec 3.1, 3.3.6, 9.1): the start page and its examples, the styles by
meaning, the repeat command, and the progress line. Always simulated; the subprocess tests send a
real Ctrl+C (SIGINT) in the middle of a burst.
"""

from __future__ import annotations

import io
import json
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console

from phycovolt import __main__ as cli
from phycovolt import circuits as ci
from phycovolt import recording
from phycovolt import signals
from phycovolt import waveforms as wf
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import Role

PROJECT = Path(__file__).resolve().parents[1]
SGR = re.compile(r"\x1b\[[0-9;]*m")                  # a style escape code
SINE = "source-i sine freq=100Hz pp=2mA circuit=resistor r=10 return=2.5V"
SHORT_DC = "source-i dc level=10mA seconds=0.3"


@pytest.fixture(autouse=True)
def keep_signal_handlers():
    """main() installs SIGTERM and SIGHUP handlers: put pytest's back afterwards."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)


COMMAND_LINE = re.compile(r"^\s*(?:(?:Plot|Repeat)\s+)?(phycovolt .*)$")


@pytest.fixture(scope="module")
def typed_calibration(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path, list[str]]:
    """One simulated typed calibration of measure-v, shared: (its output, its folder, its words)."""
    import contextlib
    output = tmp_path_factory.mktemp("typed")
    words = ["--simulate", "--output", str(output), "calibrate", "--stage", "measure-v", "--typed"]
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            assert cli.main(words) == cli.EXIT_DONE
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)
    [folder] = sorted((output / "calibrations").iterdir())
    return out.getvalue(), folder, words


def commands_in(text: str) -> list[list[str]]:
    """The commands shown in a text (alone on their line, or after a Plot or Repeat label), their
    " \\" continuations joined, split as a shell would."""
    joined = re.sub(r" \\\n\s*", " ", text)
    return [shlex.split(match.group(1)) for match in map(COMMAND_LINE.match, joined.splitlines()) if match]


def accepted(words: list[str]) -> None:
    """The parser takes the command, and a signals command describes channels that can be used."""
    assert words[0] == "phycovolt"
    args = cli.build_parser().parse_args(words[1:])
    if args.command == "signals":
        a, b = cli.parse_channel(args.a), cli.parse_channel(args.b)
        ci.check_pair(a, b)
        seconds = None if args.duration is None else cli.units.value(args.duration, "s")
        signals.burst_chunks(a, b, seconds)


# --- The start page and the examples --------------------------------------------------------------


def test_the_start_page_opens_no_instrument(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the start page opened an instrument")

    monkeypatch.setattr(cli, "open_instrument", never)
    assert cli.main([]) == cli.EXIT_DONE
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert f"  {name} " in out
    assert "--simulate" in out and "phycovolt signals --help" in out
    assert "Ctrl+C stops at once: both channels go to HI_Z." in out


def test_the_start_page_fits_80_columns() -> None:
    assert max(len(line.plain) for line in cli.start_page(80)) <= 80


def test_every_example_is_accepted_by_the_parser() -> None:
    for example in cli.EXAMPLES:
        for words in example.commands:
            accepted(list(words))
            accepted([words[0], "--simulate", *words[1:]])          # how the start page says to try them
    for name in cli.COMMANDS:
        assert cli.examples_for(name), f"{name} --help has no example"


def test_the_examples_as_printed_are_the_ones_tested(capsys: pytest.CaptureFixture) -> None:
    every = [list(words) for example in cli.EXAMPLES for words in example.commands]
    cli.main([])
    shown = commands_in(capsys.readouterr().out)
    assert shown and all(words in every for words in shown)
    for name in cli.COMMANDS:
        with pytest.raises(SystemExit):
            cli.main([name, "--help"])
        help_text = capsys.readouterr().out
        assert "examples:" in help_text
        shown = commands_in(help_text.split("examples:")[1])
        assert shown == [list(words) for example in cli.examples_for(name) for words in example.commands]


# --- Repeatable runs ------------------------------------------------------------------------------


def every_configuration() -> list[ChannelConfig]:
    configs = [ChannelConfig(), ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT, "5V", r=4.7e3))]
    for kind, role in ((CircuitKind.RESISTOR, Role.SOURCE_I), (CircuitKind.LOAD_SHUNT, Role.SOURCE_V)):
        for ret in ci.RETURNS[kind]:
            circuit = ci.circuit_from_text(kind, {"return": ret})
            for shape in wf.offered_shapes(role, ret):
                configs.append(ChannelConfig(role, circuit, wf.default_waveform(role, shape, ret)))
    board = Circuit(CircuitKind.IVI_BOARD, wf.BOARD, scale=-0.0123, board_input=2.47)
    configs.append(ChannelConfig(Role.SOURCE_I, board, wf.default_waveform(Role.SOURCE_I, Shape.BIPHASIC, wf.BOARD)))
    configs.append(cli.parse_channel("source-i sine freq=33.3 pp=0.0000123 center=1,5mA"))
    configs.append(cli.parse_channel("source-v square low=0.1 high=4.9 duty=33% return=2.5V"))
    return configs


Shape = wf.Shape


@pytest.mark.parametrize("simulate", [True, False])
def test_the_repeat_command_gives_the_same_configuration(tmp_path: Path, simulate: bool) -> None:
    for a in every_configuration():
        for output in (None, tmp_path):
            b = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE)) \
                if a.circuit.kind is CircuitKind.LOAD_SHUNT and a.circuit.shunt else ChannelConfig()
            seconds = 2.5 if signals.measure_only(a, b) else None
            words = shlex.split(recording.command_text(cli.repeat_signals(simulate, output, a, b, seconds)))
            assert "--confirm-output" not in words                   # it shows the before-checks first
            assert ("--simulate" in words) is simulate               # a simulated repeat stays simulated
            args = cli.build_parser().parse_args(words[1:])
            assert getattr(args, "output", None) == (None if output is None else output.resolve())
            assert cli.parse_channel(args.a) == a and cli.parse_channel(args.b) == b
            assert (args.duration is None) is (seconds is None)


def test_a_recording_stores_its_command_and_its_summary_ends_with_it(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture) -> None:
    typed = ["--simulate", "--output", str(tmp_path), "signals", "--a", SHORT_DC, "--confirm-output"]
    assert cli.main(typed) == cli.EXIT_DONE
    out = capsys.readouterr().out
    [folder] = sorted((tmp_path / "recordings").iterdir())
    command = json.loads((folder / recording.METADATA_FILE).read_text(encoding="utf-8"))["command"]
    assert shlex.split(command["typed"]) == ["phycovolt", *typed]
    repeat = shlex.split(command["repeat"])
    assert "--confirm-output" not in repeat and "--simulate" in repeat
    assert cli.parse_channel(repeat[repeat.index("--a") + 1]) == cli.parse_channel(SHORT_DC)
    summary = (folder / recording.SUMMARY_FILE).read_text(encoding="utf-8")
    assert commands_in(summary) == [["phycovolt", "plot", str(folder)], repeat]
    assert commands_in(out) == [["phycovolt", "plot", str(folder)], repeat]
    assert cli.main(["analyze", str(folder)]) == cli.EXIT_DONE
    assert commands_in(capsys.readouterr().out) == [["phycovolt", "plot", str(folder)], repeat]


def test_a_calibration_stores_its_command(typed_calibration: tuple[str, Path, list[str]]) -> None:
    out, folder, typed = typed_calibration
    command = json.loads((folder / recording.METADATA_FILE).read_text(encoding="utf-8"))["command"]
    assert shlex.split(command["typed"]) == ["phycovolt", *typed]
    repeat = shlex.split(command["repeat"])
    args = cli.build_parser().parse_args(repeat[1:])
    assert (args.command, args.stage, args.typed, args.simulate) == ("calibrate", "measure-v", True, True)
    assert "\x1b" not in out


# --- Styles ---------------------------------------------------------------------------------------


def test_no_style_codes_when_the_output_is_not_a_terminal(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SINE, "--confirm-output"])
    cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", "source-i sine pp=200mA r=100"])
    captured = capsys.readouterr()
    assert "\x1b" not in captured.out + captured.err
    for path in tmp_path.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        assert "\x1b" not in text and "[danger]" not in text, path


def test_no_style_codes_in_a_calibration_folder(typed_calibration: tuple[str, Path, list[str]]) -> None:
    out, folder, _ = typed_calibration
    assert "\x1b" not in out
    for path in folder.iterdir():
        assert "\x1b" not in path.read_text(encoding="utf-8"), path


def styled_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path) -> str:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a",
                     "source-i sine pp=200mA circuit=resistor r=100 return=2.5V"])
    assert code == cli.EXIT_NOT_STARTED                          # a before-check error (✗)
    return capsys.readouterr().out


def test_styles_on_a_terminal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = styled_run(monkeypatch, capsys, tmp_path)
    error = next(line for line in out.splitlines() if "✗" in line)
    assert "\x1b[1;31m" in error                                  # danger: bold red
    assert "\x1b[32m✓" in out                                     # safe: green


def test_no_color_turns_every_style_off(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                                        tmp_path: Path) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    out = styled_run(monkeypatch, capsys, tmp_path)
    assert "✗" in out and not SGR.search(out)


def test_every_mark_has_a_look_in_the_theme(tmp_path: Path) -> None:
    from phycovolt import simulator as sim
    a, b = cli.parse_channel(SINE), cli.parse_channel("hi-z")
    m1k = sim.SimulatedM1K(sim.bench_for(a, b))
    prepared = signals.prepare(m1k, a, b)
    measurement = signals.run(m1k, prepared)
    view = recording.View(terminal=True, width=80)
    lines = (recording.before_lines(prepared, view) + recording.switch_lines(prepared, True, view)
             + recording.switch_lines(prepared, False, view)
             + recording.summary_lines(measurement, tmp_path, view, ["phycovolt"], before_shown=True)
             + recording.burst_verdict(replace(measurement, ended="stopped"), view)
             + recording.burst_verdict(replace(measurement, overcurrent=True), view)
             + recording.status_lines("x", prepared.records, prepared.rest, {"A": 0, "B": 1}, view)
             + cli.start_page(80))
    used = {span.style for line in lines for span in line.spans} | {line.style for line in lines if line.style}
    assert used and used <= set(cli.THEME.styles) and set(recording.STYLES) <= set(cli.THEME.styles)


def test_a_bracket_in_a_path_or_a_description_is_printed_as_it_is(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "run[bold]x[/bold]"
    assert cli.main(["--simulate", "--output", str(output), "signals", "--a", SHORT_DC, "--confirm-output"]) == 0
    [folder] = sorted((output / "recordings").iterdir())
    assert f"Folder  {folder}" in capsys.readouterr().out
    assert cli.main(["--simulate", "signals", "--a", "source-i sine pp=[red]2"]) == cli.EXIT_NOT_STARTED
    assert "[red]2" in capsys.readouterr().err


def test_a_wrapped_line_keeps_its_indent_and_a_value_its_unit() -> None:
    text = "  voltage: amplitude 10.40 mV (expected 10.02 mV), level 2.446 V (expected 2.446 V) " * 3
    lines = recording.wrapped(text.strip(), "  ", 40)
    assert lines[0].plain.startswith("  voltage") and all(line.plain.startswith("    ") for line in lines[1:])
    assert all(len(line.plain) <= 40 for line in lines)
    assert not any(line.plain.strip().startswith(("V", "mV")) for line in lines)


# --- The progress line ----------------------------------------------------------------------------


def test_the_progress_line_starts_no_thread_and_gives_the_cursor_back() -> None:
    file = io.StringIO()
    console = Console(file=file, force_terminal=True, width=80, theme=cli.THEME, color_system="standard")
    before = threading.active_count()
    started = time.perf_counter()
    with cli.ProgressLine(console, 30) as line:
        for done in range(1, 31):
            line.chunk(done)
            assert threading.active_count() == before              # redrawn by the loop itself
    per_chunk = (time.perf_counter() - started) / 30
    out = file.getvalue()
    assert "3.0 s of 3.0 s" in out
    assert out.rindex("\x1b[?25h") > out.index("\x1b[?25l")        # the cursor is back
    assert per_chunk < 0.02                                       # a chunk lasts 100 ms


def test_without_a_terminal_one_progress_line_per_second() -> None:
    file = io.StringIO()
    with cli.ProgressLine(Console(file=file, width=80), 25) as line:
        for done in range(1, 13):
            line.chunk(done)
        line.stopped(12)
    assert file.getvalue().splitlines() == ["  1.0 s of 2.5 s", "  1.2 s of 2.5 s · stopped"]


@pytest.mark.skipif(not hasattr(select, "select") or os.name == "nt", reason="select on pipes")
def test_ctrl_c_in_the_middle_of_a_burst_on_a_styled_terminal(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "phycovolt", "--simulate", "--output", str(tmp_path), "signals",
               "--a", "source-i dc level=10mA seconds=6", "--confirm-output"]
    env = {**os.environ, "FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "80"}
    env.pop("NO_COLOR", None)
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    seen = b""
    try:
        deadline = time.monotonic() + 30
        while b"0.5 s of 6.0 s" not in seen and time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if ready:
                seen += os.read(process.stdout.fileno(), 65536)
        process.send_signal(signal.SIGINT)                        # Ctrl+C
        rest, err = process.communicate(timeout=30)
    finally:
        process.kill()
    out = (seen + rest).decode("utf-8")
    assert process.returncode == cli.EXIT_INTERRUPTED, err
    assert "stopped at" in out and out.rindex("\x1b[?25h") > out.index("\x1b[?25l")
    shown = SGR.sub("", out)
    stopped = next(line for line in shown.splitlines() if "STOPPED" in line)
    assert "by the operator at" in stopped and "both channels are in HI_Z" in stopped
    assert shown.rindex("STOPPED") > shown.rindex("After the measurement") > shown.rindex("stopped at")
    [folder] = sorted((tmp_path / "recordings").iterdir())
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ended"] == "stopped" and meta["interrupted"] is True


# --- The hierarchy (spec 3.1, Hierarchy) -----------------------------------------------------------

BOX = re.compile("[─-╿]")                       # box-drawing characters


def burst_output(tmp_path: Path, capsys: pytest.CaptureFixture) -> tuple[str, Path]:
    assert cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SINE, "--confirm-output"]) == 0
    [folder] = sorted((tmp_path / "recordings").iterdir())
    return capsys.readouterr().out, folder


def test_the_verdict_and_the_next_actions_come_last(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out, folder = burst_output(tmp_path, capsys)
    lines = out.rstrip("\n").splitlines()
    verdict = next(k for k, line in enumerate(lines) if line.startswith(" COMPLETE "))
    assert "both channels are in HI_Z" in lines[verdict] and "checks:" in lines[verdict + 1]
    assert verdict > next(k for k, line in enumerate(lines) if "After the measurement" in line)
    rest = lines[verdict + 2:]
    labels = [line.split()[0] for line in rest if re.match(r"^  \S", line)]
    assert labels == ["Folder", "Plot", "Repeat"] and rest[0] == f"  Folder  {folder}"
    assert lines[-1].strip() == "it shows the before-checks first; add --confirm-output to switch on"


def test_passed_checks_are_counted_on_the_terminal_and_listed_in_the_file(tmp_path: Path,
                                                                         capsys: pytest.CaptureFixture) -> None:
    out, folder = burst_output(tmp_path, capsys)
    after_result = out.split("── Result")[1]
    measurement = recording.read(folder).measurement
    passed = sum(check.level is cli.Level.OK for check in measurement.after)
    assert f"  ✓ {passed} passed" in after_result
    assert "✓ CH" not in after_result                            # the passes are not written out again
    assert "Before switching on: ✓ 8 passed, ◆ 1 note, shown above" in after_result
    summary = (folder / recording.SUMMARY_FILE).read_text(encoding="utf-8")
    for check in measurement.prepared.checks + measurement.after:
        assert check.text.split()[0] in summary
    assert summary.count("✓ CH") == sum(c.level is cli.Level.OK for c in measurement.prepared.checks + measurement.after)


def test_the_file_keeps_the_method_paragraph(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out, folder = burst_output(tmp_path, capsys)
    summary = (folder / recording.SUMMARY_FILE).read_text(encoding="utf-8")
    assert "== How the numbers were obtained" in summary and "statistical only" in " ".join(summary.split())
    assert "statistical only" not in out and "summary.txt in the folder" in out


@pytest.mark.parametrize("a, badge", [(SINE, "COMPLETE"), ("source-i sine pp=200mA r=100", "NOT SWITCHED ON")])
def test_badges_survive_as_words_with_no_color(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                                               tmp_path: Path, a: str, badge: str) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", a, "--confirm-output"])
    out = capsys.readouterr().out
    assert f" {badge} " in out and not SGR.search(out)


def test_nothing_is_drawn_around_a_command(tmp_path: Path, capsys: pytest.CaptureFixture,
                                          typed_calibration: tuple[str, Path, list[str]]) -> None:
    out, _ = burst_output(tmp_path, capsys)
    cli.main([])
    shown = out + capsys.readouterr().out + typed_calibration[0]
    commands = [line for line in shown.splitlines() if "phycovolt " in line or "smu " in line]
    assert len(commands) > 5 and not any(BOX.search(line) for line in commands)


def test_a_calibration_opens_each_stage_and_ends_with_smu_w(typed_calibration: tuple[str, Path, list[str]]) -> None:
    out, folder, _ = typed_calibration
    assert "── measure-v ─" in out
    assert sum(line.startswith(" PASS ") for line in out.splitlines()) == 2        # CH A and CH B at 5V
    lines = out.rstrip("\n").splitlines()
    end = max(k for k, line in enumerate(lines) if line.startswith(" COMPLETE "))
    assert "verifications: 2 pass" in lines[end + 1]
    assert lines[end + 2] == f"  Folder  {folder}" and lines[end + 3] == f"  Write   smu -w {folder / recording.CALIB_FILE}"
    assert lines[end + 4] == "  Check   smu -d" and all(line.startswith(" " * 10) for line in lines[end + 5:])
    report = (folder / recording.REPORT_FILE).read_text(encoding="utf-8")
    assert "PASS · verification at 5V pin" in report and "== measure-v ==" in report


def test_a_failed_verification_is_a_fail_badge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                               capsys: pytest.CaptureFixture) -> None:
    from phycovolt import simulator as sim
    original = sim.SimulatedM1K

    def bent(*args: object, **kwargs: object) -> sim.SimulatedM1K:       # H-06 near 5 V
        return original(*args, faults=sim.Faults(source_nonlinearity=True), **kwargs)

    monkeypatch.setattr(cli.simulator, "SimulatedM1K", bent)
    assert cli.main(["--simulate", "--output", str(tmp_path), "calibrate", "--stage", "source-v"]) == 0
    out = capsys.readouterr().out
    fails = [line for line in out.splitlines() if line.startswith(" FAIL ")]
    assert len(fails) == 2 and all("verification at 4.5 V" in line for line in fails)          # CH A and CH B
    assert "FAIL (above)" in out and " COMPLETE " in out


# --- The badge's color is the verdict (spec 3.1, Hierarchy 2) ---------------------------------------

TERMINAL_80 = recording.View(terminal=True, width=80)


@pytest.fixture(scope="module")
def clean_burst() -> "signals.Measurement":
    from phycovolt import simulator as sim
    a, b = cli.parse_channel(SINE), cli.parse_channel("hi-z")
    m1k = sim.SimulatedM1K(sim.bench_for(a, b))
    measurement = signals.run(m1k, signals.prepare(m1k, a, b))
    assert not any(c.level in (cli.Level.WARNING, cli.Level.ERROR) for c in measurement.prepared.checks + measurement.after)
    return measurement


def badge(lines: list) -> tuple[str, str]:
    """(the badge's word, its style): a span of its own, not the line's base style."""
    first = lines[0]
    assert first.style == ""                                   # the rest of the line is not colored
    span = first.spans[0]
    return first.plain[span.start:span.end].strip(), span.style


def note(level: "cli.Level") -> "cli.checks.Check":
    return cli.checks.Check(level, "drift", "A")


@pytest.mark.parametrize("case, word, style", [
    ("clean", "COMPLETE", recording.GOOD),                     # nothing to look at
    ("note", "COMPLETE", recording.GOOD),                      # a note does not change it
    ("warning", "COMPLETE", recording.STOP),                   # look above
    ("stopped", "STOPPED", recording.STOP),
    ("error", "FAILED", recording.BAD),                        # it failed
    ("limiter", "FAILED", recording.BAD),
])
def test_the_badge_color_of_a_burst_is_its_verdict(clean_burst, case: str, word: str, style: str) -> None:
    measurement = {
        "clean": clean_burst,
        "note": replace(clean_burst, after=clean_burst.after + [note(cli.Level.NOTE)]),
        "warning": replace(clean_burst, after=clean_burst.after + [note(cli.Level.WARNING)]),
        "stopped": replace(clean_burst, ended="stopped"),
        "error": replace(clean_burst, after=clean_burst.after + [note(cli.Level.ERROR)]),
        "limiter": replace(clean_burst, overcurrent=True),
    }[case]
    lines = recording.burst_verdict(measurement, TERMINAL_80)
    assert badge(lines) == (word, style)
    if case == "warning":
        assert "1 warning (⚠ above)" in lines[0].plain
    assert recording.plain(recording.burst_verdict(measurement, recording.FILE)).startswith(f"{word} · ")


def test_a_warning_before_switching_on_makes_the_complete_yellow(clean_burst) -> None:
    prepared = replace(clean_burst.prepared, checks=clean_burst.prepared.checks + [note(cli.Level.WARNING)])
    assert badge(recording.burst_verdict(replace(clean_burst, prepared=prepared), TERMINAL_80)) == \
        ("COMPLETE", recording.STOP)


def test_the_badges_before_switching_on(clean_burst) -> None:
    prepared = clean_burst.prepared
    assert badge(recording.switch_lines(prepared, True, TERMINAL_80)[1:]) == ("SWITCHING ON", recording.LIVE)
    assert badge(recording.switch_lines(prepared, False, TERMINAL_80)[1:]) == ("NOT SWITCHED ON", recording.STOP)
    blocked = replace(prepared, checks=prepared.checks + [note(cli.Level.ERROR)])
    assert badge(recording.switch_lines(blocked, True, TERMINAL_80)[1:]) == ("NOT SWITCHED ON", recording.BAD)


def test_a_lost_read_is_a_red_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                     capsys: pytest.CaptureFixture) -> None:
    from phycovolt import simulator as sim
    original = sim.SimulatedM1K
    monkeypatch.setattr(cli.simulator, "SimulatedM1K",
                        lambda *args, **kwargs: original(*args, faults=sim.Faults(usb_failure_at_chunk=3), **kwargs))
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SINE, "--confirm-output"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_NOT_STARTED
    assert not (tmp_path / "recordings").exists() or not list((tmp_path / "recordings").iterdir())
    lines = out.splitlines()
    start = next(k for k, line in enumerate(lines) if "FAILED" in line)
    assert "\x1b[1;37;41m FAILED " in lines[start]                  # white on red
    block = " ".join(SGR.sub("", line).strip() for line in lines[start:])
    assert "both channels are in HI_Z" in block and "no recording: run the command again." in block


def test_the_calibration_badges_and_the_caution_under_smu_w(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                            capsys: pytest.CaptureFixture,
                                                            typed_calibration: tuple[str, Path, list[str]]) -> None:
    caution = "a verification failed (above): write it only if you accept the record there"
    clean, folder, _ = typed_calibration
    assert " COMPLETE " in clean and caution not in clean
    assert caution not in (folder / recording.REPORT_FILE).read_text(encoding="utf-8")
    from phycovolt import simulator as sim
    original = sim.SimulatedM1K
    monkeypatch.setattr(cli.simulator, "SimulatedM1K",
                        lambda *args, **kwargs: original(*args, faults=sim.Faults(source_nonlinearity=True), **kwargs))
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.main(["--simulate", "--output", str(tmp_path), "calibrate", "--stage", "source-v"]) == cli.EXIT_DONE
    lines = SGR.sub("", capsys.readouterr().out).splitlines()
    raw = None
    for k, line in enumerate(lines):
        if line.startswith("  Write   smu -w "):
            assert caution in " ".join(" ".join(lines[k + 1:k + 3]).split())
            raw = k
    assert raw is not None
    [folder] = sorted((tmp_path / "calibrations").iterdir())
    report = (folder / recording.REPORT_FILE).read_text(encoding="utf-8")
    assert caution in " ".join(report.split()) and "COMPLETE · source-v" in report


def test_a_calibration_complete_is_green_only_without_a_fail_or_a_warning() -> None:
    from phycovolt.checks import Check, Level
    ok = recording.verdict_style([Check(Level.OK, "x"), Check(Level.NOTE, "y")])
    assert ok == recording.GOOD
    assert recording.verdict_style([Check(Level.OK, "x")], look=True) == recording.STOP      # a FAIL above
    assert recording.verdict_style([Check(Level.WARNING, "x")]) == recording.STOP            # kept warnings
    assert recording.verdict_style([Check(Level.ERROR, "x")]) == recording.BAD


def test_a_stored_record_at_the_default_is_yellow_in_status(monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture) -> None:
    from phycovolt import simulator as sim
    original = sim.SimulatedM1K
    monkeypatch.setattr(cli.simulator, "SimulatedM1K",
                        lambda *args, **kwargs: original(*args, faults=sim.Faults(eeprom_default=True), **kwargs))
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli.main(["--simulate", "status"]) == cli.EXIT_DONE
    records = [line for line in capsys.readouterr().out.splitlines() if " CH " in line and "⚠" in line]
    assert records and all("\x1b[33m" in line for line in records) and not any("\x1b[32m✓" in r for r in records)


def test_method_b_is_decided_in_the_files(tmp_path: Path, capsys: pytest.CaptureFixture,
                                          typed_calibration: tuple[str, Path, list[str]]) -> None:
    """Method B is decided (spec 8, 2026-11-21): no "provisional" next to it; the
    robust spread (open item 3) keeps its own."""
    _, folder = burst_output(tmp_path, capsys)
    summary = " ".join((folder / recording.SUMMARY_FILE).read_text(encoding="utf-8").split())
    assert "(method B, spec §8)" in summary and "provisional" not in summary
    report = " ".join((typed_calibration[1] / recording.REPORT_FILE).read_text(encoding="utf-8").split())
    assert "uses method B: the standard deviation of the 100 ms block means." in report
    assert "provisional" not in report
    from phycovolt import analysis
    assert "Method B (spec 8, decided 2026-11-21)" in analysis.uncertainty_of_mean.__doc__
    assert "Provisional (spec 8, open item 3)" in analysis.robust_spread.__doc__
