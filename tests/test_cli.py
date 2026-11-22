"""The command line end to end, always with --simulate (spec 3.1, 3.9, 9.1).

The in-process tests call `__main__.main`; the signal tests start a real subprocess and send it
SIGTERM or SIGHUP in the middle of a burst. None of them can reach the instrument: --simulate never
imports m1k.py, and the conftest guard blocks the real pysmu in this process.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from phycovolt import __main__ as cli
from phycovolt import checks as ch
from phycovolt import m1k
from phycovolt import simulator as sim
from phycovolt.hardware import InstrumentError, Record

PROJECT = Path(__file__).resolve().parents[1]
REAL_OPEN_M1K = m1k.open_m1k
SHORT_DC = "source-i dc level=10mA seconds=0.3"


@pytest.fixture(autouse=True)
def keep_signal_handlers():
    """main() installs SIGTERM and SIGHUP handlers: put pytest's back afterwards."""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)


def recordings(output: Path) -> list[Path]:
    folder = output / "recordings"
    return sorted(folder.iterdir()) if folder.exists() else []


def test_without_confirmation_nothing_is_switched_on(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SHORT_DC])
    assert code == cli.EXIT_DONE
    assert "NOT SWITCHED ON  without --confirm-output" in capsys.readouterr().out
    assert recordings(tmp_path) == []


def test_with_confirmation_a_recording_is_made(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["signals", "--simulate", "--output", str(tmp_path), "--a", SHORT_DC, "--b", "hi-z",
                     "--confirm-output"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_DONE
    [folder] = recordings(tmp_path)
    assert folder.name.endswith("_source-i-dc")
    assert f"Folder  {folder}" in out


def test_a_before_check_error_keeps_the_outputs_off(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a",
                     "source-i sine pp=200mA circuit=resistor r=100 return=2.5V", "--confirm-output"])
    assert code == cli.EXIT_NOT_STARTED
    assert "the outputs stay off" in capsys.readouterr().out
    assert recordings(tmp_path) == []


@pytest.mark.parametrize("text", ["source-x sine", "source-i", "source-i sine pp=2 parsecs",
                                  "source-i sine circuit=load-shunt", "hi-z freq=10"])
def test_a_bad_channel_description(tmp_path: Path, text: str, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", text]) == cli.EXIT_NOT_STARTED
    assert "cannot be used" in capsys.readouterr().err


def test_measure_only_on_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a",
                     "hi-z circuit=shunt r=10 return=2.5V", "--duration", "0.5 s"])
    assert code == cli.EXIT_DONE                             # nothing to switch on: no confirmation needed
    assert "MEASURING  nothing is sourced" in capsys.readouterr().out
    [folder] = recordings(tmp_path)
    assert folder.name.endswith("_hi-z")


@pytest.mark.parametrize("a, duration", [(SHORT_DC, "2 s"), ("hi-z", "0.2 s"), ("hi-z", "two")])
def test_a_duration_that_cannot_be_used(tmp_path: Path, a: str, duration: str) -> None:
    code = cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", a, "--duration", duration])
    assert code == cli.EXIT_NOT_STARTED
    assert recordings(tmp_path) == []


def test_channel_descriptions() -> None:
    config = cli.parse_channel("source-i sine freq=100Hz pp=2mA center=0 circuit=resistor r=10 return=2.5V")
    assert (config.waveform["freq"], config.waveform["pp"], config.circuit.r) == (100.0, 2e-3, 10.0)
    into_5v = cli.parse_channel("source-i sine return=5V")
    assert into_5v.waveform["center"] == pytest.approx(-0.06)       # the defaults follow the return
    board = cli.parse_channel("source-i sine circuit=ivi-board scale=0.01")
    assert board.waveform["pp"] == pytest.approx(4e-3)
    assert cli.parse_channel("hi-z circuit=shunt r=10 return=GND").circuit.r == 10.0


def test_analyze_a_recording_again(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SHORT_DC, "--confirm-output"])
    capsys.readouterr()
    [folder] = recordings(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    assert cli.main(["analyze", str(folder)]) == cli.EXIT_DONE
    assert "CH A source-i dc" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == before     # nothing rewritten


def test_analyze_something_that_is_not_a_recording(tmp_path: Path) -> None:
    assert cli.main(["analyze", str(tmp_path)]) == cli.EXIT_NOT_STARTED


def test_status(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["status", "--simulate"]) == cli.EXIT_DONE
    out = capsys.readouterr().out
    assert "simulated (no board) · fw 2.17 · hw D" in out and "no V/I swap" in out


def status_through_the_fake(monkeypatch: pytest.MonkeyPatch, calibration: list | None) -> tuple[int, list]:
    """`phycovolt status` on the real driver (m1k.py), with the fake pysmu in place of the M1K."""
    from tests import fake_pysmu

    fake = fake_pysmu.make(fake_pysmu.Faults(calibration=calibration))
    monkeypatch.setattr(m1k, "open_m1k", lambda: REAL_OPEN_M1K(fake.module()))
    return cli.main(["status"]), fake.log


def test_status_shows_an_implausible_record_with_the_same_rest_only_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Spec 3.2: status shows the 8 records, the implausible one as an error, the rest
    voltages and the board check, and exits with 1; its calls are the same rest-only ones."""
    plain_code, plain_log = status_through_the_fake(monkeypatch, None)
    capsys.readouterr()
    implausible = [[0.0, 1.0, 1.0] for _ in range(8)]
    implausible[2] = [0.0, 1.5, 1.0]                                  # CH A source V: gain+ 1.5
    code, log = status_through_the_fake(monkeypatch, implausible)
    out = capsys.readouterr().out
    assert (plain_code, code) == (cli.EXIT_DONE, cli.EXIT_NOT_STARTED)
    assert log == plain_log                                           # the same sequence, call by call
    assert all(entry[2] == "HI_Z" for entry in log if entry[0] == "mode")
    assert not any(entry[0] in ("write", "start") for entry in log)   # nothing sourced, no burst
    assert "✗ CH A source V" in out and "Rest" in out and "Board check" in out
    assert out.rstrip().endswith("calibrate again.") and "reset it with `smu -r`" in " ".join(out.split())
    monkeypatch.setattr(sim.SimulatedM1K, "calibration", lambda self: [Record(*r) for r in implausible])
    assert cli.main(["--simulate", "--output", str(tmp_path), "signals", "--a", SHORT_DC]) == cli.EXIT_NOT_STARTED
    assert recordings(tmp_path) == []                                 # the other commands still refuse


def test_plot_starts_its_own_process_and_frees_the_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture) -> None:
    """Spec 3.3.7: the window is `python -m phycovolt.plot <folder>`, started and not waited for."""
    started = []

    class Started:
        def __init__(self, args: list[str], **options: object) -> None:
            started.append((args, options))

        def wait(self, *args: object) -> None:
            raise AssertionError("the plot window must not be waited for")

    monkeypatch.setattr(cli.subprocess, "Popen", Started)
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert cli.main(["plot", str(tmp_path)]) == cli.EXIT_DONE
    [(args, options)] = started
    assert args == [sys.executable, "-m", "phycovolt.plot", str(tmp_path.resolve())]
    assert not options.get("shell") and "this terminal is free" in capsys.readouterr().out
    assert cli.main(["plot", str(tmp_path / "nothing")]) == cli.EXIT_NOT_STARTED and len(started) == 1


def test_without_pysmu_the_program_does_not_start(monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture) -> None:
    from phycovolt import m1k

    def no_pysmu(*args: object, **kwargs: object) -> None:
        raise InstrumentError("no_pysmu", "No module named 'pysmu'")

    monkeypatch.setattr(m1k, "open_m1k", no_pysmu)             # the real one is never called
    assert cli.main(["status"]) == cli.EXIT_NOT_STARTED
    assert "activate the lab environment" in capsys.readouterr().err


def test_vi_swap_at_opening_stops_everything() -> None:
    m1k = sim.SimulatedM1K(faults=sim.Faults(vi_swaps_per_second={"B": 5}), seed=4)
    assert cli.board_check(m1k)["B"] > 0
    assert "unplug" in ch.INSTRUMENT_ERROR_TEXTS["vi_swap"]


def test_signal_handlers_become_an_interrupt() -> None:
    cli.install_signal_handlers()
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGTERM)


@pytest.mark.parametrize("which", ["SIGTERM", "SIGHUP"])
def test_a_signal_in_the_middle_of_a_burst(tmp_path: Path, which: str) -> None:
    if not hasattr(signal, which):
        pytest.skip(f"{which} does not exist on this system")
    command = [sys.executable, "-m", "phycovolt", "--simulate", "--output", str(tmp_path), "signals",
               "--a", "source-i dc level=10mA seconds=6", "--confirm-output"]
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = ""
        deadline = time.monotonic() + 30
        while "of 6.0 s" not in line and time.monotonic() < deadline:   # the first progress line
            line = process.stdout.readline()
        process.send_signal(getattr(signal, which))
        out, err = process.communicate(timeout=30)
    finally:
        process.kill()
    assert process.returncode == cli.EXIT_INTERRUPTED, err
    assert "both channels are in HI_Z" in out + err
    [folder] = recordings(tmp_path)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ended"] == "stopped" and meta["interrupted"] is True
    assert 0 < meta["burst"]["samples"] < 60 * 10_000


def test_c10_a_full_calibration_on_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = cli.main(["--simulate", "--output", str(tmp_path), "calibrate", "--stage", "all", "--skip-optional"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_DONE
    [folder] = sorted((tmp_path / "calibrations").iterdir())
    assert folder.name.endswith("_all")
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["calib.txt", "blocks.txt", "readings.csv", "dmm.csv", "report.txt", "metadata.json"])
    assert f"smu -w {folder / 'calib.txt'}" in out


def test_a_calibration_never_touches_the_base(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    base = PROJECT / "calib.txt"
    before = base.read_bytes()
    code = cli.main(["--simulate", "--output", str(tmp_path), "calibrate", "--stage", "measure-v", "--base", str(base)])
    assert code == cli.EXIT_DONE
    assert base.read_bytes() == before
    [folder] = sorted((tmp_path / "calibrations").iterdir())
    assert folder.name.endswith("_measure-v") and (folder / "calib.txt").exists()


def test_a_template_profile_is_refused(tmp_path: Path) -> None:
    from phycovolt import multimeter
    code = cli.main(["--simulate", "--output", str(tmp_path), "calibrate", "--dmm",
                     str(multimeter.PROFILES / "template.ini")])
    assert code == cli.EXIT_NOT_STARTED


def test_simulated_runs_never_default_to_the_operators_folders(tmp_path: Path) -> None:
    from phycovolt import recording
    assert cli.output_folder(True, None) == recording.PROJECT_ROOT / "scratch"
    assert cli.output_folder(True, tmp_path) == tmp_path
    assert cli.output_folder(False, None) is None                                      # the real bench
    assert recording.recordings_folder(cli.output_folder(True, None)) != recording.PROJECT_ROOT / "recordings"


def test_a_signal_in_the_middle_of_a_calibration(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "phycovolt", "--simulate", "--output", str(tmp_path), "calibrate",
               "--stage", "all", "--skip-optional"]
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = ""
        deadline = time.monotonic() + 60
        while "source-v ·" not in line and time.monotonic() < deadline:     # measure-v has finished
            line = process.stdout.readline()
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=60)
    finally:
        process.kill()
    assert process.returncode == cli.EXIT_INTERRUPTED, err
    [folder] = sorted((tmp_path / "calibrations").iterdir())
    assert folder.name.endswith("_interrupted")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ended"] == "interrupted" and "measure-v" in meta["stages"]
