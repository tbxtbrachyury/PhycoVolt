"""Is a recording complete, readable by any tool, and analyzable again from its files? (spec 3.4)"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from phycovolt import circuits as ci
from phycovolt import recording as rec
from phycovolt import signals
from phycovolt import simulator as sim
from phycovolt import waveforms as wf
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import Role
from phycovolt.waveforms import Shape

HI_Z = ChannelConfig()


def measured(a: ChannelConfig, b: ChannelConfig = HI_Z) -> signals.Measurement:
    m1k = sim.SimulatedM1K(sim.bench_for(a, b), seed=3)
    return signals.run(m1k, signals.prepare(m1k, a, b))


SINE = ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I),
                     wf.Waveform(Shape.SINE, {"freq": 100, "pp": 0.2, "center": 0.0, "periods": 20}))


def test_files_and_folder_name(tmp_path: Path) -> None:
    measurement = measured(SINE)
    folder = rec.write(measurement, tmp_path)
    assert folder.parent == tmp_path
    assert folder.name.endswith("_source-i-sine")
    assert sorted(p.name for p in folder.iterdir()) == ["metadata.json", "samples.csv.gz", "summary.txt"]


def test_same_second_gets_a_suffix(tmp_path: Path) -> None:
    started = datetime(2026, 11, 22, 10, 15, 0)
    names = []
    for _ in range(3):
        folder = rec.free_folder(tmp_path, started, "source-i-sine")
        folder.mkdir()
        names.append(folder.name)
    assert names == ["2026-11-22_101500_source-i-sine", "2026-11-22_101500_source-i-sine_2",
                     "2026-11-22_101500_source-i-sine_3"]
    (tmp_path / (rec.INCOMPLETE_PREFIX + "2026-11-22_101500_source-i-sine_4")).mkdir()
    assert rec.free_folder(tmp_path, started, "source-i-sine").name.endswith("_5")


def test_an_interrupted_write_never_looks_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    measurement = measured(SINE)

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(rec.np, "savetxt", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rec.write(measurement, tmp_path)
    [left] = list(tmp_path.iterdir())
    assert left.name.startswith(rec.INCOMPLETE_PREFIX)        # named for what it is


def test_samples_csv_is_plain_csv(tmp_path: Path) -> None:
    measurement = measured(SINE)
    folder = rec.write(measurement, tmp_path)
    with gzip.open(folder / rec.SAMPLES_FILE, "rt", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == list(rec.COLUMNS)
    assert len(rows) == 1 + measurement.samples.count
    first = rows[1]
    assert first[0] == "0" and first[1] == "0.00000"
    assert first[6] == "" and first[7] == ""                    # CH B is hi-z: no requested, no dac
    assert float(first[2]) == pytest.approx(0.0)                # the sine starts at its center
    assert float(rows[251][2]) == pytest.approx(0.1)            # a quarter period later, the crest
    assert np.float32(first[4]) == np.float32(measurement.samples.v_a[0])   # 9 digits name a float32


def test_metadata_has_what_the_analysis_needs(tmp_path: Path) -> None:
    folder = rec.write(measured(SINE), tmp_path)
    meta = json.loads((folder / rec.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["program"]["version"] == "0.1.0"
    assert meta["instrument"]["simulated"] is True
    assert len(meta["calibration"]) == 8
    assert meta["channels"]["A"]["circuit"] == {"kind": "resistor", "return": "2.5V", "r_ohm": 10.0,
                                               "shunt_ohm": 0.0, "scale": 0.0, "board_input_v": 2.5}
    assert meta["channels"]["A"]["waveform"]["real_frequency_hz"] == 100.0
    assert meta["ended"] == "complete" and meta["interrupted"] is False
    assert {c["code"] for c in meta["checks"]["before"]} >= {"wiring", "pin_limit", "frequency"}
    assert meta["results"]["A"]["i_fit_amplitude_a"] == pytest.approx(0.1, rel=0.01)
    assert meta["results"]["statistical_only"] is True and meta["results"]["uncertainty_method"] == "B"
    assert meta["files"] == ["samples.csv.gz", "metadata.json", "summary.txt"]


@pytest.mark.parametrize("a, b", [
    (SINE, HI_Z),
    (ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I),
                   wf.default_waveform(Role.SOURCE_I, Shape.DC, "2.5V")), HI_Z),
    (ChannelConfig(Role.SOURCE_V, Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0),
                   wf.Waveform(Shape.SQUARE, {"freq": 100, "low": 1.5, "high": 3.5, "duty": 0.5, "periods": 20})),
     ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE))),
])
def test_analyzed_again_from_the_files_alone(tmp_path: Path, a: ChannelConfig, b: ChannelConfig) -> None:
    measurement = measured(a, b)
    folder = rec.write(measurement, tmp_path)
    again = rec.read(folder).measurement
    for name in ("A", "B"):
        original = measurement.results[name].numbers()
        repeated = again.results[name].numbers()
        assert repeated.keys() == original.keys()
        for key, value in original.items():
            if isinstance(value, float):
                assert repeated[key] == pytest.approx(value, rel=1e-12, abs=1e-18), key
            else:
                assert repeated[key] == value, key
    assert [c.text for c in again.after] == [c.text for c in measurement.after]


def test_summary_is_plain_text_within_100_columns(tmp_path: Path) -> None:
    folder = rec.write(measured(SINE), tmp_path)
    text = (folder / rec.SUMMARY_FILE).read_text(encoding="utf-8")
    commands = [line for line in text.splitlines() if "phycovolt " in line]
    assert commands == [f"  Plot    phycovolt plot {folder}"]   # a command is never cut (spec 3.1)
    assert all(len(line) <= rec.WIDTH for line in text.splitlines() if line not in commands)
    assert text.startswith("CH A source-i sine 100.0 Hz → R 10.00 Ω → 2.5V · CH B hi-z")
    assert "statistical only" in " ".join(text.split())
    assert "\x1b" not in text                                   # no colors


def test_nothing_is_overwritten(tmp_path: Path) -> None:
    measurement = measured(SINE)
    first = rec.write(measurement, tmp_path)
    second = rec.write(measurement, tmp_path)
    assert first != second and first.exists() and second.exists()


def test_git_commit_is_read_without_running_git() -> None:
    commit = rec.git_commit()
    assert commit is None or (len(commit) == 40 and int(commit, 16) >= 0)
    assert isinstance(np.zeros(1), np.ndarray)
