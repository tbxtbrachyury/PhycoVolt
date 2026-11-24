"""Can every recording be drawn? (spec 3.3.7; the tests use matplotlib's Agg, no window)"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phycovolt import circuits as ci
from phycovolt import plot
from phycovolt import recording as rec
from phycovolt import signals
from phycovolt import simulator as sim
from phycovolt import waveforms as wf
from phycovolt.circuits import ChannelConfig, Circuit, CircuitKind
from phycovolt.hardware import Role
from phycovolt.waveforms import Shape

HI_Z = ChannelConfig()


@pytest.mark.parametrize("a, b, expected", [
    (ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I),
                   wf.Waveform(Shape.SINE, {"freq": 100, "pp": 20e-6, "center": 0.0, "periods": 10})), HI_Z, 1),
    (ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I),
                   wf.default_waveform(Role.SOURCE_I, Shape.DC, "2.5V")), HI_Z, 1),
    (ChannelConfig(Role.SOURCE_V, Circuit(CircuitKind.LOAD_SHUNT, "2.5V", r=47.0, shunt=10.0),
                   wf.Waveform(Shape.SINE, {"freq": 100, "pp": 2.0, "center": 2.5, "periods": 10})),
     ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE)), 2),
])
def test_figures_are_drawn(tmp_path: Path, a: ChannelConfig, b: ChannelConfig, expected: int) -> None:
    m1k = sim.SimulatedM1K(sim.bench_for(a, b), seed=5)
    folder = rec.write(signals.run(m1k, signals.prepare(m1k, a, b)), tmp_path)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figures = plot.figures(rec.read(folder))
    assert len(figures) == expected
    for figure in figures:
        FigureCanvasAgg(figure).draw()                        # a bare Figure's canvas draws nothing
        assert figure.axes


def test_the_plot_command_needs_a_folder() -> None:
    assert plot.main([]) == 1


def test_matplotlib_is_not_imported_by_the_program_itself() -> None:
    import subprocess
    import sys
    code = "import sys, phycovolt.__main__, phycovolt.signals, phycovolt.recording; " \
           "print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    assert out == "False"                                    # NF-09: the command line starts fast


def test_the_window_backend_loads(tmp_path: Path) -> None:
    """Qt (PySide6) draws the figures; offscreen, so no display is needed."""
    import os
    import subprocess
    import sys
    a = ChannelConfig(Role.SOURCE_I, ci.default_circuit(Role.SOURCE_I),
                      wf.Waveform(Shape.SINE, {"freq": 100, "pp": 0.2, "center": 0.0, "periods": 10}))
    m1k = sim.SimulatedM1K(sim.bench_for(a, HI_Z), seed=6)
    folder = rec.write(signals.run(m1k, signals.prepare(m1k, a, HI_Z)), tmp_path)
    code = ("import sys, matplotlib; matplotlib.use('QtAgg'); import matplotlib.pyplot as plt; "
            "from pathlib import Path; from phycovolt import plot, recording; "
            "figures = plot.figures(recording.read(Path(sys.argv[1])), plt.figure); "
            "[f.canvas.draw() for f in figures]; print(matplotlib.get_backend(), len(figures))")
    result = subprocess.run([sys.executable, "-c", code, str(folder)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, timeout=120)
    assert result.stdout.split() == ["QtAgg", "1"], result.stderr


def test_no_axis_uses_offset_notation(tmp_path: Path) -> None:
    """Full values on every axis: a "+1" in a corner would make 0.9999868 V look negative (spec 3.3.7)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    a = ChannelConfig(Role.SOURCE_V, ci.default_circuit(Role.SOURCE_V),
                      wf.Waveform(Shape.DC, {"level": 0.99998, "seconds": 1.0}))
    b = ChannelConfig(Role.HI_Z, Circuit(CircuitKind.SHUNT_NODE))
    m1k = sim.SimulatedM1K(sim.bench_for(a, b), seed=7)
    folder = rec.write(signals.run(m1k, signals.prepare(m1k, a, b)), tmp_path)
    figures = plot.figures(rec.read(folder))
    assert figures
    for figure in figures:
        FigureCanvasAgg(figure).draw()                        # a bare Figure's canvas draws nothing
        for axes in figure.axes:
            for axis in (axes.xaxis, axes.yaxis):
                formatter = axis.get_major_formatter()
                assert getattr(formatter, "get_useOffset", lambda: False)() is False
                assert axis.get_offset_text().get_text() == ""


# --- A calibration (spec 3.7) ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A simulated calibration of one voltage and one current stage, written as the program writes it."""
    from datetime import datetime

    from phycovolt import calibration as cal
    from phycovolt import multimeter
    profile = multimeter.load_profile(multimeter.PROFILES / "agilent_34401a.ini")
    m1k = sim.SimulatedM1K(seed=8)
    dmm = sim.VirtualMultimeter(m1k, profile, seed=9)
    context = cal.Context(m1k, dmm, profile, sim.SimulatedOperator(m1k, dmm), m1k.clock, m1k.calibration())
    run = cal.calibrate(context, ["measure-v", "measure-i"])
    return rec.write_calibration_folder(run, None, tmp_path_factory.mktemp("calibrations"),
                                        datetime(2026, 11, 22, 10), "simulated", profile.name)


def test_a_calibration_is_drawn(calibration_folder: Path, tmp_path: Path) -> None:
    """One figure per stage: the error before and with the new record, what remains +-U, each
    verification against its limit, and the budget; no axis uses offset notation."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    assert rec.is_calibration(calibration_folder) and not rec.is_calibration(tmp_path)
    meta = rec.read_calibration(calibration_folder)
    figures = plot.calibration_figures(meta)
    assert len(figures) == 2                                  # measure-v and measure-i
    for figure, stage in zip(figures, ("measure-v", "measure-i")):
        FigureCanvasAgg(figure).draw()
        effect, remains, verify, budget = figure.axes
        channels = meta["results"][stage]
        assert len(effect.lines) == 2 * len(channels) + 1     # before and after per channel, and zero
        made = sum(len(item["verifications"]) for item in channels.values())
        labels = [label.get_text() for label in verify.get_xticklabels()]
        assert len(labels) == made > 0 and all(label.endswith("PASS") for label in labels)
        assert len(budget.patches) == len(budget.get_yticklabels()) > 0
        for axes in figure.axes:
            for axis in (axes.xaxis, axes.yaxis):
                assert axis.get_offset_text().get_text() == ""
    with pytest.raises(ValueError):
        rec.read_calibration(tmp_path) if (tmp_path / rec.METADATA_FILE).write_text("{}", encoding="utf-8") else None


def test_a_calibration_that_names_the_uncertainty_b_is_still_drawn(calibration_folder: Path, tmp_path: Path) -> None:
    """Calibration folders made before the calculation got its descriptive name hold "B" in
    `uncertainty_method`; they are still read and drawn."""
    meta = json.loads((calibration_folder / rec.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["uncertainty_method"] == "std_of_block_means"
    meta["uncertainty_method"] = "B"
    (tmp_path / rec.METADATA_FILE).write_text(json.dumps(meta), encoding="utf-8")
    assert len(plot.calibration_figures(rec.read_calibration(tmp_path))) == 2
