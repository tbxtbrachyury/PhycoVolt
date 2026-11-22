"""The figures of a recording or a calibration, in a matplotlib window (Qt), run as a separate process.

Questions it answers: what did each channel do, sample by sample and period by period, against what
was asked and what the DAC delivered? And what did a calibration change, how sure is it, did its
verifications pass, and which error sources weigh most?

Main choices (spec 3.3.7): `python -m phycovolt.plot <folder>` opens the window, so the terminal is
free at once and the command line does not import matplotlib (NF-09). Every axis shows full values:
matplotlib's offset notation is off (a "+1" in a corner would make 0.9999868 V look like a value
below zero). The figures are built by `figures`, which works
on any backend (the tests use Agg, without a display). Nothing is saved automatically: the operator
saves a figure from the window's toolbar if wanted.

- Periodic: the imposed and the other quantity in time, opening on the first 5 periods (zoom for
  the rest), the request dashed and the DAC-delivered waveform dotted (as a staircase below 200
  levels), a rail or the 200 mA limit in red when the curve comes close; the cycle average with a
  +-1 sigma band.
- DC: the 100 ms block means in time with the DAC-delivered target, and a histogram of the samples
  with bins on the ADC step, the mean and +-1 sigma.
- A calibration (spec 3.7), one figure per stage, both channels, from the folder's metadata.json:
  the error at each value of the effect table without calibration (or with the stored record) and
  with the new one; what remains with the new record, +-U; each verification's residual against its
  limit; the uncertainty budget at the main point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

from phycovolt import analysis, checks, units
from phycovolt import hardware as hw
from phycovolt import recording as rec
from phycovolt.hardware import Kind, Record, Role

OPENING_PERIODS = 5
NO_OFFSET = {"axes.formatter.useoffset": False}
NEAR_RAIL_V = 0.5                      # draw the rail when the voltage comes this close
NEAR_LIMIT_A = 0.05                    # draw the 200 mA limit when the current comes this close


def _quantity(role: Role) -> tuple[str, str]:
    return ("current", "A") if role is Role.SOURCE_I else ("voltage", "V")


def _limits(axes: object, unit: str, values: np.ndarray) -> None:
    """The relevant limit in red, when the curve gets close."""
    if values.size == 0:
        return
    if unit == "V":
        for rail in (hw.V_MIN, hw.V_MAX):
            if np.min(np.abs(values - rail)) < NEAR_RAIL_V:
                axes.axhline(rail, color="red", linewidth=1)
    else:
        for limit in (-hw.CHANNEL_CURRENT_LIMIT, hw.CHANNEL_CURRENT_LIMIT):
            if np.min(np.abs(values - limit)) < NEAR_LIMIT_A:
                axes.axhline(limit, color="red", linewidth=1)


def _periodic(figure: object, recording: rec.Recording, name: str) -> None:
    measurement = recording.measurement
    config = measurement.prepared.config(name)
    result = measurement.results[name]
    prediction = measurement.prepared.predictions[name]
    v, i = measurement.samples.channel(name)
    n = result.period
    time = np.arange(v.size) / hw.SAMPLE_RATE
    imposed_name, imposed_unit = _quantity(config.role)
    imposed, other = (i, v) if config.role is Role.SOURCE_I else (v, i)
    other_unit = "V" if imposed_unit == "A" else "A"
    requested = np.resize(prediction.requested, v.size)
    delivered = np.resize(prediction.delivered, v.size)
    staircase = np.unique(prediction.codes).size < checks.DAC_STAIRCASE_PLOT_LEVELS
    top, middle, bottom = figure.subplots(3, 1)
    top.plot(time, imposed, linewidth=0.8, label="measured")
    top.plot(time, requested, "--", linewidth=1, label="asked")
    top.plot(time, delivered, ":", linewidth=1.2, label="DAC-delivered",
             drawstyle="steps-post" if staircase else "default")
    _limits(top, imposed_unit, imposed)
    top.set_ylabel(f"{imposed_name} ({imposed_unit})")
    top.set_xlim(0, min(OPENING_PERIODS * n, v.size) / hw.SAMPLE_RATE)
    top.legend(loc="upper right")
    middle.plot(time, other, linewidth=0.8, color="tab:green")
    _limits(middle, other_unit, other)
    middle.set_ylabel(f"{'voltage' if other_unit == 'V' else 'current'} ({other_unit})")
    middle.set_xlim(top.get_xlim())
    middle.set_xlabel("time (s)")
    cycle = result.i_cycle if config.role is Role.SOURCE_I else result.v_cycle
    if cycle is not None:
        phase = np.arange(n) / hw.SAMPLE_RATE
        bottom.fill_between(phase, cycle.mean - cycle.std, cycle.mean + cycle.std, alpha=0.3,
                            label="±1σ")
        bottom.plot(phase, cycle.mean, linewidth=1, label=f"cycle average ({cycle.periods} periods)")
        bottom.plot(phase, prediction.requested, "--", linewidth=1, label="asked")
        bottom.plot(phase, prediction.delivered, ":", linewidth=1.2, label="DAC-delivered",
                    drawstyle="steps-post" if staircase else "default")
        bottom.set_xlabel("time in the period (s)")
        bottom.set_ylabel(f"{imposed_name} ({imposed_unit})")
        bottom.legend(loc="upper right")
    figure.suptitle(rec.describe_channel(name, config))


def _dc(figure: object, recording: rec.Recording, name: str) -> None:
    measurement = recording.measurement
    config = measurement.prepared.config(name)
    result = measurement.results[name]
    v, i = measurement.samples.channel(name)
    imposed_name, imposed_unit = _quantity(config.role) if config.role is not Role.HI_Z else ("voltage", "V")
    imposed = i if imposed_unit == "A" else v
    step = hw.ADC_I_STEP if imposed_unit == "A" else hw.V_STEP
    samples = imposed[result.settling_samples:]
    means = analysis.block_means(samples)
    left, right = figure.subplots(1, 2)
    times = (np.arange(means.size) + 0.5) * hw.CHUNK_SECONDS + result.settling_samples / hw.SAMPLE_RATE
    left.plot(times, means, "o-", label="100 ms block means")
    if name in measurement.prepared.predictions:
        target = float(measurement.prepared.predictions[name].delivered[0])
        left.axhline(target, linestyle=":", color="tab:orange", label="DAC-delivered")
    left.set_xlabel("time (s)")
    left.set_ylabel(f"{imposed_name} ({imposed_unit})")
    left.legend(loc="best")
    if samples.size:
        low, high = np.min(samples), np.max(samples)
        edges = np.arange(np.floor(low / step) - 0.5, np.ceil(high / step) + 1.5) * step
        if edges.size > 2_000:                   # very noisy: let numpy choose
            edges = 200
        right.hist(samples, bins=edges)
        mean, sigma = float(np.mean(samples)), float(np.std(samples))
        for x, style in ((mean, "-"), (mean - sigma, "--"), (mean + sigma, "--")):
            right.axvline(x, color="black", linestyle=style, linewidth=1)
        right.set_xlabel(f"{imposed_name} ({imposed_unit}); bins of one ADC step")
        right.set_ylabel("samples")
    figure.suptitle(rec.describe_channel(name, config))


def _hi_z_periodic(figure: object, recording: rec.Recording, name: str) -> None:
    measurement = recording.measurement
    result = measurement.results[name]
    v, _ = measurement.samples.channel(name)
    n = result.period
    top, bottom = figure.subplots(2, 1)
    top.plot(np.arange(v.size) / hw.SAMPLE_RATE, v, linewidth=0.8)
    _limits(top, "V", v)
    top.set_xlim(0, min(OPENING_PERIODS * n, v.size) / hw.SAMPLE_RATE)
    top.set_xlabel("time (s)")
    top.set_ylabel("voltage (V)")
    if result.v_cycle is not None:
        phase = np.arange(n) / hw.SAMPLE_RATE
        cycle = result.v_cycle
        bottom.fill_between(phase, cycle.mean - cycle.std, cycle.mean + cycle.std, alpha=0.3, label="±1σ")
        bottom.plot(phase, cycle.mean, linewidth=1, label=f"cycle average ({cycle.periods} periods)")
        bottom.set_xlabel("time in the period (s)")
        bottom.set_ylabel("voltage (V)")
        bottom.legend(loc="upper right")
    figure.suptitle(rec.describe_channel(name, measurement.prepared.config(name)))


MEASURING_CIRCUITS = ("shunt", "shunt-node")     # a hi-z channel is plotted when it measures these


def figures(recording: rec.Recording, new_figure: Callable[..., object] | None = None) -> list[object]:
    """One figure per sourcing channel, and per hi-z channel on a shunt; `new_figure` makes a figure
    (pyplot.figure in the window process; matplotlib.figure.Figure otherwise)."""
    import matplotlib
    if new_figure is None:
        from matplotlib.figure import Figure
        new_figure = Figure
    made = []
    with matplotlib.rc_context(NO_OFFSET):          # each axis's formatter takes it when created
        for name in hw.CHANNELS:
            figure = _figure(recording, name, new_figure)
            if figure is not None:
                made.append(figure)
    return made


def _figure(recording: rec.Recording, name: str, new_figure: Callable[..., object]) -> object | None:
    """The figure of one channel, or None when it has none (a hi-z channel on no shunt)."""
    measurement = recording.measurement
    config = measurement.prepared.config(name)
    result = measurement.results[name]
    if config.role is Role.HI_Z and config.circuit.kind.value not in MEASURING_CIRCUITS:
        return None
    figure = new_figure(figsize=(10, 7), layout="constrained")
    if result.period is None:
        _dc(figure, recording, name)
    elif config.role is Role.HI_Z:
        _hi_z_periodic(figure, recording, name)
    else:
        _periodic(figure, recording, name)
    return figure


# --- A calibration (spec 3.7) ---------------------------------------------------------------------

# A calibration's errors are drawn in µV or µA: microvolts are what the M1K's errors are made of.
ERROR_SCALE = 1 / units.PREFIXES["µ"]
BUDGET_SOURCES = 6                     # per channel; the rest is drawn as one "other" bar
CHANNEL_COLORS = {"A": "tab:blue", "B": "tab:orange"}
# The budget's source names, shortened so that the labels leave room for the other panels.
SHORT_NAMES = (("multimeter spread", "DMM spread"), ("multimeter ", "DMM "), ("dc_voltage ", ""),
               ("dc_current ", ""), (" range ", " "))


def _short(source: str) -> str:
    for long, short in SHORT_NAMES:
        source = source.replace(long, short)
    return source


def _number(value: object) -> float:
    return float("nan") if value is None else float(value)


def calibration_figures(meta: dict, new_figure: Callable[..., object] | None = None) -> list[object]:
    """One figure per stage of a calibration folder's metadata (`recording.read_calibration`)."""
    import matplotlib
    if new_figure is None:
        from matplotlib.figure import Figure
        new_figure = Figure
    with matplotlib.rc_context(NO_OFFSET):
        return [_calibration_figure(stage, channels, meta, new_figure)
                for stage, channels in meta["results"].items() if channels]


def _calibration_figure(stage: str, channels: dict, meta: dict, new_figure: Callable[..., object]) -> object:
    figure = new_figure(figsize=(12, 8), layout="constrained")
    (effect, remains), (verify, budget) = figure.subplots(2, 2)
    names = sorted(channels)
    kind = Kind(channels[names[0]]["kind"])
    unit = "V" if kind in (Kind.MEASURE_V, Kind.SOURCE_V) else "A"
    error_scale, error_unit = ERROR_SCALE, f"µ{unit}"
    xs = [row["x"] for row in channels[names[0]]["effect"]]
    width = 0.8 / len(names)
    verifications: list[str] = []
    budget_labels: list[str] = []
    for k, name in enumerate(names):
        item = channels[name]
        rows = item["effect"]
        shift = (k - (len(names) - 1) / 2) * width
        at = np.arange(len(rows)) + shift
        stored = Record(*meta["stored_calibration"][hw.record_index(name, kind)])
        before = "without calibration" if stored == Record() else "stored record"
        color = CHANNEL_COLORS.get(name)
        effect.plot(at, [_number(r["before"]) * error_scale for r in rows], "x", color=color,
                    label=f"CH {name}, {before}")
        effect.plot(at, [_number(r["after"]) * error_scale for r in rows], "o", color=color,
                    label=f"CH {name}, new record")
        remains.errorbar(at, [_number(r["after"]) * error_scale for r in rows], color=color,
                         yerr=[checks.COVERAGE * _number(r["u"]) * error_scale for r in rows],
                         fmt="o", capsize=3, label=f"CH {name}")
        for check in item["verifications"]:
            position = len(verifications)
            verdict = "PASS" if check["passed"] else "FAIL"
            verifications.append(f"CH {name} {check['name']}\n{verdict}")
            verify.errorbar(position, 0.0, yerr=_number(check["limit"]) * error_scale, fmt="none", capsize=8,
                            color="gray")
            verify.plot(position, _number(check["residual"]) * error_scale, "o",
                        color="tab:green" if check["passed"] else "tab:red")
        shares = sorted(item["budget"], key=lambda pair: -pair[1])
        kept, rest = shares[:BUDGET_SOURCES], sum(share for _, share in shares[BUDGET_SOURCES:])
        kept = kept + ([("other sources", rest)] if rest > 0 else [])
        start = len(budget_labels)
        budget_labels += [_short(source) for source, _ in kept]
        budget.barh(np.arange(start, len(budget_labels)), [100 * share for _, share in kept], color=color,
                    label=f"CH {name}")
    labels = [units.show(x, unit) for x in xs]
    for axes in (effect, remains):
        axes.set_xticks(np.arange(len(xs)), labels)
        axes.tick_params(axis="x", labelrotation=45)             # nine current values side by side
        axes.axhline(0.0, color="black", linewidth=0.8)
        axes.set_xlabel("value")
        axes.set_ylabel(f"error ({error_unit})")
        axes.legend(loc="best")
    effect.set_title("Error at each value: before, and with the new record")
    remains.set_title(f"With the new record, ±U (U = {checks.COVERAGE:g}u)")
    verify.set_title("Verifications: the residual (dot) inside its limit, U + one step (bar)")
    verify.axhline(0.0, color="black", linewidth=0.8)
    verify.set_ylabel(f"residual ({error_unit})")
    if verifications:
        verify.set_xticks(np.arange(len(verifications)), verifications)
        verify.set_xlim(-0.5, len(verifications) - 0.5)
    else:
        verify.text(0.5, 0.5, "no verification (the optional points were skipped)", ha="center",
                    va="center", transform=verify.transAxes)
    budget.set_yticks(np.arange(len(budget_labels)), budget_labels)
    budget.invert_yaxis()
    budget.set_xlabel("share of the variance (%)")
    budget.legend(loc="lower right")
    budget.set_title("Uncertainty budget at the main point")
    figure.suptitle(f"Calibration · {stage} · started {meta.get('started', '')}")
    return figure


def main(argv: list[str] | None = None) -> int:
    """Open the window: `python -m phycovolt.plot <recording or calibration folder>`."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m phycovolt.plot <recording or calibration folder>", file=sys.stderr)
        return 1
    import matplotlib
    matplotlib.use("QtAgg")
    matplotlib.rcParams.update(NO_OFFSET)
    import matplotlib.pyplot as plt
    folder = Path(argv[0])
    if rec.is_calibration(folder):
        made = calibration_figures(rec.read_calibration(folder), plt.figure)
    else:
        made = figures(rec.read(folder), plt.figure)
    for figure in made:
        figure.canvas.manager.set_window_title(f"PhycoVolt · {folder.resolve().name}")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
