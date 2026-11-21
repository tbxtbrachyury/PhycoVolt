"""The calibration acceptance scenarios C1 to C14 (spec 9.2), on the simulated bench.

The simulated operator wires the simulated bench as each step asks; the virtual multimeter reads
it with range errors drawn within the 34401A's specification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phycovolt import calibration as cal
from phycovolt import hardware as hw
from phycovolt import libsmu_math
from phycovolt import multimeter
from phycovolt import recording as rec
from phycovolt import simulator as sim
from phycovolt.hardware import Kind, Record

PROFILE = multimeter.load_profile(multimeter.PROFILES / "agilent_34401a.ini")
UNIT = hw.load_unit()
PROJECT_CALIB = Path(__file__).resolve().parents[1] / "calib.txt"


def bench(faults: sim.Faults | None = None, typed: bool = False, seed: int = 1, **operator: object
          ) -> tuple[cal.Context, sim.SimulatedOperator, sim.VirtualMultimeter]:
    m1k = sim.SimulatedM1K(faults=faults, seed=seed)
    dmm = sim.VirtualMultimeter(m1k, PROFILE, seed=seed + 1)
    person = sim.SimulatedOperator(m1k, dmm, **operator)
    return cal.Context(m1k, None if typed else dmm, PROFILE, person, m1k.clock, m1k.calibration()), person, dmm


def within_u(found: Record, u: tuple[float, float, float], true: Record) -> bool:
    pairs = zip((found.offset, found.gain_pos, found.gain_neg), (true.offset, true.gain_pos, true.gain_neg), u)
    return all(abs(a - b) <= 2 * s + 1e-6 * abs(b) + 1e-9 for a, b, s in pairs)


@pytest.fixture(scope="module")
def full_run() -> cal.CalibrationRun:
    context, _, _ = bench()
    return cal.calibrate(context, list(cal.STAGES))


# --- C1 to C4: each stage, the right wiring ------------------------------------------------------


@pytest.mark.parametrize("stage", cal.STAGES)
def test_c1_to_c4_true_records_within_u_and_verifications_pass(full_run: cal.CalibrationRun, stage: str) -> None:
    assert full_run.stopped is None
    result = full_run.stages[stage]
    for channel, item in result.channels.items():
        true = UNIT.true_records[hw.record_index(channel, item.kind)]
        if item.kind is Kind.SOURCE_I:
            # The zero pair is written "0.0000" (B.1) while code 26 214 is -3.8 uA: the written offset
            # is the source's output at that code.
            zero = float(libsmu_math.source_output(hw.DAC_REST_CODE, Kind.SOURCE_I, true))
            true = Record(zero, true.gain_pos, true.gain_neg)
        assert within_u(item.record, item.record_u, true), (stage, channel, item.record, true, item.record_u)
        assert item.verifications and all(v.passed for v in item.verifications)
        assert item.budget and abs(sum(share for _, share in item.budget) - 1) < 0.01


def test_the_multimeter_dominates_measure_v(full_run: cal.CalibrationRun) -> None:   # H-15
    budget = dict(full_run.stages["measure-v"].channels["A"].budget)
    assert budget["multimeter dc_voltage 10 V range gain"] + budget["multimeter dc_voltage 10 V range zero"] > 0.9


def test_effect_table_values(full_run: cal.CalibrationRun) -> None:
    xs = [row.x for row in full_run.stages["source-i"].channels["A"].effect]
    assert xs == [-0.15, -0.1, -0.01, -1e-3, 0.0, 1e-3, 0.01, 0.1, 0.15]
    reference = full_run.stages["measure-v"].channels["A"].effect[3].x
    assert reference == pytest.approx(UNIT.pins["2.5V"], abs=1e-3)


def test_measure_i_slope_is_measured(full_run: cal.CalibrationRun) -> None:
    slope = full_run.stages["measure-i"].channels["A"].slope
    assert slope == pytest.approx(UNIT.current_zero_slope, abs=3e-6)


def test_every_reading_is_logged(full_run: cal.CalibrationRun) -> None:
    points = {row.point for row in full_run.readings}
    assert {"zero", "reference", "A 0V", "A +fit", "B -verify", "A probe 3V"} <= points
    assert all(row.status == "accepted" for row in full_run.readings)
    assert len(full_run.dmm_rows) > 100


# --- C5 to C12 -----------------------------------------------------------------------------------


def test_c5_a_crooked_channel_fails_its_verification() -> None:
    context, _, _ = bench(sim.Faults(crooked_v_at_5v={"B": 3e-3}))
    result = cal.calibrate(context, ["measure-v"]).stages["measure-v"]
    assert result.channels["A"].verifications[0].passed
    assert not result.channels["B"].verifications[0].passed


def test_c6_typed_value_with_two_digits_swapped_is_redone() -> None:
    context, person, _ = bench(typed=True, swap_digits_once=True)
    run = cal.calibrate(context, ["measure-v"])
    assert run.stopped is None
    assert any(w.variant == "odd_typed" for w in run.warnings)
    reference = [row for row in run.dmm_rows if row.point == "reference"]
    assert len(reference) == 4                                   # three, and the redone one
    item = run.stages["measure-v"].channels["A"]
    assert within_u(item.record, item.record_u, UNIT.true_records[0])


def test_c7_a_jump_in_one_block_redoes_the_reading() -> None:
    context, _, _ = bench(sim.Faults(jump=("A", "v", 20, 1e-3)))
    run = cal.calibrate(context, ["measure-v"])
    zero = [row for row in run.readings if row.point == "zero" and row.channel == "A"]
    assert [row.status for row in zero] == ["reading:jump", "accepted"]
    assert run.stopped is None


def test_c8_vi_swap_stops_the_stage() -> None:
    context, _, _ = bench(sim.Faults(vi_swaps_per_second={"B": 5}))
    run = cal.calibrate(context, ["measure-v"])
    assert run.stopped is not None and run.stopped.variant == "vi_swap"
    assert "measure-v" not in run.stages


def test_c9_one_stage_merged_into_the_base(tmp_path: Path) -> None:
    context, _, _ = bench()
    run = cal.calibrate(context, ["measure-v"])
    folder = rec.write_calibration_folder(run, PROJECT_CALIB, tmp_path, cal_started(), "simulated", PROFILE.name)
    written = libsmu_math.parse_calib((folder / rec.CALIB_FILE).read_text(encoding="utf-8"))
    base = libsmu_math.parse_calib(PROJECT_CALIB.read_text(encoding="utf-8"))
    for index, block in enumerate(written):
        if index in (hw.record_index("A", Kind.MEASURE_V), hw.record_index("B", Kind.MEASURE_V)):
            assert block == run.stages["measure-v"].channels[hw.RECORD_ORDER[index][0]].block
        else:
            assert block == base[index]
    assert PROJECT_CALIB.read_text(encoding="utf-8") == PROJECT_CALIB.read_text(encoding="utf-8")


def cal_started():
    from datetime import datetime
    return datetime(2026, 11, 22, 10, 0, 0)


def test_c10_files_of_a_complete_calibration(tmp_path: Path, full_run: cal.CalibrationRun) -> None:
    folder = rec.write_calibration_folder(full_run, None, tmp_path, cal_started(), "simulated", PROFILE.name)
    assert folder.name == "2026-11-22_100000_all"
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["calib.txt", "blocks.txt", "readings.csv", "dmm.csv", "report.txt", "metadata.json"])
    text = (folder / rec.CALIB_FILE).read_text(encoding="utf-8")
    assert libsmu_math.calib_problems(text) == []
    assert len(libsmu_math.libsmu_reads(text)) == 8
    report = (folder / rec.REPORT_FILE).read_text(encoding="utf-8")
    assert f"smu -w {folder / rec.CALIB_FILE}" in report and "smu -d" in report
    assert "NOT for smu -w" in (folder / rec.BLOCKS_FILE).read_text(encoding="utf-8")
    # Every line fits in 100 columns, except the smu command, which must stay one line to be copied.
    assert all(len(line) <= rec.WIDTH for line in report.splitlines() if "smu -w " not in line)


def test_c11_the_multimeter_stops_answering() -> None:
    context, person, dmm = bench()
    dmm.fail_reads = 3                     # three readings fail in a row, then it answers again
    run = cal.calibrate(context, ["measure-v"])
    statuses = [row.status for row in run.readings if row.point == "reference" and row.channel == "A"]
    assert statuses == ["reading:dmm_failed"] * 3 + ["accepted"]
    assert [p.variant for p in person.problems] == ["dmm_pause"]          # the stage paused and asked
    assert run.stopped is None


def test_c12_eeprom_at_the_default() -> None:
    context, _, _ = bench(sim.Faults(eeprom_default=True))
    run = cal.calibrate(context, ["measure-v", "source-v"])
    for stage in ("measure-v", "source-v"):
        item = run.stages[stage].channels["A"]
        assert item.before_label == "no calibration"
        assert within_u(item.record, item.record_u, UNIT.true_records[hw.record_index("A", item.kind)])
    assert abs(run.stages["measure-v"].channels["A"].effect[3].before) > 0.05     # 2.3 % low (H-02)


# --- Wiring mistakes the physics checks catch ----------------------------------------------------


def test_the_multimeter_on_the_wrong_channel_is_caught() -> None:
    context, person, _ = bench(miswire={"dmm_voltage": ("v", "B")})
    run = cal.calibrate(context, ["source-v"])
    assert run.stopped is not None and run.stopped.variant == "attempts"
    # On its 100 mV range (0 V expected) it sees the open CH B's 0.34 V: an overload, the wiring.
    assert person.problems[0].variant == "dmm_overload"


def test_an_open_current_loop_is_caught_by_the_probe() -> None:            # B-01
    context, person, _ = bench(loop_ohms=1e6)
    run = cal.calibrate(context, ["source-i"])
    assert run.stopped is not None
    assert person.problems[0].variant == "probe_open"


def test_skipping_the_optional_points() -> None:
    context, _, _ = bench()
    context.skip_optional = True
    run = cal.calibrate(context, ["measure-i"])
    assert run.stopped is None
    item = run.stages["measure-i"].channels["A"]
    assert item.verifications == () and item.slope == 0.0
    assert [note.variant for note in run.stages["measure-i"].notes] == ["aux_skipped"]


def test_the_generated_calib_txt_is_what_libsmu_reads(full_run: cal.CalibrationRun) -> None:
    blocks = cal.complete_calib(full_run, None)
    text = libsmu_math.format_calib(blocks, ["test"])
    found = libsmu_math.libsmu_reads(text)
    for result in full_run.stages.values():
        for channel, item in result.channels.items():
            assert found[hw.record_index(channel, item.kind)] == item.record


# --- The multimeter is set to DC current before the load probe -----------------------------------


def test_the_multimeter_is_set_to_current_before_each_probe() -> None:
    context, person, dmm = bench()
    dmm.configure(multimeter.Function.DC_VOLTAGE, 10.0)          # as source-v leaves it
    events: list[tuple] = []
    configure, acquiring = dmm.configure, person.acquiring

    def logged_configure(function: multimeter.Function, full_scale: float) -> None:
        events.append(("configure", function, full_scale))
        configure(function, full_scale)

    def logged_acquiring(stage: str, point: str, attempt: int, seconds: float) -> None:
        events.append(("acquiring", point))
        acquiring(stage, point, attempt, seconds)

    dmm.configure, person.acquiring = logged_configure, logged_acquiring
    run = cal.calibrate(context, ["measure-i"])
    assert run.stopped is None
    for channel in "AB":
        probe = events.index(("acquiring", f"{channel} probe 3V"))
        configured = [event for event in events[:probe] if event[0] == "configure"]
        assert configured[-1] == ("configure", multimeter.Function.DC_CURRENT, 0.1)


def test_typed_mode_asks_the_operator_to_set_current_before_the_probe() -> None:
    context, person, _ = bench(typed=True)
    context.skip_optional = True
    cal.calibrate(context, ["source-i"])
    kinds = [step.kind for step in person.steps]
    assert kinds[:2] == ["current_loop", "dmm_current"]
    assert person.steps[1].full_scale == 0.1


# --- I-16: nothing is sourced while the operator mounts ------------------------------------------


def test_both_channels_are_in_hi_z_at_every_prompt() -> None:
    """I-16: at every wiring prompt, also right after points that sourced, both channels are
    in HI_Z, so the operator mounts and unmounts with nothing applied."""
    context, person, _ = bench()
    wire, seen = person.wire, []

    def look(step: cal.WiringStep) -> bool:
        seen.append((step.kind, step.channel, dict(context.instrument.modes)))
        return wire(step)

    person.wire = look
    skipping = cal.Context(context.instrument, context.dmm, PROFILE, person, context.clock, context.stored, True)
    cal.calibrate(skipping, ["measure-i"])            # SVMI zeros, then a loop per channel
    assert [(kind, channel) for kind, channel, _ in seen] == [("open", None), ("current_loop", "A"),
                                                              ("current_loop", "B")]
    assert all(modes == {"A": hw.Role.HI_Z, "B": hw.Role.HI_Z} for _, _, modes in seen)


# --- An interrupted calibration keeps what finished ----------------------------------------------


def test_an_interrupted_calibration_keeps_what_finished(tmp_path: Path) -> None:
    context, person, _ = bench()
    wire = person.wire

    def ctrl_c_at_source_v(step: cal.WiringStep) -> bool:
        if step.stage == "source-v":
            raise KeyboardInterrupt
        return wire(step)

    person.wire = ctrl_c_at_source_v
    run = cal.calibrate(context, ["measure-v", "source-v"])
    assert run.interrupted and list(run.stages) == ["measure-v"]
    assert {row.point for row in run.readings} >= {"zero", "reference"}
    assert context.instrument.modes == {"A": hw.Role.HI_Z, "B": hw.Role.HI_Z}
    folder = rec.write_calibration_folder(run, PROJECT_CALIB, tmp_path, cal_started(), "simulated", PROFILE.name)
    assert folder.name == "2026-11-22_100000_measure-v_interrupted"
    import json
    meta = json.loads((folder / rec.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["ended"] == "interrupted" and meta["interrupted"] is True and meta["stages"] == ["measure-v"]
    assert (folder / rec.CALIB_FILE).exists() and (folder / rec.READINGS_FILE).exists()
    assert "interrupted" in (folder / rec.REPORT_FILE).read_text(encoding="utf-8")


def test_a_stage_that_stops_is_marked_in_the_folder_name(tmp_path: Path) -> None:
    context, _, _ = bench(sim.Faults(vi_swaps_per_second={"B": 5}))
    run = cal.calibrate(context, ["measure-v"])
    folder = rec.write_calibration_folder(run, None, tmp_path, cal_started(), "simulated", PROFILE.name)
    assert folder.name.endswith("_interrupted")


# --- C13, C14: the 2026-11-21 bench findings ------------------------------------------------------


def test_c13_the_bend_near_5v_fails_the_verification_and_the_report_names_h06(tmp_path: Path) -> None:
    context, _, _ = bench(sim.Faults(source_nonlinearity=True))
    run = cal.calibrate(context, ["source-v"])
    for channel, size in (("A", -0.29e-3), ("B", -1.37e-3)):
        [verification] = run.stages["source-v"].channels[channel].verifications
        assert not verification.passed
        assert verification.residual == pytest.approx(size, abs=0.1e-3)
    folder = rec.write_calibration_folder(run, PROJECT_CALIB, tmp_path, cal_started(), "simulated", PROFILE.name)
    report = " ".join((folder / rec.REPORT_FILE).read_text(encoding="utf-8").split())
    assert report.count("(H-06)") == 2 and "-290.0 µV on CH A and -1.370 mV on CH B" in report


def test_c14_fewer_decimals_than_the_display_shows() -> None:
    assert cal.display_decimals(PROFILE, multimeter.Function.DC_VOLTAGE, 10.0) == ("V", 5)
    assert cal.display_decimals(PROFILE, multimeter.Function.DC_CURRENT, 0.1) == ("mA", 4)
    budgets = {}
    for fewer in (0, 1):
        context, _, _ = bench(typed=True, fewer_decimals=fewer)
        run = cal.calibrate(context, ["measure-v"])
        warned = [w for w in run.warnings if w.variant == "few_decimals"]
        assert bool(warned) == bool(fewer)
        if warned:
            assert "typed with 4 decimals, and the display shows 5" in warned[0].text
        budgets[fewer] = dict(run.stages["measure-v"].channels["A"].budget).get("multimeter spread at reference", 0.0)
    assert budgets[1] > 0.03 > budgets[0]                   # the coarser resolution enters the budget
