"""Does libsmu_math compute what libsmu 1.0.3 computes, and does the calib.txt check catch misreads?"""

from __future__ import annotations

import numpy as np
import pytest

from phycovolt import hardware as hw
from phycovolt import libsmu_math as lm
from phycovolt.hardware import Kind, Record

# The 24 coefficients of the calibration stored on 2026-11-18 (spec A.3, confirmed by `smu -d`).
STORED_2026_11_18 = [
    (0.000566, 1.023517, 1.0),
    (0.0005275, 1.019914, 1.017460),
    (0.002752, 1.001784, 1.0),
    (-0.0003932, 1.008366, 1.008412),
    (0.000538, 1.023770, 1.0),
    (0.0005136, 1.020277, 1.018271),
    (0.001327, 1.001694, 1.0),
    (-0.0006588, 1.008778, 1.009436),
]


def stored(project_calib_text: str, channel: str, kind: Kind) -> Record:
    return lm.libsmu_reads(project_calib_text)[hw.record_index(channel, kind)]


# --- The project's calib.txt ----------------------------------------------------------------------


def test_project_calib_passes_the_check(project_calib_text: str) -> None:
    assert lm.calib_problems(project_calib_text) == []


def test_project_calib_reproduces_its_24_coefficients(project_calib_text: str) -> None:
    records = lm.libsmu_reads(project_calib_text)
    assert len(records) == 8
    for record, (offset, gain_pos, gain_neg) in zip(records, STORED_2026_11_18):
        assert record.offset == pytest.approx(offset, abs=5e-8)
        assert record.gain_pos == pytest.approx(gain_pos, abs=5e-7)
        assert record.gain_neg == pytest.approx(gain_neg, abs=5e-7)


def test_strict_parse_and_libsmu_reader_agree(project_calib_text: str) -> None:
    blocks = lm.parse_calib(project_calib_text)
    assert [block.record() for block in blocks] == lm.libsmu_reads(project_calib_text)


# --- C.1: smu -w in float32 -----------------------------------------------------------------------


def test_record_is_computed_in_float32() -> None:
    record = lm.record_from_values([0.0, 2.4451], [0.000566, 2.389485])
    f = np.float32
    offset = f(f(0.000566) - f(0.0))
    gain = f(f(2.4451) / f(f(2.389485) - offset))
    assert record == Record(float(offset), float(gain), 1.0)


def test_gains_are_means_of_ratios_by_the_sign_of_the_reference() -> None:
    record = lm.record_from_values([0.0, 0.1, 0.2, -0.1], [0.001, 0.101, 0.199, -0.099])
    expected_pos = np.mean([0.1 / 0.1, 0.2 / 0.198])
    assert record.gain_pos == pytest.approx(expected_pos, rel=1e-6)
    assert record.gain_neg == pytest.approx(-0.1 / -0.1, rel=1e-6)


def test_strtof_rounds_like_c() -> None:
    assert lm.strtof("0.1") == np.float32(0.1)
    assert lm.strtof("-0.1030510") == np.float32(-0.1030510)


# --- C.2: measurement and its inversion -----------------------------------------------------------


def test_adc_scale_at_the_edges() -> None:
    assert lm.adc_voltage(np.array([0, hw.CODE_MAX])) == pytest.approx([0.0, 4.99992], abs=1e-5)
    assert lm.adc_current(np.array([0, hw.CODE_MAX])) == pytest.approx([-0.24375, 0.25624], abs=1e-5)


def test_voltage_inversion_is_exact(project_calib_text: str) -> None:
    record = stored(project_calib_text, "B", Kind.MEASURE_V)
    raw = lm.adc_voltage(np.arange(0, hw.CODES, 7))
    corrected = lm.correct_voltage(raw, record)
    back = lm.raw_voltage(corrected, record)
    assert np.max(np.abs(back - raw)) < 1e-6          # float32 rounding only (1 ADC step = 76 uV)


def test_current_gain_is_chosen_by_the_previous_corrected_sample() -> None:
    record = Record(0.0, 2.0, 3.0)
    raw = np.array([0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1])
    corrected = lm.correct_current(raw, record, transfer=4)
    #   sample:   0      1      2       3      4 (transfer starts: previous = 0 -> gain-)
    expected = [0.3, 0.2, -0.2, -0.3, 0.3, 0.2, 0.2, -0.2]
    assert corrected == pytest.approx(expected, rel=1e-6)


def test_current_inversion_is_exact_including_transfer_starts(project_calib_text: str) -> None:
    record = stored(project_calib_text, "A", Kind.MEASURE_I)
    rng = np.random.default_rng(1)
    raw = lm.adc_current(rng.integers(30_000, 34_000, size=5_000))   # both signs around zero
    transfer = hw.samples_per_transfer()
    corrected = lm.correct_current(raw, record, transfer)
    back = lm.raw_current(corrected, record, transfer)
    assert np.max(np.abs(back - raw)) < 1e-8


def test_adc_words_undo_the_scale() -> None:
    words = np.array([0, 31_950, hw.CODE_MAX])
    v_word, i_word = lm.adc_words(lm.adc_voltage(words), lm.adc_current(words))
    assert v_word == pytest.approx(words, abs=1e-2)
    assert i_word == pytest.approx(words, abs=1e-2)


# --- C.3: the source ------------------------------------------------------------------------------


def test_zero_amperes_is_code_26214_and_minus_3_8_microamperes() -> None:
    code = lm.dac_code(0.0, Kind.SOURCE_I, Record())
    assert int(code) == hw.DAC_REST_CODE == 26_214
    assert float(lm.code_value(code, Kind.SOURCE_I)) == pytest.approx(-3.815e-6, abs=1e-9)


def test_rest_code_is_two_volts_in_svmi() -> None:
    assert float(lm.code_value(hw.DAC_REST_CODE, Kind.SOURCE_V)) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("request_a, delivered", [(50e-6, 47.52e-6), (100e-6, 94.81e-6)])
def test_micro_amperes_on_channel_a_with_the_stored_calibration(
    project_calib_text: str, request_a: float, delivered: float
) -> None:
    record = stored(project_calib_text, "A", Kind.SOURCE_I)
    assert float(lm.dac_delivers(request_a, Kind.SOURCE_I, record)) == pytest.approx(delivered, abs=0.01e-6)


def test_every_voltage_output_is_up_to_one_step_below_the_request() -> None:
    # pysmu hands libsmu a float: the request is the float32 of what Python asked.
    requests = np.linspace(0.01, 4.99, 2_001).astype(np.float32).astype(np.float64)
    delivered = lm.dac_delivers(requests, Kind.SOURCE_V, Record())
    shortfall = requests - delivered
    assert np.all(shortfall >= 0)
    assert np.all(shortfall < hw.V_STEP)


def test_codes_are_clamped() -> None:
    assert lm.dac_code(np.array([-1.0, 6.0]), Kind.SOURCE_V, Record()).tolist() == [0, hw.CODE_MAX]
    codes = lm.dac_code(np.array([-0.5, 0.5]), Kind.SOURCE_I, Record())
    assert codes.tolist() == [
        int(hw.CODES * (hw.DAC_I_BASE - hw.DAC_I_SCALE_LIBSMU * np.float32(0.2))),
        int(hw.CODES * (hw.DAC_I_BASE + hw.DAC_I_SCALE_LIBSMU * np.float32(0.2))),
    ]


def test_current_gain_is_chosen_by_the_request_sign() -> None:
    record = Record(0.0, 1.1, 0.9)
    positive = lm.dac_code(0.1, Kind.SOURCE_I, record)
    negative = lm.dac_code(-0.1, Kind.SOURCE_I, record)
    assert float(lm.code_value(positive, Kind.SOURCE_I)) == pytest.approx(0.11, abs=2 * hw.DAC_I_STEP)
    assert float(lm.code_value(negative, Kind.SOURCE_I)) == pytest.approx(-0.09, abs=2 * hw.DAC_I_STEP)


@pytest.mark.parametrize("kind", [Kind.SOURCE_V, Kind.SOURCE_I])
@pytest.mark.parametrize("channel", ["A", "B"])
def test_request_for_code_reaches_the_code(project_calib_text: str, kind: Kind, channel: str) -> None:
    record = stored(project_calib_text, channel, kind)
    # SIMV codes only span +-0.2 A: 65536 * (0.4 +- 1.6 * 0.2) = 5 243 to 47 185.
    low, high = (100, hw.CODES - 100) if kind is Kind.SOURCE_V else (5_400, 47_000)
    for code in range(low, high, 997):
        request = lm.request_for_code(code, kind, record)
        assert int(lm.dac_code(request, kind, record)) == code


def test_request_for_a_code_beyond_the_clamp_is_refused() -> None:
    with pytest.raises(ValueError):
        lm.request_for_code(47_874, Kind.SOURCE_I, Record())           # 0.2066 A


def test_source_output_with_the_true_record() -> None:
    true = Record(0.002, 1.01, 1.0)
    out = lm.source_output(32_768, Kind.SOURCE_V, true)       # code value 2.5 V
    assert float(out) == pytest.approx(0.002 + 2.5 / 1.01)


# --- calib.txt: writing, refusing, merging --------------------------------------------------------


def test_round_trip(project_calib_text: str) -> None:
    blocks = lm.parse_calib(project_calib_text)
    text = lm.format_calib(blocks, ["round trip"])
    assert lm.parse_calib(text) == blocks
    assert lm.libsmu_reads(text) == lm.libsmu_reads(project_calib_text)


def test_written_comments_carry_the_float32_coefficients(project_calib_text: str) -> None:
    text = lm.format_calib(lm.parse_calib(project_calib_text))
    # smu -d would show 1.0084 and 1.0084 (P-05); the comment shows the float32 values.
    assert "# Channel A, source I   ->  offset -0.0003932, gain+ 1.0083665, gain- 1.0084121" in text


def _replace(text: str, old: str, new: str) -> str:
    assert old in text
    return text.replace(old, new, 1)


BAD_EDITS = {
    "leading space": ("<2.445100, 2.389485>", " <2.445100, 2.389485>", "not_a_pair"),
    "comment in a block": ("<2.445100, 2.389485>", "# reference\n<2.445100, 2.389485>", "not_a_pair"),
    "blank line in a block": ("<2.445100, 2.389485>", "\n<2.445100, 2.389485>", "not_a_pair"),
    "no space after the comma": ("<2.445100, 2.389485>", "<2.445100,2.389485>", "not_a_pair"),
    "marker in a comment": ("# Channel A, measure V", "# Channel A </> measure V", "marker_in_comment"),
    "zero not first": ("<0.0000, 0.000566>", "<0.0010, 0.000566>", "first_pair_not_zero"),
    "zero repeated": ("<2.445100, 2.389485>", "<0.0000, 2.389485>", "zero_repeated"),
    "current with one sign": ("<-0.1030510, -0.1007551>\n", "", "current_needs_both_signs"),
    "line too long": ("<2.445100, 2.389485>", "<2.445100, 2.389485" + "0" * 110 + ">", "line_too_long"),
}


@pytest.mark.parametrize("name", BAD_EDITS)
def test_misreadable_files_are_refused(project_calib_text: str, name: str) -> None:
    old, new, code = BAD_EDITS[name]
    problems = lm.calib_problems(_replace(project_calib_text, old, new))
    assert code in [problem.code for problem in problems]


def test_a_missing_block_is_refused(project_calib_text: str) -> None:
    start = project_calib_text.index("# Channel B, source I")
    problems = lm.calib_problems(project_calib_text[:start])
    assert "block_count" in [problem.code for problem in problems]


def test_too_many_pairs_are_refused(project_calib_text: str) -> None:
    extra = "".join(f"<{2.0 + k / 100:.6f}, {2.0 + k / 100:.6f}>\n" for k in range(16))
    text = _replace(project_calib_text, "<2.445100, 2.389485>\n", "<2.445100, 2.389485>\n" + extra)
    assert "pair_count" in [problem.code for problem in lm.calib_problems(text)]


def test_libsmu_counts_a_comment_in_a_block_as_a_repeated_point(project_calib_text: str) -> None:
    # Why the check exists: libsmu would compute another gain without any warning.
    text = _replace(project_calib_text, "<0.1015730, 0.1001173>", "<0.1015730, 0.1001173>\n# note")
    good = stored(project_calib_text, "A", Kind.MEASURE_I)
    misread = lm.libsmu_reads(text)[hw.record_index("A", Kind.MEASURE_I)]
    assert misread.gain_pos == pytest.approx(good.gain_pos, rel=1e-6)   # the duplicate repeats gain+
    assert lm.calib_problems(text) != []


def test_libsmu_reads_the_rest_of_a_long_line_as_another_point(project_calib_text: str) -> None:
    # fgets(128) returns the first 127 bytes, then the rest: "<1.000000, 0.977000>" becomes a point.
    long_pair = "<2.445100, 2.389485>" + " " * 107 + "<1.000000, 0.977000>"
    text = _replace(project_calib_text, "<2.445100, 2.389485>", long_pair)
    record = lm.libsmu_reads(text)[hw.record_index("A", Kind.MEASURE_V)]
    assert record != stored(project_calib_text, "A", Kind.MEASURE_V)


def test_merge_keeps_the_base_blocks(project_calib_text: str) -> None:
    base = lm.parse_calib(project_calib_text)
    new_block = lm.CalibBlock((("0.0000", "0.000600"), ("2.445000", "2.390000")))
    merged = lm.merge_blocks(base, {0: new_block})
    assert merged[0] == new_block
    assert merged[1:] == base[1:]


def test_format_refuses_seven_blocks(project_calib_text: str) -> None:
    with pytest.raises(ValueError):
        lm.format_calib(lm.parse_calib(project_calib_text)[:7])


def test_pair_formatting() -> None:
    assert lm.format_pair(0.0, 0.000566, Kind.MEASURE_V, first=True) == ("0.0000", "0.000566")
    assert lm.format_pair(0.101573, 0.1001173, Kind.MEASURE_I) == ("0.1015730", "0.1001173")
    assert lm.format_pair(2.5, 2.4983, Kind.SOURCE_V) == ("2.5000", "2.498300")
    assert lm.format_pair(-0.1, -0.099559, Kind.SOURCE_I) == ("-0.1000", "-0.0995590")
