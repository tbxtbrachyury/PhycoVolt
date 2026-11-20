"""The arithmetic libsmu 1.0.3 does with the calibration and the DAC, and the calib.txt file.

Questions it answers:

- What does libsmu do to a sample before we see it, and how do we undo it (spec C.2)?
- Which DAC code does libsmu send for a requested value, and what comes out of the M1K (C.3)?
- Which records does `smu -w` compute from a calib.txt (C.1), and would it misread a file (B.1)?

Main choices: everything is computed the way libsmu computes it, in float32 where libsmu uses
`float` and in double where it uses `double`, with C truncation, so that predictions match the
instrument code for code (read in libsmu 1.0.3's src/device_m1000.cpp on 2026-11-20). The libsmu
quirks that matter are reproduced, not fixed: the current gain chosen by the sign of the PREVIOUS
corrected sample, reset at every USB transfer; the DAC truncation; `smu -w` averaging ratios. The
calib.txt validator is stricter than libsmu's reader: it refuses every file that libsmu would read
differently from what it says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Sequence

import numpy as np

from phycovolt import hardware as hw
from phycovolt.hardware import Kind, Record

F32 = np.float32


def _f32(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# --- Measurement (C.2) ----------------------------------------------------------------------------


def adc_voltage(word: np.ndarray) -> np.ndarray:
    """Raw voltage of ADC words, as libsmu computes it (float32)."""
    return _f32(np.asarray(word, dtype=np.float64) * hw.V_STEP)


def adc_current(word: np.ndarray) -> np.ndarray:
    """Raw current of ADC words, as libsmu computes it (in double, stored as float32)."""
    span = hw.ADC_I_SPAN / hw.CODES
    return _f32((np.asarray(word, dtype=np.float64) * span - hw.ADC_I_OFFSET) * hw.ADC_I_GAIN)


def correct_voltage(raw: np.ndarray, record: Record) -> np.ndarray:
    """libsmu's correction of a measured voltage: (raw - offset) * gain+ (float32)."""
    return (_f32(raw) - F32(record.offset)) * F32(record.gain_pos)


def _previous_positive(difference: np.ndarray, transfer: int) -> np.ndarray:
    """For each sample: was the previous corrected sample > 0? False at every transfer's start."""
    previous = np.empty(difference.shape, dtype=bool)
    previous[0] = False
    previous[1:] = difference[:-1] > 0
    previous[::transfer] = False
    return previous


def correct_current(raw: np.ndarray, record: Record, transfer: int | None = None) -> np.ndarray:
    """libsmu's correction of a measured current (float32).

    (raw - offset) * g, with g = gain+ if the previous corrected sample of the channel is > 0, else
    gain-; the previous sample counts as 0 at the start of every USB transfer (`transfer` samples,
    hardware.samples_per_transfer). With positive gains, the corrected sample has the sign of
    raw - offset, which makes this computable without a loop.
    """
    transfer = transfer or hw.samples_per_transfer()
    difference = _f32(raw) - F32(record.offset)
    gain = np.where(_previous_positive(difference, transfer), F32(record.gain_pos), F32(record.gain_neg))
    return difference * gain.astype(np.float32)


def raw_voltage(corrected: np.ndarray, record: Record) -> np.ndarray:
    """Undo libsmu's voltage correction (I-09): raw = corrected / gain+ + offset."""
    return np.asarray(corrected, dtype=np.float64) / record.gain_pos + record.offset


def raw_current(corrected: np.ndarray, record: Record, transfer: int | None = None) -> np.ndarray:
    """Undo libsmu's current correction (I-09), choosing the gain exactly as libsmu did."""
    transfer = transfer or hw.samples_per_transfer()
    corrected = np.asarray(corrected, dtype=np.float64)
    gain = np.where(_previous_positive(corrected, transfer), record.gain_pos, record.gain_neg)
    return corrected / gain + record.offset


def adc_words(v_raw: np.ndarray, i_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ADC words (as floats) behind raw voltage and current samples; used to find the V/I swap."""
    v_word = np.asarray(v_raw, dtype=np.float64) / hw.V_STEP
    i_word = (np.asarray(i_raw, dtype=np.float64) - hw.ADC_I_MIN) / hw.ADC_I_STEP
    return v_word, i_word


# --- Source (C.3) ---------------------------------------------------------------------------------


def dac_code(request: np.ndarray, kind: Kind, record: Record) -> np.ndarray:
    """The DAC code libsmu sends for each requested value (SVMI: volts; SIMV: amperes).

    Voltage: c = (request - offset) * gain+, clamped to 0..5 V; code = trunc(c / step).
    Current: gain+ if the request is > 0, else gain-; c clamped to +-0.2 A;
    code = trunc(65536 * (0.4 + 1.6 * c)). The code is clamped to 0..65535.
    """
    request = _f32(request)
    if kind is Kind.SOURCE_V:
        corrected = (request - F32(record.offset)) * F32(record.gain_pos)
        corrected = np.clip(corrected, F32(hw.V_MIN), F32(hw.V_MAX))
        code = np.trunc(corrected.astype(np.float64) * (1 / hw.V_STEP))
    elif kind is Kind.SOURCE_I:
        gain = np.where(request > 0, F32(record.gain_pos), F32(record.gain_neg)).astype(np.float32)
        corrected = (request - F32(record.offset)) * gain
        corrected = np.clip(corrected, F32(hw.I_MIN), F32(hw.I_MAX))
        code = np.trunc(hw.CODES * (hw.DAC_I_BASE + hw.DAC_I_SCALE_LIBSMU * corrected.astype(np.float64)))
    else:
        raise ValueError(f"{kind} is not a source")
    return np.clip(code, 0, hw.CODE_MAX).astype(np.int64)


def code_value(code: np.ndarray, kind: Kind) -> np.ndarray:
    """The raw value a DAC code represents (V or A): the request the M1K really receives (I-11)."""
    code = np.asarray(code, dtype=np.float64)
    if kind is Kind.SOURCE_V:
        return code * hw.V_STEP
    if kind is Kind.SOURCE_I:
        return (code / hw.CODES - hw.DAC_I_BASE) / hw.DAC_I_SCALE
    raise ValueError(f"{kind} is not a source")


def source_output(code: np.ndarray, kind: Kind, record: Record) -> np.ndarray:
    """What a source whose behaviour is `record` puts out for a DAC code.

    output = offset + value / gain, with gain+ for a positive value and gain- otherwise (the same
    model `smu -w` fits to the <request, multimeter> pairs). With the unit's true record this is the
    true output; with the stored record it is "what the DAC delivers" (C.3).
    """
    value = code_value(code, kind)
    gain = np.where(value > 0, record.gain_pos, record.gain_neg) if kind is Kind.SOURCE_I else record.gain_pos
    return record.offset + value / gain


def dac_delivers(request: np.ndarray, kind: Kind, record: Record) -> np.ndarray:
    """What the DAC delivers for a request, assuming the stored calibration `record` is exact."""
    return source_output(dac_code(request, kind, record), kind, record)


def request_for_code(code: int, kind: Kind, record: Record) -> float:
    """The request that makes libsmu, with `record` stored, send exactly DAC code `code` (I-09).

    The request is pre-distorted (C.3) so that libsmu's correction lands in the middle of the code,
    which keeps float32 rounding away from the truncation edge; the result is checked.
    """
    target = float(code_value(code + 0.5, kind))
    request = target / record.gain_pos + record.offset
    if kind is Kind.SOURCE_I and not F32(request) > 0:
        request = target / record.gain_neg + record.offset
    if int(dac_code(request, kind, record)) != code:
        raise ValueError(f"no request gives code {code} with {record}")
    return request


# --- calib.txt: reading as libsmu does (C.1, B.1) -------------------------------------------------

FGETS_SIZE = 128                      # libsmu reads lines with fgets(str, 128, fp)
MAX_LINE_BYTES = FGETS_SIZE - 2       # 126 bytes, plus the newline, fit in one fgets
MAX_PAIRS = 16
BLOCKS = len(hw.RECORD_ORDER)
BLOCK_OPEN = "</>"
BLOCK_CLOSE = "<\\>"

_FLOAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_SCANF_FIRST = re.compile(rf"<\s*({_FLOAT})")      # sscanf "<%f": '<' in the first column
_SCANF_SECOND = re.compile(rf",\s*({_FLOAT})")     # then ", %f": the comma right after the number
_STRICT_PAIR = re.compile(rf"<({_FLOAT}), ({_FLOAT})>")


def strtof(text: str) -> np.float32:
    """A decimal text as C's strtof/sscanf("%f") reads it: the nearest float32."""
    exact = Fraction(Decimal(text))
    nearest = F32(float(exact))
    candidates = (np.nextafter(nearest, F32(-np.inf)), nearest, np.nextafter(nearest, F32(np.inf)))
    return min(candidates, key=lambda value: abs(Fraction(float(value)) - exact))


def _fgets_lines(text: str) -> list[str]:
    """The pieces libsmu's fgets(128) returns: lines, with lines over 127 bytes split."""
    pieces = []
    for line in text.encode("utf-8").splitlines(keepends=True):
        while len(line) > FGETS_SIZE - 1:
            pieces.append(line[: FGETS_SIZE - 1])
            line = line[FGETS_SIZE - 1:]
        pieces.append(line)
    return [piece.decode("utf-8", errors="replace") for piece in pieces]


def record_from_values(refs: Sequence[float], vals: Sequence[float], float32: bool = True) -> Record:
    """`smu -w`'s computation for one block (C.1), in float32 as libsmu does it.

    offset = val0 - ref0; gain+ = mean of ref / (val - offset) over the pairs after the first with
    ref > 0; gain- the same over ref <= 0; a gain with no pairs is 1. With `float32=False` the same
    steps run in double precision: the uncertainty propagation (C.6) uses that, so that float32's
    rounding does not hide perturbations of a few microvolts.
    """
    number = F32 if float32 else float
    refs = [number(ref) for ref in refs]
    vals = [number(val) for val in vals]
    offset = number(vals[0] - refs[0])
    gain_pos, gain_neg, count_pos, count_neg = number(0), number(0), 0, 0
    for ref, val in zip(refs[1:], vals[1:]):
        ratio = number(ref / number(val - offset))
        if ref > 0:
            gain_pos, count_pos = number(gain_pos + ratio), count_pos + 1
        else:
            gain_neg, count_neg = number(gain_neg + ratio), count_neg + 1
    return Record(
        float(offset),
        float(number(gain_pos / number(count_pos))) if count_pos else 1.0,
        float(number(gain_neg / number(count_neg))) if count_neg else 1.0,
    )


def libsmu_reads(text: str) -> list[Record]:
    """The records libsmu 1.0.3's `smu -w` would compute from a calib.txt, quirks included.

    A block opens on any line containing "</>" and closes on a line containing "<\\>" once it has a
    point; every other line in a block is parsed with sscanf("<%f, %f>") and counts as a point even
    when the parse fails (the failed values keep what the previous line left). This function says
    what libsmu would do; `calib_problems` says whether the file is safe to give it.
    """
    records: list[Record] = []
    pieces = iter(_fgets_lines(text))
    for piece in pieces:
        if BLOCK_OPEN not in piece:
            continue
        refs: list[float] = []
        vals: list[float] = []
        ref = val = 0.0
        for inner in pieces:
            if BLOCK_CLOSE in inner and refs:
                break
            first = _SCANF_FIRST.match(inner)
            if first:
                ref = float(strtof(first.group(1)))
                second = _SCANF_SECOND.match(inner, first.end())
                if second:
                    val = float(strtof(second.group(1)))
            refs.append(ref)
            vals.append(val)
        if refs:
            records.append(record_from_values(refs, vals))
    return records


# --- calib.txt: what PhycoVolt writes, and the strict check -----------------------------------------


@dataclass(frozen=True)
class CalibBlock:
    """One block of calib.txt: its pairs as written (text), and the comment line above it."""

    pairs: tuple[tuple[str, str], ...]
    comment: str = ""

    def values(self) -> tuple[list[float], list[float]]:
        refs = [float(strtof(a)) for a, _ in self.pairs]
        vals = [float(strtof(b)) for _, b in self.pairs]
        return refs, vals

    def record(self) -> Record:
        return record_from_values(*self.values())


@dataclass(frozen=True)
class CalibProblem:
    """Something in a calib.txt that libsmu would misread, or that breaks B.1. `line` is 1-based."""

    code: str
    line: int
    detail: str = ""


def _pair_decimals(kind: Kind, first: bool) -> tuple[int, int]:
    """Decimals of (a, b) in a pair (B.1 rule 6)."""
    measured = 7 if kind in (Kind.MEASURE_I, Kind.SOURCE_I) else 6
    if kind in (Kind.SOURCE_V, Kind.SOURCE_I):
        return 4, measured
    return (4 if first else measured), measured


def format_pair(a: float, b: float, kind: Kind, first: bool = False) -> tuple[str, str]:
    """A pair as calib.txt text: the zero as 0.0000; source requests with 4 decimals; measured values
    with 6 (V) or 7 (A) decimals (B.1 rule 6)."""
    decimals_a, decimals_b = _pair_decimals(kind, first)
    text_a = "0.0000" if first else f"{a:.{decimals_a}f}"
    return text_a, f"{b:.{decimals_b}f}"


def calib_problems(text: str) -> list[CalibProblem]:
    """Every way a calib.txt breaks the rules of B.1 (an empty list: libsmu reads what it says)."""
    problems: list[CalibProblem] = []
    lines = text.replace("\r\n", "\n").split("\n")
    block: list[tuple[int, float]] | None = None
    blocks = 0
    for number, line in enumerate(lines, start=1):
        if block is None:
            # Outside blocks a line is free (a comment) unless it carries a marker libsmu would see.
            if line.strip() == BLOCK_OPEN:
                block = []
            elif BLOCK_OPEN in line:
                problems.append(CalibProblem("marker_in_comment", number, BLOCK_OPEN))
            elif BLOCK_CLOSE in line:
                problems.append(CalibProblem("close_without_open", number))
            continue
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            problems.append(CalibProblem("line_too_long", number, f"{len(line.encode('utf-8'))} bytes"))
        if line.strip() == BLOCK_CLOSE:
            problems.extend(_block_problems(block, blocks, number))
            blocks += 1
            block = None
            continue
        match = _STRICT_PAIR.fullmatch(line)
        if not match:
            problems.append(CalibProblem("not_a_pair", number, line[:40]))
            continue
        block.append((number, float(strtof(match.group(1)))))
    if block is not None:
        problems.append(CalibProblem("block_not_closed", len(lines)))
    if blocks != BLOCKS:
        problems.append(CalibProblem("block_count", len(lines), f"{blocks} blocks, not {BLOCKS}"))
    return problems


def _block_problems(block: list[tuple[int, float]], index: int, close_line: int) -> list[CalibProblem]:
    problems = []
    if not 2 <= len(block) <= MAX_PAIRS:
        problems.append(CalibProblem("pair_count", close_line, f"{len(block)} pairs"))
    if not block:
        return problems
    first_line, first_ref = block[0]
    if first_ref != 0:
        problems.append(CalibProblem("first_pair_not_zero", first_line))
    for line, ref in block[1:]:
        if ref == 0:
            problems.append(CalibProblem("zero_repeated", line))
    kind = hw.RECORD_ORDER[index][1] if index < BLOCKS else None
    if kind in (Kind.MEASURE_I, Kind.SOURCE_I):
        refs = [ref for _, ref in block[1:]]
        if len(block) < 3 or not any(ref > 0 for ref in refs) or not any(ref < 0 for ref in refs):
            problems.append(CalibProblem("current_needs_both_signs", close_line))
    return problems


def format_calib(blocks: Sequence[CalibBlock], header: Sequence[str] = ()) -> str:
    """The text of a calib.txt with 8 blocks, each with its predicted float32 coefficients in the
    comment (P-05). Raises ValueError if the result would not be read as written (B.1)."""
    if len(blocks) != BLOCKS:
        raise ValueError(f"calib.txt needs {BLOCKS} blocks, not {len(blocks)}")
    lines = [f"# {line}" if line else "#" for line in header]
    for (channel, kind), block in zip(hw.RECORD_ORDER, blocks):
        record = block.record()
        lines += [
            "",
            f"# Channel {channel}, {kind.value}   ->  offset {record.offset:.7g}, "
            f"gain+ {record.gain_pos:.7f}, gain- {record.gain_neg:.7f}"
            + (f"  ({block.comment})" if block.comment else ""),
            BLOCK_OPEN,
            *(f"<{a}, {b}>" for a, b in block.pairs),
            BLOCK_CLOSE,
        ]
    text = "\n".join(lines) + "\n"
    problems = calib_problems(text)
    if problems:
        raise ValueError(f"the generated calib.txt would be misread: {problems}")
    if libsmu_reads(text) != [block.record() for block in blocks]:
        raise ValueError("libsmu would compute other records from the generated calib.txt")
    return text


def parse_calib(text: str) -> list[CalibBlock]:
    """The 8 blocks of a calib.txt that passes `calib_problems`; raises ValueError otherwise."""
    problems = calib_problems(text)
    if problems:
        raise ValueError(f"calib.txt breaks the format: {problems}")
    blocks: list[CalibBlock] = []
    pairs: list[tuple[str, str]] = []
    inside = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if not inside:
            if line.strip() == BLOCK_OPEN:
                inside, pairs = True, []
            continue
        if line.strip() == BLOCK_CLOSE:
            blocks.append(CalibBlock(tuple(pairs)))
            inside = False
            continue
        match = _STRICT_PAIR.fullmatch(line)
        assert match is not None          # calib_problems checked every line
        pairs.append((match.group(1), match.group(2)))
    return blocks


def merge_blocks(base: Sequence[CalibBlock], new: dict[int, CalibBlock]) -> list[CalibBlock]:
    """The base's 8 blocks with some replaced: `new` maps a RECORD_ORDER index to its new block."""
    if len(base) != BLOCKS:
        raise ValueError(f"the base has {len(base)} blocks, not {BLOCKS}")
    return [new.get(index, block) for index, block in enumerate(base)]
